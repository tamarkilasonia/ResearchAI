"""Run an evaluation against the real stack: PostgreSQL index, MCP tools, the agent.

Used by the command line (in-process MCP server by default, or ``mcp_url``) and
by ``POST /evaluations/run`` (the MCP server over HTTP). Long runs must survive
infrastructure trouble, so:

* the run refuses to start unless the index holds exactly this corpus with this
  embedder;
* every result is saved as soon as it exists (:class:`RunStore`) and an
  interrupted run is resumed with ``resume`` (missing and failed cases only);
* each case has a hard limit that also covers opening and closing its tool
  session, and in-process runs get a fresh tool server per case, so one broken
  session cannot freeze the run;
* after a failed case the backend is checked; if it is down the run stops
  instead of recording a string of failures.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator, Callable, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import httpx
from mcp.server.mcpserver import MCPServer
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from researchai.agent.llm import LLMClient, build_llm
from researchai.agent.models import ResearchResult, RunStatus
from researchai.agent.runner import ResearchAgent
from researchai.config import Settings
from researchai.corpus.embeddings import Embedder, build_embedder
from researchai.corpus.loader import Corpus
from researchai.db.session import create_engine, session_factory
from researchai.evaluation.dataset import Dataset
from researchai.evaluation.models import Split
from researchai.evaluation.runner import (
    EvaluationError,
    Mode,
    RunCase,
    RunConfig,
    RunReport,
    RunStore,
    check_index,
    new_run_id,
    pending_cases,
    run_cases,
    score_store,
    select_cases,
)
from researchai.evaluation.scoring import CaseScore
from researchai.mcp_client.client import ResearchTools
from researchai.mcp_server.server import ToolServices, build_server
from researchai.tools.documents import DocumentStore
from researchai.tools.search import CorpusSearch

logger = logging.getLogger(__name__)

SESSION_MARGIN_SECONDS = 120.0  # opening/closing a tool session, beyond the agent's own limit
HEALTH_TIMEOUT_SECONDS = 15.0

Target = Callable[[], str | MCPServer]


@dataclass(frozen=True)
class Backend:
    sessions: async_sessionmaker[AsyncSession]
    embedder: Embedder
    manifest: str
    target: Target
    corpus: Corpus
    mcp_url: str | None

    async def healthy(self) -> bool:
        """Database (and MCP server, if remote) answer within a short time."""
        try:
            async with asyncio.timeout(HEALTH_TIMEOUT_SECONDS):
                await check_index(self.sessions, self.corpus, self.embedder.name)
                if self.mcp_url is not None:
                    base = self.mcp_url.rsplit("/", 1)[0]
                    async with httpx.AsyncClient(timeout=HEALTH_TIMEOUT_SECONDS) as client:
                        (await client.get(f"{base}/health")).raise_for_status()
        except Exception:
            logger.exception("evaluation backend check failed")
            return False
        return True


@asynccontextmanager
async def open_backend(
    settings: Settings, corpus: Corpus, *, mcp_url: str | None = None
) -> AsyncIterator[Backend]:
    engine = create_engine(settings)
    try:
        sessions = session_factory(engine)
        embedder = build_embedder(settings)
        manifest = await check_index(sessions, corpus, embedder.name)

        def target() -> str | MCPServer:
            if mcp_url is not None:
                return mcp_url
            return build_server(
                ToolServices(
                    search=CorpusSearch(sessions, embedder), documents=DocumentStore(sessions)
                )
            )

        yield Backend(sessions, embedder, manifest, target, corpus, mcp_url)
    finally:
        await engine.dispose()


def agent_runner(settings: Settings, target: Target, llm: LLMClient | None) -> RunCase:
    limit = settings.run_timeout_seconds + SESSION_MARGIN_SECONDS
    mode: Mode = "offline" if llm is None else "llm"

    def failed(question: str, run_id: str, reason: str) -> ResearchResult:
        return ResearchResult(
            run_id=run_id, question=question, status=RunStatus.FAILED, mode=mode, reason=reason
        )

    async def run(question: str, run_id: str) -> ResearchResult:
        # A session (and in-process server) per case: tool logs and server state never mix.
        try:
            async with (
                asyncio.timeout(limit),
                ResearchTools(target(), settings.tool_timeout_seconds) as tools,
            ):
                return await ResearchAgent(tools=tools, llm=llm, settings=settings).run(
                    question, run_id=run_id
                )
        except TimeoutError:
            return failed(question, run_id, f"case exceeded {limit:g} s including its tool session")
        except Exception as exc:
            logger.exception("evaluation case could not run", extra={"run": run_id})
            return failed(question, run_id, f"tool session failed: {type(exc).__name__}")

    return run


async def evaluate(
    settings: Settings,
    dataset: Dataset,
    corpus: Corpus,
    *,
    split: Split,
    mode: Mode,
    case_ids: Sequence[str] = (),
    limit: int | None = None,
    concurrency: int = 1,
    mcp_url: str | None = None,
    git_commit: str | None = None,
    resume: Path | None = None,
    on_result: Callable[[int, CaseScore], None] | None = None,
) -> tuple[RunReport, Path]:
    """Run or resume an evaluation.

    When resuming, ``split``, ``mode`` and the cases come from the stored run.
    """
    started = datetime.now(UTC)
    if resume is not None:
        store = RunStore(resume)
        config = store.config()
        split, mode = config.split, config.mode
        cases = [c for c in dataset.split(split) if c.id in set(config.case_ids)]
    else:
        cases = select_cases(dataset, split, case_ids, limit)
    llm = build_llm(settings) if mode == "llm" else None
    if mode == "llm" and llm is None:
        raise EvaluationError("mode 'llm' needs LLM_PROVIDER to be set (it is 'none')")
    if not cases:
        raise EvaluationError("no cases selected")

    async with open_backend(settings, corpus, mcp_url=mcp_url) as backend:
        if resume is None:
            config = RunConfig(
                run_id=new_run_id(split, mode, started),
                dataset=dataset.info.name,
                dataset_version=dataset.info.version,
                split=split,
                mode=mode,
                model=llm.model if llm is not None else None,
                embedder=backend.embedder.name,
                search_mode=settings.agent_search_mode,
                corpus_version=corpus.version,
                manifest_sha256=backend.manifest,
                git_commit=git_commit,
                started_at=started,
                finished_at=started,
                case_ids=[c.id for c in cases],
            )
            store = RunStore.create(settings.eval_output_dir, config)
        elif (backend.manifest, backend.embedder.name) != (config.manifest_sha256, config.embedder):
            raise EvaluationError("the index changed since this run started; start a new run")
        else:
            config = config.model_copy(
                update={"resumed": [*config.resumed, f"{started.isoformat()}@{git_commit}"]}
            )
        todo = pending_cases(cases, store.results())

        def save(done: int, score: CaseScore, result: ResearchResult) -> None:
            store.append(score.case_id, result)
            if on_result is not None:
                on_result(done, score)

        async def stop_if_down(_: ResearchResult) -> bool:
            return not await backend.healthy()

        outcomes = await run_cases(
            todo,
            agent_runner(settings, backend.target, llm),
            dataset,
            corpus,
            concurrency=concurrency,
            on_result=save,
            should_stop=stop_if_down,
        )
    config = config.model_copy(update={"finished_at": datetime.now(UTC)})
    report = score_store(
        store, config, dataset, corpus, cost_per_1k_tokens=settings.eval_cost_per_1k_tokens
    )
    if len(outcomes) < len(todo):
        raise EvaluationError(
            f"the backend became unavailable; {len(outcomes)} of {len(todo)} pending case(s) ran. "
            f"Results so far are in {store.path}; fix the backend, then resume with "
            f"--resume {store.path}"
        )
    return report, store.path
