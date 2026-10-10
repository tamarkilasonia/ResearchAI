"""Run an evaluation against the real stack: PostgreSQL index, MCP tools, the agent.

Used by the command line (in-process MCP server, no extra process needed) and
by ``POST /evaluations/run`` (the MCP server over HTTP). The run refuses to
start if the index does not hold exactly this corpus with this embedder.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from mcp.server.mcpserver import MCPServer
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from researchai.agent.llm import LLMClient, build_llm
from researchai.agent.models import ResearchResult
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
    build_report,
    check_index,
    new_run_id,
    run_cases,
    select_cases,
    write_run,
)
from researchai.evaluation.scoring import CaseScore
from researchai.mcp_client.client import ResearchTools
from researchai.mcp_server.server import ToolServices, build_server
from researchai.tools.documents import DocumentStore
from researchai.tools.search import CorpusSearch


@dataclass(frozen=True)
class Backend:
    sessions: async_sessionmaker[AsyncSession]
    embedder: Embedder
    manifest: str
    target: str | MCPServer


@asynccontextmanager
async def open_backend(
    settings: Settings, corpus: Corpus, *, mcp_url: str | None = None
) -> AsyncIterator[Backend]:
    engine = create_engine(settings)
    try:
        sessions = session_factory(engine)
        embedder = build_embedder(settings)
        manifest = await check_index(sessions, corpus, embedder.name)
        target: str | MCPServer = mcp_url or build_server(
            ToolServices(search=CorpusSearch(sessions, embedder), documents=DocumentStore(sessions))
        )
        yield Backend(sessions, embedder, manifest, target)
    finally:
        await engine.dispose()


def agent_runner(settings: Settings, target: str | MCPServer, llm: LLMClient | None) -> RunCase:
    async def run(question: str, run_id: str) -> ResearchResult:
        # A session per case: tool-call logs never mix, even when cases run concurrently.
        async with ResearchTools(target, settings.tool_timeout_seconds) as tools:
            return await ResearchAgent(tools=tools, llm=llm, settings=settings).run(
                question, run_id=run_id
            )

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
    on_result: Callable[[int, CaseScore], None] | None = None,
) -> tuple[RunReport, Path]:
    llm = build_llm(settings) if mode == "llm" else None
    if mode == "llm" and llm is None:
        raise EvaluationError("mode 'llm' needs LLM_PROVIDER to be set (it is 'none')")
    cases = select_cases(dataset, split, case_ids, limit)
    if not cases:
        raise EvaluationError("no cases selected")
    started = datetime.now(UTC)
    async with open_backend(settings, corpus, mcp_url=mcp_url) as backend:
        outcomes = await run_cases(
            cases,
            agent_runner(settings, backend.target, llm),
            dataset,
            corpus,
            concurrency=concurrency,
            on_result=on_result,
        )
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
        finished_at=datetime.now(UTC),
        case_ids=[c.id for c in cases],
    )
    report = build_report(config, outcomes, cost_per_1k_tokens=settings.eval_cost_per_1k_tokens)
    path = write_run(report, [result for _, result in outcomes], settings.eval_output_dir)
    return report, path
