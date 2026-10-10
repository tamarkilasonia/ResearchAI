"""Run the agent over dataset cases, score every result, and write the run artefacts.

A run directory holds ``summary.json`` (configuration, metrics, per-case scores),
``results.jsonl`` (every full research result: the audit trail) and
``report.md`` (tables and failed cases).
"""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from collections.abc import Awaitable, Callable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from researchai.agent.models import ResearchResult
from researchai.corpus.ingest import latest_build
from researchai.corpus.loader import Corpus
from researchai.evaluation.dataset import Dataset
from researchai.evaluation.metrics import RunMetrics, aggregate
from researchai.evaluation.models import Case, Split
from researchai.evaluation.report import render_report
from researchai.evaluation.scoring import CaseScore, score_case

logger = logging.getLogger(__name__)

Mode = Literal["offline", "llm"]
RunCase = Callable[[str, str], Awaitable[ResearchResult]]
"""``(question, run_id) -> result`` (the agent; must not raise for research outcomes)."""


class EvaluationError(RuntimeError):
    """The run cannot produce trustworthy numbers (stale index, wrong configuration)."""


class _Model(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class RunConfig(_Model):
    run_id: str
    dataset: str
    dataset_version: str
    split: Split
    mode: Mode
    model: str | None
    embedder: str
    search_mode: str
    corpus_version: str
    manifest_sha256: str
    git_commit: str | None
    started_at: datetime
    finished_at: datetime
    case_ids: list[str]


class RunReport(_Model):
    config: RunConfig
    metrics: RunMetrics
    scores: list[CaseScore]


async def check_index(
    sessions: async_sessionmaker[AsyncSession], corpus: Corpus, embedder: str
) -> str:
    """The ingested index must be this corpus, embedded with this embedder; returns its hash."""
    async with sessions() as session:
        build = await latest_build(session)
    manifest = str(corpus.manifest()["sha256"])
    if build is None:
        raise EvaluationError(
            "the corpus is not ingested; run `python -m researchai.corpus ingest`"
        )
    if (build.manifest_sha256, build.embedder) != (manifest, embedder):
        raise EvaluationError(
            f"the index holds corpus {build.corpus_version} embedded with {build.embedder}; "
            f"this run needs corpus {corpus.version} with {embedder}: re-run ingestion"
        )
    return manifest


def select_cases(
    dataset: Dataset, split: Split, case_ids: Sequence[str] = (), limit: int | None = None
) -> list[Case]:
    cases = dataset.split(split)
    if case_ids:
        wanted = set(case_ids)
        unknown = wanted - {c.id for c in cases}
        if unknown:
            raise EvaluationError(f"not in the {split} split: {sorted(unknown)}")
        cases = [c for c in cases if c.id in wanted]
    return cases[:limit] if limit is not None else cases


async def run_cases(
    cases: Sequence[Case],
    run_case: RunCase,
    dataset: Dataset,
    corpus: Corpus,
    *,
    concurrency: int = 1,
    on_result: Callable[[int, CaseScore], None] | None = None,
) -> list[tuple[CaseScore, ResearchResult]]:
    traps = {t.id: t for t in dataset.traps}
    bodies = {d.id: d.body for d in corpus.documents}
    slots = asyncio.Semaphore(concurrency)
    done = 0

    async def one(case: Case) -> tuple[CaseScore, ResearchResult]:
        nonlocal done
        async with slots:
            result = await run_case(case.question, f"eval-{case.id}-{uuid.uuid4().hex[:8]}")
        score = score_case(case, result, traps, bodies)
        done += 1
        if on_result is not None:
            on_result(done, score)
        return score, result

    return list(await asyncio.gather(*(one(case) for case in cases)))


def build_report(
    config: RunConfig,
    outcomes: Sequence[tuple[CaseScore, ResearchResult]],
    *,
    cost_per_1k_tokens: float = 0.0,
) -> RunReport:
    scores = [score for score, _ in outcomes]
    return RunReport(
        config=config,
        metrics=aggregate(scores, cost_per_1k_tokens=cost_per_1k_tokens),
        scores=scores,
    )


def write_run(report: RunReport, results: Sequence[ResearchResult], out_dir: Path) -> Path:
    path = out_dir / report.config.run_id
    path.mkdir(parents=True, exist_ok=False)
    (path / "summary.json").write_text(report.model_dump_json(indent=2) + "\n", encoding="utf-8")
    with (path / "results.jsonl").open("w", encoding="utf-8") as handle:
        for score, result in zip(report.scores, results, strict=True):
            handle.write(
                f'{{"case_id": "{score.case_id}", "result": {result.model_dump_json()}}}\n'
            )
    (path / "report.md").write_text(render_report(report), encoding="utf-8")
    return path


def read_results(path: Path) -> list[tuple[str, ResearchResult]]:
    rows: list[tuple[str, ResearchResult]] = []
    with (path / "results.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            rows.append((row["case_id"], ResearchResult.model_validate(row["result"])))
    return rows


def rescore(
    path: Path, dataset: Dataset, corpus: Corpus, *, cost_per_1k_tokens: float = 0.0
) -> RunReport:
    """Score a finished run again with the current scorers (the agent is not re-run)."""
    previous = RunReport.model_validate_json((path / "summary.json").read_text(encoding="utf-8"))
    cases = {c.id: c for c in dataset.cases}
    traps = {t.id: t for t in dataset.traps}
    bodies = {d.id: d.body for d in corpus.documents}
    rows = read_results(path)
    unknown = [case_id for case_id, _ in rows if case_id not in cases]
    if unknown:
        raise EvaluationError(f"cases no longer in the dataset: {unknown}")
    outcomes = [(score_case(cases[i], r, traps, bodies), r) for i, r in rows]
    report = build_report(previous.config, outcomes, cost_per_1k_tokens=cost_per_1k_tokens)
    (path / "summary.json").write_text(report.model_dump_json(indent=2) + "\n", encoding="utf-8")
    (path / "report.md").write_text(render_report(report), encoding="utf-8")
    return report


def new_run_id(split: Split, mode: Mode, now: datetime | None = None) -> str:
    stamp = (now or datetime.now(UTC)).strftime("%Y%m%dT%H%M%SZ")
    return f"{stamp}-{split}-{mode}"
