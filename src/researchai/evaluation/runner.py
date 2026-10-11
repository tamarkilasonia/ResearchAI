"""Run the agent over dataset cases, score every result, and write the run artefacts.

A run directory holds ``config.json``, ``results.jsonl`` (every full research
result, appended as it finishes: the audit trail), ``summary.json``
(configuration, metrics, per-case scores) and ``report.md`` (tables and failed
cases). An interrupted run is resumed from its directory.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import uuid
from collections.abc import Awaitable, Callable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from researchai.agent.models import ResearchResult, RunStatus
from researchai.corpus.ingest import latest_build
from researchai.corpus.loader import Corpus
from researchai.evaluation.dataset import Dataset
from researchai.evaluation.judge import judge_summary, read_judgements
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
    search_mode: str = "unrecorded"  # runs written before the mode was recorded
    corpus_version: str
    manifest_sha256: str
    git_commit: str | None
    started_at: datetime
    finished_at: datetime
    case_ids: list[str]
    resumed: list[str] = Field(
        default_factory=list, description="'<time>@<commit>' of every resumed segment"
    )


class JudgeCriterion(_Model):
    n: int
    mean: float = Field(description="Mean judge score, 0..2")
    kappa: float | None = None
    weighted_kappa: float | None = None
    reliable: bool | None = Field(default=None, description="None: not calibrated yet")


class JudgeSection(_Model):
    model: str
    calibrated_on: int = Field(description="Human-labelled items behind the kappas (0: none)")
    criteria: dict[str, JudgeCriterion]


class RunReport(_Model):
    config: RunConfig
    metrics: RunMetrics
    scores: list[CaseScore]
    judge: JudgeSection | None = None


CALIBRATION_FILE = "calibration.json"


def judge_section(path: Path) -> JudgeSection | None:
    """Judge means for a run, with the calibration kappas if the run has been calibrated."""
    judgements = read_judgements(path)
    if not judgements:
        return None
    calibration: dict[str, Any] = {}
    if (path / CALIBRATION_FILE).exists():
        calibration = json.loads((path / CALIBRATION_FILE).read_text("utf-8"))
    by_criterion = {row["criterion"]: row for row in calibration.get("criteria", [])}
    summary = judge_summary(judgements)
    return JudgeSection(
        model=next(iter(judgements.values())).model,
        calibrated_on=int(calibration.get("items", 0)),
        criteria={
            criterion: JudgeCriterion(
                n=int(values["n"]),
                mean=float(values["mean"]),
                kappa=by_criterion.get(criterion, {}).get("kappa"),
                weighted_kappa=by_criterion.get(criterion, {}).get("weighted_kappa"),
                reliable=by_criterion.get(criterion, {}).get("reliable"),
            )
            for criterion, values in summary.items()
        },
    )


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


StopCheck = Callable[[ResearchResult], Awaitable[bool]]
"""Called after a case fails; returns True when the run must stop (backend down)."""


async def run_cases(
    cases: Sequence[Case],
    run_case: RunCase,
    dataset: Dataset,
    corpus: Corpus,
    *,
    concurrency: int = 1,
    on_result: Callable[[int, CaseScore, ResearchResult], None] | None = None,
    should_stop: StopCheck | None = None,
) -> list[tuple[CaseScore, ResearchResult]]:
    """Run and score cases. After ``should_stop`` says so, cases not yet started are skipped."""
    traps = {t.id: t for t in dataset.traps}
    bodies = {d.id: d.body for d in corpus.documents}
    slots = asyncio.Semaphore(concurrency)
    stopped = asyncio.Event()
    outcomes: list[tuple[CaseScore, ResearchResult]] = []

    async def one(case: Case) -> None:
        async with slots:
            if stopped.is_set():
                return
            result = await run_case(case.question, f"eval-{case.id}-{uuid.uuid4().hex[:8]}")
        score = score_case(case, result, traps, bodies)
        outcomes.append((score, result))
        if on_result is not None:
            on_result(len(outcomes), score, result)
        if (
            result.status is RunStatus.FAILED
            and should_stop is not None
            and await should_stop(result)
        ):
            stopped.set()

    await asyncio.gather(*(one(case) for case in cases))
    order = {case.id: i for i, case in enumerate(cases)}
    return sorted(outcomes, key=lambda pair: order[pair[0].case_id])


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


class RunStore:
    """A run directory, written as the run goes so a crash loses at most one case.

    ``config.json`` is written first, every result is appended to
    ``results.jsonl`` as soon as it exists (a re-run case's newer line wins), and
    ``summary.json`` / ``report.md`` are (re)written at the end.
    """

    def __init__(self, path: Path) -> None:
        self.path = path

    @classmethod
    def create(cls, out_dir: Path, config: RunConfig) -> RunStore:
        store = cls(out_dir / config.run_id)
        store.path.mkdir(parents=True, exist_ok=False)
        store.write_config(config)
        (store.path / "results.jsonl").touch()
        return store

    def config(self) -> RunConfig:
        return RunConfig.model_validate_json((self.path / "config.json").read_text("utf-8"))

    def write_config(self, config: RunConfig) -> None:
        (self.path / "config.json").write_text(
            config.model_dump_json(indent=2) + "\n", encoding="utf-8"
        )

    def append(self, case_id: str, result: ResearchResult) -> None:
        line = json.dumps({"case_id": case_id, "result": result.model_dump(mode="json")})
        with (self.path / "results.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")
            handle.flush()
            os.fsync(handle.fileno())

    def results(self) -> dict[str, ResearchResult]:
        """Latest result per case."""
        latest: dict[str, ResearchResult] = {}
        with (self.path / "results.jsonl").open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    row = json.loads(line)
                    latest[row["case_id"]] = ResearchResult.model_validate(row["result"])
        return latest

    def write_report(self, report: RunReport) -> None:
        self.write_config(report.config)
        (self.path / "summary.json").write_text(
            report.model_dump_json(indent=2) + "\n", encoding="utf-8"
        )
        (self.path / "report.md").write_text(render_report(report), encoding="utf-8")


def score_store(
    store: RunStore,
    config: RunConfig,
    dataset: Dataset,
    corpus: Corpus,
    *,
    cost_per_1k_tokens: float = 0.0,
) -> RunReport:
    """Score the latest result of every case in the run, in the run's case order."""
    cases = {c.id: c for c in dataset.cases}
    traps = {t.id: t for t in dataset.traps}
    bodies = {d.id: d.body for d in corpus.documents}
    results = store.results()
    unknown = sorted(set(results) - set(cases))
    if unknown:
        raise EvaluationError(f"cases no longer in the dataset: {unknown}")
    outcomes = [
        (score_case(cases[case_id], results[case_id], traps, bodies), results[case_id])
        for case_id in config.case_ids
        if case_id in results
    ]
    report = build_report(config, outcomes, cost_per_1k_tokens=cost_per_1k_tokens)
    report = report.model_copy(update={"judge": judge_section(store.path)})
    store.write_report(report)
    return report


def rescore(
    path: Path, dataset: Dataset, corpus: Corpus, *, cost_per_1k_tokens: float = 0.0
) -> RunReport:
    """Score a finished run again with the current scorers (the agent is not re-run)."""
    store = RunStore(path)
    return score_store(
        store, store.config(), dataset, corpus, cost_per_1k_tokens=cost_per_1k_tokens
    )


def pending_cases(cases: Sequence[Case], done: Mapping[str, ResearchResult]) -> list[Case]:
    """Cases without a result, or whose result is an infrastructure failure."""
    return [c for c in cases if c.id not in done or done[c.id].status is RunStatus.FAILED]


def new_run_id(split: Split, mode: Mode, now: datetime | None = None) -> str:
    stamp = (now or datetime.now(UTC)).strftime("%Y%m%dT%H%M%SZ")
    return f"{stamp}-{split}-{mode}"
