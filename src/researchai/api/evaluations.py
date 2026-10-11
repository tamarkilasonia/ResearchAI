"""Evaluation runs started over the API, and their stored summaries.

The runner (:func:`dataset_runner`) evaluates a dataset split against the
running MCP tool server and stores the metrics; the full run (results and
report) is written under ``EVAL_OUTPUT_DIR``. Without a runner
(``AppServices.evaluation_runner is None``) ``POST /evaluations/run`` answers
503 ``EVALUATION_UNAVAILABLE`` instead of pretending to evaluate anything.
"""

from __future__ import annotations

import uuid
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from typing import Any, Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from researchai.config import Settings
from researchai.corpus.loader import load_corpus
from researchai.db.models import EvaluationRunRow
from researchai.evaluation.dataset import load_dataset, validate
from researchai.evaluation.live import evaluate
from researchai.evaluation.runner import EvaluationError

Split = Literal["dev", "holdout"]
Mode = Literal["offline", "llm"]


class EvaluationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dataset: str = Field(default="default", pattern=r"^[a-z0-9_-]{1,64}$")
    split: Split = "dev"
    mode: Mode = "offline"


class EvaluationRecord(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str
    dataset: str
    split: Split
    mode: Mode
    status: Literal["running", "completed", "failed"]
    created_at: datetime
    finished_at: datetime | None = None
    error: str | None = None
    summary: dict[str, Any] | None = None


EvaluationRunner = Callable[[EvaluationRequest], Awaitable[dict[str, Any]]]


class EvaluationStoreProtocol(Protocol):
    async def list(self, limit: int = 50) -> list[EvaluationRecord]: ...
    async def start(self, request: EvaluationRequest) -> EvaluationRecord: ...
    async def finish(
        self, run_id: str, *, summary: dict[str, Any] | None = None, error: str | None = None
    ) -> EvaluationRecord: ...


def _record(row: EvaluationRunRow) -> EvaluationRecord:
    return EvaluationRecord(
        id=row.id,
        dataset=row.dataset,
        split=row.split,  # type: ignore[arg-type]
        mode=row.mode,  # type: ignore[arg-type]
        status=row.status,  # type: ignore[arg-type]
        created_at=row.created_at,
        finished_at=row.finished_at,
        error=row.error,
        summary=row.summary,
    )


class EvaluationStore:
    def __init__(self, sessions: async_sessionmaker[AsyncSession]) -> None:
        self._sessions = sessions

    async def list(self, limit: int = 50) -> list[EvaluationRecord]:
        async with self._sessions() as session:
            rows = await session.scalars(
                select(EvaluationRunRow).order_by(EvaluationRunRow.created_at.desc()).limit(limit)
            )
            return [_record(row) for row in rows.all()]

    async def start(self, request: EvaluationRequest) -> EvaluationRecord:
        async with self._sessions.begin() as session:
            row = EvaluationRunRow(
                id=uuid.uuid4().hex,
                dataset=request.dataset,
                split=request.split,
                mode=request.mode,
                status="running",
                created_at=datetime.now(UTC),
            )
            session.add(row)
        return _record(row)

    async def finish(
        self, run_id: str, *, summary: dict[str, Any] | None = None, error: str | None = None
    ) -> EvaluationRecord:
        async with self._sessions.begin() as session:
            await session.execute(
                update(EvaluationRunRow)
                .where(EvaluationRunRow.id == run_id)
                .values(
                    status="failed" if error else "completed",
                    summary=summary,
                    error=error,
                    finished_at=datetime.now(UTC),
                )
            )
            row = await session.get(EvaluationRunRow, run_id, populate_existing=True)
        assert row is not None  # noqa: S101 - created by start() in this process
        return _record(row)


def dataset_runner(settings: Settings) -> EvaluationRunner:
    """Evaluate the requested dataset split through the MCP server at ``MCP_SERVER_URL``."""

    async def run(request: EvaluationRequest) -> dict[str, Any]:
        dataset = load_dataset(request.dataset, settings.eval_dir)
        corpus = load_corpus(settings.corpus_dir)
        problems = validate(dataset, corpus)
        if problems:
            raise EvaluationError(f"dataset {request.dataset} is invalid: {problems[:3]}")
        report, path = await evaluate(
            settings,
            dataset,
            corpus,
            split=request.split,
            mode=request.mode,
            mcp_url=settings.mcp_server_url,
        )
        return {
            "run_id": report.config.run_id,
            "path": str(path),
            "cases": report.metrics.cases,
            "metrics": report.metrics.model_dump(
                mode="json", exclude={"by_category", "by_language"}
            ),
        }

    return run
