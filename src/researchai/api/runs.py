"""Research runs: storage and background execution.

A run is stored as soon as it is requested (``queued``), executed in the
background under a concurrency limit (``running``), and finished with its full
result, the audit trail, as JSONB (``completed``), or with a safe error message
(``failed``). Runs a previous process left unfinished are marked failed on
startup, so nothing stays "running" forever.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import uuid
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from typing import Literal, Protocol

from pydantic import BaseModel, ConfigDict
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from researchai.agent.models import ResearchResult, RunStatus
from researchai.db.models import ResearchRunRow

logger = logging.getLogger(__name__)

RunState = Literal["queued", "running", "completed", "failed"]
INTERRUPTED = "interrupted: the server stopped before the run finished"


class RunRecord(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str
    question: str
    status: RunState
    created_at: datetime
    started_at: datetime | None = None
    finished_at: datetime | None = None
    error: str | None = None
    result: ResearchResult | None = None


class RunRepository(Protocol):
    async def create(self, run_id: str, question: str) -> RunRecord: ...
    async def mark_running(self, run_id: str) -> None: ...
    async def complete(self, run_id: str, result: ResearchResult) -> None: ...
    async def fail(self, run_id: str, error: str) -> None: ...
    async def get(self, run_id: str) -> RunRecord | None: ...
    async def interrupt_unfinished(self) -> int: ...


def _now() -> datetime:
    return datetime.now(UTC)


def _record(row: ResearchRunRow) -> RunRecord:
    return RunRecord(
        id=row.id,
        question=row.question,
        status=row.status,  # type: ignore[arg-type]  # constrained by the table's CHECK
        created_at=row.created_at,
        started_at=row.started_at,
        finished_at=row.finished_at,
        error=row.error,
        result=ResearchResult.model_validate(row.result) if row.result is not None else None,
    )


class PostgresRunRepository:
    def __init__(self, sessions: async_sessionmaker[AsyncSession]) -> None:
        self._sessions = sessions

    async def create(self, run_id: str, question: str) -> RunRecord:
        async with self._sessions.begin() as session:
            row = ResearchRunRow(id=run_id, question=question, status="queued", created_at=_now())
            session.add(row)
        return _record(row)

    async def _update(self, run_id: str, **values: object) -> None:
        async with self._sessions.begin() as session:
            await session.execute(
                update(ResearchRunRow).where(ResearchRunRow.id == run_id).values(**values)
            )

    async def mark_running(self, run_id: str) -> None:
        await self._update(run_id, status="running", started_at=_now())

    async def complete(self, run_id: str, result: ResearchResult) -> None:
        await self._update(
            run_id,
            status="completed",
            outcome=result.status.value,
            mode=result.mode,
            total_ms=result.total_ms,
            result=result.model_dump(mode="json"),
            finished_at=_now(),
        )

    async def fail(self, run_id: str, error: str) -> None:
        await self._update(
            run_id,
            status="failed",
            outcome=RunStatus.FAILED.value,
            error=error,
            finished_at=_now(),
        )

    async def get(self, run_id: str) -> RunRecord | None:
        async with self._sessions() as session:
            row = await session.get(ResearchRunRow, run_id)
        return None if row is None else _record(row)

    async def interrupt_unfinished(self) -> int:
        async with self._sessions.begin() as session:
            result = await session.execute(
                update(ResearchRunRow)
                .where(ResearchRunRow.status.in_(["queued", "running"]))
                .values(status="failed", outcome="failed", error=INTERRUPTED, finished_at=_now())
                .returning(ResearchRunRow.id)
            )
            return len(result.all())

    async def recent(self, limit: int = 20) -> list[RunRecord]:
        async with self._sessions() as session:
            rows = await session.scalars(
                select(ResearchRunRow).order_by(ResearchRunRow.created_at.desc()).limit(limit)
            )
            return [_record(row) for row in rows.all()]


AgentRunner = Callable[[str, str], Awaitable[ResearchResult]]
"""``(question, run_id) -> result``; must not raise for research outcomes."""


class RunService:
    def __init__(
        self, repository: RunRepository, runner: AgentRunner, *, max_concurrent: int
    ) -> None:
        self._repository = repository
        self._runner = runner
        self._slots = asyncio.Semaphore(max_concurrent)
        self._tasks: dict[str, asyncio.Task[None]] = {}

    async def submit(self, question: str) -> RunRecord:
        run_id = uuid.uuid4().hex
        record = await self._repository.create(run_id, question)
        task = asyncio.create_task(self._execute(run_id, question), name=f"research-{run_id}")
        self._tasks[run_id] = task
        task.add_done_callback(lambda _: self._tasks.pop(run_id, None))
        return record

    async def _execute(self, run_id: str, question: str) -> None:
        async with self._slots:
            try:
                await self._repository.mark_running(run_id)
                result = await self._runner(question, run_id)
                await self._repository.complete(run_id, result)
            except asyncio.CancelledError:
                await self._repository.fail(run_id, INTERRUPTED)
                raise
            except Exception:
                logger.exception("research run failed", extra={"run": run_id})
                await self._repository.fail(run_id, "internal error")

    async def wait(self, run_id: str, seconds: float) -> None:
        """Wait up to ``seconds`` for a run started by this process (never cancels it)."""
        task = self._tasks.get(run_id)
        if task is None:
            return
        with contextlib.suppress(TimeoutError):
            async with asyncio.timeout(seconds):
                await asyncio.shield(task)

    @property
    def active(self) -> int:
        return len(self._tasks)

    async def shutdown(self) -> None:
        tasks = list(self._tasks.values())
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
