"""In-memory fakes for API tests."""

from __future__ import annotations

import asyncio
import uuid
from datetime import UTC, datetime
from typing import Any

from researchai.agent.models import ResearchResult, RunStatus
from researchai.api.evaluations import EvaluationRecord, EvaluationRequest
from researchai.api.runs import INTERRUPTED, RunRecord





class MemoryRuns:
    def __init__(self) -> None:
        self.records: dict[str, RunRecord] = {}

    async def create(self, run_id: str, question: str) -> RunRecord:
        record = RunRecord(
            id=run_id, question=question, status="queued", created_at=datetime.now(UTC)
        )
        self.records[run_id] = record
        return record

    def _set(self, run_id: str, **values: Any) -> None:
        self.records[run_id] = self.records[run_id].model_copy(update=values)

    async def mark_running(self, run_id: str) -> None:
        self._set(run_id, status="running", started_at=datetime.now(UTC))

    async def complete(self, run_id: str, result: ResearchResult) -> None:
        self._set(run_id, status="completed", result=result, finished_at=datetime.now(UTC))

    async def fail(self, run_id: str, error: str) -> None:
        self._set(run_id, status="failed", error=error, finished_at=datetime.now(UTC))

    async def get(self, run_id: str) -> RunRecord | None:
        return self.records.get(run_id)

    
    async def interrupt_unfinished(self) -> int:
        unfinished = [r.id for r in self.records.values() if r.status in ("queued", "running")]
        for run_id in unfinished:
            await self.fail(run_id, INTERRUPTED)
        return len(unfinished)


def result_for(question: str, run_id: str) -> ResearchResult:
    return ResearchResult(
        run_id=run_id, question=question, status=RunStatus.ABSTAINED, mode="offline"
    )


class FakeRunner:
    def __init__(self, delay: float = 0.0, error: Exception | None = None) -> None:
        self.delay = delay
        self.error = error
        self.started: list[str] = []

    async def __call__(self, question: str, run_id: str) -> ResearchResult:
        self.started.append(run_id)
        await asyncio.sleep(self.delay)
        if self.error is not None:
            raise self.error
        return result_for(question, run_id)


class MemoryEvaluations:
    def __init__(self) -> None:
        self.records: dict[str, EvaluationRecord] = {}

    async def list(self, limit: int = 50) -> list[EvaluationRecord]:
        return sorted(self.records.values(), key=lambda r: r.created_at, reverse=True)[:limit]

    async def start(self, request: EvaluationRequest) -> EvaluationRecord:
        record = EvaluationRecord(
            id=uuid.uuid4().hex, dataset=request.dataset, split=request.split, mode=request.mode,
            status="running", created_at=datetime.now(UTC),
        )  # fmt: skip
        self.records[record.id] = record
        return record

    async def finish(
        self, run_id: str, *, summary: dict[str, Any] | None = None, error: str | None = None
    ) -> EvaluationRecord:
        update = {
            "status": "failed" if error else "completed",
            "summary": summary,
            "error": error,
            "finished_at": datetime.now(UTC),
        }
        record = self.records[run_id].model_copy(update=update)
        self.records[run_id] = record
        return record
