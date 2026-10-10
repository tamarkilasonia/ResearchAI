"""The API against PostgreSQL, and end to end over the real MCP tool server."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator
from typing import Any

import httpx
import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from researchai.agent.models import ResearchResult, RunStatus
from researchai.api.app import create_app
from researchai.api.evaluations import EvaluationRequest, EvaluationStore
from researchai.api.runs import INTERRUPTED, PostgresRunRepository
from tests.integration.conftest import scratch_database_url, serving_real_corpus
from tests.support import make_settings

API_DATABASE_URL = scratch_database_url("_test_api")


async def test_run_repository_lifecycle(sessions: async_sessionmaker[AsyncSession]) -> None:
    repo = PostgresRunRepository(sessions)
    record = await repo.create("a" * 32, "How many chargers?")
    assert record.status == "queued"
    await repo.mark_running(record.id)
    result = ResearchResult(run_id=record.id, question="How many chargers?",
                            status=RunStatus.ANSWERED, mode="offline", total_ms=12.5)  # fmt: skip
    await repo.complete(record.id, result)
    stored = await repo.get(record.id)
    assert stored is not None
    assert stored.status == "completed"
    assert stored.result == result
    assert stored.started_at is not None
    assert stored.finished_at is not None
    assert await repo.get("b" * 32) is None
    await repo.create("c" * 32, "Pending?")
    await repo.create("d" * 32, "Failing?")
    await repo.fail("d" * 32, "internal error")
    assert await repo.interrupt_unfinished() == 1
    interrupted = await repo.get("c" * 32)
    assert interrupted is not None
    assert (interrupted.status, interrupted.error) == ("failed", INTERRUPTED)
    assert [r.id for r in await repo.recent()] == ["d" * 32, "c" * 32, "a" * 32]


async def test_evaluation_store(sessions: async_sessionmaker[AsyncSession]) -> None:
    store = EvaluationStore(sessions)
    started = await store.start(EvaluationRequest(split="dev", mode="offline"))
    assert started.status == "running"
    done = await store.finish(started.id, summary={"cases": 2})
    assert (done.status, done.summary) == ("completed", {"cases": 2})
    failed = await store.finish((await store.start(EvaluationRequest())).id, error="boom")
    assert failed.status == "failed"
    assert [r.id for r in await store.list()] == [failed.id, done.id]


@pytest_asyncio.fixture(scope="module", loop_scope="module")
async def mcp_url() -> AsyncIterator[str]:
    async with serving_real_corpus(API_DATABASE_URL) as url:
        yield url


@pytest.mark.asyncio(loop_scope="module")
async def test_research_end_to_end_through_the_api(mcp_url: str) -> None:
    root = logging.getLogger()
    saved = root.handlers[:], root.level
    settings = make_settings(
        database_url=API_DATABASE_URL, mcp_server_url=mcp_url, embedding_dim=64
    )
    app = create_app(settings)
    try:
        async with app.router.lifespan_context(app):
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(transport=transport, base_url="http://api") as client:
                health = await client.get("/health")
                response = await client.post(
                    "/research?wait=true",
                    json={"question": "How many public charging points are in the registry?"},
                )
                stored = await client.get(response.headers["Location"])
    finally:
        root.handlers[:] = saved[0]
        root.setLevel(saved[1])
    assert health.status_code == 200
    assert health.json()["components"] == {"database": "ok", "tool_server": "ok"}
    assert response.status_code == 200
    body: dict[str, Any] = response.json()
    assert body["status"] == "completed"
    assert body["result"]["status"] == "answered"
    assert body["result"]["citation_report"]["precision"] == 1.0
    assert any(call["tool"] == "search_corpus" for call in body["result"]["tool_calls"])
    assert "## Sources" in body["report_markdown"]
    assert stored.json()["result"] == body["result"]  # read back from PostgreSQL


@pytest.mark.asyncio(loop_scope="module")
async def test_health_reports_an_unreachable_tool_server(mcp_url: str) -> None:
    settings = make_settings(
        database_url=API_DATABASE_URL, mcp_server_url="http://127.0.0.1:9/mcp", embedding_dim=64
    )
    app = create_app(settings)
    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://api") as client:
            health = await client.get("/health")
    assert health.status_code == 503
    assert health.json()["components"] == {"database": "ok", "tool_server": "error"}
    await asyncio.sleep(0)
