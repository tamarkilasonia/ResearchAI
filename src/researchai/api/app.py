"""FastAPI application.

    POST /research              start a research run (202 + id; ?wait=true blocks)
    GET  /research/{id}         status, full result with audit trail, Markdown report
    GET  /evaluations           evaluation runs, newest first
    POST /evaluations/run       evaluate a dataset split in the background
    GET  /health                database and MCP tool server status

Run with ``uvicorn researchai.api.app:create_app --factory``. Tests pass
``services`` directly; otherwise the lifespan builds them from settings.
"""

from __future__ import annotations

import asyncio
import logging
import re
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Annotated, Literal

import httpx
from fastapi import FastAPI, Query, Request, Response
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import text

from researchai import __version__
from researchai.agent.llm import build_llm
from researchai.agent.models import ResearchResult
from researchai.agent.render import render_markdown
from researchai.agent.runner import ResearchAgent
from researchai.api import errors
from researchai.api.errors import ApiError, ErrorResponse
from researchai.api.evaluations import (
    EvaluationRecord,
    EvaluationRequest,
    EvaluationRunner,
    EvaluationStore,
    EvaluationStoreProtocol,
    dataset_runner,
)
from researchai.api.runs import (
    PostgresRunRepository,
    RunRecord,
    RunRepository,
    RunService,
)
from researchai.config import LLMProvider, Settings, get_settings
from researchai.db.session import create_engine, session_factory
from researchai.log import configure_logging
from researchai.mcp_client.client import ResearchTools

logger = logging.getLogger(__name__)

_RUN_ID = re.compile(r"^[0-9a-f]{32}$")
MAX_QUESTION_CHARS = 2000

HealthCheck = Callable[[], Awaitable[bool]]


@dataclass
class AppServices:
    runs: RunService
    repository: RunRepository
    evaluations: EvaluationStoreProtocol | None
    evaluation_runner: EvaluationRunner | None = None
    health_checks: dict[str, HealthCheck] = field(default_factory=dict)
    close: Callable[[], Awaitable[None]] | None = None
    background: set[asyncio.Task[None]] = field(default_factory=set)


# ----------------------------------------------------------------- schemas


class ResearchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    question: str = Field(min_length=3, max_length=MAX_QUESTION_CHARS)


class RunView(BaseModel):
    id: str
    status: Literal["queued", "running", "completed", "failed"]
    question: str
    created_at: str
    started_at: str | None
    finished_at: str | None
    error: str | None
    result: ResearchResult | None
    report_markdown: str | None

    @classmethod
    def of(cls, record: RunRecord) -> RunView:
        return cls(
            id=record.id,
            status=record.status,
            question=record.question,
            created_at=record.created_at.isoformat(),
            started_at=record.started_at.isoformat() if record.started_at else None,
            finished_at=record.finished_at.isoformat() if record.finished_at else None,
            error=record.error,
            result=record.result,
            report_markdown=render_markdown(record.result) if record.result else None,
        )


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    version: str
    llm_provider: LLMProvider
    components: dict[str, Literal["ok", "error"]]


ERRORS: dict[int | str, dict[str, object]] = {
    code: {"model": ErrorResponse} for code in (404, 422, 500, 503)
}


# ----------------------------------------------------------------- production wiring


def build_services(settings: Settings) -> AppServices:
    engine = create_engine(settings)
    sessions = session_factory(engine)
    repository = PostgresRunRepository(sessions)
    llm = build_llm(settings)

    async def run_agent(question: str, run_id: str) -> ResearchResult:
        # One MCP session per run: tool-call logs of concurrent runs never mix.
        async with ResearchTools(settings.mcp_server_url, settings.tool_timeout_seconds) as tools:
            agent = ResearchAgent(tools=tools, llm=llm, settings=settings)
            return await agent.run(question, run_id=run_id)

    async def database() -> bool:
        async with engine.connect() as connection:
            await connection.execute(text("SELECT 1"))
        return True

    async def tool_server() -> bool:
        base = settings.mcp_server_url.rsplit("/", 1)[0]
        async with httpx.AsyncClient(timeout=3) as client:
            return (await client.get(f"{base}/health")).status_code == 200

    return AppServices(
        runs=RunService(repository, run_agent, max_concurrent=settings.api_max_concurrent_runs),
        repository=repository,
        evaluations=EvaluationStore(sessions),
        evaluation_runner=dataset_runner(settings),
        health_checks={"database": database, "tool_server": tool_server},
        close=engine.dispose,
    )


# ----------------------------------------------------------------- app


def create_app(settings: Settings | None = None, services: AppServices | None = None) -> FastAPI:
    settings = settings or get_settings()
    configure_logging(settings)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        owned = services is None
        state: AppServices = services or build_services(settings)
        app.state.services = state
        if owned:
            interrupted = await state.repository.interrupt_unfinished()
            if interrupted:
                logger.warning("marked unfinished runs as failed", extra={"runs": interrupted})
        try:
            yield
        finally:
            await state.runs.shutdown()
            for task in list(state.background):
                task.cancel()
            if state.close is not None:
                await state.close()

    app = FastAPI(title="ResearchAI", version=__version__, lifespan=lifespan)
    if services is not None:
        app.state.services = services  # usable without running the lifespan (tests)
    errors.install(app)

    def get_services(request: Request) -> AppServices:
        state: AppServices = request.app.state.services
        return state

    @app.post("/research", status_code=202, responses=ERRORS)
    async def start_research(
        body: ResearchRequest,
        request: Request,
        response: Response,
        wait: Annotated[bool, Query(description="Block until the run finishes")] = False,
    ) -> RunView:
        state = get_services(request)
        record = await state.runs.submit(body.question)
        response.headers["Location"] = f"/research/{record.id}"
        if wait:
            await state.runs.wait(record.id, settings.api_wait_seconds)
            record = await state.repository.get(record.id) or record
            if record.status in ("completed", "failed"):
                response.status_code = 200
        return RunView.of(record)

    @app.get("/research/{run_id}", responses=ERRORS)
    async def get_research(run_id: str, request: Request) -> RunView:
        if not _RUN_ID.match(run_id):
            raise ApiError(404, "NOT_FOUND", "no such research run")
        record = await get_services(request).repository.get(run_id)
        if record is None:
            raise ApiError(404, "NOT_FOUND", "no such research run")
        return RunView.of(record)

    @app.get("/evaluations", responses=ERRORS)
    async def list_evaluations(
        request: Request, limit: Annotated[int, Query(ge=1, le=200)] = 50
    ) -> list[EvaluationRecord]:
        store = get_services(request).evaluations
        return [] if store is None else await store.list(limit)

    @app.post("/evaluations/run", status_code=202, responses=ERRORS)
    async def run_evaluation(body: EvaluationRequest, request: Request) -> EvaluationRecord:
        state = get_services(request)
        runner, store = state.evaluation_runner, state.evaluations
        if runner is None or store is None:
            raise ApiError(
                503, "EVALUATION_UNAVAILABLE", "no evaluation runner is installed in this build"
            )
        record = await store.start(body)

        async def execute() -> None:
            try:
                summary = await runner(body)
            except Exception:
                logger.exception("evaluation run failed", extra={"evaluation": record.id})
                await store.finish(record.id, error="internal error")
            else:
                await store.finish(record.id, summary=summary)

        task = asyncio.create_task(execute(), name=f"evaluation-{record.id}")
        state.background.add(task)
        task.add_done_callback(state.background.discard)
        return record

    @app.get("/health", responses=ERRORS)
    async def health(request: Request, response: Response) -> HealthResponse:
        components: dict[str, Literal["ok", "error"]] = {}
        for name, check in get_services(request).health_checks.items():
            try:
                components[name] = "ok" if await check() else "error"
            except Exception:  # a failing dependency is a status, not an API error
                logger.warning("health check failed", extra={"component": name}, exc_info=True)
                components[name] = "error"
        healthy = all(value == "ok" for value in components.values())
        if not healthy:
            response.status_code = 503
        return HealthResponse(
            status="ok" if healthy else "degraded",
            version=__version__,
            llm_provider=settings.llm_provider,
            components=components,
        )

    return app
