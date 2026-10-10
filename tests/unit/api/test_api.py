"""HTTP contract of the API, with in-memory storage and a fake agent."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

import httpx
import pytest

from researchai.api.app import AppServices, create_app
from researchai.api.runs import INTERRUPTED, RunService
from tests.api_fakes import FakeRunner, MemoryEvaluations, MemoryRuns
from tests.support import make_settings


@pytest.fixture(autouse=True)
def _restore_logging() -> Any:
    root = logging.getLogger()
    saved = root.handlers[:], root.level
    yield
    root.handlers[:] = saved[0]
    root.setLevel(saved[1])


def services(
    runner: FakeRunner | None = None, *, max_concurrent: int = 2, **kwargs: Any
) -> AppServices:
    repo = MemoryRuns()
    return AppServices(
        runs=RunService(repo, runner or FakeRunner(), max_concurrent=max_concurrent),
        repository=repo,
        evaluations=kwargs.pop("evaluations", None),
        **kwargs,
    )


async def client_for(state: AppServices, **settings: Any) -> httpx.AsyncClient:
    app = create_app(make_settings(**settings), services=state)
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test")


async def until(state: AppServices, run_id: str, status: str) -> None:
    for _ in range(200):
        record = await state.repository.get(run_id)
        if record is not None and record.status == status:
            return
        await asyncio.sleep(0.01)
    raise AssertionError(f"run never reached {status}")


def assert_error(response: httpx.Response, status: int, code: str) -> dict[str, Any]:
    assert response.status_code == status
    body: dict[str, Any] = response.json()
    assert set(body) == {"error"}
    assert body["error"]["code"] == code
    assert body["error"]["request_id"] == response.headers["X-Request-ID"]
    assert "Traceback" not in response.text
    return body


async def test_start_then_poll() -> None:
    state = services()
    async with await client_for(state) as client:
        started = await client.post("/research", json={"question": "How many chargers?"})
        assert started.status_code == 202
        run_id = started.json()["id"]
        assert started.headers["Location"] == f"/research/{run_id}"
        assert started.json()["status"] == "queued"
        await until(state, run_id, "completed")
        done = (await client.get(f"/research/{run_id}")).json()
    assert done["status"] == "completed"
    assert done["result"]["run_id"] == run_id
    assert done["report_markdown"].startswith("**abstained**")


async def test_wait_returns_the_finished_run() -> None:
    async with await client_for(services()) as client:
        response = await client.post("/research?wait=true", json={"question": "How many chargers?"})
    assert response.status_code == 200
    assert response.json()["status"] == "completed"


async def test_wait_gives_up_after_the_configured_time() -> None:
    state = services(FakeRunner(delay=0.5))
    async with await client_for(state, api_wait_seconds=0.05) as client:
        response = await client.post("/research?wait=true", json={"question": "How many chargers?"})
        assert response.status_code == 202
        assert response.json()["status"] in ("queued", "running")
        await until(state, response.json()["id"], "completed")  # waiting never cancels the run


async def test_runner_crash_is_a_failed_run_without_details() -> None:
    state = services(FakeRunner(error=RuntimeError("password=hunter2")))
    async with await client_for(state) as client:
        response = await client.post("/research?wait=true", json={"question": "How many chargers?"})
    assert response.status_code == 200
    assert response.json()["status"] == "failed"
    assert response.json()["error"] == "internal error"
    assert "hunter2" not in response.text


async def test_runs_beyond_the_limit_wait_their_turn() -> None:
    state = services(FakeRunner(delay=0.2), max_concurrent=1)
    async with await client_for(state) as client:
        first = (await client.post("/research", json={"question": "First question?"})).json()["id"]
        second = (await client.post("/research", json={"question": "Second question?"})).json()[
            "id"
        ]
        await until(state, first, "running")
        assert (await state.repository.get(second)).status == "queued"  # type: ignore[union-attr]
        await until(state, second, "completed")


async def test_shutdown_marks_running_work_as_interrupted() -> None:
    state = services(FakeRunner(delay=5))
    async with await client_for(state) as client:
        run_id = (await client.post("/research", json={"question": "Slow question?"})).json()["id"]
        await until(state, run_id, "running")
    await state.runs.shutdown()
    record = await state.repository.get(run_id)
    assert record is not None
    assert (record.status, record.error) == ("failed", INTERRUPTED)
    assert state.runs.active == 0


@pytest.mark.parametrize(
    "payload",
    [{"question": "hi"}, {"question": "x" * 2001}, {"question": "ok?", "extra": 1}, {}, []],
)
async def test_invalid_requests_use_the_error_format(payload: Any) -> None:
    async with await client_for(services()) as client:
        response = await client.post("/research", json=payload)
    body = assert_error(response, 422, "INVALID_REQUEST")
    assert body["error"]["message"]


@pytest.mark.parametrize("run_id", ["0" * 32, "not-a-run-id", "../../etc/passwd"])
async def test_unknown_runs_are_404(run_id: str) -> None:
    async with await client_for(services()) as client:
        response = await client.get(f"/research/{run_id}")
    assert_error(response, 404, "NOT_FOUND")


async def test_unknown_paths_and_methods_use_the_error_format() -> None:
    async with await client_for(services()) as client:
        assert_error(await client.get("/nope"), 404, "NOT_FOUND")
        assert_error(await client.delete("/research/" + "0" * 32), 405, "METHOD_NOT_ALLOWED")


async def test_unexpected_errors_are_opaque_500s() -> None:
    state = services()

    async def broken_get(run_id: str) -> None:
        raise RuntimeError("SELECT * FROM secrets")

    state.repository.get = broken_get  # type: ignore[method-assign]
    async with await client_for(state) as client:
        response = await client.get("/research/" + "0" * 32)
    body = assert_error(response, 500, "INTERNAL")
    assert "secrets" not in response.text
    assert body["error"]["message"] == "internal error"


async def test_request_ids_are_echoed_or_generated() -> None:
    async with await client_for(services()) as client:
        echoed = await client.get("/health", headers={"X-Request-ID": "abc-123"})
        replaced = await client.get("/health", headers={"X-Request-ID": "bad id with spaces"})
        generated = await client.get("/health")
    assert echoed.headers["X-Request-ID"] == "abc-123"
    assert replaced.headers["X-Request-ID"] != "bad id with spaces"
    assert len(generated.headers["X-Request-ID"]) == 32


async def test_evaluations_without_a_runner() -> None:
    async with await client_for(services()) as client:
        assert (await client.get("/evaluations")).json() == []
        response = await client.post("/evaluations/run", json={})
    assert_error(response, 503, "EVALUATION_UNAVAILABLE")


async def test_evaluations_with_a_runner() -> None:
    store = MemoryEvaluations()

    async def runner(request: Any) -> dict[str, Any]:
        return {"cases": 3, "split": request.split}

    async def failing(request: Any) -> dict[str, Any]:
        raise RuntimeError("boom")

    state = services(evaluations=store, evaluation_runner=runner)
    async with await client_for(state) as client:
        started = await client.post("/evaluations/run", json={"split": "dev", "mode": "offline"})
        assert started.status_code == 202
        for _ in range(100):
            if store.records[started.json()["id"]].status != "running":
                break
            await asyncio.sleep(0.01)
        listed = (await client.get("/evaluations")).json()
        assert listed[0]["summary"] == {"cases": 3, "split": "dev"}
        assert_error(
            await client.post("/evaluations/run", json={"split": "test"}), 422, "INVALID_REQUEST"
        )
        state.evaluation_runner = failing
        failed = (await client.post("/evaluations/run", json={})).json()["id"]
        for _ in range(100):
            if store.records[failed].status != "running":
                break
            await asyncio.sleep(0.01)
    assert (store.records[failed].status, store.records[failed].error) == (
        "failed",
        "internal error",
    )


async def test_health_reports_components() -> None:
    async def ok() -> bool:
        return True

    async def down() -> bool:
        return False

    async def broken() -> bool:
        raise ConnectionError("db at 10.0.0.5")

    async with await client_for(services(health_checks={"database": ok})) as client:
        healthy = await client.get("/health")
    assert healthy.status_code == 200
    assert healthy.json()["components"] == {"database": "ok"}
    async with await client_for(
        services(health_checks={"database": down, "tool_server": broken})
    ) as client:
        degraded = await client.get("/health")
    assert degraded.status_code == 503
    assert degraded.json()["status"] == "degraded"
    assert degraded.json()["components"] == {"database": "error", "tool_server": "error"}
    assert "10.0.0.5" not in degraded.text


async def test_lifespan_shuts_down_and_closes() -> None:
    closed: list[bool] = []

    async def close() -> None:
        closed.append(True)

    state = services(close=close)
    app = create_app(make_settings(), services=state)
    async with app.router.lifespan_context(app):
        pass
    assert closed == [True]


async def test_waiting_for_a_run_of_another_process_returns_at_once() -> None:
    state = services()
    await state.runs.wait("f" * 32, seconds=5)  # not started here: nothing to wait for


async def test_owned_lifespan_interrupts_leftovers_and_cancels_background_work(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    from researchai.api import app as app_module

    state = services()
    await state.repository.create("e" * 32, "Left over from a crash?")
    monkeypatch.setattr(app_module, "build_services", lambda settings: state)
    app = create_app(make_settings())
    logging.getLogger().addHandler(caplog.handler)  # create_app replaced the root handlers
    caplog.set_level(logging.WARNING)
    async with app.router.lifespan_context(app):
        never = asyncio.create_task(asyncio.sleep(60))
        state.background.add(never)
    left_over = await state.repository.get("e" * 32)
    assert left_over is not None
    assert (left_over.status, left_over.error) == ("failed", INTERRUPTED)
    assert "marked unfinished runs as failed" in caplog.text
    await asyncio.sleep(0)
    assert never.cancelled()
