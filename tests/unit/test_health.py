from __future__ import annotations

import logging

import httpx

from researchai import __version__
from researchai.api.app import AppServices, create_app
from researchai.api.runs import RunService
from tests.api_fakes import FakeRunner, MemoryRuns
from tests.support import make_settings


async def test_health_reports_version_provider_and_components() -> None:
    root = logging.getLogger()
    saved_handlers, saved_level = root.handlers[:], root.level
    repo = MemoryRuns()
    services = AppServices(
        runs=RunService(repo, FakeRunner(), max_concurrent=1), repository=repo, evaluations=None
    )
    try:
        app = create_app(make_settings(llm_provider="none"), services=services)
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get("/health")
    finally:
        root.handlers[:] = saved_handlers
        root.setLevel(saved_level)
    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "version": __version__,
        "llm_provider": "none",
        "components": {},
    }
