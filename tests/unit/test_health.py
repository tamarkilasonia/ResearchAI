from __future__ import annotations

import logging

import httpx

from researchai import __version__
from researchai.api.app import create_app
from tests.support import make_settings


async def test_health_reports_version_and_provider() -> None:
    root = logging.getLogger()
    saved_handlers, saved_level = root.handlers[:], root.level
    try:
        app = create_app(make_settings(llm_provider="none"))
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get("/health")
    finally:
        root.handlers[:] = saved_handlers
        root.setLevel(saved_level)
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "version": __version__, "llm_provider": "none"}
