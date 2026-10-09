"""FastAPI application factory.

Run with ``uvicorn researchai.api.app:create_app --factory``.
"""

from __future__ import annotations

from typing import Literal

from fastapi import FastAPI
from pydantic import BaseModel

from researchai import __version__
from researchai.config import LLMProvider, Settings, get_settings
from researchai.log import configure_logging


class HealthResponse(BaseModel):
    status: Literal["ok"]
    version: str
    llm_provider: LLMProvider


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    configure_logging(settings)
    app = FastAPI(title="ResearchAI", version=__version__)

    @app.get("/health")
    async def health() -> HealthResponse:
        return HealthResponse(status="ok", version=__version__, llm_provider=settings.llm_provider)

    return app
