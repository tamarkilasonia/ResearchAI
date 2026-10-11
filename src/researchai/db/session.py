"""Async engine and session factory."""

from __future__ import annotations

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from researchai.config import Settings

CONNECT_TIMEOUT_SECONDS = 10


def create_engine(settings: Settings) -> AsyncEngine:
    # pre-ping replaces connections a database restart killed; the connect timeout
    # makes a dead server an error instead of a hang.
    return create_async_engine(
        settings.database_url,
        pool_pre_ping=True,
        connect_args={"timeout": CONNECT_TIMEOUT_SECONDS},
    )


def session_factory(engine: AsyncEngine) -> async_sessionmaker[AsyncSession]:
    return async_sessionmaker(engine, expire_on_commit=False)
