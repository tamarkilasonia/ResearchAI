"""Run Alembic migrations programmatically (used by the CLI, tests and the init job)."""

from __future__ import annotations

from pathlib import Path

from alembic import command
from alembic.config import Config
from sqlalchemy import Connection
from sqlalchemy.ext.asyncio import AsyncEngine

MIGRATIONS_DIR = Path(__file__).resolve().parent / "migrations"


def alembic_config(*, embedding_dim: int) -> Config:
    config = Config()
    config.set_main_option("script_location", str(MIGRATIONS_DIR))
    config.attributes["embedding_dim"] = embedding_dim
    return config


async def upgrade(engine: AsyncEngine, *, embedding_dim: int, revision: str = "head") -> None:
    config = alembic_config(embedding_dim=embedding_dim)

    def _run(connection: Connection) -> None:
        config.attributes["connection"] = connection
        command.upgrade(config, revision)

    async with engine.begin() as connection:
        await connection.run_sync(_run)


async def downgrade(engine: AsyncEngine, *, embedding_dim: int, revision: str = "base") -> None:
    config = alembic_config(embedding_dim=embedding_dim)

    def _run(connection: Connection) -> None:
        config.attributes["connection"] = connection
        command.downgrade(config, revision)

    async with engine.begin() as connection:
        await connection.run_sync(_run)
