"""Integration fixtures: a real PostgreSQL + pgvector database.

Tests run against ``<DATABASE_URL database>_test`` so a developer's data is
never touched. Start Postgres with ``docker compose up -d db`` (local) or the
CI service container, then run ``pytest -m integration``.
"""

from __future__ import annotations

import asyncio
import os
import socket
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

import pytest
import uvicorn
from sqlalchemy import make_url, text
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.pool import NullPool

from researchai.config import Settings
from researchai.corpus.embeddings import HashEmbedder
from researchai.corpus.ingest import ingest
from researchai.corpus.loader import load_corpus
from researchai.db import migrate
from researchai.db.session import session_factory
from researchai.mcp_server.__main__ import build_app
from researchai.mcp_server.server import ToolServices
from researchai.tools.documents import DocumentStore
from researchai.tools.search import CorpusSearch
from tests.support import make_settings

TEST_DIM = 64
CORPUS_ROOT = Path(__file__).resolve().parents[2] / "corpus"

# Read once at import, before the suite-wide fixture clears settings env vars.
_BASE_URL = make_url(
    os.environ.get("DATABASE_URL") or str(Settings.model_fields["database_url"].default)
)


def scratch_database_url(suffix: str = "_test") -> str:
    """URL of a throwaway database next to the configured one."""
    return _BASE_URL.set(database=f"{_BASE_URL.database}{suffix}").render_as_string(
        hide_password=False
    )


TEST_DATABASE_URL = scratch_database_url()


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    here = Path(__file__).parent
    for item in items:
        if here in Path(str(item.fspath)).parents:
            item.add_marker(pytest.mark.integration)


async def ensure_database(url: str) -> None:
    target = make_url(url)
    admin = create_async_engine(
        target.set(database="postgres"), isolation_level="AUTOCOMMIT", poolclass=NullPool
    )
    try:
        async with admin.connect() as connection:
            exists = await connection.scalar(
                text("SELECT 1 FROM pg_database WHERE datname = :name"),
                {"name": target.database},
            )
            if not exists:
                await connection.execute(text(f'CREATE DATABASE "{target.database}"'))
    finally:
        await admin.dispose()


@pytest.fixture(scope="session")
def database_url() -> str:
    asyncio.run(ensure_database(TEST_DATABASE_URL))
    return TEST_DATABASE_URL


@pytest.fixture
async def engine(database_url: str) -> AsyncIterator[AsyncEngine]:
    """A freshly migrated schema for every test."""
    engine = create_async_engine(database_url, poolclass=NullPool)
    await migrate.downgrade(engine, embedding_dim=TEST_DIM)
    await migrate.upgrade(engine, embedding_dim=TEST_DIM)
    try:
        yield engine
    finally:
        await engine.dispose()


@pytest.fixture
def sessions(engine: AsyncEngine) -> async_sessionmaker[AsyncSession]:
    return session_factory(engine)


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


@asynccontextmanager
async def serving_real_corpus(database_url: str) -> AsyncIterator[str]:
    """Migrate + ingest the real corpus into ``database_url`` and serve MCP over HTTP.

    Yields the MCP endpoint URL. Uses the offline hash embedder (dimension TEST_DIM).
    """
    await ensure_database(database_url)
    engine = create_async_engine(database_url, poolclass=NullPool)
    embedder = HashEmbedder(TEST_DIM)
    await migrate.downgrade(engine, embedding_dim=TEST_DIM)
    await migrate.upgrade(engine, embedding_dim=TEST_DIM)
    sessions = session_factory(engine)
    await ingest(load_corpus(CORPUS_ROOT), embedder, sessions, batch_size=128)
    port = _free_port()
    services = ToolServices(
        search=CorpusSearch(sessions, embedder), documents=DocumentStore(sessions)
    )
    app = build_app(make_settings(mcp_port=port), services)
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_config=None))
    task = asyncio.create_task(server.serve())
    while not server.started:  # noqa: ASYNC110 - uvicorn exposes only a flag
        await asyncio.sleep(0.05)
    try:
        yield f"http://127.0.0.1:{port}/mcp"
    finally:
        server.should_exit = True
        await task
        await engine.dispose()
