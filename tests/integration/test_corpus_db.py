from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest
from sqlalchemy import delete, func, select, text
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.pool import NullPool

from researchai.corpus import __main__ as corpus_cli
from researchai.corpus.embeddings import HashEmbedder
from researchai.corpus.ingest import IngestError, ingest, stored_embedding_dim
from researchai.corpus.loader import load_corpus
from researchai.db import __main__ as db_cli
from researchai.db import migrate
from researchai.db.models import ChunkRow, CorpusBuildRow, DocumentRow
from tests.integration.conftest import TEST_DIM
from tests.support import make_settings, write_corpus, write_document

EMBEDDER = HashEmbedder(TEST_DIM)
CORPUS_ROOT = Path(__file__).resolve().parents[2] / "corpus"


async def count(session: AsyncSession, model: type[object]) -> int:
    return int(await session.scalar(select(func.count()).select_from(model)) or 0)


async def test_migration_roundtrip(
    engine: AsyncEngine, sessions: async_sessionmaker[AsyncSession]
) -> None:
    async with sessions() as session:
        assert await stored_embedding_dim(session) == TEST_DIM
    await migrate.downgrade(engine, embedding_dim=TEST_DIM)
    async with sessions() as session:
        assert await session.scalar(text("SELECT to_regclass('chunks')")) is None
        with pytest.raises(IngestError, match="upgrade"):
            await stored_embedding_dim(session)
    await migrate.upgrade(engine, embedding_dim=TEST_DIM)
    async with sessions() as session:
        assert await stored_embedding_dim(session) == TEST_DIM


async def test_ingest_is_idempotent(
    tmp_path: Path, sessions: async_sessionmaker[AsyncSession]
) -> None:
    corpus = load_corpus(write_corpus(tmp_path))
    first = await ingest(corpus, EMBEDDER, sessions, batch_size=3)
    assert first.status == "ingested"
    assert first.document_count == 2
    async with sessions() as session:
        assert await count(session, DocumentRow) == 2
        assert await count(session, ChunkRow) == first.chunk_count
        build = await session.scalar(select(CorpusBuildRow))
        assert build is not None
        assert (build.manifest_sha256, build.embedder) == (first.manifest_sha256, EMBEDDER.name)

    assert (await ingest(corpus, EMBEDDER, sessions, batch_size=3)).status == "unchanged"
    assert (await ingest(corpus, EMBEDDER, sessions, batch_size=3, force=True)).status == "ingested"
    async with sessions() as session:
        assert await count(session, CorpusBuildRow) == 2
        assert await count(session, DocumentRow) == 2


async def test_changed_corpus_replaces_rows(
    tmp_path: Path, sessions: async_sessionmaker[AsyncSession]
) -> None:
    root = write_corpus(tmp_path)
    await ingest(load_corpus(root), EMBEDDER, sessions, batch_size=8)
    write_document(root, "tax-en-0001", body="Turnover tax for small businesses is one percent.")
    (root / "ev" / "ka" / "ev-ka-0001.md").unlink()
    report = await ingest(load_corpus(root), EMBEDDER, sessions, batch_size=8)
    assert report.status == "ingested"
    async with sessions() as session:
        ids = set((await session.scalars(select(DocumentRow.id))).all())
        assert ids == {"ev-en-0001", "tax-en-0001"}
        orphans = await session.scalar(
            select(func.count()).select_from(ChunkRow).where(ChunkRow.doc_id == "ev-ka-0001")
        )
        assert orphans == 0


async def test_dimension_mismatch_is_refused(
    tmp_path: Path, sessions: async_sessionmaker[AsyncSession]
) -> None:
    with pytest.raises(IngestError, match="64-dim"):
        await ingest(load_corpus(write_corpus(tmp_path)), HashEmbedder(32), sessions, batch_size=8)


async def test_chunk_spans_match_stored_body(
    tmp_path: Path, sessions: async_sessionmaker[AsyncSession]
) -> None:
    await ingest(load_corpus(write_corpus(tmp_path)), EMBEDDER, sessions, batch_size=8)
    async with sessions() as session:
        rows = (await session.execute(select(ChunkRow, DocumentRow.body).join(DocumentRow))).all()
    assert rows
    for chunk, body in rows:
        assert body[chunk.char_start : chunk.char_end] == chunk.text


async def test_vector_search_ranks_relevant_document_first(
    tmp_path: Path, sessions: async_sessionmaker[AsyncSession]
) -> None:
    root = write_corpus(tmp_path)
    write_document(root, "tax-en-0001", body="Small business turnover tax rate is one percent.")
    await ingest(load_corpus(root), EMBEDDER, sessions, batch_size=8)
    query = EMBEDDER.embed_one("turnover tax rate for a small business")
    async with sessions() as session:
        best = await session.scalar(
            select(ChunkRow.doc_id).order_by(ChunkRow.embedding.cosine_distance(query)).limit(1)
        )
    assert best == "tax-en-0001"


async def test_full_text_search_works_for_georgian(
    tmp_path: Path, sessions: async_sessionmaker[AsyncSession]
) -> None:
    await ingest(load_corpus(write_corpus(tmp_path)), EMBEDDER, sessions, batch_size=8)
    async with sessions() as session:
        hits = (
            await session.scalars(
                select(ChunkRow.doc_id).where(
                    ChunkRow.tsv.op("@@")(func.plainto_tsquery("simple", "დატენვა"))
                )
            )
        ).all()
    assert set(hits) == {"ev-ka-0001"}


async def test_deleting_a_document_cascades_to_chunks(
    tmp_path: Path, sessions: async_sessionmaker[AsyncSession]
) -> None:
    await ingest(load_corpus(write_corpus(tmp_path)), EMBEDDER, sessions, batch_size=8)
    async with sessions.begin() as session:
        await session.execute(delete(DocumentRow).where(DocumentRow.id == "ev-en-0001"))
    async with sessions() as session:
        remaining = set((await session.scalars(select(ChunkRow.doc_id))).all())
    assert remaining == {"ev-ka-0001"}


def test_cli_upgrade_then_ingest(
    database_url: str, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    settings = make_settings(database_url=database_url, embedding_dim=TEST_DIM)
    assert db_cli.main(["downgrade"], settings=settings) == 0
    assert db_cli.main(["upgrade"], settings=settings) == 0
    root = write_corpus(tmp_path)
    assert corpus_cli.main(["--dir", str(root), "ingest"], settings=settings) == 0
    out = capsys.readouterr().out
    report = json.loads(out[out.index("{") :])
    assert report["status"] == "ingested"

    async def documents() -> int:
        engine = create_async_engine(database_url, poolclass=NullPool)
        async with engine.connect() as connection:
            value = await connection.scalar(text("SELECT count(*) FROM documents"))
        await engine.dispose()
        return int(value or 0)

    assert asyncio.run(documents()) == 2


async def test_real_corpus_ingests_with_exact_spans(
    sessions: async_sessionmaker[AsyncSession],
) -> None:
    corpus = load_corpus(CORPUS_ROOT)
    report = await ingest(corpus, EMBEDDER, sessions, batch_size=64)
    assert report.document_count == len(corpus.documents)
    async with sessions() as session:
        assert await count(session, DocumentRow) == len(corpus.documents)
        assert await count(session, ChunkRow) == report.chunk_count
        mismatched = await session.scalar(
            select(func.count())
            .select_from(ChunkRow)
            .join(DocumentRow)
            .where(
                func.substr(
                    DocumentRow.body,
                    ChunkRow.char_start + 1,
                    ChunkRow.char_end - ChunkRow.char_start,
                )
                != ChunkRow.text
            )
        )
        georgian_hits = await session.scalar(
            select(func.count())
            .select_from(ChunkRow)
            .where(ChunkRow.tsv.op("@@")(func.plainto_tsquery("simple", "ელექტრომობილები")))
        )
    assert mismatched == 0  # Python offsets == PostgreSQL character offsets, also for Georgian
    assert georgian_hits is not None
    assert georgian_hits > 0
