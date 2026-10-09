"""Ingest the frozen corpus into PostgreSQL + pgvector.

Ingestion is idempotent: if the latest build already holds the same manifest,
embedder and chunker, nothing is written. Otherwise all corpus rows are
replaced in one transaction, so readers never see a half-built corpus.
Embedding happens before the transaction opens (it can take minutes).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from sqlalchemy import delete, select, text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from researchai.corpus.chunking import MAX_CHARS, chunk_document
from researchai.corpus.embeddings import Embedder, Vector
from researchai.corpus.loader import Corpus
from researchai.corpus.models import Chunk, Document
from researchai.db.models import ChunkRow, CorpusBuildRow, DocumentRow

CHUNKER_ID = f"section-v1:{MAX_CHARS}"


class IngestError(RuntimeError):
    pass


@dataclass(frozen=True)
class IngestReport:
    status: Literal["ingested", "unchanged"]
    corpus_version: str
    manifest_sha256: str
    embedder: str
    document_count: int
    chunk_count: int


def embedding_text(document: Document, chunk: Chunk) -> str:
    """Text sent to the embedder: title and section give the chunk context."""
    header = (
        document.meta.title if not chunk.section else f"{document.meta.title} — {chunk.section}"
    )
    return f"{header}\n{chunk.text}"


def plan_chunks(corpus: Corpus) -> list[tuple[Document, list[Chunk]]]:
    return [(document, chunk_document(document)) for document in corpus.documents]


async def embed_all(texts: Sequence[str], embedder: Embedder, *, batch_size: int) -> list[Vector]:
    vectors: list[Vector] = []
    for start in range(0, len(texts), batch_size):
        vectors.extend(await embedder.embed(texts[start : start + batch_size]))
    if len(vectors) != len(texts):
        raise IngestError(f"embedder returned {len(vectors)} vectors for {len(texts)} texts")
    return vectors


def document_row(document: Document) -> DocumentRow:
    meta = document.meta
    return DocumentRow(
        id=meta.id,
        title=meta.title,
        url=meta.url,
        published=meta.published,
        snapshot=meta.snapshot,
        language=meta.language.value,
        topic=meta.topic.value,
        source_type=meta.source_type.value,
        publisher=meta.publisher,
        synthetic=meta.synthetic,
        license=meta.license,
        body=document.body,
        sha256=document.sha256,
    )


def chunk_row(chunk: Chunk, embedding: Vector) -> ChunkRow:
    return ChunkRow(
        id=chunk.id,
        doc_id=chunk.doc_id,
        ordinal=chunk.ordinal,
        section=chunk.section,
        text=chunk.text,
        char_start=chunk.start,
        char_end=chunk.end,
        embedding=embedding,
    )


async def latest_build(session: AsyncSession) -> CorpusBuildRow | None:
    result = await session.scalars(
        select(CorpusBuildRow).order_by(CorpusBuildRow.id.desc()).limit(1)
    )
    return result.first()


async def stored_embedding_dim(session: AsyncSession) -> int:
    """Dimension of chunks.embedding as created by the migration."""
    dim = await session.scalar(
        text(
            "SELECT atttypmod FROM pg_attribute "
            "WHERE attrelid = to_regclass('chunks') AND attname = 'embedding'"
        )
    )
    if dim is None:
        raise IngestError("chunks.embedding not found; run `python -m researchai.db upgrade`")
    return int(dim)


async def ingest(
    corpus: Corpus,
    embedder: Embedder,
    sessions: async_sessionmaker[AsyncSession],
    *,
    batch_size: int,
    force: bool = False,
) -> IngestReport:
    manifest_sha = str(corpus.manifest()["sha256"])
    planned = plan_chunks(corpus)
    chunk_count = sum(len(chunks) for _, chunks in planned)

    def report(status: Literal["ingested", "unchanged"]) -> IngestReport:
        return IngestReport(
            status=status,
            corpus_version=corpus.version,
            manifest_sha256=manifest_sha,
            embedder=embedder.name,
            document_count=len(corpus.documents),
            chunk_count=chunk_count,
        )

    async with sessions() as session:
        stored_dim = await stored_embedding_dim(session)
        if stored_dim != embedder.dim:
            raise IngestError(
                f"database stores {stored_dim}-dim vectors but the embedder produces "
                f"{embedder.dim}; recreate the database or fix EMBEDDING_DIM"
            )
        build = await latest_build(session)
    if (
        not force
        and build is not None
        and (build.manifest_sha256, build.embedder, build.chunker)
        == (manifest_sha, embedder.name, CHUNKER_ID)
    ):
        return report("unchanged")

    texts = [embedding_text(doc, chunk) for doc, chunks in planned for chunk in chunks]
    vectors = iter(await embed_all(texts, embedder, batch_size=batch_size))

    async with sessions.begin() as session:
        await session.execute(delete(ChunkRow))
        await session.execute(delete(DocumentRow))
        session.add_all(document_row(doc) for doc, _ in planned)
        await session.flush()
        session.add_all(
            chunk_row(chunk, next(vectors)) for _, chunks in planned for chunk in chunks
        )
        session.add(
            CorpusBuildRow(
                corpus_version=corpus.version,
                manifest_sha256=manifest_sha,
                embedder=embedder.name,
                embedding_dim=embedder.dim,
                chunker=CHUNKER_ID,
                document_count=len(corpus.documents),
                chunk_count=chunk_count,
            )
        )
    return report("ingested")
