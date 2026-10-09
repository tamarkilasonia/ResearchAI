"""ORM models. The schema itself is owned by the Alembic migrations."""

from __future__ import annotations

from datetime import date, datetime

from pgvector.sqlalchemy import Vector
from sqlalchemy import Computed, DateTime, ForeignKey, Index, Integer, String, Text, func
from sqlalchemy.dialects.postgresql import TSVECTOR
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


class DocumentRow(Base):
    __tablename__ = "documents"

    id: Mapped[str] = mapped_column(String(32), primary_key=True)
    title: Mapped[str] = mapped_column(Text)
    url: Mapped[str] = mapped_column(Text)
    published: Mapped[date]
    snapshot: Mapped[date]
    language: Mapped[str] = mapped_column(String(8), index=True)
    topic: Mapped[str] = mapped_column(String(16), index=True)
    source_type: Mapped[str] = mapped_column(String(32))
    publisher: Mapped[str] = mapped_column(Text)
    synthetic: Mapped[bool]
    license: Mapped[str] = mapped_column(String(100))
    body: Mapped[str] = mapped_column(Text)
    sha256: Mapped[str] = mapped_column(String(64))

    chunks: Mapped[list[ChunkRow]] = relationship(
        back_populates="document", cascade="all, delete-orphan", passive_deletes=True
    )


class ChunkRow(Base):
    __tablename__ = "chunks"
    __table_args__ = (Index("ix_chunks_tsv", "tsv", postgresql_using="gin"),)

    id: Mapped[str] = mapped_column(String(48), primary_key=True)
    doc_id: Mapped[str] = mapped_column(ForeignKey("documents.id", ondelete="CASCADE"), index=True)
    ordinal: Mapped[int] = mapped_column(Integer)
    section: Mapped[str] = mapped_column(Text)
    text: Mapped[str] = mapped_column(Text)
    char_start: Mapped[int] = mapped_column(Integer)
    char_end: Mapped[int] = mapped_column(Integer)
    # Dimension is fixed by the migration (EMBEDDING_DIM at creation time).
    embedding: Mapped[list[float]] = mapped_column(Vector())
    # 'simple' config: no stemming, works the same for English and Georgian.
    tsv: Mapped[str] = mapped_column(
        TSVECTOR, Computed("to_tsvector('simple', text)", persisted=True)
    )

    document: Mapped[DocumentRow] = relationship(back_populates="chunks")


class CorpusBuildRow(Base):
    """One successful ingestion. The latest row describes what is in the tables."""

    __tablename__ = "corpus_builds"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    corpus_version: Mapped[str] = mapped_column(String(32))
    manifest_sha256: Mapped[str] = mapped_column(String(64))
    embedder: Mapped[str] = mapped_column(String(200))
    embedding_dim: Mapped[int] = mapped_column(Integer)
    chunker: Mapped[str] = mapped_column(String(100))
    document_count: Mapped[int] = mapped_column(Integer)
    chunk_count: Mapped[int] = mapped_column(Integer)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
