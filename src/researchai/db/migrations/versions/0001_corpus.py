"""Corpus tables: documents, chunks (pgvector + full-text), corpus builds.

There is deliberately no ANN index on chunks.embedding: at a few thousand
chunks an exact sequential scan is fast, and exact search keeps evaluation
runs reproducible (HNSW results can depend on build order).

Revision ID: 0001
Revises:
Create Date: 2026-10-09
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import context, op
from pgvector.sqlalchemy import Vector
from sqlalchemy.dialects.postgresql import TSVECTOR

revision: str = "0001"
down_revision: str | None = None
branch_labels: str | None = None
depends_on: str | None = None


def upgrade() -> None:
    dim = int(context.config.attributes["embedding_dim"])
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")
    op.create_table(
        "documents",
        sa.Column("id", sa.String(32), primary_key=True),
        sa.Column("title", sa.Text, nullable=False),
        sa.Column("url", sa.Text, nullable=False),
        sa.Column("published", sa.Date, nullable=False),
        sa.Column("snapshot", sa.Date, nullable=False),
        sa.Column("language", sa.String(8), nullable=False),
        sa.Column("topic", sa.String(16), nullable=False),
        sa.Column("source_type", sa.String(32), nullable=False),
        sa.Column("publisher", sa.Text, nullable=False),
        sa.Column("synthetic", sa.Boolean, nullable=False),
        sa.Column("license", sa.String(100), nullable=False),
        sa.Column("body", sa.Text, nullable=False),
        sa.Column("sha256", sa.String(64), nullable=False),
    )
    op.create_index("ix_documents_language", "documents", ["language"])
    op.create_index("ix_documents_topic", "documents", ["topic"])
    op.create_table(
        "chunks",
        sa.Column("id", sa.String(48), primary_key=True),
        sa.Column(
            "doc_id",
            sa.String(32),
            sa.ForeignKey("documents.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("ordinal", sa.Integer, nullable=False),
        sa.Column("section", sa.Text, nullable=False),
        sa.Column("text", sa.Text, nullable=False),
        sa.Column("char_start", sa.Integer, nullable=False),
        sa.Column("char_end", sa.Integer, nullable=False),
        sa.Column("embedding", Vector(dim), nullable=False),
        sa.Column(
            "tsv",
            TSVECTOR,
            sa.Computed("to_tsvector('simple', text)", persisted=True),
            nullable=False,
        ),
        sa.CheckConstraint("char_end > char_start", name="ck_chunks_span"),
    )
    op.create_index("ix_chunks_doc_id", "chunks", ["doc_id"])
    op.create_index("ix_chunks_tsv", "chunks", ["tsv"], postgresql_using="gin")
    op.create_table(
        "corpus_builds",
        sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
        sa.Column("corpus_version", sa.String(32), nullable=False),
        sa.Column("manifest_sha256", sa.String(64), nullable=False),
        sa.Column("embedder", sa.String(200), nullable=False),
        sa.Column("embedding_dim", sa.Integer, nullable=False),
        sa.Column("chunker", sa.String(100), nullable=False),
        sa.Column("document_count", sa.Integer, nullable=False),
        sa.Column("chunk_count", sa.Integer, nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
    )


def downgrade() -> None:
    op.drop_table("corpus_builds")
    op.drop_table("chunks")
    op.drop_table("documents")
