"""Database-free parts of ingestion. Database behaviour: tests/integration."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import pytest

from researchai.corpus.chunking import chunk_document
from researchai.corpus.embeddings import HashEmbedder, Vector
from researchai.corpus.ingest import (
    IngestError,
    chunk_row,
    document_row,
    embed_all,
    embedding_text,
    plan_chunks,
)
from researchai.corpus.loader import load_corpus
from tests.support import make_document, write_corpus


class CountingEmbedder:
    def __init__(self, drop_last: bool = False) -> None:
        self.calls: list[int] = []
        self.drop_last = drop_last

    @property
    def name(self) -> str:
        return "counting"

    @property
    def dim(self) -> int:
        return 1

    async def embed(self, texts: Sequence[str]) -> list[Vector]:
        self.calls.append(len(texts))
        vectors = [[float(len(t))] for t in texts]
        return vectors[:-1] if self.drop_last else vectors


def test_embedding_text_includes_title_and_section() -> None:
    document = make_document()
    first, second = chunk_document(document)
    assert embedding_text(document, first).startswith("Test document ev-en-0001 — Charging")
    assert "— Charging at home > Cost\n" in embedding_text(document, second)


def test_embedding_text_without_section() -> None:
    document = make_document(body="Plain text without headings.")
    (chunk,) = chunk_document(document)
    assert (
        embedding_text(document, chunk) == "Test document ev-en-0001\nPlain text without headings."
    )


def test_plan_chunks_covers_every_document(tmp_path: Path) -> None:
    planned = plan_chunks(load_corpus(write_corpus(tmp_path)))
    assert [doc.id for doc, _ in planned] == ["ev-en-0001", "ev-ka-0001"]
    assert all(chunks for _, chunks in planned)


async def test_embed_all_batches() -> None:
    embedder = CountingEmbedder()
    vectors = await embed_all(["a", "bb", "ccc", "dddd", "e"], embedder, batch_size=2)
    assert embedder.calls == [2, 2, 1]
    assert vectors == [[1.0], [2.0], [3.0], [4.0], [1.0]]


async def test_embed_all_detects_missing_vectors() -> None:
    with pytest.raises(IngestError, match="returned 2 vectors for 3"):
        await embed_all(["a", "b", "c"], CountingEmbedder(drop_last=True), batch_size=5)


async def test_rows_mirror_models() -> None:
    document = make_document("ev-ka-0001")
    row = document_row(document)
    assert (row.id, row.language, row.topic, row.synthetic) == ("ev-ka-0001", "ka", "ev", True)
    assert row.sha256 == document.sha256
    chunk = chunk_document(document)[0]
    (vector,) = await HashEmbedder(8).embed([chunk.text])
    crow = chunk_row(chunk, vector)
    assert (crow.id, crow.char_start, crow.char_end) == (chunk.id, chunk.start, chunk.end)
    assert crow.embedding == vector
