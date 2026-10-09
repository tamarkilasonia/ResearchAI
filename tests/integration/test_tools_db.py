"""Tool backends against PostgreSQL, called directly (no MCP layer)."""

from __future__ import annotations

from pathlib import Path

import pytest
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from researchai.corpus.embeddings import HashEmbedder
from researchai.corpus.ingest import ingest
from researchai.corpus.loader import load_corpus
from researchai.tools.documents import DocumentStore
from researchai.tools.errors import ErrorCode, ToolFailure
from researchai.tools.search import CorpusSearch
from tests.integration.conftest import TEST_DIM
from tests.support import write_corpus

EMBEDDER = HashEmbedder(TEST_DIM)


class RenamedEmbedder(HashEmbedder):
    @property
    def name(self) -> str:
        return "some-other-model"


async def _ingested(tmp_path: Path, sessions: async_sessionmaker[AsyncSession]) -> None:
    await ingest(load_corpus(write_corpus(tmp_path)), EMBEDDER, sessions, batch_size=8)


async def test_search_refuses_an_empty_index(sessions: async_sessionmaker[AsyncSession]) -> None:
    with pytest.raises(ToolFailure) as excinfo:
        await CorpusSearch(sessions, EMBEDDER).search("anything")
    assert excinfo.value.code is ErrorCode.INDEX_NOT_READY


async def test_search_refuses_an_index_from_another_embedder(
    tmp_path: Path, sessions: async_sessionmaker[AsyncSession]
) -> None:
    await _ingested(tmp_path, sessions)
    with pytest.raises(ToolFailure) as excinfo:
        await CorpusSearch(sessions, RenamedEmbedder(TEST_DIM)).search("anything")
    assert excinfo.value.code is ErrorCode.INDEX_NOT_READY
    assert "some-other-model" in excinfo.value.message


async def test_reingest_with_another_embedder_is_detected_by_a_running_searcher(
    tmp_path: Path, sessions: async_sessionmaker[AsyncSession]
) -> None:
    await _ingested(tmp_path, sessions)
    search = CorpusSearch(sessions, EMBEDDER)
    assert (await search.search("charging")).hits
    corpus = load_corpus(tmp_path)
    await ingest(corpus, RenamedEmbedder(TEST_DIM), sessions, batch_size=8)
    with pytest.raises(ToolFailure) as excinfo:
        await search.search("charging")
    assert excinfo.value.code is ErrorCode.INDEX_NOT_READY


@pytest.mark.parametrize(
    ("query", "top_k"), [("", 5), ("  ", 5), ("x" * 501, 5), ("q", 0), ("q", 21)]
)
async def test_search_validates_arguments(
    tmp_path: Path, sessions: async_sessionmaker[AsyncSession], query: str, top_k: int
) -> None:
    await _ingested(tmp_path, sessions)
    with pytest.raises(ToolFailure) as excinfo:
        await CorpusSearch(sessions, EMBEDDER).search(query, top_k=top_k)
    assert excinfo.value.code is ErrorCode.INVALID_ARGUMENT


async def test_stop_word_only_query_still_uses_vectors(
    tmp_path: Path, sessions: async_sessionmaker[AsyncSession]
) -> None:
    await _ingested(tmp_path, sessions)
    search = CorpusSearch(sessions, EMBEDDER)
    response = await search.search("what is the", top_k=2)
    assert response.hits
    assert all(hit.keyword_rank is None for hit in response.hits)
    assert (await search.search("what is the", mode="keyword")).hits == []


async def test_document_store_validation(
    tmp_path: Path, sessions: async_sessionmaker[AsyncSession]
) -> None:
    await _ingested(tmp_path, sessions)
    store = DocumentStore(sessions)
    for bad in ("../etc/passwd", "EV-EN-0001", "ev-en-1"):
        with pytest.raises(ToolFailure) as excinfo:
            await store.fetch(bad)
        assert excinfo.value.code is ErrorCode.INVALID_ARGUMENT
    for start, end in ((5, 5), (-1, 3), (3, 2)):
        with pytest.raises(ToolFailure):
            await store.fetch("ev-en-0001", start=start, end=end)
    with pytest.raises(ToolFailure):
        await store.metadata([])
    with pytest.raises(ToolFailure):
        await store.metadata([f"ev-en-{i:04d}" for i in range(21)])
    deduped = await store.metadata(["ev-en-0001", "ev-en-0001"])
    assert [s.doc_id for s in deduped.sources] == ["ev-en-0001"]
