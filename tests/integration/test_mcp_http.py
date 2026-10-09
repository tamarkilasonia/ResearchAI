"""The real MCP server over streamable HTTP, backed by PostgreSQL and the real corpus.

Runs against its own database (``<db>_test_mcp``), migrated and ingested once
per module, with uvicorn serving the app on a free local port.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from datetime import date

import httpx
import pytest
import pytest_asyncio

from researchai.corpus.models import Language, Topic
from researchai.mcp_client.client import ResearchTools, ToolCallError
from researchai.tools.errors import ErrorCode
from tests.integration.conftest import scratch_database_url, serving_real_corpus

pytestmark = pytest.mark.asyncio(loop_scope="module")

MCP_DATABASE_URL = scratch_database_url("_test_mcp")


@pytest_asyncio.fixture(scope="module", loop_scope="module")
async def mcp_url() -> AsyncIterator[str]:
    async with serving_real_corpus(MCP_DATABASE_URL) as url:
        yield url


async def test_tools_are_listed_over_http(mcp_url: str) -> None:
    async with ResearchTools(mcp_url) as tools:
        assert await tools.list_tools() == [
            "calculate",
            "fetch_document",
            "get_source_metadata",
            "search_corpus",
        ]


async def test_hits_carry_exact_spans(mcp_url: str) -> None:
    async with ResearchTools(mcp_url) as tools:
        response = await tools.search_corpus("public charging points in Georgia", top_k=8)
        assert response.hits
        for hit in response.hits:
            document = await tools.fetch_document(
                hit.doc_id, start=hit.char_start, end=hit.char_end
            )
            assert document.text == hit.text


async def test_keyword_leg_finds_current_registry(mcp_url: str) -> None:
    async with ResearchTools(mcp_url) as tools:
        response = await tools.search_corpus("public charging points registry June 2026")
    assert "ev-en-0013" in {hit.doc_id for hit in response.hits}


async def test_georgian_query_with_language_filter(mcp_url: str) -> None:
    async with ResearchTools(mcp_url) as tools:
        response = await tools.search_corpus("ელექტრომობილის დატენვის ფასი", language=Language.KA)
    assert response.hits
    assert {hit.language for hit in response.hits} == {Language.KA}
    assert any(hit.keyword_rank is not None for hit in response.hits)


async def test_topic_and_date_filters(mcp_url: str) -> None:
    async with ResearchTools(mcp_url) as tools:
        tax = await tools.search_corpus("threshold turnover tax rate", topic=Topic.TAX, top_k=10)
        old = await tools.search_corpus(
            "electricity tariff", published_to=date(2022, 12, 31), top_k=10
        )
    assert tax.hits
    assert {hit.topic for hit in tax.hits} == {Topic.TAX}
    assert old.hits
    assert all(hit.published <= date(2022, 12, 31) for hit in old.hits)


async def test_single_leg_modes(mcp_url: str) -> None:
    async with ResearchTools(mcp_url) as tools:
        keyword = await tools.search_corpus("VAT exemption", mode="keyword")
        vector = await tools.search_corpus("VAT exemption", mode="vector")
    assert keyword.hits
    assert all(h.keyword_rank is not None and h.vector_rank is None for h in keyword.hits)
    assert all(h.vector_rank is not None and h.keyword_rank is None for h in vector.hits)


async def test_search_is_deterministic(mcp_url: str) -> None:
    async with ResearchTools(mcp_url) as tools:
        first = await tools.search_corpus("used Nissan Leaf price", top_k=10)
        second = await tools.search_corpus("used Nissan Leaf price", top_k=10)
    assert first == second


async def test_document_errors(mcp_url: str) -> None:
    async with ResearchTools(mcp_url) as tools:
        with pytest.raises(ToolCallError) as missing:
            await tools.fetch_document("ev-en-9999")
        with pytest.raises(ToolCallError) as out_of_range:
            await tools.fetch_document("ev-en-0013", start=0, end=10**6)
        whole = await tools.fetch_document("ev-en-0013")
    assert missing.value.code is ErrorCode.NOT_FOUND
    assert out_of_range.value.code is ErrorCode.INVALID_ARGUMENT
    assert whole.char_end == whole.length == len(whole.text)
    assert whole.metadata.synthetic


async def test_metadata_reports_missing_ids(mcp_url: str) -> None:
    async with ResearchTools(mcp_url) as tools:
        response = await tools.get_source_metadata(["ev-ka-0005", "ev-en-0013", "tax-en-0999"])
    assert [s.doc_id for s in response.sources] == ["ev-ka-0005", "ev-en-0013"]
    assert response.missing == ["tax-en-0999"]


async def test_parallel_calls_share_one_session(mcp_url: str) -> None:
    queries = ["VAT", "tariff", "winter range", "Batumi chargers", "insurance", "1% tax"]
    async with ResearchTools(mcp_url) as tools:
        results = await asyncio.gather(*(tools.search_corpus(q) for q in queries))
        calculation = await tools.calculate("16.8 * 0.292")
    assert [r.query for r in results] == queries
    assert calculation.result == pytest.approx(4.9056)
    assert len(tools.calls) == len(queries) + 1


async def test_dns_rebinding_protection(mcp_url: str) -> None:
    async with httpx.AsyncClient() as client:
        rejected = await client.post(
            mcp_url, headers={"Host": "attacker.example"}, json={"jsonrpc": "2.0"}
        )
        health = await client.get(mcp_url.removesuffix("/mcp") + "/health")
    assert rejected.status_code == 421
    assert health.json()["status"] == "ok"
