"""MCP contract tests: the real server and client, in-process, with fake backends."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date

import httpx
import pytest
from sqlalchemy.exc import OperationalError

from researchai.corpus.embeddings import EmbeddingError
from researchai.corpus.models import Language, SourceType, Topic
from researchai.mcp_client.client import ResearchTools, ToolCallError, parse_error
from researchai.mcp_server.__main__ import build_app
from researchai.mcp_server.server import ToolServices, build_server
from researchai.tools.errors import ErrorCode, ToolFailure
from researchai.tools.schemas import (
    DocumentResponse,
    MetadataResponse,
    SearchHit,
    SearchMode,
    SearchResponse,
    SourceMetadata,
    WebResult,
    WebSearchResponse,
)
from researchai.tools.search import SearchFilters
from tests.support import make_settings

META = SourceMetadata(
    doc_id="ev-en-0013",
    title="Registry",
    url="https://ministry.invalid/ev-en-0013",
    publisher="Ministry of Transport and Mobility",
    source_type=SourceType.GOVERNMENT,
    published=date(2026, 7, 15),
    snapshot=date(2026, 9, 1),
    language=Language.EN,
    topic=Topic.EV,
    synthetic=True,
)
HIT = SearchHit(
    chunk_id="ev-en-0013#c0",
    doc_id="ev-en-0013",
    title="Registry",
    section="Totals",
    text="Public charging points: 512",
    char_start=10,
    char_end=37,
    score=0.032,
    vector_rank=1,
    keyword_rank=None,
    published=date(2026, 7, 15),
    language=Language.EN,
    topic=Topic.EV,
    source_type=SourceType.GOVERNMENT,
    publisher="Ministry of Transport and Mobility",
)


class FakeSearch:
    def __init__(self, error: Exception | None = None) -> None:
        self.error = error
        self.calls: list[tuple[str, int, SearchMode, SearchFilters | None]] = []

    async def search(
        self,
        query: str,
        *,
        top_k: int = 8,
        mode: SearchMode = "hybrid",
        filters: SearchFilters | None = None,
    ) -> SearchResponse:
        self.calls.append((query, top_k, mode, filters))
        if self.error is not None:
            raise self.error
        return SearchResponse(query=query, mode=mode, hits=[HIT])


class FakeDocuments:
    async def fetch(
        self, doc_id: str, *, start: int | None = None, end: int | None = None
    ) -> DocumentResponse:
        if doc_id != META.doc_id:
            raise ToolFailure(ErrorCode.NOT_FOUND, f"no document {doc_id}")
        return DocumentResponse(
            metadata=META, text="512", char_start=start or 0, char_end=end or 3, length=40
        )

    async def metadata(self, doc_ids: Sequence[str]) -> MetadataResponse:
        return MetadataResponse(
            sources=[META] if META.doc_id in doc_ids else [],
            missing=[i for i in doc_ids if i != META.doc_id],
        )


class FakeWeb:
    async def search(self, query: str, *, max_results: int = 5) -> WebSearchResponse:
        return WebSearchResponse(
            query=query, results=[WebResult(title="t", url="https://example.org", snippet="s")]
        )


def tools_for(search: FakeSearch | None = None, *, web: bool = False) -> ResearchTools:
    services = ToolServices(
        search=search or FakeSearch(), documents=FakeDocuments(), web=FakeWeb() if web else None
    )
    return ResearchTools(build_server(services))


async def test_tool_listing_hides_disabled_web_search() -> None:
    async with tools_for() as tools:
        assert await tools.list_tools() == [
            "calculate",
            "fetch_document",
            "get_source_metadata",
            "search_corpus",
        ]
    async with tools_for(web=True) as tools:
        assert "live_web_search" in await tools.list_tools()


async def test_input_schemas_carry_limits() -> None:
    async with tools_for() as tools:
        listed = {t.name: t for t in (await tools._connected().list_tools()).tools}
    search = listed["search_corpus"].input_schema["properties"]
    assert search["query"]["maxLength"] == 500
    assert search["top_k"]["maximum"] == 20
    fetch = listed["fetch_document"].input_schema["properties"]
    assert fetch["doc_id"]["pattern"] == r"^[a-z]+-(en|ka)-\d{4}$"
    assert listed["search_corpus"].annotations is not None
    assert listed["search_corpus"].annotations.read_only_hint


async def test_search_roundtrip_passes_typed_filters() -> None:
    fake = FakeSearch()
    async with tools_for(fake) as tools:
        response = await tools.search_corpus(
            "chargers",
            top_k=3,
            mode="keyword",
            language=Language.KA,
            topic=Topic.EV,
            published_from=date(2025, 1, 1),
            source_types=[SourceType.GOVERNMENT],
        )
    assert response.hits == [HIT]
    assert response.content_trust == "untrusted"
    query, top_k, mode, filters = fake.calls[0]
    assert (query, top_k, mode) == ("chargers", 3, "keyword")
    assert filters == SearchFilters(
        language=Language.KA,
        topic=Topic.EV,
        published_from=date(2025, 1, 1),
        source_types=(SourceType.GOVERNMENT,),
    )
    record = tools.calls[0]
    assert record.ok
    assert record.arguments["published_from"] == "2025-01-01"
    assert "published_to" not in record.arguments
    assert record.latency_ms >= 0


async def test_semantic_validation_inside_tool_gets_stable_code() -> None:
    async with tools_for() as tools:
        with pytest.raises(ToolCallError) as excinfo:
            await tools.search_corpus(
                "q", published_from=date(2026, 1, 1), published_to=date(2025, 1, 1)
            )
    assert excinfo.value.code is ErrorCode.INVALID_ARGUMENT


async def test_anticipated_failure_code_and_message() -> None:
    fake = FakeSearch(ToolFailure(ErrorCode.INDEX_NOT_READY, "the corpus has not been ingested"))
    async with tools_for(fake) as tools:
        with pytest.raises(ToolCallError) as excinfo:
            await tools.search_corpus("q")
    assert excinfo.value.code is ErrorCode.INDEX_NOT_READY
    assert excinfo.value.message == "the corpus has not been ingested"
    assert tools.calls[0].error_code is ErrorCode.INDEX_NOT_READY


@pytest.mark.parametrize(
    "error",
    [
        OperationalError("SELECT 1", {}, Exception("password=hunter2 host=db")),
        EmbeddingError("ollama at http://10.0.0.5 refused"),
        ConnectionRefusedError("10.0.0.5:5432"),
    ],
)
async def test_backend_failures_are_reported_without_internals(error: Exception) -> None:
    async with tools_for(FakeSearch(error)) as tools:
        with pytest.raises(ToolCallError) as excinfo:
            await tools.search_corpus("q")
    assert excinfo.value.code is ErrorCode.BACKEND_UNAVAILABLE
    for secret in ("hunter2", "10.0.0.5"):
        assert secret not in str(excinfo.value)


async def test_unexpected_errors_are_opaque() -> None:
    async with tools_for(FakeSearch(RuntimeError("stack detail: /srv/secret.py"))) as tools:
        with pytest.raises(ToolCallError) as excinfo:
            await tools.search_corpus("q")
    assert excinfo.value.code is ErrorCode.INTERNAL
    assert "secret" not in str(excinfo.value)


async def test_schema_violation_is_invalid_argument() -> None:
    async with tools_for() as tools:
        with pytest.raises(ToolCallError) as excinfo:
            await tools.search_corpus("q", top_k=99)
        assert excinfo.value.code is ErrorCode.INVALID_ARGUMENT
        with pytest.raises(ToolCallError) as excinfo:
            await tools.fetch_document("../../etc/passwd")
        assert excinfo.value.code is ErrorCode.INVALID_ARGUMENT


async def test_disabled_web_search_is_unknown_tool() -> None:
    async with tools_for() as tools:
        with pytest.raises(ToolCallError) as excinfo:
            await tools.live_web_search("ev prices")
    assert excinfo.value.code is ErrorCode.UNKNOWN_TOOL


async def test_enabled_web_search() -> None:
    async with tools_for(web=True) as tools:
        response = await tools.live_web_search("ev prices", max_results=2)
    assert response.results[0].url == "https://example.org"


async def test_calculate_through_mcp() -> None:
    async with tools_for() as tools:
        assert (await tools.calculate("16.8 * 0.292")).result == pytest.approx(4.9056)
        with pytest.raises(ToolCallError) as excinfo:
            await tools.calculate("1/0")
    assert excinfo.value.code is ErrorCode.DIVISION_BY_ZERO


async def test_documents_through_mcp() -> None:
    async with tools_for() as tools:
        document = await tools.fetch_document("ev-en-0013", start=0, end=3)
        assert (document.text, document.metadata.publisher) == ("512", META.publisher)
        metadata = await tools.get_source_metadata(["ev-en-0013", "ev-ka-0999"])
        assert metadata.missing == ["ev-ka-0999"]
        with pytest.raises(ToolCallError) as excinfo:
            await tools.fetch_document("ev-en-0999")
    assert excinfo.value.code is ErrorCode.NOT_FOUND


async def test_malformed_result_is_internal() -> None:
    async with tools_for() as tools:
        with pytest.raises(ToolCallError) as excinfo:
            await tools.call("calculate", {"expression": "1+1"}, SearchResponse)
    assert excinfo.value.code is ErrorCode.INTERNAL
    assert not tools.calls[-1].ok


async def test_client_requires_context_manager() -> None:
    with pytest.raises(RuntimeError, match="context manager"):
        await tools_for().calculate("1+1")


async def test_exit_without_enter_is_harmless() -> None:
    await tools_for().__aexit__(None, None, None)


@pytest.mark.parametrize(
    ("text", "code"),
    [
        ('Error executing tool x: {"code": "NOT_FOUND", "message": "m"}', ErrorCode.NOT_FOUND),
        ('Error executing tool x: {"code": "NO_SUCH_CODE", "message": "m"}', ErrorCode.INTERNAL),
        ("Error executing tool x: {not json}", ErrorCode.INTERNAL),
        ("Unknown tool: live_web_search", ErrorCode.UNKNOWN_TOOL),
        ("1 validation error for searchArguments", ErrorCode.INVALID_ARGUMENT),
        ("Error executing tool search_corpus", ErrorCode.INTERNAL),
    ],
)
def test_parse_error(text: str, code: ErrorCode) -> None:
    assert parse_error(text)[0] is code


async def test_http_app_health_route() -> None:
    services = ToolServices(search=FakeSearch(), documents=FakeDocuments())
    app = build_app(make_settings(), services)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://localhost") as client:
        response = await client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
