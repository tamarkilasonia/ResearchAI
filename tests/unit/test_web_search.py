from __future__ import annotations

import httpx
import pytest

from researchai.tools.errors import ErrorCode, ToolFailure
from researchai.tools.web import MAX_SNIPPET_CHARS, LiveWebSearch


def searcher(payload: object, status: int = 200) -> LiveWebSearch:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/search"
        assert request.url.params["format"] == "json"
        return httpx.Response(status, json=payload)

    return LiveWebSearch("http://searx:8080/", transport=httpx.MockTransport(handler))


async def test_results_are_mapped_truncated_and_limited() -> None:
    items = [
        {"title": f"T{i}", "url": f"https://example.org/{i}", "content": "x" * 900}
        for i in range(8)
    ]
    response = await searcher({"results": items}).search("ev prices", max_results=3)
    assert [r.title for r in response.results] == ["T0", "T1", "T2"]
    assert len(response.results[0].snippet) == MAX_SNIPPET_CHARS
    assert response.source == "live_web"
    assert response.content_trust == "untrusted"


@pytest.mark.parametrize(
    ("payload", "status"),
    [({"results": []}, 500), ({"nope": 1}, 200), ({"results": [{"title": "no url"}]}, 200)],
)
async def test_backend_failures(payload: object, status: int) -> None:
    with pytest.raises(ToolFailure) as excinfo:
        await searcher(payload, status).search("q")
    assert excinfo.value.code is ErrorCode.BACKEND_UNAVAILABLE


@pytest.mark.parametrize(("query", "max_results"), [("", 5), ("x" * 501, 5), ("q", 0), ("q", 11)])
async def test_invalid_arguments(query: str, max_results: int) -> None:
    with pytest.raises(ToolFailure) as excinfo:
        await searcher({"results": []}).search(query, max_results=max_results)
    assert excinfo.value.code is ErrorCode.INVALID_ARGUMENT


def test_requires_http_url() -> None:
    with pytest.raises(ValueError, match="http"):
        LiveWebSearch("searx:8080")
