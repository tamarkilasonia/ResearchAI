"""Optional live web search through a SearxNG instance (``/search?format=json``).

Disabled by default and never used in evaluation runs: when
``LIVE_WEB_SEARCH_ENABLED`` is false the MCP server does not even register the
tool, so the agent cannot see it. Results are untrusted, like corpus text.
"""

from __future__ import annotations

import httpx

from researchai.tools.errors import ErrorCode, ToolFailure
from researchai.tools.schemas import MAX_QUERY_CHARS, MAX_WEB_RESULTS, WebResult, WebSearchResponse

MAX_SNIPPET_CHARS = 500


class LiveWebSearch:
    def __init__(
        self,
        base_url: str,
        *,
        timeout: float = 15.0,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        if not base_url.startswith(("http://", "https://")):
            raise ValueError("LIVE_WEB_SEARCH_URL must be an http(s) URL")
        self._url = f"{base_url.rstrip('/')}/search"
        self._timeout = timeout
        self._transport = transport

    async def search(self, query: str, *, max_results: int = 5) -> WebSearchResponse:
        query = query.strip()
        if not query or len(query) > MAX_QUERY_CHARS:
            raise ToolFailure(
                ErrorCode.INVALID_ARGUMENT, f"query must be 1-{MAX_QUERY_CHARS} characters"
            )
        if not 1 <= max_results <= MAX_WEB_RESULTS:
            raise ToolFailure(
                ErrorCode.INVALID_ARGUMENT, f"max_results must be 1-{MAX_WEB_RESULTS}"
            )
        try:
            async with httpx.AsyncClient(
                timeout=self._timeout, transport=self._transport
            ) as client:
                response = await client.get(self._url, params={"q": query, "format": "json"})
                response.raise_for_status()
                items = response.json()["results"]
            results = [
                WebResult(
                    title=str(item.get("title", ""))[:300],
                    url=str(item["url"]),
                    snippet=str(item.get("content", ""))[:MAX_SNIPPET_CHARS],
                )
                for item in items[:max_results]
            ]
        except (httpx.HTTPError, KeyError, TypeError, ValueError, AttributeError) as exc:
            raise ToolFailure(ErrorCode.BACKEND_UNAVAILABLE, "web search backend failed") from exc
        return WebSearchResponse(query=query, results=results)
