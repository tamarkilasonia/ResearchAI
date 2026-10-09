"""Typed client for the research tools.

This is the only way the agent reaches data: it talks MCP to the tool server
(over streamable HTTP, or in-process in tests) and never imports data-access
code. Every call is recorded in :attr:`ResearchTools.calls`, which the audit
trace and the tool-selection metrics use.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from datetime import date
from enum import Enum
from types import TracebackType
from typing import Any, Self, TypeVar

from mcp import Client
from mcp.server.mcpserver import MCPServer
from pydantic import BaseModel, ValidationError

from researchai.corpus.models import Language, SourceType, Topic
from researchai.tools.errors import ErrorCode
from researchai.tools.schemas import (
    CalculationResponse,
    DocumentResponse,
    MetadataResponse,
    SearchMode,
    SearchResponse,
    WebSearchResponse,
)

ModelT = TypeVar("ModelT", bound=BaseModel)

_JSON_OBJECT = re.compile(r"\{.*\}", re.DOTALL)


class ToolCallError(Exception):
    def __init__(self, tool: str, code: ErrorCode, message: str) -> None:
        super().__init__(f"{tool}: {code.value}: {message}")
        self.tool = tool
        self.code = code
        self.message = message


@dataclass(frozen=True)
class ToolCallRecord:
    tool: str
    arguments: dict[str, Any]
    ok: bool
    error_code: ErrorCode | None
    latency_ms: float


def parse_error(text: str) -> tuple[ErrorCode, str]:
    """Recover the stable error code from an ``is_error`` tool result."""
    if match := _JSON_OBJECT.search(text):
        try:
            payload = json.loads(match.group(0))
            return ErrorCode(payload["code"]), str(payload["message"])
        except (ValueError, KeyError, TypeError):
            pass
    if "Unknown tool" in text:
        return ErrorCode.UNKNOWN_TOOL, text
    if "validation error" in text:
        return ErrorCode.INVALID_ARGUMENT, text
    return ErrorCode.INTERNAL, text


def _jsonable(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, list | tuple):
        return [_jsonable(item) for item in value]
    return value


@dataclass
class ResearchTools:
    """Async context manager: ``async with ResearchTools(url) as tools: ...``."""

    target: str | MCPServer
    timeout_seconds: float = 60.0
    calls: list[ToolCallRecord] = field(default_factory=list)
    _client: Client | None = field(default=None, init=False, repr=False)

    async def __aenter__(self) -> Self:
        client = Client(self.target, read_timeout_seconds=self.timeout_seconds)
        await client.__aenter__()
        self._client = client
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        client, self._client = self._client, None
        if client is not None:
            await client.__aexit__(exc_type, exc, tb)

    def _connected(self) -> Client:
        if self._client is None:
            raise RuntimeError("use ResearchTools as an async context manager")
        return self._client

    async def list_tools(self) -> list[str]:
        result = await self._connected().list_tools()
        return sorted(tool.name for tool in result.tools)

    async def call(self, tool: str, arguments: dict[str, Any], model: type[ModelT]) -> ModelT:
        args = {key: _jsonable(value) for key, value in arguments.items() if value is not None}
        started = time.perf_counter()
        result = await self._connected().call_tool(tool, args)
        latency = round((time.perf_counter() - started) * 1000, 1)
        if result.is_error:
            text = " ".join(getattr(part, "text", "") for part in result.content)
            code, message = parse_error(text)
            self.calls.append(ToolCallRecord(tool, args, False, code, latency))
            raise ToolCallError(tool, code, message)
        try:
            parsed = model.model_validate(result.structured_content)
        except ValidationError as exc:
            self.calls.append(ToolCallRecord(tool, args, False, ErrorCode.INTERNAL, latency))
            raise ToolCallError(tool, ErrorCode.INTERNAL, "malformed tool result") from exc
        self.calls.append(ToolCallRecord(tool, args, True, None, latency))
        return parsed

    async def search_corpus(
        self,
        query: str,
        *,
        top_k: int = 8,
        mode: SearchMode = "hybrid",
        language: Language | None = None,
        topic: Topic | None = None,
        published_from: date | None = None,
        published_to: date | None = None,
        source_types: list[SourceType] | None = None,
    ) -> SearchResponse:
        return await self.call(
            "search_corpus",
            {
                "query": query,
                "top_k": top_k,
                "mode": mode,
                "language": language,
                "topic": topic,
                "published_from": published_from,
                "published_to": published_to,
                "source_types": source_types,
            },
            SearchResponse,
        )

    async def fetch_document(
        self, doc_id: str, *, start: int | None = None, end: int | None = None
    ) -> DocumentResponse:
        return await self.call(
            "fetch_document", {"doc_id": doc_id, "start": start, "end": end}, DocumentResponse
        )

    async def get_source_metadata(self, doc_ids: list[str]) -> MetadataResponse:
        return await self.call("get_source_metadata", {"doc_ids": doc_ids}, MetadataResponse)

    async def calculate(self, expression: str) -> CalculationResponse:
        return await self.call("calculate", {"expression": expression}, CalculationResponse)

    async def live_web_search(self, query: str, *, max_results: int = 5) -> WebSearchResponse:
        return await self.call(
            "live_web_search", {"query": query, "max_results": max_results}, WebSearchResponse
        )
