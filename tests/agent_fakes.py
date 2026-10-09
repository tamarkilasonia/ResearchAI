"""Fakes for agent tests: scripted LLM and recording tools."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date
from typing import Any

from researchai.agent.llm import Completion, LLMError, LLMUsage
from researchai.corpus.models import Language, SourceType, Topic
from researchai.mcp_client.client import ToolCallError, ToolCallRecord
from researchai.tools.calculator import evaluate
from researchai.tools.errors import ErrorCode
from researchai.tools.schemas import CalculationResponse, SearchHit, SearchResponse


def hit(
    chunk: str,
    text: str,
    *,
    score: float = 0.03,
    published: date = date(2026, 7, 15),
    language: Language = Language.EN,
    topic: Topic = Topic.EV,
    source_type: SourceType = SourceType.GOVERNMENT,
    start: int = 0,
) -> SearchHit:
    doc_id = chunk.split("#")[0]
    return SearchHit(
        chunk_id=chunk,
        doc_id=doc_id,
        title=f"Title of {doc_id}",
        section="Section",
        text=text,
        char_start=start,
        char_end=start + len(text),
        score=score,
        vector_rank=1,
        keyword_rank=1,
        published=published,
        language=language,
        topic=topic,
        source_type=source_type,
        publisher="Publisher",
    )


CHARGING = hit(
    "ev-en-0013#c0",
    "Public charging points: 512. Of which DC fast charging points (50 kW or more): 146.",
    start=100,
)
TARIFF = hit(
    "ev-en-0029#c1",
    "In Tbilisi the tariff for 101-301 kWh per month is 0.292 GEL per kWh.",
    published=date(2024, 11, 28),
    source_type=SourceType.REGULATOR,
)
CONSUMPTION = hit(
    "ev-en-0058#c0",
    "The Nissan Leaf 40 kWh uses 16.8 kWh per 100 km in real driving.",
    source_type=SourceType.NGO,
)
VAT = hit(
    "ev-en-0001#c2",
    "From 1 January 2024 until 31 December 2026, the import of a battery electric passenger "
    "car is exempt from import VAT.",
)


@dataclass
class FakeTools:
    """Returns hits whose text shares a word with the query; records every call."""

    corpus: list[SearchHit] = field(default_factory=lambda: [CHARGING, TARIFF, CONSUMPTION, VAT])
    fail_with: ErrorCode | None = None
    calls: list[ToolCallRecord] = field(default_factory=list)
    topics: list[Topic | None] = field(default_factory=list)

    async def search_corpus(
        self, query: str, *, top_k: int = 8, topic: Topic | None = None
    ) -> SearchResponse:
        self.topics.append(topic)
        if self.fail_with is not None:
            self.calls.append(
                ToolCallRecord("search_corpus", {"query": query}, False, self.fail_with, 1.0)
            )
            raise ToolCallError("search_corpus", self.fail_with, "fake failure")
        words = {w.strip("?.,").casefold() for w in query.split() if len(w) > 3}
        hits = [
            h for h in self.corpus if words & {w.strip(".,:").casefold() for w in h.text.split()}
        ]
        self.calls.append(ToolCallRecord("search_corpus", {"query": query}, True, None, 1.0))
        return SearchResponse(query=query, mode="hybrid", hits=hits[:top_k])

    async def calculate(self, expression: str) -> CalculationResponse:
        self.calls.append(ToolCallRecord("calculate", {"expression": expression}, True, None, 1.0))
        return CalculationResponse(expression=expression, result=evaluate(expression))


Responder = Callable[[list[dict[str, str]]], str | dict[str, Any] | Exception]


@dataclass
class ScriptedLLM:
    """Answers by response schema title (LLMPlan, ProposedCalculations, LLMReport)."""

    script: dict[str, list[Responder]]
    context_tokens: int = 16_384
    usage: LLMUsage = field(
        default_factory=lambda: LLMUsage(prompt_tokens=500, completion_tokens=100)
    )
    requests: list[tuple[str, list[dict[str, str]]]] = field(default_factory=list)

    @property
    def model(self) -> str:
        return "scripted"

    async def chat(
        self, messages: list[dict[str, str]], *, schema: dict[str, Any], max_tokens: int
    ) -> Completion:
        title = str(schema.get("title"))
        self.requests.append((title, messages))
        queue = self.script.get(title)
        if not queue:
            raise LLMError("transport", f"no scripted reply for {title}")
        reply = queue.pop(0)(messages) if len(queue) > 1 else queue[0](messages)
        if isinstance(reply, Exception):
            raise reply
        content = reply if isinstance(reply, str) else json.dumps(reply, ensure_ascii=False)
        return Completion(content=content, usage=self.usage)


def always(value: str | dict[str, Any] | Exception) -> Responder:
    return lambda _messages: value
