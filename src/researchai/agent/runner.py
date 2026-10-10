"""Run one research question end to end and package the result with its audit trail."""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from typing import Any, Protocol

from researchai.agent.budget import Limits, RunBudget
from researchai.agent.graph import AgentContext, AgentTools, FatalToolError, build_graph
from researchai.agent.llm import LLMClient
from researchai.agent.models import ResearchResult, RunStatus, ToolCall, Usage
from researchai.config import Settings
from researchai.log import bind_run_id
from researchai.mcp_client.client import ToolCallRecord
from researchai.safety.models import SafetyReport

logger = logging.getLogger(__name__)


class RecordingTools(AgentTools, Protocol):
    calls: list[ToolCallRecord]


class ResearchAgent:
    def __init__(self, *, tools: RecordingTools, llm: LLMClient | None, settings: Settings) -> None:
        self._tools = tools
        self._llm = llm
        self._settings = settings
        self._graph = build_graph()

    async def run(self, question: str, *, run_id: str | None = None) -> ResearchResult:
        run_id = run_id or uuid.uuid4().hex
        budget = RunBudget(Limits.from_settings(self._settings))
        context = AgentContext(
            tools=self._tools,
            llm=self._llm,
            budget=budget,
            max_output_tokens=self._settings.llm_max_output_tokens,
            max_evidence_tokens=self._settings.llm_writer_evidence_tokens,
        )
        first_call = len(self._tools.calls)
        started = time.perf_counter()
        state: dict[str, Any] = {}
        status: RunStatus | None = None
        reason: str | None = None
        with bind_run_id(run_id):
            try:
                state = await asyncio.wait_for(
                    self._graph.ainvoke({"question": question}, context=context),
                    timeout=self._settings.run_timeout_seconds,
                )
            except FatalToolError as exc:
                status, reason = RunStatus.FAILED, f"tool backend unavailable: {exc}"
            except TimeoutError:
                status = RunStatus.FAILED
                reason = f"run exceeded {self._settings.run_timeout_seconds:g} s"
            except Exception:
                logger.exception("research run crashed")
                status, reason = RunStatus.FAILED, "internal error"
        calls = self._tools.calls[first_call:]
        result = ResearchResult(
            run_id=run_id,
            question=question,
            status=status or state.get("status", RunStatus.FAILED),
            mode="offline" if self._llm is None else "llm",
            profile=state.get("profile"),
            plan=state.get("plan"),
            evidence=state.get("evidence", []),
            assessments=state.get("assessments", []),
            claims=state.get("claims", []),
            conflicts=state.get("conflicts", []),
            sufficiency=state.get("verdicts", []),
            missing_entities=state.get("missing_entities", []),
            calculations=state.get("calculations", []),
            report=state.get("report") if status is None else None,
            citation_report=state.get("citation_report") if status is None else None,
            reason=reason or state.get("reason"),
            safety=SafetyReport(
                input=state.get("guard"),
                injections=state.get("injections", []),
                output_issues=state.get("output_issues", []),
            ),
            trace=state.get("trace", []),
            tool_calls=[
                ToolCall(
                    tool=c.tool,
                    arguments=c.arguments,
                    ok=c.ok,
                    error_code=c.error_code.value if c.error_code else None,
                    latency_ms=c.latency_ms,
                )
                for c in calls
            ],
            usage=Usage(
                searches=sum(1 for c in calls if c.tool == "search_corpus"),
                tool_calls=len(calls),
                llm_calls=budget.llm_calls,
                prompt_tokens=budget.prompt_tokens,
                completion_tokens=budget.completion_tokens,
                iterations=budget.iterations,
            ),
            budget_exhausted=list(budget.exhausted),
            total_ms=round((time.perf_counter() - started) * 1000, 1),
        )
        logger.info(
            "research run finished",
            extra={
                "status": result.status.value,
                "ms": result.total_ms,
                "searches": result.usage.searches,
            },
        )
        return result
