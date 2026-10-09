"""The research graph (LangGraph).

    validate_input -> input_guard -> profile -> planner
        -> search_worker x N (parallel, one per sub-question)  <-+
        -> collect --(sub-questions without evidence, budget left)-+
        -> analyze (calculations; LLM only) -> synthesize -> finalize

Later phases insert verification (source assessment, claims, conflicts,
sufficiency) between collect and synthesize, and citation verification
between synthesize and finalize.

Nodes reach data only through the MCP client in the runtime context.
Infrastructure failures (index not ready, backend down) end the run as
``failed``; the agent never answers without evidence.
"""

from __future__ import annotations

import logging
import operator
import time
from collections.abc import Awaitable
from dataclasses import dataclass
from typing import Annotated, Any, Protocol, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.runtime import Runtime
from langgraph.types import Send

from researchai.agent.analysis import run_calculations
from researchai.agent.budget import BudgetExceeded, RunBudget
from researchai.agent.llm import LLMClient, LLMError
from researchai.agent.models import (
    Calculation,
    Evidence,
    QuestionKind,
    QuestionProfile,
    Report,
    ResearchPlan,
    RunStatus,
    SubQuestion,
    TraceEvent,
)
from researchai.agent.planner import make_plan
from researchai.agent.profile import profile_question
from researchai.agent.synthesis import abstention_report, extractive_report, llm_report
from researchai.corpus.models import Topic
from researchai.mcp_client.client import ToolCallError
from researchai.tools.errors import ErrorCode
from researchai.tools.schemas import CalculationResponse, SearchHit, SearchResponse

logger = logging.getLogger(__name__)

MIN_QUESTION_CHARS = 3
MAX_QUESTION_CHARS = 2000
FATAL_TOOL_CODES = frozenset({ErrorCode.INDEX_NOT_READY, ErrorCode.BACKEND_UNAVAILABLE})
CALCULATION_KINDS = (QuestionKind.NUMERIC, QuestionKind.COMPARISON, QuestionKind.RECOMMENDATION)


class AgentTools(Protocol):
    async def search_corpus(
        self, query: str, *, top_k: int = 8, topic: Topic | None = None
    ) -> SearchResponse: ...

    async def calculate(self, expression: str) -> CalculationResponse: ...


@dataclass
class AgentContext:
    tools: AgentTools
    llm: LLMClient | None
    budget: RunBudget
    max_output_tokens: int
    max_evidence_tokens: int = 4000


class ResearchState(TypedDict, total=False):
    question: str
    status: RunStatus
    reason: str
    profile: QuestionProfile
    plan: ResearchPlan
    round: int
    pending: list[str]
    hits: Annotated[list[dict[str, Any]], operator.add]
    evidence: list[Evidence]
    calculations: list[Calculation]
    report: Report
    trace: Annotated[list[TraceEvent], operator.add]


class WorkerInput(TypedDict):
    sub_question: SubQuestion
    topic: Topic | None
    round: int


class FatalToolError(RuntimeError):
    """A tool failure that makes any answer untrustworthy (ends the run as failed)."""


class Node(Protocol):
    def __call__(
        self, state: ResearchState, *, runtime: Runtime[AgentContext]
    ) -> Awaitable[dict[str, Any]]: ...


def traced(name: str, node: Node) -> Node:
    async def run(state: ResearchState, *, runtime: Runtime[AgentContext]) -> dict[str, Any]:
        started = time.perf_counter()
        update = await node(state, runtime=runtime)
        detail = update.pop("_detail", {})
        event = TraceEvent(
            node=name, ms=round((time.perf_counter() - started) * 1000, 1), detail=detail
        )
        return {**update, "trace": [event]}

    return run


# ----------------------------------------------------------------- nodes


async def validate_input(state: ResearchState, *, runtime: Runtime[AgentContext]) -> dict[str, Any]:
    question = " ".join(state.get("question", "").split())
    if not MIN_QUESTION_CHARS <= len(question) <= MAX_QUESTION_CHARS:
        reason = f"question must be {MIN_QUESTION_CHARS}-{MAX_QUESTION_CHARS} characters"
        return {"status": RunStatus.REJECTED, "reason": reason, "_detail": {"ok": False}}
    return {"question": question, "_detail": {"ok": True, "chars": len(question)}}


async def input_guard(state: ResearchState, *, runtime: Runtime[AgentContext]) -> dict[str, Any]:
    # Placeholder until phase 8 (prompt injection, harmful requests). Allows everything.
    return {"_detail": {"verdict": "allow", "implemented": False}}


async def profile(state: ResearchState, *, runtime: Runtime[AgentContext]) -> dict[str, Any]:
    result = profile_question(state["question"])
    return {"profile": result, "_detail": result.model_dump(mode="json")}


async def planner(state: ResearchState, *, runtime: Runtime[AgentContext]) -> dict[str, Any]:
    ctx = runtime.context
    plan, fallback = await make_plan(
        state["question"],
        state["profile"],
        ctx.budget,
        ctx.llm,
        max_output_tokens=ctx.max_output_tokens,
    )
    detail: dict[str, Any] = {
        "planned_by": plan.planned_by,
        "sub_questions": len(plan.sub_questions),
        "queries": sum(len(sq.queries) for sq in plan.sub_questions),
    }
    if fallback:
        detail["llm_fallback"] = fallback
    return {
        "plan": plan,
        "round": 0,
        "pending": [sq.id for sq in plan.sub_questions],
        "_detail": detail,
    }


def dispatch_searches(state: ResearchState, *, runtime: Runtime[AgentContext]) -> list[Send] | str:
    """Fan out one worker per pending sub-question, within the search budget."""
    budget = runtime.context.budget
    plan = state["plan"]
    pending = [sq for sq in plan.sub_questions if sq.id in state.get("pending", [])]
    try:
        budget.start_iteration()
    except BudgetExceeded:
        return "collect"
    sends: list[Send] = []
    for sq in pending:
        granted = budget.reserve_searches(len(sq.queries))
        if granted == 0:
            break
        worker_sq = sq.model_copy(update={"queries": sq.queries[:granted]})
        topic = state["profile"].topic if state.get("round", 0) == 0 else None
        sends.append(
            Send(
                "search_worker",
                WorkerInput(sub_question=worker_sq, topic=topic, round=state.get("round", 0)),
            )
        )
    return sends or "collect"


async def search_worker(state: WorkerInput, *, runtime: Runtime[AgentContext]) -> dict[str, Any]:
    ctx = runtime.context
    sq = state["sub_question"]
    best: dict[str, tuple[SearchHit, str]] = {}
    started = time.perf_counter()
    errors: list[str] = []
    for query in sq.queries:
        try:
            response = await ctx.tools.search_corpus(
                query, top_k=ctx.budget.limits.hits_per_subquestion * 2, topic=state["topic"]
            )
        except ToolCallError as exc:
            if exc.code in FATAL_TOOL_CODES:
                raise FatalToolError(f"{exc.code.value}: {exc.message}") from exc
            errors.append(exc.code.value)
            continue
        for hit in response.hits:
            if hit.chunk_id not in best or hit.score > best[hit.chunk_id][0].score:
                best[hit.chunk_id] = (hit, query)
    ranked = sorted(best.values(), key=lambda pair: (-pair[0].score, pair[0].chunk_id))
    kept = ranked[: ctx.budget.limits.hits_per_subquestion]
    hits = [
        {"sub_question_id": sq.id, "query": query, "round": state["round"], "hit": hit}
        for hit, query in kept
    ]
    event = TraceEvent(
        node="search_worker",
        ms=round((time.perf_counter() - started) * 1000, 1),
        detail={
            "sub_question_id": sq.id,
            "round": state["round"],
            "queries": sq.queries,
            "hits": len(kept),
            "errors": errors,
        },
    )
    return {"hits": hits, "trace": [event]}


def _evidence_from_hits(plan: ResearchPlan, hits: list[dict[str, Any]]) -> list[Evidence]:
    order = {sq.id: i for i, sq in enumerate(plan.sub_questions)}
    seen: set[str] = set()
    ordered = sorted(
        hits, key=lambda h: (order[h["sub_question_id"]], -h["hit"].score, h["hit"].chunk_id)
    )
    evidence: list[Evidence] = []
    for item in ordered:
        hit: SearchHit = item["hit"]
        if hit.chunk_id in seen:
            continue  # first (highest-priority) sub-question keeps a shared passage
        seen.add(hit.chunk_id)
        evidence.append(
            Evidence(
                id=f"E{len(evidence) + 1}",
                sub_question_id=item["sub_question_id"],
                query=item["query"],
                chunk_id=hit.chunk_id,
                doc_id=hit.doc_id,
                title=hit.title,
                section=hit.section,
                text=hit.text,
                char_start=hit.char_start,
                char_end=hit.char_end,
                published=hit.published,
                language=hit.language,
                source_type=hit.source_type,
                publisher=hit.publisher,
                score=hit.score,
            )
        )
    return evidence


async def collect(state: ResearchState, *, runtime: Runtime[AgentContext]) -> dict[str, Any]:
    plan = state["plan"]
    evidence = _evidence_from_hits(plan, state.get("hits", []))
    covered = {e.sub_question_id for e in evidence}
    empty = [sq.id for sq in plan.sub_questions if sq.id not in covered]
    budget = runtime.context.budget
    retry = bool(empty) and budget.can_follow_up()
    pending: list[str] = []
    new_plan = plan
    if retry:
        # Follow-up round: search the sub-question text itself, without the topic filter.
        pending = empty
        subs = [
            sq.model_copy(update={"queries": [sq.text]}) if sq.id in empty else sq
            for sq in plan.sub_questions
        ]
        new_plan = plan.model_copy(update={"sub_questions": subs})
    return {
        "evidence": evidence,
        "plan": new_plan,
        "pending": pending,
        "round": state.get("round", 0) + (1 if retry else 0),
        "_detail": {
            "evidence": len(evidence),
            "documents": len({e.doc_id for e in evidence}),
            "empty_sub_questions": empty,
            "follow_up_round": retry,
        },
    }


def after_collect(state: ResearchState, *, runtime: Runtime[AgentContext]) -> list[Send] | str:
    if state.get("pending"):
        return dispatch_searches(state, runtime=runtime)
    return "analyze"


async def analyze(state: ResearchState, *, runtime: Runtime[AgentContext]) -> dict[str, Any]:
    ctx = runtime.context
    profile_ = state["profile"]
    evidence = state.get("evidence", [])
    if ctx.llm is None or not evidence or profile_.kind not in CALCULATION_KINDS:
        return {"calculations": [], "_detail": {"skipped": True}}
    calculations, notes = await run_calculations(
        state["question"],
        evidence,
        llm=ctx.llm,
        budget=ctx.budget,
        tools=ctx.tools,
        max_output_tokens=ctx.max_output_tokens,
    )
    return {
        "calculations": calculations,
        "_detail": {"calculations": len(calculations), "notes": notes},
    }


async def synthesize(state: ResearchState, *, runtime: Runtime[AgentContext]) -> dict[str, Any]:
    ctx = runtime.context
    profile_ = state["profile"]
    plan = state["plan"]
    evidence = state.get("evidence", [])
    if not evidence:
        report = abstention_report(state["question"], profile_.language, plan)
        return {"report": report, "_detail": {"generated_by": "rules", "abstained": True}}
    detail: dict[str, Any] = {}
    if ctx.llm is not None:
        covered = {e.sub_question_id for e in evidence}
        insufficient = [sq.id for sq in plan.sub_questions if sq.id not in covered]
        try:
            report, stats, trimmed = await llm_report(
                state["question"],
                profile_,
                plan,
                evidence,
                state.get("calculations", []),
                insufficient,
                llm=ctx.llm,
                budget=ctx.budget,
                max_output_tokens=ctx.max_output_tokens,
                max_evidence_tokens=ctx.max_evidence_tokens,
            )
            detail = {"generated_by": "llm", "trimmed_evidence": trimmed, **stats.__dict__}
            return {"report": report, "_detail": detail}
        except (LLMError, BudgetExceeded) as exc:
            detail = {"llm_fallback": f"{type(exc).__name__}: {exc}"}
    report = extractive_report(state["question"], profile_, plan, evidence)
    return {"report": report, "_detail": {"generated_by": "rules", **detail}}


async def finalize(state: ResearchState, *, runtime: Runtime[AgentContext]) -> dict[str, Any]:
    if state.get("status") is RunStatus.REJECTED:
        return {"_detail": {"status": "rejected"}}
    report = state.get("report")
    answered = report is not None and any(s.kind == "claim" for s in report.sentences())
    status = RunStatus.ANSWERED if answered else RunStatus.ABSTAINED
    return {"status": status, "_detail": {"status": status.value}}


def _route_after_validation(state: ResearchState) -> str:
    return "finalize" if state.get("status") is RunStatus.REJECTED else "input_guard"


def build_graph() -> CompiledStateGraph[ResearchState, AgentContext, ResearchState, ResearchState]:
    graph = StateGraph(ResearchState, context_schema=AgentContext)
    graph.add_node("validate_input", traced("validate_input", validate_input))
    graph.add_node("input_guard", traced("input_guard", input_guard))
    graph.add_node("profile", traced("profile", profile))
    graph.add_node("planner", traced("planner", planner))
    graph.add_node("search_worker", search_worker)
    graph.add_node("collect", traced("collect", collect))
    graph.add_node("analyze", traced("analyze", analyze))
    graph.add_node("synthesize", traced("synthesize", synthesize))
    graph.add_node("finalize", traced("finalize", finalize))

    graph.add_edge(START, "validate_input")
    graph.add_conditional_edges(
        "validate_input", _route_after_validation, ["input_guard", "finalize"]
    )
    graph.add_edge("input_guard", "profile")
    graph.add_edge("profile", "planner")
    graph.add_conditional_edges("planner", dispatch_searches, ["search_worker", "collect"])
    graph.add_edge("search_worker", "collect")
    graph.add_conditional_edges("collect", after_collect, ["search_worker", "analyze"])
    graph.add_edge("analyze", "synthesize")
    graph.add_edge("synthesize", "finalize")
    graph.add_edge("finalize", END)
    return graph.compile()
