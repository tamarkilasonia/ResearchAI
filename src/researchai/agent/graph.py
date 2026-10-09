"""The research graph (LangGraph).

    validate_input -> input_guard -> profile -> planner
      -> search_worker x N (parallel)  <------------------------------+
      -> collect -> assess_sources -> extract_claims -> check_sufficiency
           (insufficient sub-questions, budget left: follow-up round) -+
      -> detect_conflicts -> analyze -> synthesize -> finalize

Rules first, LLM where rules are unsure: source assessment, claim extraction
and sufficiency are rules; conflict detection is rules plus an optional LLM
pass for disagreements without numbers. Phase 6 adds citation verification
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
from researchai.agent.assess import assess_sources as assess_rules
from researchai.agent.assess import rerank
from researchai.agent.budget import BudgetExceeded, RunBudget
from researchai.agent.claims import extract_claims as extract_rules
from researchai.agent.conflicts import llm_conflicts, rule_conflicts
from researchai.agent.llm import LLMClient, LLMError
from researchai.agent.models import (
    Calculation,
    Claim,
    Conflict,
    Evidence,
    QuestionKind,
    QuestionProfile,
    Report,
    ResearchPlan,
    RunStatus,
    SourceAssessment,
    SubQuestion,
    SufficiencyVerdict,
    TraceEvent,
)
from researchai.agent.planner import make_plan
from researchai.agent.profile import profile_question
from researchai.agent.sufficiency import key_entities, missing_entities, verdicts
from researchai.agent.synthesis import (
    abstention_report,
    extractive_report,
    fit_evidence,
    llm_report,
)
from researchai.agent.verification import Findings
from researchai.corpus.models import Topic
from researchai.mcp_client.client import ToolCallError
from researchai.textutil import keyword_terms
from researchai.tools.errors import ErrorCode
from researchai.tools.schemas import (
    MAX_METADATA_IDS,
    CalculationResponse,
    MetadataResponse,
    SearchHit,
    SearchMode,
    SearchResponse,
    SourceMetadata,
)

logger = logging.getLogger(__name__)

MIN_QUESTION_CHARS = 3
MAX_QUESTION_CHARS = 2000
FATAL_TOOL_CODES = frozenset({ErrorCode.INDEX_NOT_READY, ErrorCode.BACKEND_UNAVAILABLE})
CALCULATION_KINDS = (QuestionKind.NUMERIC, QuestionKind.COMPARISON, QuestionKind.RECOMMENDATION)
FOCUSED_KINDS = (QuestionKind.FACTUAL, QuestionKind.NUMERIC)


class AgentTools(Protocol):
    async def search_corpus(
        self,
        query: str,
        *,
        top_k: int = 8,
        mode: SearchMode = "hybrid",
        topic: Topic | None = None,
    ) -> SearchResponse: ...

    async def get_source_metadata(self, doc_ids: list[str]) -> MetadataResponse: ...

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
    assessments: list[SourceAssessment]
    claims: list[Claim]
    verdicts: list[SufficiencyVerdict]
    missing_entities: list[str]
    conflicts: list[Conflict]
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
    evidence = _evidence_from_hits(state["plan"], state.get("hits", []))
    return {
        "evidence": evidence,
        "_detail": {"evidence": len(evidence), "documents": len({e.doc_id for e in evidence})},
    }


def _fatal(exc: ToolCallError) -> FatalToolError:
    return FatalToolError(f"{exc.code.value}: {exc.message}")


async def _metadata(tools: AgentTools, doc_ids: list[str]) -> list[SourceMetadata]:
    found: list[SourceMetadata] = []
    for start in range(0, len(doc_ids), MAX_METADATA_IDS):
        response = await tools.get_source_metadata(doc_ids[start : start + MAX_METADATA_IDS])
        found.extend(response.sources)
    return found


async def assess_sources(state: ResearchState, *, runtime: Runtime[AgentContext]) -> dict[str, Any]:
    evidence = state.get("evidence", [])
    if not evidence:
        return {"assessments": [], "_detail": {"sources": 0}}
    doc_ids = list(dict.fromkeys(e.doc_id for e in evidence))
    try:
        metadata = await _metadata(runtime.context.tools, doc_ids)
    except ToolCallError as exc:
        if exc.code in FATAL_TOOL_CODES:
            raise _fatal(exc) from exc
        metadata = []  # fall back to dates carried by the evidence itself
    assessments = assess_rules(evidence, metadata)
    flagged = {a.doc_id: a.flags for a in assessments if a.flags}
    return {
        "assessments": assessments,
        "evidence": rerank(evidence, assessments),
        "_detail": {"sources": len(assessments), "flagged": flagged},
    }


async def extract_claims(state: ResearchState, *, runtime: Runtime[AgentContext]) -> dict[str, Any]:
    reliability = {a.doc_id: a.reliability for a in state.get("assessments", [])}
    claims = extract_rules(state.get("evidence", []), reliability)
    facts = sum(len(c.facts) for c in claims)
    return {"claims": claims, "_detail": {"claims": len(claims), "numeric_facts": facts}}


async def check_sufficiency(
    state: ResearchState, *, runtime: Runtime[AgentContext]
) -> dict[str, Any]:
    ctx = runtime.context
    plan = state["plan"]
    evidence = state.get("evidence", [])
    results = verdicts(plan, evidence, state.get("assessments", []))
    missing = state.get("missing_entities", [])
    entity_searches = 0
    if state.get("round", 0) == 0:
        entities = key_entities(state["question"], state["profile"].language)
        try:
            missing, entity_searches = await missing_entities(
                entities, evidence, ctx.tools, ctx.budget.searches_left
            )
        except ToolCallError as exc:
            if exc.code in FATAL_TOOL_CODES:
                raise _fatal(exc) from exc
            missing = []
        ctx.budget.reserve_searches(entity_searches)
    insufficient = [v.sub_question_id for v in results if not v.sufficient]
    follow_up = bool(insufficient) and not missing and ctx.budget.can_follow_up()
    update: dict[str, Any] = {
        "verdicts": results,
        "missing_entities": missing,
        "pending": [],
        "_detail": {
            "insufficient": insufficient,
            "missing_entities": missing,
            "entity_searches": entity_searches,
            "follow_up_round": follow_up,
        },
    }
    if follow_up:
        # Follow-up round: search the sub-question text itself, without the topic filter.
        subs = [
            sq.model_copy(update={"queries": [sq.text]}) if sq.id in insufficient else sq
            for sq in plan.sub_questions
        ]
        update |= {
            "plan": plan.model_copy(update={"sub_questions": subs}),
            "pending": insufficient,
            "round": state.get("round", 0) + 1,
        }
    return update


def after_sufficiency(state: ResearchState, *, runtime: Runtime[AgentContext]) -> list[Send] | str:
    if state.get("pending"):
        return dispatch_searches(state, runtime=runtime)
    return "detect_conflicts"


async def detect_conflicts(
    state: ResearchState, *, runtime: Runtime[AgentContext]
) -> dict[str, Any]:
    ctx = runtime.context
    evidence = state.get("evidence", [])
    claims = list(state.get("claims", []))
    conflicts = rule_conflicts(claims)
    notes: list[str] = []
    llm_found = 0
    if ctx.llm is not None and len({e.doc_id for e in evidence}) >= 2:
        fitted, _ = fit_evidence(evidence, ctx.max_evidence_tokens)
        reliability = {a.doc_id: a.reliability for a in state.get("assessments", [])}
        new_claims, new_conflicts, notes = await llm_conflicts(
            state["question"],
            fitted,
            claims,
            conflicts,
            reliability,
            llm=ctx.llm,
            budget=ctx.budget,
            max_output_tokens=ctx.max_output_tokens,
        )
        claims += new_claims
        conflicts += new_conflicts
        llm_found = len(new_conflicts)
    return {
        "claims": claims,
        "conflicts": conflicts,
        "_detail": {
            "rules": len(conflicts) - llm_found,
            "llm": llm_found,
            "outdated": sum(c.kind == "outdated" for c in conflicts),
            "contradictions": sum(c.kind == "contradiction" for c in conflicts),
            "notes": notes,
        },
    }


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


def findings_of(state: ResearchState) -> Findings:
    question_terms = keyword_terms(state["question"])
    relevance = tuple(
        (sq.id, frozenset(keyword_terms(" ".join([*question_terms, sq.text, *sq.queries]))))
        for sq in state["plan"].sub_questions
    )
    return Findings(
        assessments=tuple(state.get("assessments", [])),
        claims=tuple(state.get("claims", [])),
        conflicts=tuple(state.get("conflicts", [])),
        verdicts=tuple(state.get("verdicts", [])),
        missing_entities=tuple(state.get("missing_entities", [])),
        relevance_terms=relevance,
    )


def abstain_reason(state: ResearchState, findings: Findings) -> str | None:
    """Why the run must abstain instead of writing a report, if it must."""
    if not state.get("evidence"):
        return "no evidence"
    if findings.verdicts and not any(v.sufficient for v in findings.verdicts):
        return "no sub-question has sufficient evidence"
    if findings.missing_entities and state["profile"].kind in FOCUSED_KINDS:
        return f"the corpus never mentions {', '.join(findings.missing_entities)}"
    return None


async def synthesize(state: ResearchState, *, runtime: Runtime[AgentContext]) -> dict[str, Any]:
    ctx = runtime.context
    profile_ = state["profile"]
    plan = state["plan"]
    evidence = state.get("evidence", [])
    findings = findings_of(state)
    if reason := abstain_reason(state, findings):
        report = abstention_report(
            state["question"], profile_.language, plan, findings.missing_entities
        )
        return {"report": report, "_detail": {"generated_by": "rules", "abstained": reason}}
    detail: dict[str, Any] = {}
    if ctx.llm is not None:
        try:
            report, stats, trimmed = await llm_report(
                state["question"],
                profile_,
                plan,
                evidence,
                state.get("calculations", []),
                findings,
                llm=ctx.llm,
                budget=ctx.budget,
                max_output_tokens=ctx.max_output_tokens,
                max_evidence_tokens=ctx.max_evidence_tokens,
            )
            detail = {"generated_by": "llm", "trimmed_evidence": trimmed, **stats.__dict__}
            return {"report": report, "_detail": detail}
        except (LLMError, BudgetExceeded) as exc:
            detail = {"llm_fallback": f"{type(exc).__name__}: {exc}"}
    report = extractive_report(state["question"], profile_, plan, evidence, findings)
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
    graph.add_node("assess_sources", traced("assess_sources", assess_sources))
    graph.add_node("extract_claims", traced("extract_claims", extract_claims))
    graph.add_node("check_sufficiency", traced("check_sufficiency", check_sufficiency))
    graph.add_node("detect_conflicts", traced("detect_conflicts", detect_conflicts))
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
    graph.add_edge("collect", "assess_sources")
    graph.add_edge("assess_sources", "extract_claims")
    graph.add_edge("extract_claims", "check_sufficiency")
    graph.add_conditional_edges(
        "check_sufficiency", after_sufficiency, ["search_worker", "detect_conflicts"]
    )
    graph.add_edge("detect_conflicts", "analyze")
    graph.add_edge("analyze", "synthesize")
    graph.add_edge("synthesize", "finalize")
    graph.add_edge("finalize", END)
    return graph.compile()
