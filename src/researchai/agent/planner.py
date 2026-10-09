"""Research planning: rules always work; the LLM refines when available."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from researchai.agent.budget import BudgetExceeded, Limits, RunBudget
from researchai.agent.llm import LLMClient, LLMError, structured_call
from researchai.agent.models import QuestionKind, QuestionProfile, ResearchPlan, SubQuestion
from researchai.agent.profile import aspect
from researchai.agent.prompts import PLANNER_SYSTEM, planner_user
from researchai.corpus.models import Language

MAX_QUERY_CHARS = 200
FOCUSED_KINDS = (QuestionKind.FACTUAL, QuestionKind.NUMERIC)


def rule_plan(question: str, profile: QuestionProfile, limits: Limits) -> ResearchPlan:
    """Deterministic plan from the question profile and the aspect catalogue."""
    georgian = profile.language is Language.KA
    subs: list[SubQuestion] = []
    if profile.kind in FOCUSED_KINDS or not profile.aspects:
        # The question itself is the best query for a focused question.
        first_queries = [question[:MAX_QUERY_CHARS]]
        if profile.aspects:
            first = aspect(profile.aspects[0])
            first_queries.append(first.query_en)
        subs.append(
            SubQuestion(
                id="sq1",
                text=question[:300],
                queries=first_queries,
                aspect=profile.aspects[0] if profile.aspects else None,
            )
        )
        remaining = profile.aspects[1:]
    else:
        remaining = profile.aspects
    for key in remaining:
        if len(subs) >= limits.max_subquestions:
            break
        item = aspect(key)
        subs.append(
            SubQuestion(
                id=f"sq{len(subs) + 1}",
                text=item.question_ka if georgian else item.question_en,
                queries=[item.query_ka, item.query_en] if georgian else [item.query_en],
                aspect=key,
            )
        )
    return ResearchPlan(sub_questions=subs, planned_by="rules")


def fit_to_budget(plan: ResearchPlan, searches: int) -> ResearchPlan:
    """Keep at most ``searches`` queries: every sub-question's first query before any second.

    Sub-questions that get no query are dropped. Returns the plan unchanged if it fits.
    """
    if searches < 1:
        raise ValueError("no searches left to plan with")
    total = sum(len(sq.queries) for sq in plan.sub_questions)
    if total <= searches:
        return plan
    # total > searches, so this always stops on the budget before running out of queries.
    kept: dict[str, int] = {sq.id: 0 for sq in plan.sub_questions}
    granted = 0
    depth = 0
    while granted < searches:
        for sq in plan.sub_questions:
            if depth < len(sq.queries) and granted < searches:
                kept[sq.id] += 1
                granted += 1
        depth += 1
    subs = [
        sq.model_copy(update={"queries": sq.queries[: kept[sq.id]]})
        for sq in plan.sub_questions
        if kept[sq.id] > 0
    ]
    return ResearchPlan(sub_questions=subs, planned_by=plan.planned_by)


class _LLMSubQuestion(BaseModel):
    model_config = ConfigDict(extra="forbid")
    text: str = Field(min_length=3, max_length=300)
    queries: list[str] = Field(min_length=1, max_length=3)


class LLMPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")
    sub_questions: list[_LLMSubQuestion] = Field(min_length=1, max_length=10)


def from_llm(raw: LLMPlan, limits: Limits) -> ResearchPlan:
    subs: list[SubQuestion] = []
    for item in raw.sub_questions[: limits.max_subquestions]:
        queries = [q.strip()[:MAX_QUERY_CHARS] for q in item.queries if q.strip()]
        if not queries:
            continue
        subs.append(
            SubQuestion(id=f"sq{len(subs) + 1}", text=item.text.strip(), queries=queries[:3])
        )
    if not subs:
        raise LLMError("schema", "plan has no usable sub-questions")
    return ResearchPlan(sub_questions=subs, planned_by="llm")


async def make_plan(
    question: str,
    profile: QuestionProfile,
    budget: RunBudget,
    llm: LLMClient | None,
    *,
    max_output_tokens: int,
) -> tuple[ResearchPlan, str | None]:
    """Plan with the LLM if available, else (or on any failure) with rules.

    Returns the plan and, when the rule path was used despite an LLM, the reason.
    """
    limits = budget.limits
    fallback: str | None = None
    plan: ResearchPlan | None = None
    if llm is not None:
        try:
            result = await structured_call(
                llm,
                budget,
                system=PLANNER_SYSTEM.replace("{max_subquestions}", str(limits.max_subquestions)),
                user=planner_user(question, profile),
                schema=LLMPlan,
                max_tokens=min(max_output_tokens, 1024),
            )
            plan = from_llm(result.value, limits)
        except (LLMError, BudgetExceeded) as exc:
            fallback = f"{type(exc).__name__}: {exc}"
    if plan is None:
        plan = rule_plan(question, profile, limits)
    return fit_to_budget(plan, budget.searches_left), fallback
