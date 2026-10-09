from __future__ import annotations

from datetime import date

import pytest
from pydantic import ValidationError

from researchai.agent.budget import BudgetExceeded, Limits, RunBudget
from researchai.agent.models import (
    Citation,
    Report,
    ReportSection,
    ReportSentence,
    ResearchPlan,
    SubQuestion,
    Usage,
)
from researchai.corpus.models import Language
from tests.support import make_settings

LIMITS = Limits(max_searches=3, max_iterations=2, max_tokens=1000, max_subquestions=4,
                hits_per_subquestion=2)  # fmt: skip
CITE = Citation(evidence_id="E1", doc_id="ev-en-0001", chunk_id="ev-en-0001#c0", quote="q")


def test_claims_and_conflicts_must_cite() -> None:
    for kind in ("claim", "conflict"):
        with pytest.raises(ValidationError, match="citation"):
            ReportSentence(text="x", kind=kind)
        assert ReportSentence(text="x", kind=kind, citations=[CITE])
    assert ReportSentence(text="No data.", kind="insufficient_evidence")


def test_citation_anchoring_flag() -> None:
    assert not CITE.anchored
    assert CITE.model_copy(update={"char_start": 1, "char_end": 2}).anchored


def test_plan_ids_unique() -> None:
    sq = SubQuestion(id="sq1", text="What?", queries=["q"])
    with pytest.raises(ValidationError, match="unique"):
        ResearchPlan(sub_questions=[sq, sq], planned_by="rules")


def test_report_helpers() -> None:
    claim = ReportSentence(text="x", kind="claim", citations=[CITE])
    report = Report(
        language=Language.EN,
        title="t",
        summary=[ReportSentence(text="n", kind="note")],
        sections=[ReportSection(heading="h", sub_question_id="sq1", sentences=[claim])],
        generated_by="rules",
    )
    assert [s.kind for s in report.sentences()] == ["note", "claim"]
    assert report.citations() == [CITE]


def test_usage_total() -> None:
    assert Usage(prompt_tokens=3, completion_tokens=4).total_tokens == 7


def test_limits_from_settings() -> None:
    limits = Limits.from_settings(make_settings(max_searches=7, max_iterations=3))
    assert (limits.max_searches, limits.max_iterations) == (7, 3)


def test_search_budget_grants_what_is_left_and_records_exhaustion() -> None:
    budget = RunBudget(LIMITS)
    assert budget.reserve_searches(2) == 2
    assert budget.reserve_searches(2) == 1
    assert budget.searches_left == 0
    assert budget.reserve_searches(1) == 0
    assert budget.exhausted == ["searches"]


def test_iteration_cap() -> None:
    budget = RunBudget(LIMITS)
    budget.start_iteration()
    budget.start_iteration()
    with pytest.raises(BudgetExceeded) as excinfo:
        budget.start_iteration()
    assert excinfo.value.budget == "iterations"
    assert budget.iterations == 2


def test_token_budget_checks_worst_case_before_the_call() -> None:
    budget = RunBudget(LIMITS)
    budget.check_tokens(estimated_prompt=400, max_output=500)
    budget.record_tokens(prompt=450, completion=100)
    assert (budget.tokens, budget.llm_calls) == (550, 1)
    with pytest.raises(BudgetExceeded, match="1050"):
        budget.check_tokens(estimated_prompt=300, max_output=200)
    assert budget.exhausted == ["tokens"]


def test_exhaustion_is_recorded_once() -> None:
    budget = RunBudget(LIMITS)
    for _ in range(3):
        with pytest.raises(BudgetExceeded):
            budget.check_tokens(5000, 1)
    assert budget.exhausted == ["tokens"]


def test_dates_are_plain_dates() -> None:
    assert isinstance(date(2026, 1, 1), date)


def test_follow_up_check_records_the_limiting_budget() -> None:
    rounds = RunBudget(LIMITS)
    rounds.start_iteration()
    rounds.start_iteration()
    assert not rounds.can_follow_up()
    assert rounds.exhausted == ["iterations"]
    searches = RunBudget(LIMITS)
    searches.start_iteration()
    searches.reserve_searches(3)
    assert not searches.can_follow_up()
    assert searches.exhausted == ["searches"]
    fresh = RunBudget(LIMITS)
    fresh.start_iteration()
    assert fresh.can_follow_up()
    assert fresh.exhausted == []
