from __future__ import annotations

import pytest

from researchai.agent.budget import Limits, RunBudget
from researchai.agent.llm import LLMError
from researchai.agent.models import QuestionKind, ResearchPlan, SubQuestion
from researchai.agent.planner import LLMPlan, fit_to_budget, from_llm, make_plan, rule_plan
from researchai.agent.profile import detect_language, profile_question
from researchai.corpus.models import Language, Topic
from tests.agent_fakes import ScriptedLLM, always

LIMITS = Limits(max_searches=12, max_iterations=2, max_tokens=40_000, max_subquestions=6,
                hits_per_subquestion=4)  # fmt: skip


@pytest.mark.parametrize(
    ("question", "language", "topic", "kind"),
    [
        ("Compare the advantages and disadvantages of electric cars in Georgia and give me a "
         "recommendation.", "en", "ev", "recommendation"),
        ("How many public charging points are there in Georgia?", "en", "ev", "numeric"),
        ("What is the small business tax rate for an individual entrepreneur?", "en", "tax",
         "factual"),
        ("შეადარე ელექტრომობილის დადებითი და უარყოფითი მხარეები", "ka", "ev", "comparison"),
        ("რამდენი ღირს Nissan Leaf-ის დატენვა 100 კმ-ზე?", "ka", "ev", "numeric"),
        ("ღირს თუ არა ელექტრომობილის ყიდვა?", "ka", "ev", "recommendation"),
        ("Is the VAT exemption for EVs still valid?", "en", "ev", "factual"),
        ("What is the capital of France?", "en", None, "factual"),
        ("Does every bus need 16 seats?", "en", None, "factual"),
    ],
)  # fmt: skip
def test_profile(question: str, language: str, topic: str | None, kind: str) -> None:
    profile = profile_question(question)
    assert profile.language.value == language
    assert (profile.topic.value if profile.topic else None) == topic
    assert profile.kind.value == kind


def test_broad_questions_cover_the_topic_and_focused_ones_do_not() -> None:
    broad = profile_question("Compare electric cars in Georgia")
    assert len(broad.aspects) == 6
    focused = profile_question("Should a freelancer register for VAT?")
    assert focused.kind is QuestionKind.RECOMMENDATION
    assert focused.aspects == ["registration", "vat"]


def test_detect_language() -> None:
    assert detect_language("Tesla Model 3 ფასი რამდენია?") is Language.KA
    assert detect_language("Price of a Tesla in ლარი") is Language.EN


def test_rule_plan_for_broad_georgian_question_uses_both_languages() -> None:
    profile = profile_question("შეადარე ელექტრომობილის დადებითი და უარყოფითი მხარეები")
    plan = rule_plan("q", profile, LIMITS)
    assert plan.planned_by == "rules"
    assert len(plan.sub_questions) == 6
    first = plan.sub_questions[0]
    assert first.aspect == "prices"
    assert len(first.queries) == 2
    assert any(ord(ch) > 0x10CF for ch in first.text)


def test_rule_plan_for_focused_question_searches_the_question_first() -> None:
    question = "How many public charging points are there in Georgia?"
    plan = rule_plan(question, profile_question(question), LIMITS)
    assert plan.sub_questions[0].queries[0] == question
    assert plan.sub_questions[0].aspect == "charging"


def test_rule_plan_without_topic_is_one_sub_question() -> None:
    plan = rule_plan(
        "What is the capital of France?", profile_question("What is the capital of France?"), LIMITS
    )
    assert [sq.queries for sq in plan.sub_questions] == [["What is the capital of France?"]]


def test_rule_plan_respects_max_subquestions() -> None:
    limits = Limits(12, 2, 40_000, 3, 4)
    plan = rule_plan("Compare electric cars", profile_question("Compare electric cars"), limits)
    assert len(plan.sub_questions) == 3


def _plan(*query_counts: int) -> ResearchPlan:
    return ResearchPlan(
        sub_questions=[
            SubQuestion(
                id=f"sq{i + 1}", text=f"Question {i}", queries=[f"q{i}-{j}" for j in range(n)]
            )
            for i, n in enumerate(query_counts)
        ],
        planned_by="rules",
    )


def test_fit_to_budget_keeps_first_queries_first() -> None:
    fitted = fit_to_budget(_plan(2, 2, 2), 4)
    assert [len(sq.queries) for sq in fitted.sub_questions] == [2, 1, 1]
    dropped = fit_to_budget(_plan(1, 1, 1), 2)
    assert [sq.id for sq in dropped.sub_questions] == ["sq1", "sq2"]
    assert fit_to_budget(_plan(1, 2), 5) == _plan(1, 2)
    with pytest.raises(ValueError, match="no searches"):
        fit_to_budget(_plan(1), 0)


def test_from_llm_assigns_ids_trims_and_rejects_empty() -> None:
    raw = LLMPlan.model_validate(
        {
            "sub_questions": [
                {"text": " What A? ", "queries": [" a ", "  "]},
                {"text": "What B?", "queries": ["b"]},
            ]
        }
    )
    plan = from_llm(raw, Limits(12, 2, 40_000, 1, 4))
    assert [(sq.id, sq.text, sq.queries) for sq in plan.sub_questions] == [
        ("sq1", "What A?", ["a"])
    ]
    assert plan.planned_by == "llm"
    with pytest.raises(LLMError):
        from_llm(
            LLMPlan.model_validate({"sub_questions": [{"text": "Why?", "queries": [" "]}]}), LIMITS
        )


async def test_make_plan_uses_llm_and_falls_back_to_rules() -> None:
    question = "How many public charging points are there?"
    profile = profile_question(question)
    llm = ScriptedLLM(
        {
            "LLMPlan": [
                always(
                    {
                        "sub_questions": [
                            {"text": "How many chargers?", "queries": ["chargers 2026"]}
                        ]
                    }
                )
            ]
        }
    )
    plan, fallback = await make_plan(
        question, profile, RunBudget(LIMITS), llm, max_output_tokens=500
    )
    assert (plan.planned_by, fallback) == ("llm", None)

    broken = ScriptedLLM({"LLMPlan": [always("garbage")]})
    plan, fallback = await make_plan(
        question, profile, RunBudget(LIMITS), broken, max_output_tokens=500
    )
    assert plan.planned_by == "rules"
    assert fallback is not None
    assert "LLMError" in fallback

    plan, fallback = await make_plan(
        question, profile, RunBudget(LIMITS), None, max_output_tokens=500
    )
    assert (plan.planned_by, fallback) == ("rules", None)


async def test_make_plan_fits_the_remaining_search_budget() -> None:
    budget = RunBudget(Limits(2, 2, 40_000, 6, 4))
    plan, _ = await make_plan(
        "Compare electric cars",
        profile_question("Compare electric cars"),
        budget,
        None,
        max_output_tokens=500,
    )
    assert sum(len(sq.queries) for sq in plan.sub_questions) == 2


def test_topic_enum_values() -> None:
    assert Topic("ev") is Topic.EV
