from __future__ import annotations

import pytest

from researchai.agent.analysis import (
    ProposedCalculations,
    check_grounded,
    evidence_numbers,
    operands,
    run_calculations,
)
from researchai.agent.budget import Limits, RunBudget
from researchai.agent.graph import _evidence_from_hits
from researchai.agent.llm import LLMError
from researchai.agent.models import ResearchPlan, SubQuestion
from researchai.mcp_client.client import ToolCallError
from researchai.tools.errors import ErrorCode
from tests.agent_fakes import CONSUMPTION, TARIFF, FakeTools, ScriptedLLM, always

PLAN = ResearchPlan(
    sub_questions=[SubQuestion(id="sq1", text="Cost per 100 km?", queries=["q"])],
    planned_by="rules",
)
EVIDENCE = _evidence_from_hits(
    PLAN,
    [
        {"sub_question_id": "sq1", "query": "q", "hit": CONSUMPTION},
        {"sub_question_id": "sq1", "query": "q", "hit": TARIFF},
    ],
)
BY_ID = {e.id: e for e in EVIDENCE}


def budget() -> RunBudget:
    return RunBudget(Limits(10, 2, 40_000, 5, 3))


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("1,800 USD", {1800.0, 1.8}),
        ("0,55 ლარი", {55.0, 0.55}),
        ("86 900 ლარი", {86900.0, 86.0, 900.0}),
        ("16.8 kWh", {16.8}),
        ("1.800,5", {1800.5, 1.8005}),
        ("no numbers", set()),
        ("version 1.2.3", set()),  # not a number under any convention
    ],
)
def test_evidence_numbers(text: str, expected: set[float]) -> None:
    assert evidence_numbers(text) == expected


def test_operands_and_rejections() -> None:
    assert sorted(operands("(16.8 * 0.292) / 2")) == [0.292, 2.0, 16.8]
    for bad in ("x * 2", "abs(-1)", "a.b", "'1' + 1", "1 +", "True * 2", "210", "-5"):
        with pytest.raises(LLMError):
            operands(bad)


def _proposal(expression: str, ids: list[str]) -> ProposedCalculations:
    return ProposedCalculations.model_validate(
        {
            "calculations": [
                {"description": "cost per 100 km", "expression": expression, "evidence_ids": ids}
            ]
        }
    )


def test_grounding_check() -> None:
    ok = _proposal("16.8 * 0.292", ["E1", "E2"]).calculations[0]
    assert check_grounded(ok, BY_ID) is None
    unit = _proposal("16.8 * 0.292 * 100 / 100", ["E1", "E2"]).calculations[0]
    assert check_grounded(unit, BY_ID) is None
    invented = _proposal("16.8 * 0.35", ["E1", "E2"]).calculations[0]
    assert "0.35" in str(check_grounded(invented, BY_ID))
    wrong_source = _proposal("16.8 * 0.292", ["E1"]).calculations[0]
    assert check_grounded(wrong_source, BY_ID) is not None
    unknown = _proposal("16.8 * 2", ["E9"]).calculations[0]
    assert "unknown" in str(check_grounded(unknown, BY_ID))
    not_math = _proposal("sqrt(16.8) * 2", ["E1"]).calculations[0]
    assert "not allowed" in str(check_grounded(not_math, BY_ID))


async def test_run_calculations_executes_only_grounded_proposals() -> None:
    llm = ScriptedLLM(
        {
            "ProposedCalculations": [
                always(
                    {
                        "calculations": [
                            {
                                "description": "home cost per 100 km",
                                "expression": "16.8 * 0.292",
                                "evidence_ids": ["E1", "E2"],
                            },
                            {
                                "description": "made up",
                                "expression": "16.8 * 0.4",
                                "evidence_ids": ["E1", "E2"],
                            },
                        ]
                    }
                )
            ]
        }
    )
    tools = FakeTools()
    results, notes = await run_calculations(
        "q", EVIDENCE, llm=llm, budget=budget(), tools=tools, max_output_tokens=500
    )
    assert [(c.id, round(c.result, 4)) for c in results] == [("C1", 4.9056)]
    assert any("rejected" in note for note in notes)
    assert [c.tool for c in tools.calls] == ["calculate"]


async def test_run_calculations_tolerates_llm_and_tool_failures() -> None:
    results, notes = await run_calculations(
        "q",
        EVIDENCE,
        llm=ScriptedLLM({}),
        budget=budget(),
        tools=FakeTools(),
        max_output_tokens=500,
    )
    assert results == []
    assert "no calculations" in notes[0]

    class BrokenCalculator(FakeTools):
        async def calculate(self, expression: str):  # type: ignore[no-untyped-def]
            raise ToolCallError("calculate", ErrorCode.DIVISION_BY_ZERO, "x")

    llm = ScriptedLLM(
        {
            "ProposedCalculations": [
                always(
                    {
                        "calculations": [
                            {
                                "description": "ratio",
                                "expression": "16.8 / 0.292",
                                "evidence_ids": ["E1", "E2"],
                            }
                        ]
                    }
                )
            ]
        }
    )
    results, notes = await run_calculations(
        "q", EVIDENCE, llm=llm, budget=budget(), tools=BrokenCalculator(), max_output_tokens=500
    )
    assert results == []
    assert "DIVISION_BY_ZERO" in notes[0]
