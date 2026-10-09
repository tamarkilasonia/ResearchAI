"""End-to-end graph runs with fake tools and a scripted LLM."""

from __future__ import annotations

import asyncio
import time
from typing import Any

from researchai.agent.models import RunStatus
from researchai.agent.runner import ResearchAgent
from researchai.corpus.models import Topic
from researchai.tools.errors import ErrorCode
from researchai.tools.schemas import SearchResponse
from tests.agent_fakes import FakeTools, ScriptedLLM, always
from tests.support import make_settings

CHARGING_Q = "How many public charging points are there in Georgia?"
COST_Q = "How much does charging cost per 100 km compared with the tariff?"


def agent(tools: FakeTools, llm: ScriptedLLM | None = None, **settings: Any) -> ResearchAgent:
    return ResearchAgent(tools=tools, llm=llm, settings=make_settings(**settings))


async def test_offline_run_answers_with_cited_claims() -> None:
    tools = FakeTools()
    result = await agent(tools).run(CHARGING_Q, run_id="run-1")
    assert result.run_id == "run-1"
    assert (result.status, result.mode) == (RunStatus.ANSWERED, "offline")
    assert result.report is not None
    assert result.report.generated_by == "rules"
    claims = [s for s in result.report.sentences() if s.kind == "claim"]
    assert claims
    assert all(c.anchored for s in claims for c in s.citations)
    assert result.evidence[0].id == "E1"
    assert tools.topics[0] is Topic.EV  # round 0 uses the detected topic as a filter
    nodes = [event.node for event in result.trace]
    assert nodes[:4] == ["validate_input", "input_guard", "profile", "planner"]
    assert nodes[-3:] == ["analyze", "synthesize", "finalize"]
    assert result.usage.searches == len(result.tool_calls) > 0
    assert result.usage.llm_calls == 0
    assert result.total_ms > 0


async def test_abstains_after_a_follow_up_round_without_topic_filter() -> None:
    tools = FakeTools(corpus=[])
    result = await agent(tools).run("What is the capital of France?")
    assert result.status is RunStatus.ABSTAINED
    assert result.usage.iterations == 2
    assert result.usage.searches == 2
    assert result.report is not None
    assert result.report.insufficient == ["sq1"]
    collects = [e for e in result.trace if e.node == "collect"]
    assert [c.detail["follow_up_round"] for c in collects] == [True, False]


async def test_iteration_cap_stops_follow_up_rounds() -> None:
    result = await agent(FakeTools(corpus=[]), max_iterations=1).run(
        "What is the capital of France?"
    )
    assert result.usage.iterations == 1
    assert result.status is RunStatus.ABSTAINED
    assert result.budget_exhausted == ["iterations"]


async def test_search_budget_caps_parallel_fan_out() -> None:
    tools = FakeTools()
    result = await agent(tools, max_searches=2).run("Compare electric cars in Georgia")
    assert result.usage.searches == 2
    assert "searches" in result.budget_exhausted or result.plan is not None
    assert result.plan is not None
    assert sum(len(sq.queries) for sq in result.plan.sub_questions) <= 2


async def test_search_budget_exhaustion_is_reported_in_follow_up_rounds() -> None:
    result = await agent(FakeTools(corpus=[]), max_searches=1).run("What is the capital of France?")
    assert result.usage.searches == 1
    assert result.usage.iterations == 1  # no budget left for a second round
    assert result.budget_exhausted == ["searches"]


async def test_rejects_invalid_input_without_calling_tools() -> None:
    tools = FakeTools()
    result = await agent(tools).run("  ?  ")
    assert result.status is RunStatus.REJECTED
    assert result.reason is not None
    assert tools.calls == []
    assert result.report is None


async def test_fatal_tool_error_fails_the_run_without_an_answer() -> None:
    result = await agent(FakeTools(fail_with=ErrorCode.INDEX_NOT_READY)).run(CHARGING_Q)
    assert result.status is RunStatus.FAILED
    assert result.report is None
    assert result.reason is not None
    assert "INDEX_NOT_READY" in result.reason
    assert result.tool_calls[0].error_code == "INDEX_NOT_READY"


async def test_non_fatal_tool_error_degrades_to_abstention() -> None:
    result = await agent(FakeTools(fail_with=ErrorCode.INVALID_ARGUMENT)).run(CHARGING_Q)
    assert result.status is RunStatus.ABSTAINED
    worker = next(e for e in result.trace if e.node == "search_worker")
    assert worker.detail["errors"]
    assert set(worker.detail["errors"]) == {"INVALID_ARGUMENT"}


async def test_timeout_fails_the_run() -> None:
    class Slow(FakeTools):
        async def search_corpus(
            self, query: str, *, top_k: int = 8, topic: Topic | None = None
        ) -> SearchResponse:
            await asyncio.sleep(1)
            return await super().search_corpus(query, top_k=top_k, topic=topic)

    result = await agent(Slow(), run_timeout_seconds=0.05).run(CHARGING_Q)
    assert result.status is RunStatus.FAILED
    assert result.reason is not None
    assert "exceeded" in result.reason


async def test_unexpected_crash_fails_the_run_with_a_generic_reason() -> None:
    class Crashing(FakeTools):
        async def search_corpus(
            self, query: str, *, top_k: int = 8, topic: Topic | None = None
        ) -> SearchResponse:
            raise RuntimeError("secret internals")

    result = await agent(Crashing()).run(CHARGING_Q)
    assert (result.status, result.reason) == (RunStatus.FAILED, "internal error")


async def test_search_workers_run_in_parallel() -> None:
    class Slow(FakeTools):
        async def search_corpus(
            self, query: str, *, top_k: int = 8, topic: Topic | None = None
        ) -> SearchResponse:
            await asyncio.sleep(0.2)
            return await super().search_corpus(query, top_k=top_k, topic=topic)

    started = time.perf_counter()
    result = await agent(Slow(), max_iterations=1).run("Compare electric cars in Georgia")
    elapsed = time.perf_counter() - started
    assert result.usage.searches == 6
    assert elapsed < 0.6  # six 0.2 s searches in one round: concurrent, not 1.2 s


async def test_tool_calls_are_sliced_per_run() -> None:
    tools = FakeTools()
    research = agent(tools)
    first = await research.run(CHARGING_Q)
    second = await research.run(CHARGING_Q)
    assert len(tools.calls) == len(first.tool_calls) + len(second.tool_calls)


def _llm_script(report: dict[str, Any] | str) -> ScriptedLLM:
    return ScriptedLLM(
        {
            "LLMPlan": [
                always(
                    {
                        "sub_questions": [
                            {"text": "What is the home tariff?", "queries": ["tariff kWh"]},
                            {
                                "text": "How much energy does a Leaf use?",
                                "queries": ["Nissan Leaf kWh"],
                            },
                        ]
                    }
                )
            ],
            "ProposedCalculations": [
                always(
                    {
                        "calculations": [
                            {
                                "description": "cost per 100 km",
                                "expression": "16.8 * 0.292",
                                "evidence_ids": ["E1", "E2"],
                            }
                        ]
                    }
                )
            ],
            "LLMReport": [always(report)],
        }
    )


REPORT = {
    "title": "Charging cost",
    "summary": [{"text": "Charging at home costs about 4.91 GEL per 100 km.", "kind": "claim",
                 "citations": [{"evidence_id": "E1", "quote": "0.292 GEL per kWh"},
                               {"evidence_id": "E2", "quote": "16.8 kWh per 100 km"}],
                 "calculation_ids": ["C1"]}],
    "sections": [
        {"sub_question_id": "sq1", "heading": "Tariff", "sentences": [
            {"text": "The middle block costs 0.292 GEL per kWh.", "kind": "claim",
             "citations": [{"evidence_id": "E1", "quote": "0.292 GEL per kWh"}]}]},
        {"sub_question_id": "sq2", "heading": "Consumption", "sentences": [
            {"text": "A Leaf uses 16.8 kWh per 100 km.", "kind": "claim",
             "citations": [{"evidence_id": "E2", "quote": "16.8 kWh per 100 km"}]}]},
    ],
}  # fmt: skip


async def test_llm_run_plans_computes_and_writes() -> None:
    tools = FakeTools()
    llm = _llm_script(REPORT)
    result = await agent(tools, llm).run(COST_Q)
    assert (result.status, result.mode) == (RunStatus.ANSWERED, "llm")
    assert result.plan is not None
    assert result.plan.planned_by == "llm"
    assert [round(c.result, 4) for c in result.calculations] == [4.9056]
    assert result.report is not None
    assert result.report.generated_by == "llm"
    assert all(c.anchored for c in result.report.citations())
    assert result.usage.llm_calls == 3
    assert result.usage.total_tokens == 3 * 600
    assert [c.tool for c in result.tool_calls].count("calculate") == 1
    synth = next(e for e in result.trace if e.node == "synthesize")
    assert synth.detail["anchored"] == 4


async def test_llm_writer_failure_falls_back_to_extractive() -> None:
    result = await agent(FakeTools(), _llm_script("not json")).run(COST_Q)
    assert result.status is RunStatus.ANSWERED
    assert result.report is not None
    assert result.report.generated_by == "rules"
    synth = next(e for e in result.trace if e.node == "synthesize")
    assert "LLMError" in synth.detail["llm_fallback"]


async def test_token_budget_forces_rule_fallbacks_but_still_answers() -> None:
    result = await agent(FakeTools(), _llm_script(REPORT), max_tokens_per_run=1000).run(COST_Q)
    assert result.status is RunStatus.ANSWERED
    assert "tokens" in result.budget_exhausted
    assert result.plan is not None
    assert result.plan.planned_by == "rules"
    assert result.report is not None
    assert result.report.generated_by == "rules"
    assert result.usage.llm_calls == 0


async def test_follow_up_round_with_fewer_searches_than_empty_sub_questions() -> None:
    result = await agent(FakeTools(corpus=[]), max_searches=7).run(
        "Compare electric cars in Georgia"
    )
    assert result.usage.searches == 7  # 6 in round one, the last search left in round two
    assert result.usage.iterations == 2
    assert "searches" in result.budget_exhausted


async def test_dispatch_enforces_the_iteration_cap_itself() -> None:
    from langgraph.runtime import Runtime

    from researchai.agent.budget import Limits, RunBudget
    from researchai.agent.graph import AgentContext, dispatch_searches
    from researchai.agent.planner import rule_plan
    from researchai.agent.profile import profile_question

    budget = RunBudget(Limits(10, 1, 1000, 6, 4))
    budget.start_iteration()  # the only allowed round is already used
    context = AgentContext(tools=FakeTools(), llm=None, budget=budget, max_output_tokens=100)
    profile = profile_question(CHARGING_Q)
    plan = rule_plan(CHARGING_Q, profile, budget.limits)
    state = {"plan": plan, "profile": profile, "pending": ["sq1"], "round": 1}
    assert dispatch_searches(state, runtime=Runtime(context=context)) == "collect"  # type: ignore[arg-type]
    assert budget.searches == 0
    assert budget.exhausted == ["iterations"]
