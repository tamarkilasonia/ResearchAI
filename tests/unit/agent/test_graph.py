"""End-to-end graph runs with fake tools and a scripted LLM."""

from __future__ import annotations

import asyncio
import time
from datetime import date
from typing import Any

from researchai.agent.models import RunStatus
from researchai.agent.runner import ResearchAgent
from researchai.corpus.models import SourceType, Topic
from researchai.tools.errors import ErrorCode
from researchai.tools.schemas import SearchMode, SearchResponse
from tests.agent_fakes import FakeTools, ScriptedLLM, always, hit
from tests.support import make_settings

CHARGING_Q = "How many public charging points are there?"
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
    assert nodes[-4:] == ["analyze", "synthesize", "verify_citations", "finalize"]
    assert result.citation_report is not None
    assert result.citation_report.precision == 1.0
    assert result.usage.searches == sum(c.tool == "search_corpus" for c in result.tool_calls) > 0
    assert any(c.tool == "get_source_metadata" for c in result.tool_calls)
    assert result.usage.llm_calls == 0
    assert result.total_ms > 0


async def test_abstains_after_a_follow_up_round_without_topic_filter() -> None:
    tools = FakeTools(corpus=[])
    result = await agent(tools).run("what is the airspeed of an unladen swallow?")
    assert result.status is RunStatus.ABSTAINED
    assert result.usage.iterations == 2
    assert result.usage.searches == 2
    assert result.report is not None
    assert result.report.insufficient == ["sq1"]
    collects = [e for e in result.trace if e.node == "check_sufficiency"]
    assert [c.detail["follow_up_round"] for c in collects] == [True, False]


async def test_iteration_cap_stops_follow_up_rounds() -> None:
    result = await agent(FakeTools(corpus=[]), max_iterations=1).run(
        "what is the airspeed of an unladen swallow?"
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
    result = await agent(FakeTools(corpus=[]), max_searches=1).run(
        "what is the airspeed of an unladen swallow?"
    )
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
            self,
            query: str,
            *,
            top_k: int = 8,
            mode: SearchMode = "hybrid",
            topic: Topic | None = None,
        ) -> SearchResponse:
            await asyncio.sleep(1)
            return await super().search_corpus(query, top_k=top_k, mode=mode, topic=topic)

    result = await agent(Slow(), run_timeout_seconds=0.05).run(CHARGING_Q)
    assert result.status is RunStatus.FAILED
    assert result.reason is not None
    assert "exceeded" in result.reason


async def test_unexpected_crash_fails_the_run_with_a_generic_reason() -> None:
    class Crashing(FakeTools):
        async def search_corpus(
            self,
            query: str,
            *,
            top_k: int = 8,
            mode: SearchMode = "hybrid",
            topic: Topic | None = None,
        ) -> SearchResponse:
            raise RuntimeError("secret internals")

    result = await agent(Crashing()).run(CHARGING_Q)
    assert (result.status, result.reason) == (RunStatus.FAILED, "internal error")


async def test_search_workers_run_in_parallel() -> None:
    class Slow(FakeTools):
        async def search_corpus(
            self,
            query: str,
            *,
            top_k: int = 8,
            mode: SearchMode = "hybrid",
            topic: Topic | None = None,
        ) -> SearchResponse:
            await asyncio.sleep(0.2)
            return await super().search_corpus(query, top_k=top_k, mode=mode, topic=topic)

    started = time.perf_counter()
    result = await agent(Slow(), max_iterations=1).run("Compare electric cars")
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
    result = await agent(FakeTools(corpus=[]), max_searches=7).run("Compare electric cars")
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


async def test_focused_question_about_an_unknown_entity_abstains_and_names_it() -> None:
    tools = FakeTools()
    result = await agent(tools).run("How many public charging points are there in Japan?")
    assert result.status is RunStatus.ABSTAINED
    assert result.missing_entities == ["Japan"]
    assert result.usage.iterations == 1  # searching again cannot find what the corpus lacks
    assert result.report is not None
    assert any("Japan" in s.text for s in result.report.summary)
    keyword = [c for c in tools.calls if c.arguments.get("query") == "Japan"]
    assert len(keyword) == 1


async def test_broad_question_with_an_unknown_entity_answers_the_rest() -> None:
    result = await agent(FakeTools()).run("Compare public charging and tariff kWh costs in Japan")
    assert result.missing_entities == ["Japan"]
    assert result.status is RunStatus.ANSWERED
    assert result.report is not None
    assert result.report.summary[1].kind == "insufficient_evidence"
    assert "Japan" in result.report.summary[1].text


async def test_entities_found_in_evidence_cost_no_extra_search() -> None:
    tools = FakeTools()
    result = await agent(tools).run("What does the Nissan Leaf use per 100 km?")
    assert result.missing_entities == []
    assert not any(c.arguments.get("query") in ("Nissan", "Leaf") for c in tools.calls)


OLDER_REGISTRY = hit(
    "ev-en-0014#c0",
    "Public charging points registry: 385 public charging points.",
    published=date(2025, 2, 12),
)
NEWER_REGISTRY = hit(
    "ev-en-0013#c1",
    "Public charging points registry: 512 public charging points.",
    published=date(2026, 7, 15),
)


async def test_outdated_values_are_flagged_and_not_reported_as_current() -> None:
    tools = FakeTools(corpus=[NEWER_REGISTRY, OLDER_REGISTRY])
    result = await agent(tools).run(CHARGING_Q)
    (conflict,) = result.conflicts
    assert conflict.kind == "outdated"
    assert result.report is not None
    sentences = result.report.sentences()
    claims = [s for s in sentences if s.kind == "claim"]
    assert all(s.citations[0].doc_id == "ev-en-0013" for s in claims)
    assert sum(s.kind == "conflict" for s in sentences) == 1
    assert {a.doc_id for a in result.assessments} == {"ev-en-0013", "ev-en-0014"}
    assert result.sufficiency[0].sufficient


async def test_metadata_failure_degrades_but_backend_failure_fails() -> None:
    soft = await agent(FakeTools(metadata_error=ErrorCode.INVALID_ARGUMENT)).run(CHARGING_Q)
    assert soft.status is RunStatus.ANSWERED
    assert soft.assessments  # assessed from the evidence's own dates
    hard = await agent(FakeTools(metadata_error=ErrorCode.BACKEND_UNAVAILABLE)).run(CHARGING_Q)
    assert hard.status is RunStatus.FAILED


async def test_entity_check_failures() -> None:
    question = "How many public charging points are there in Japan?"
    soft = await agent(FakeTools(keyword_error=ErrorCode.INVALID_ARGUMENT)).run(question)
    assert soft.missing_entities == []  # unconfirmed: benefit of the doubt
    hard = await agent(FakeTools(keyword_error=ErrorCode.INDEX_NOT_READY)).run(question)
    assert hard.status is RunStatus.FAILED


async def test_only_low_reliability_evidence_means_abstention() -> None:
    forum = hit(
        "ev-en-0025#c0",
        "Someone said there are public charging points everywhere.",
        source_type=SourceType.FORUM,
    )
    result = await agent(FakeTools(corpus=[forum]), max_iterations=1).run(CHARGING_Q)
    assert result.status is RunStatus.ABSTAINED
    synth = next(e for e in result.trace if e.node == "synthesize")
    assert synth.detail["abstained"] == "no sub-question has sufficient evidence"


async def test_llm_run_adds_conflicts_the_writer_left_out() -> None:
    tools = FakeTools(corpus=[NEWER_REGISTRY, OLDER_REGISTRY])
    report = {
        "title": "Chargers",
        "summary": [],
        "sections": [
            {"sub_question_id": "sq1", "heading": "Count", "sentences": [
                {"text": "There are 512 public charging points.", "kind": "claim",
                 "citations": [{"evidence_id": "E1", "quote": "512 public charging points"}]}]}
        ],
    }  # fmt: skip
    llm = ScriptedLLM(
        {
            "LLMPlan": [always({"sub_questions": [{"text": "How many public charging points?",
                                                   "queries": ["public charging points"]}]})],
            "Disagreements": [always({"disagreements": []})],
            "ProposedCalculations": [always({"calculations": []})],
            "LLMReport": [always(report)],
        }
    )  # fmt: skip
    result = await agent(tools, llm).run(CHARGING_Q)
    assert result.status is RunStatus.ANSWERED
    synth = next(e for e in result.trace if e.node == "synthesize")
    assert synth.detail["templated_conflicts"] == 1
    assert result.report is not None
    assert any(s.kind == "conflict" for s in result.report.sentences())
    conflicts_node = next(e for e in result.trace if e.node == "detect_conflicts")
    assert (conflicts_node.detail["rules"], conflicts_node.detail["llm"]) == (1, 0)


async def test_llm_report_names_missing_entities_first() -> None:
    llm = _llm_script(REPORT)
    result = await agent(FakeTools(), llm).run(
        "Compare the tariff kWh charging cost per 100 km in Georgia and Japan"
    )
    assert "Japan" in result.missing_entities
    assert result.report is not None
    assert result.report.generated_by == "llm"
    assert "Japan" in result.report.summary[0].text


async def test_unverifiable_citations_fail_the_run() -> None:
    result = await agent(FakeTools(fetch_error=ErrorCode.INTERNAL)).run(CHARGING_Q)
    assert result.status is RunStatus.FAILED
    assert result.report is None
    assert result.citation_report is None
