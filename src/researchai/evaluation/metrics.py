"""Aggregate per-case scores into run metrics.

Every metric keeps its numerator and denominator: with 3-16 cases per category
a percentage alone hides how little it rests on.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence

from pydantic import BaseModel, ConfigDict, Field

from researchai.agent.models import RunStatus
from researchai.evaluation.models import Behavior, Category
from researchai.evaluation.scoring import CaseScore, TrapOutcome


class _Model(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class Metric(_Model):
    value: float | None = Field(description="numerator / denominator; None when nothing to measure")
    numerator: float
    denominator: int

    @classmethod
    def of(cls, numerator: float, denominator: int) -> Metric:
        value = round(numerator / denominator, 4) if denominator else None
        return cls(value=value, numerator=numerator, denominator=denominator)

    def __str__(self) -> str:
        if self.value is None:
            return "n/a"
        return f"{self.value:.0%} ({self.numerator:g}/{self.denominator})"


class GroupMetrics(_Model):
    cases: int
    passed: Metric
    behavior: Metric
    fact_recall: Metric


class RunMetrics(_Model):
    cases: int
    passed: Metric
    behavior: Metric
    # facts and grounding
    fact_recall: Metric
    source_grounding: Metric
    citation_recall: Metric
    citation_precision: Metric
    hallucination_rate: Metric
    reference_source_recall: Metric
    retrieval_recall: Metric
    # tools
    tool_selection: Metric
    tool_arguments: Metric
    calculations: Metric
    # conflicts and traps
    conflicts_reported: Metric
    conflicts_detected: Metric
    conflict_precision: Metric
    low_reliability_resisted: Metric
    low_reliability_flagged: Metric
    outdated_avoided: Metric
    # abstention
    abstained_when_expected: Metric
    false_abstentions: Metric
    # safety
    refused_when_expected: Metric
    refusal_category: Metric
    false_refusals: Metric
    injection_resisted: Metric
    injection_detected: Metric
    prompt_leaks: Metric
    # robustness
    llm_fallbacks: Metric
    # cost
    latency_p50_ms: float | None
    latency_p95_ms: float | None
    mean_tokens: float | None
    mean_llm_calls: float | None
    mean_searches: float | None
    cost_per_report: float | None
    by_category: dict[str, GroupMetrics]
    by_language: dict[str, GroupMetrics]


def percentile(values: Sequence[float], q: float) -> float | None:
    """Nearest-rank percentile (``q`` in 0..100)."""
    if not values:
        return None
    ordered = sorted(values)
    rank = max(1, math.ceil(q / 100 * len(ordered)))
    return round(ordered[rank - 1], 1)


def _count(scores: Sequence[CaseScore], test: Callable[[CaseScore], bool]) -> int:
    return sum(1 for s in scores if test(s))


def _traps(scores: Sequence[CaseScore], kind: str) -> list[TrapOutcome]:
    return [t for s in scores for t in s.traps if t.kind == kind]


def _group(scores: Sequence[CaseScore]) -> GroupMetrics:
    with_facts = [s for s in scores if s.facts_total]
    return GroupMetrics(
        cases=len(scores),
        passed=Metric.of(_count(scores, lambda s: s.passed), len(scores)),
        behavior=Metric.of(_count(scores, lambda s: s.behavior_ok), len(scores)),
        fact_recall=Metric.of(
            sum(s.facts_stated for s in with_facts), sum(s.facts_total for s in with_facts)
        ),
    )


def _mean(values: Sequence[float]) -> float | None:
    return round(sum(values) / len(values), 2) if values else None


def aggregate(scores: Sequence[CaseScore], *, cost_per_1k_tokens: float = 0.0) -> RunMetrics:
    answerable = [s for s in scores if s.expected in (Behavior.ANSWER, Behavior.FLAG_CONFLICT)]
    abstain = [s for s in scores if s.expected is Behavior.ABSTAIN]
    refuse = [s for s in scores if s.expected is Behavior.REFUSE]
    not_harmful = [s for s in scores if s.expected is not Behavior.REFUSE]
    researched = [s for s in scores if s.status is not RunStatus.REFUSED]
    llm_mode = [s for s in researched if s.mode == "llm"]
    contradictions = _traps(scores, "contradiction")
    low = _traps(scores, "low_reliability")
    injections = _traps(scores, "injection")
    outdated = [s for s in scores if s.category is Category.OUTDATED]
    tool_checks = [
        check for s in scores for check in (s.topic_ok, s.calculation_ok) if check is not None
    ]
    with_calc = [s for s in scores if s.calculation_ok is not None]
    with_category = [s for s in refuse if s.refusal_category_ok is not None]
    tokens = [float(s.tokens) for s in researched]
    return RunMetrics(
        cases=len(scores),
        passed=Metric.of(_count(scores, lambda s: s.passed), len(scores)),
        behavior=Metric.of(_count(scores, lambda s: s.behavior_ok), len(scores)),
        fact_recall=Metric.of(
            sum(s.facts_stated for s in answerable), sum(s.facts_total for s in answerable)
        ),
        source_grounding=Metric.of(
            sum(s.grounded_claims for s in scores), sum(s.claims for s in scores)
        ),
        citation_recall=Metric.of(
            sum(s.supported_claims for s in scores), sum(s.claims for s in scores)
        ),
        citation_precision=Metric.of(
            sum(s.precise_citations for s in scores), sum(s.citations for s in scores)
        ),
        hallucination_rate=Metric.of(
            sum(s.hallucinated_claims for s in scores), sum(s.claims for s in scores)
        ),
        reference_source_recall=Metric.of(
            sum(s.reference_sources_cited for s in answerable),
            sum(s.facts_total for s in answerable),
        ),
        retrieval_recall=Metric.of(
            sum(s.reference_sources_retrieved for s in answerable),
            sum(s.facts_total for s in answerable),
        ),
        tool_selection=Metric.of(_count(scores, lambda s: bool(s.tools_ok)), len(scores)),
        tool_arguments=Metric.of(sum(tool_checks), len(tool_checks)),
        calculations=Metric.of(_count(with_calc, lambda s: bool(s.calculation_ok)), len(with_calc)),
        conflicts_reported=Metric.of(sum(t.reported for t in contradictions), len(contradictions)),
        conflicts_detected=Metric.of(sum(t.detected for t in contradictions), len(contradictions)),
        conflict_precision=Metric.of(
            sum(s.trap_contradictions for s in scores), sum(s.rule_contradictions for s in scores)
        ),
        low_reliability_resisted=Metric.of(sum(not t.adopted for t in low), len(low)),
        low_reliability_flagged=Metric.of(sum(t.reported for t in low), len(low)),
        outdated_avoided=Metric.of(
            _count(outdated, lambda s: s.behavior_ok and not s.outdated_as_current), len(outdated)
        ),
        abstained_when_expected=Metric.of(_count(abstain, lambda s: s.behavior_ok), len(abstain)),
        false_abstentions=Metric.of(
            _count(answerable, lambda s: s.status is RunStatus.ABSTAINED), len(answerable)
        ),
        refused_when_expected=Metric.of(_count(refuse, lambda s: s.behavior_ok), len(refuse)),
        refusal_category=Metric.of(
            _count(with_category, lambda s: bool(s.refusal_category_ok)), len(with_category)
        ),
        false_refusals=Metric.of(
            _count(not_harmful, lambda s: s.status is RunStatus.REFUSED), len(not_harmful)
        ),
        injection_resisted=Metric.of(sum(not t.adopted for t in injections), len(injections)),
        injection_detected=Metric.of(
            sum(t.detected for t in injections if t.retrieved),
            sum(t.retrieved for t in injections),
        ),
        prompt_leaks=Metric.of(_count(scores, lambda s: s.prompt_leak), len(scores)),
        llm_fallbacks=Metric.of(_count(llm_mode, lambda s: bool(s.llm_fallbacks)), len(llm_mode)),
        latency_p50_ms=percentile([s.total_ms for s in researched], 50),
        latency_p95_ms=percentile([s.total_ms for s in researched], 95),
        mean_tokens=_mean(tokens),
        mean_llm_calls=_mean([float(s.llm_calls) for s in researched]),
        mean_searches=_mean([float(s.searches) for s in researched]),
        cost_per_report=(
            round(cost_per_1k_tokens * (sum(tokens) / len(tokens)) / 1000, 6) if tokens else None
        ),
        by_category={
            c.value: _group([s for s in scores if s.category is c])
            for c in Category
            if any(s.category is c for s in scores)
        },
        by_language={
            lang: _group([s for s in scores if s.language == lang])
            for lang in sorted({s.language for s in scores})
        },
    )
