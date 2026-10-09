"""Source assessment: how much to trust each source, by rules.

Reliability comes from the source type; freshness from the age of the source
at the corpus snapshot date (never the wall clock, so runs are reproducible).
Evidence is then re-ranked inside each sub-question by a blend of retrieval
score, reliability and freshness, so that a current official source is read
before an old forum post that happens to match the query better.

The weights are a documented judgement call, not learned. Their effect is
measured in the evaluation (phase 10), never tuned on the holdout split.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date

from researchai.agent.models import Evidence, SourceAssessment, SourceFlag
from researchai.corpus.models import SourceType
from researchai.tools.schemas import SourceMetadata

RELIABILITY: dict[SourceType, float] = {
    SourceType.GOVERNMENT: 0.9,
    SourceType.REGULATOR: 0.9,
    SourceType.STATISTICS_OFFICE: 0.9,
    SourceType.UTILITY: 0.8,
    SourceType.ACADEMIC: 0.8,
    SourceType.INDUSTRY_REPORT: 0.7,
    SourceType.NGO: 0.7,
    SourceType.COMPANY: 0.65,
    SourceType.NEWS: 0.6,
    SourceType.DEALER: 0.45,
    SourceType.BLOG: 0.35,
    SourceType.FORUM: 0.25,
}
USER_GENERATED = frozenset({SourceType.BLOG, SourceType.FORUM})
LOW_RELIABILITY = 0.5
STALE_AFTER_DAYS = 730

# (max age in days, freshness)
FRESHNESS_STEPS = ((365, 1.0), (730, 0.8), (1095, 0.6))
OLD_FRESHNESS = 0.4

WEIGHT_RELEVANCE, WEIGHT_RELIABILITY, WEIGHT_FRESHNESS = 0.5, 0.3, 0.2


def freshness(age_days: int) -> float:
    for limit, value in FRESHNESS_STEPS:
        if age_days <= limit:
            return value
    return OLD_FRESHNESS


def assess(
    source_type: SourceType, published: date, reference: date, doc_id: str
) -> SourceAssessment:
    age = max(0, (reference - published).days)
    reliability = RELIABILITY[source_type]
    flags: list[SourceFlag] = []
    if reliability < LOW_RELIABILITY:
        flags.append("low_reliability")
    if source_type in USER_GENERATED:
        flags.append("user_generated")
    if age > STALE_AFTER_DAYS:
        flags.append("stale")
    return SourceAssessment(
        doc_id=doc_id,
        source_type=source_type,
        published=published,
        age_days=age,
        reliability=reliability,
        freshness=freshness(age),
        flags=flags,
    )


def reference_date(metadata: Sequence[SourceMetadata], evidence: Sequence[Evidence]) -> date:
    """The corpus snapshot date (latest snapshot seen); latest publication if unknown."""
    if metadata:
        return max(m.snapshot for m in metadata)
    return max(e.published for e in evidence)


def assess_sources(
    evidence: Sequence[Evidence], metadata: Sequence[SourceMetadata]
) -> list[SourceAssessment]:
    reference = reference_date(metadata, evidence)
    seen: dict[str, SourceAssessment] = {}
    for item in evidence:
        if item.doc_id not in seen:
            seen[item.doc_id] = assess(item.source_type, item.published, reference, item.doc_id)
    return list(seen.values())


def rerank(evidence: Sequence[Evidence], assessments: Sequence[SourceAssessment]) -> list[Evidence]:
    """Order evidence by sub-question (plan order kept), then by blended score."""
    by_doc = {a.doc_id: a for a in assessments}
    best: dict[str, float] = {}
    for item in evidence:
        best[item.sub_question_id] = max(best.get(item.sub_question_id, 0.0), item.score)
    order = {sq: i for i, sq in enumerate(dict.fromkeys(e.sub_question_id for e in evidence))}

    def blended(item: Evidence) -> float:
        assessment = by_doc[item.doc_id]
        relevance = item.score / best[item.sub_question_id] if best[item.sub_question_id] else 0
        return (
            WEIGHT_RELEVANCE * relevance
            + WEIGHT_RELIABILITY * assessment.reliability
            + WEIGHT_FRESHNESS * assessment.freshness
        )

    return sorted(evidence, key=lambda e: (order[e.sub_question_id], -blended(e), e.id))
