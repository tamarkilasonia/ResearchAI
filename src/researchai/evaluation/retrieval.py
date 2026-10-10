"""Retrieval quality per search mode: does the source of each reference fact come back?

Used to choose the default search mode on the dev split (never the holdout).
A case's recall@k is the share of its reference facts with at least one source
document among the top-k distinct documents; reciprocal rank uses the first
such document.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from researchai.evaluation.models import Case


def case_recall(case: Case, ranked_docs: Sequence[str], k: int) -> float:
    top = set(ranked_docs[:k])
    hits = sum(bool(top & set(fact.sources)) for fact in case.facts)
    return hits / len(case.facts)


def reciprocal_rank(case: Case, ranked_docs: Sequence[str]) -> float:
    relevant = {doc for fact in case.facts for doc in fact.sources}
    for rank, doc in enumerate(ranked_docs, start=1):
        if doc in relevant:
            return 1 / rank
    return 0.0


def distinct(doc_ids: Sequence[str]) -> list[str]:
    return list(dict.fromkeys(doc_ids))


def summarize(
    cases: Sequence[Case], ranked: Mapping[str, Mapping[str, Sequence[str]]], ks: Sequence[int]
) -> dict[str, dict[str, float]]:
    """Mean recall@k and MRR per mode.

    ``ranked[mode][case_id]`` holds the distinct document ids in rank order.
    """
    summary: dict[str, dict[str, float]] = {}
    for mode, by_case in ranked.items():
        row = {
            f"recall@{k}": round(
                sum(case_recall(c, by_case[c.id], k) for c in cases) / len(cases), 4
            )
            for k in ks
        }
        row["mrr"] = round(sum(reciprocal_rank(c, by_case[c.id]) for c in cases) / len(cases), 4)
        summary[mode] = row
    return summary
