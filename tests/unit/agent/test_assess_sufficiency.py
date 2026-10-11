from __future__ import annotations

from datetime import date

import pytest

from researchai.agent.assess import assess, assess_sources, freshness, reference_date, rerank
from researchai.agent.graph import _evidence_from_hits
from researchai.agent.models import ResearchPlan, SubQuestion
from researchai.agent.sufficiency import key_entities, missing_entities, relevant, verdicts
from researchai.corpus.models import Language, SourceType
from tests.agent_fakes import CHARGING, FakeTools, hit

SNAPSHOT = date(2026, 9, 1)
PLAN = ResearchPlan(
    sub_questions=[
        SubQuestion(id="sq1", text="How many public charging points?", queries=["charging points"]),
        SubQuestion(id="sq2", text="What do forum users say?", queries=["forum opinions"]),
    ],
    planned_by="rules",
)


@pytest.mark.parametrize(
    ("age", "value"), [(0, 1.0), (365, 1.0), (366, 0.8), (1000, 0.6), (2000, 0.4)]
)
def test_freshness_steps(age: int, value: float) -> None:
    assert freshness(age) == value


def test_assessment_flags() -> None:
    forum = assess(SourceType.FORUM, date(2022, 1, 1), SNAPSHOT, "ev-en-0009")
    assert forum.reliability == 0.25
    assert forum.flags == ["low_reliability", "user_generated", "stale"]
    ministry = assess(SourceType.GOVERNMENT, date(2026, 7, 15), SNAPSHOT, "ev-en-0013")
    assert (ministry.reliability, ministry.freshness, ministry.flags) == (0.9, 1.0, [])
    assert assess(SourceType.GOVERNMENT, date(2027, 1, 1), SNAPSHOT, "x").age_days == 0


def _evidence(*hits_: tuple[str, object]) -> list:  # type: ignore[type-arg]
    return _evidence_from_hits(
        PLAN, [{"sub_question_id": sq, "query": "q", "hit": h} for sq, h in hits_]
    )


def test_reference_date_prefers_snapshot_metadata() -> None:
    from tests.agent_fakes import FakeTools as _  # noqa: F401 - keeps import order stable

    evidence = _evidence(("sq1", CHARGING))
    assert reference_date([], evidence) == CHARGING.published


async def test_rerank_puts_reliable_recent_sources_first() -> None:
    old_forum = hit("ev-en-0009#c0", "Public charging points are everywhere.", score=0.05,
                    published=date(2021, 1, 1), source_type=SourceType.FORUM)  # fmt: skip
    ministry = hit("ev-en-0013#c0", "Public charging points: 512.", score=0.04)
    evidence = _evidence(("sq1", old_forum), ("sq1", ministry))
    assert [e.doc_id for e in evidence] == ["ev-en-0009", "ev-en-0013"]  # by retrieval score
    metadata = (
        await FakeTools(corpus=[old_forum, ministry]).get_source_metadata(
            ["ev-en-0009", "ev-en-0013"]
        )
    ).sources
    assessments = assess_sources(evidence, metadata)
    assert [e.doc_id for e in rerank(evidence, assessments)] == ["ev-en-0013", "ev-en-0009"]


def test_relevance_needs_two_shared_terms() -> None:
    evidence = _evidence(("sq1", CHARGING))
    assert relevant(evidence[0], PLAN.sub_questions[0])
    off_topic = _evidence(("sq1", hit("ev-en-0084#c0", "Charging hydrogen is not possible.")))
    assert not relevant(off_topic[0], PLAN.sub_questions[0])


def test_relevance_reads_the_whole_passage() -> None:
    """The matching words come after 30 other distinct words (search terms are capped at 24)."""
    filler = " ".join(f"word{chr(97 + i // 26)}{chr(97 + i % 26)}" for i in range(30))
    long_passage = _evidence(
        ("sq1", hit("ev-en-0013#c9", f"{filler}. Public charging points: 512."))
    )
    assert relevant(long_passage[0], PLAN.sub_questions[0])


def test_verdicts() -> None:
    forum_only = hit("ev-en-0025#c0", "Forum users share opinions about charging.",
                     source_type=SourceType.FORUM)  # fmt: skip
    evidence = _evidence(("sq1", CHARGING), ("sq2", forum_only))
    assessments = assess_sources(evidence, [])
    result = {v.sub_question_id: v for v in verdicts(PLAN, evidence, assessments)}
    assert result["sq1"].sufficient
    assert not result["sq2"].sufficient
    assert "low-reliability" in result["sq2"].reason
    two_forums = _evidence(
        ("sq2", forum_only),
        (
            "sq2",
            hit("ev-en-0041#c0", "Forum opinions differ on winter.", source_type=SourceType.FORUM),
        ),
    )
    two = verdicts(PLAN, two_forums, assess_sources(two_forums, []))
    assert two[1].sufficient
    assert "independent" in two[1].reason
    assert verdicts(PLAN, [], [])[0].reason == "no relevant passage"


@pytest.mark.parametrize(
    ("question", "language", "entities"),
    [
        (
            "What is the price of a Tesla Model S in Japan?",
            Language.EN,
            ["Tesla", "Model", "Japan"],
        ),
        ("How many chargers are there in Georgia?", Language.EN, ["Georgia"]),
        ("Who won the 1998 football world cup?", Language.EN, []),
        ("Compare EVs and VAT in GEL", Language.EN, []),
        ("რა ღირს Tesla Model 3 იაპონიაში?", Language.KA, ["Tesla", "Model"]),
    ],
)
def test_key_entities(question: str, language: Language, entities: list[str]) -> None:
    assert key_entities(question, language) == entities


async def test_missing_entities_are_confirmed_by_search_within_budget() -> None:
    evidence = _evidence(("sq1", CHARGING))
    tools = FakeTools()
    missing, used = await missing_entities(["Japan", "Public", "Batumi"], evidence, tools, 5)
    assert (missing, used) == (["Japan", "Batumi"], 2)  # "Public" is in the evidence
    missing, used = await missing_entities(["Japan"], evidence, FakeTools(), 0)
    assert (missing, used) == ([], 0)  # no budget: benefit of the doubt
    found, _ = await missing_entities(
        ["Tbilisi"], evidence, FakeTools(corpus=[hit("x-en-0001#c0", "Tbilisi tariff")]), 1
    )
    assert found == []
