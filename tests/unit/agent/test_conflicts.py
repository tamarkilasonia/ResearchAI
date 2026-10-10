"""Conflict detection on cases taken from the real corpus."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from researchai.agent.budget import Limits, RunBudget
from researchai.agent.claims import extract_claims
from researchai.agent.conflicts import (
    Disagreements,
    can_supersede,
    comparable,
    consistent,
    llm_conflicts,
    rule_conflicts,
)
from researchai.agent.graph import _evidence_from_hits
from researchai.agent.models import Claim, Evidence, NumericFact, ResearchPlan, SubQuestion
from researchai.corpus.models import SourceType
from tests.agent_fakes import ScriptedLLM, always, hit

PLAN = ResearchPlan(
    sub_questions=[SubQuestion(id="sq1", text="Charging points?", queries=["q"])],
    planned_by="rules",
)


@dataclass(frozen=True)
class Doc:
    chunk: str
    text: str
    published: date
    section: str
    source_type: SourceType = SourceType.GOVERNMENT
    publisher: str = "Ministry of Transport and Mobility"


def evidence(*docs: Doc) -> list[Evidence]:
    hits = [
        {
            "sub_question_id": "sq1",
            "query": "q",
            "hit": hit(d.chunk, d.text, published=d.published, source_type=d.source_type),
        }
        for d in docs
    ]
    by_chunk = {d.chunk: d for d in docs}
    return [
        e.model_copy(
            update={
                "section": by_chunk[e.chunk_id].section,
                "publisher": by_chunk[e.chunk_id].publisher,
            }
        )
        for e in _evidence_from_hits(PLAN, hits)
    ]


def claims_of(items: list[Evidence]) -> list[Claim]:
    return extract_claims(items, {e.doc_id: 0.9 for e in items})


def summaries(*docs: Doc) -> list[str]:
    return sorted(c.summary for c in rule_conflicts(claims_of(evidence(*docs))))


REGISTRY_2026 = Doc(
    "ev-en-0013#c1",
    "Public charging points: 512",
    date(2026, 7, 15),
    "Public charging infrastructure registry: status June 2026 > Totals",
)
REGISTRY_2025 = Doc(
    "ev-en-0014#c0",
    "As of 1 January 2025 the registry lists 385 public charging points for electric vehicles, "
    "of which 102 are DC fast charging points.",
    date(2025, 2, 12),
    "Charging registry: status January 2025",
)
REGISTRY_2023 = Doc(
    "ev-en-0015#c0",
    "On 1 January 2023 Georgia had 210 public charging points. Of these, 48 were DC fast "
    "charging points. Most charging points are located in Tbilisi and Batumi.",
    date(2023, 2, 9),
    "Charging registry: status January 2023",
)
OUTLOOK = Doc(
    "ev-en-0026#c1",
    "With 512 public charging points in mid-2026, Georgia has roughly one public charging point "
    "per 55 electric cars. Our scenario requires at least 450 DC fast charging points by 2030.",
    date(2026, 4, 28),
    "Charging demand outlook 2026-2030",
    SourceType.INDUSTRY_REPORT,
    "Caucasus Mobility Institute",
)


def test_newer_registry_supersedes_older_counts_without_chaining() -> None:
    conflicts = rule_conflicts(claims_of(evidence(REGISTRY_2026, REGISTRY_2025, REGISTRY_2023)))
    assert {c.kind for c in conflicts} == {"outdated"}
    texts = [c.summary for c in conflicts]
    assert any("512" in t and "385" in t and "210" in t for t in texts)
    assert any("102" in t and "48" in t for t in texts)  # DC counts form their own series
    assert not any("512" in t and "102" in t for t in texts)  # totals never vs DC counts
    assert [c.id for c in conflicts] == [f"X{n}" for n in range(1, len(conflicts) + 1)]


def test_forecasts_and_ratios_are_not_conflicts() -> None:
    joined = " ".join(summaries(REGISTRY_2026, OUTLOOK, REGISTRY_2025))
    assert "450" not in joined  # a 2030 target is a forecast, not a current value
    assert " 55 " not in f" {joined} "  # "one point per 55 cars" is a different quantity


def test_same_period_disagreement_is_a_contradiction() -> None:
    a = Doc(
        "ev-en-0024#c0",
        "VoltLine raised the DC fast charging price to 0.85 GEL per kWh.",
        date(2026, 6, 2),
        "VoltLine raises fast-charging price",
        SourceType.NEWS,
        "Caucasus Business Daily",
    )
    b = Doc(
        "ev-en-0099#c0",
        "VoltLine DC fast charging price: 0.65 GEL per kWh.",
        date(2026, 6, 20),
        "VoltLine fast-charging price",
        SourceType.BLOG,
        "Some blog",
    )
    (conflict,) = rule_conflicts(claims_of(evidence(a, b)))
    assert conflict.kind == "contradiction"
    assert conflict.unit == "GEL/kWh"


def test_directional_thresholds_keep_rates_apart() -> None:
    base = Doc(
        "tax-en-0010#c0",
        "Who pays 1%? Entrepreneurs whose turnover is up to 500,000 GEL.",
        date(2025, 3, 3),
        "The 1% tax explained",
        SourceType.NGO,
        "Small Business Support Center",
    )
    excess = Doc(
        "tax-en-0001#c2",
        "The part of the turnover above 500,000 GEL is taxed at 3%.",
        date(2026, 2, 9),
        "Small business status for individual entrepreneurs",
        SourceType.GOVERNMENT,
        "National Revenue Office",
    )
    assert summaries(base, excess) == []


def test_tariff_tiers_compare_only_with_the_same_tier() -> None:
    old = Doc(
        "ev-en-0030#c0",
        "Up to 101 kWh per month: 0.198 GEL/kWh\n101 to 301 kWh per month: 0.248 GEL/kWh",
        date(2022, 1, 4),
        "Residential electricity tariffs 2022",
        SourceType.UTILITY,
        "Tbilisi Grid Company",  # the utility's own price list: its name qualifies each line
    )
    new = Doc(
        "ev-en-0029#c1",
        "| Monthly consumption | Tariff (GEL/kWh) |\n|---|---|\n"
        "| 0-101 kWh | 0.239 |\n| 101-301 kWh | 0.292 |",
        date(2024, 11, 28),
        "Residential electricity tariffs from 1 January 2025 > Tbilisi Grid Company (Tbilisi)",
        SourceType.REGULATOR,
        "Energy Tariff Commission",
    )
    regions = Doc(
        "ev-en-0029#c2",
        "| Monthly consumption | Tariff (GEL/kWh) |\n|---|---|\n| 0-101 kWh | 0.231 |",
        date(2024, 11, 28),
        "Residential electricity tariffs from 1 January 2025 > Kartli Power Distribution",
        SourceType.REGULATOR,
        "Energy Tariff Commission",
    )
    found = summaries(old, new, regions)
    assert len(found) == 2
    assert all(c.startswith(("0.239", "0.292")) for c in found)
    first, second = found
    assert "0.198" in first
    assert "0.248" in second
    assert not any("0.231" in c for c in found)  # another utility's region is not comparable


def _fact(
    value: float,
    high: float | None = None,
    unit: str | None = "GEL",
    subject: tuple[str, ...] = ("leaf", "price", "used"),
    labels: tuple[str, ...] = (),
) -> NumericFact:
    return NumericFact(
        value=value,
        high=high,
        unit=unit,
        as_of_year=2026,
        as_of_month=None,
        char_start=0,
        char_end=1,
        subject=list(subject),
        labels=list(labels),
    )


def test_comparable_and_consistent_rules() -> None:
    assert comparable(_fact(1), _fact(2))
    assert not comparable(_fact(1), _fact(2, unit="USD"))
    assert not comparable(_fact(1, unit=None), _fact(2, unit=None))  # small unit-less numbers
    assert comparable(_fact(100, unit=None), _fact(200, unit=None))
    assert not comparable(_fact(1, labels=("<=101",)), _fact(2, labels=(">301",)))
    assert not comparable(_fact(1, subject=("leaf", "price", "tbilisi")), _fact(2))
    generic = ("car", "electric")
    assert not comparable(_fact(1, subject=generic), _fact(2, subject=generic))
    assert not comparable(_fact(1, subject=("price", "a", "b")), _fact(2, subject=("price", "c")))
    assert consistent(_fact(100), _fact(101))
    assert not consistent(_fact(100), _fact(110))
    assert consistent(_fact(21000, 26000), _fact(25000))
    assert not consistent(_fact(21000, 26000), _fact(27000, 32000))


VAT_2022 = Doc(
    "ev-en-0003#c0",
    "VAT at 18% is charged on the customs value of every car.",
    date(2022, 5, 11),
    "Importing an EV",
    SourceType.NEWS,
    "Caucasus Business Daily",
)
VAT_2023 = Doc(
    "ev-en-0002#c0",
    "The import of battery electric passenger cars will be exempt from VAT.",
    date(2023, 12, 5),
    "Notice",
    SourceType.GOVERNMENT,
    "National Revenue Office",
)


def _budget() -> RunBudget:
    return RunBudget(Limits(10, 2, 40_000, 5, 3))


async def test_llm_pass_adds_text_disagreements_with_anchored_quotes() -> None:
    items = evidence(VAT_2022, VAT_2023)
    ids = {e.doc_id: e.id for e in items}
    old, new = ids["ev-en-0003"], ids["ev-en-0002"]
    reply = {
        "disagreements": [
            {"evidence_a": old, "quote_a": "not in the passage", "evidence_b": new,
             "quote_b": "exempt from VAT", "explanation": "unanchorable quote"},
            {"evidence_a": old, "quote_a": "VAT at 18% is charged", "evidence_b": new,
             "quote_b": "will be exempt from VAT", "explanation": "VAT applies vs exempt"},
            {"evidence_a": old, "quote_a": "VAT at 18%", "evidence_b": "E9",
             "quote_b": "some quote", "explanation": "unknown id"},
        ]
    }  # fmt: skip
    llm = ScriptedLLM({"Disagreements": [always(reply)]})
    claims, conflicts, notes = await llm_conflicts(
        "q", items, [], [], {}, llm=llm, budget=_budget(), max_output_tokens=500
    )
    (conflict,) = conflicts
    assert (conflict.kind, conflict.detected_by) == ("outdated", "llm")
    newer = next(c for c in claims if c.id == conflict.current[0])
    assert newer.doc_id == "ev-en-0002"
    assert newer.text == "will be exempt from VAT"
    assert all(c.facts == [] for c in claims)
    assert len(notes) == 2
    # An evidence pair that is already related is not reported twice.
    _, again, _ = await llm_conflicts(
        "q", items, claims, conflicts, {}, llm=llm, budget=_budget(), max_output_tokens=500
    )
    assert again == []


async def test_llm_pass_failure_is_a_note() -> None:
    claims, conflicts, notes = await llm_conflicts(
        "q", [], [], [], {}, llm=ScriptedLLM({}), budget=_budget(), max_output_tokens=500
    )
    assert (claims, conflicts) == ([], [])
    assert "no LLM conflict check" in notes[0]


def test_disagreements_schema() -> None:
    assert Disagreements.model_validate({"disagreements": []}).disagreements == []


OFFICIAL_2025 = Doc(
    "ev-en-0013#c0",
    "Public charging points registry: 512 public charging points.",
    date(2025, 7, 15),
    "Registry",
)
FORUM_2026 = Doc(
    "ev-en-0090#c0",
    "Public charging points registry: 7777 public charging points.",
    date(2026, 8, 1),
    "Thread",
    SourceType.FORUM,
    "EV owners forum",
)
OFFICIAL_2026 = Doc(
    "ev-en-0091#c0",
    "Public charging points registry: 7777 public charging points.",
    date(2026, 6, 1),
    "Registry",
)


def _with_reliability(items: list[Evidence]) -> list[Claim]:
    reliability = {"ev-en-0013": 0.9, "ev-en-0090": 0.25, "ev-en-0091": 0.9}
    return extract_claims(items, reliability)


def test_a_newer_low_reliability_source_never_supersedes_an_official_value() -> None:
    claims = _with_reliability(evidence(OFFICIAL_2025, FORUM_2026))
    (conflict,) = rule_conflicts(claims)
    assert conflict.kind == "contradiction"  # both sides shown, the official value kept
    by_id = {c.id: c for c in claims}
    assert [by_id[i].doc_id for i in conflict.current] == ["ev-en-0090"]
    assert [by_id[i].doc_id for i in conflict.others] == ["ev-en-0013"]


def test_a_reliable_source_agreeing_with_the_newest_value_supersedes_and_is_quoted() -> None:
    claims = _with_reliability(evidence(OFFICIAL_2025, FORUM_2026, OFFICIAL_2026))
    (conflict,) = rule_conflicts(claims)
    assert conflict.kind == "outdated"
    by_id = {c.id: c for c in claims}
    assert [by_id[i].doc_id for i in conflict.current] == ["ev-en-0091"]  # the forum adds nothing
    assert [by_id[i].doc_id for i in conflict.others] == ["ev-en-0013"]


def test_can_supersede() -> None:
    assert can_supersede(0.9, 0.9)
    assert can_supersede(0.6, 0.9)  # reliable enough: news may update an official figure
    assert can_supersede(0.25, 0.25)
    assert not can_supersede(0.25, 0.9)
    assert not can_supersede(0.45, 0.6)


async def test_llm_pass_applies_the_same_reliability_rule() -> None:
    items = evidence(VAT_2022, VAT_2023)
    ids = {e.doc_id: e.id for e in items}
    reply = {
        "disagreements": [
            {"evidence_a": ids["ev-en-0003"], "quote_a": "VAT at 18% is charged",
             "evidence_b": ids["ev-en-0002"], "quote_b": "will be exempt from VAT",
             "explanation": "VAT applies vs exempt"},
        ]
    }  # fmt: skip
    llm = ScriptedLLM({"Disagreements": [always(reply)]})
    # The newer notice comes from a forum this time.
    _, conflicts, _ = await llm_conflicts(
        "q", items, [], [], {"ev-en-0002": 0.25, "ev-en-0003": 0.6},
        llm=llm, budget=_budget(), max_output_tokens=500,
    )  # fmt: skip
    assert [c.kind for c in conflicts] == ["contradiction"]
