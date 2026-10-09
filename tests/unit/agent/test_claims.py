"""Numeric claim extraction, using sentences from the corpus."""

from __future__ import annotations

from datetime import date

import pytest
from hypothesis import given
from hypothesis import strategies as st

from researchai.agent.claims import (
    date_mentions,
    extract_claims,
    facts_in,
    parse_number,
    threshold_labels,
    unit_in,
)
from researchai.agent.models import Evidence
from researchai.corpus.models import Language, SourceType

PUBLISHED = date(2026, 7, 15)


FactTuple = tuple[float, float | None, str | None, tuple[int, int], list[str]]


def facts(sentence: str) -> list[FactTuple]:
    return [(f.value, f.high, f.unit, f.period, f.labels) for f in facts_in(sentence, 0, PUBLISHED)]


@pytest.mark.parametrize(
    ("token", "value"),
    [("1,800", 1800.0), ("1,800.5", 1800.5), ("1 800,5", 1800.5), ("0,55", 0.55), ("16.8", 16.8),
     ("86 900", 86900.0), ("1,240", 1240.0), ("12,5", 12.5), ("x", None)],
)  # fmt: skip
def test_parse_number(token: str, value: float | None) -> None:
    assert parse_number(token) == value


def test_date_mentions_most_specific_first() -> None:
    text = (
        "From 1 January 2025 to June 2026, and in 2027; also 2025 წლის 1 იანვრიდან and 2024-11-28."
    )
    found = {(d.year, d.month) for d in date_mentions(text)}
    assert found == {(2025, 1), (2026, 6), (2027, None), (2024, 11)}


def test_dates_are_masked_so_years_never_start_ranges() -> None:
    assert facts("The number grew from 210 in 2023 to 290 in 2024.") == [
        (210.0, None, None, (2023, 0), []),
        (290.0, None, None, (2024, 0), []),
    ]


def test_units_ranges_and_compound_units() -> None:
    assert facts("A Tesla Model 3 uses about 14.9 kWh per 100 km.") == [
        (3.0, None, None, (2026, 7), []),
        (14.9, None, "kWh/100km", (2026, 7), []),  # "100 km" is part of the unit
    ]
    assert facts("Typical winter range loss: 15 to 25% compared with mild weather.") == [
        (15.0, 25.0, "%", (2026, 7), [])
    ]
    assert facts("AC დატენვა: 0,55 ლარი 1 კვტ.სთ-ზე") == [(0.55, None, "GEL/kWh", (2026, 7), [])]


def test_relative_year_shifts_back() -> None:
    result = facts("Georgia ended 2025 with 24,100 cars, up from 17,300 a year earlier.")
    assert [(v, p) for v, _, _, p, _ in result] == [(24100.0, (2025, 0)), (17300.0, (2024, 0))]
    georgian = facts("2025 წლის ბოლოს 24 100 ელექტრომობილი იყო, წლის წინ კი 17 300.")
    assert [(v, p) for v, _, _, p, _ in georgian] == [(24100.0, (2025, 0)), (17300.0, (2024, 0))]


def test_unclaimed_years_are_not_facts() -> None:
    assert facts("Prices fell sharply in 2025.") == []


@pytest.mark.parametrize(
    ("text", "labels"),
    [("up to 500,000 GEL", ["<=500000"]), ("above 500,000 GEL", [">500000"]),
     ("0-101 kWh", ["<=101"]), ("Up to 101 kWh", ["<=101"]), ("101 to 301 kWh", ["101-301"]),
     ("above 301 kWh", [">301"]), ("101 კვტ.სთ-მდე", ["<=101"]), ("301 კვტ.სთ-ზე მეტი", [">301"]),
     ("nothing here", [])],
)  # fmt: skip
def test_threshold_labels_keep_direction(text: str, labels: list[str]) -> None:
    assert threshold_labels(text) == labels


def test_a_facts_own_range_is_not_its_label() -> None:
    assert facts("Leaf prices are 21,000-26,000 GEL.") == [(21000.0, 26000.0, "GEL", (2026, 7), [])]


def test_unit_in_header() -> None:
    assert unit_in("Tariff (GEL/kWh)") == "GEL/kWh"
    assert unit_in("Monthly consumption") is None


def _evidence(text: str, section: str = "Tariffs > Tbilisi Grid Company") -> Evidence:
    return Evidence(
        id="E1", sub_question_id="sq1", query="q", chunk_id="ev-en-0029#c1", doc_id="ev-en-0029",
        title="t", section=section, text=text, char_start=50, char_end=50 + len(text),
        published=date(2024, 11, 28), language=Language.EN, source_type=SourceType.REGULATOR,
        publisher="Energy Tariff Commission", score=0.1,
    )  # fmt: skip


def test_table_cells_take_header_unit_and_row_label() -> None:
    table = (
        "| Monthly consumption | Tariff (GEL/kWh) |\n|---|---|\n"
        "| 0-101 kWh | 0.239 |\n| 101-301 kWh | 0.292 |\n| above 301 kWh | 0.345 |"
    )
    item = _evidence(table)
    claims = extract_claims([item], {"ev-en-0029": 0.9})
    tariffs = [f for c in claims for f in c.facts if f.unit == "GEL/kWh"]
    assert [(f.value, f.labels) for f in tariffs] == [
        (0.239, ["<=101"]),
        (0.292, ["101-301"]),
        (0.345, [">301"]),
    ]
    for fact in tariffs:
        assert table[fact.char_start - 50 : fact.char_end - 50] == f"{fact.value:g}"
        assert {"tbilisi", "grid", "tariff"} <= set(fact.subject)
    assert all(c.reliability == 0.9 for c in claims)
    assert claims[0].text == "| 0-101 kWh | 0.239 |"


def test_sentences_without_numbers_are_not_claims() -> None:
    assert extract_claims([_evidence("No numbers here. None at all.")], {}) == []


def test_claim_spans_are_exact() -> None:
    text = "Intro sentence. Public charging points: 512. Of which DC fast: 146."
    claims = extract_claims([_evidence(text, section="Registry")], {})
    assert [c.text for c in claims] == ["Public charging points: 512.", "Of which DC fast: 146."]
    for claim in claims:
        assert text[claim.char_start - 50 : claim.char_end - 50] == claim.text
    assert claims[0].reliability == 0.5  # default when unassessed


@given(st.text(alphabet="0123456789 ,.-% kmGELwhto\u2013ლარი2025", max_size=80))
def test_fact_spans_always_point_at_the_number(text: str) -> None:
    for fact in facts_in(text, 0, PUBLISHED):
        token = text[fact.char_start : fact.char_end]
        assert token
        assert token[0].isdigit()


def test_unit_must_follow_closely_and_ranges_must_ascend() -> None:
    assert facts("Output: 22     kW total.") == [(22.0, None, None, (2026, 7), [])]
    assert facts("Dropped from 50 to 10 kW.") == [(50.0, None, "kW", (2026, 7), [])]


def test_day_word_year_without_a_month_is_just_a_year() -> None:
    assert [(d.year, d.month) for d in date_mentions("On 3 units 2025 shipped.")] == [(2025, None)]


def test_empty_table_cells_are_ignored() -> None:
    table = "| Model | Price (GEL) |\n|---|---|\n|  | 21,000 |"
    (claim,) = extract_claims([_evidence(table, section="Prices")], {})
    assert [(f.value, f.unit) for f in claim.facts] == [(21000.0, "GEL")]
