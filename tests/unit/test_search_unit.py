"""Database-free parts of hybrid search. Ranking against PostgreSQL: tests/integration."""

from __future__ import annotations

from datetime import date

import pytest
from hypothesis import given
from hypothesis import strategies as st
from sqlalchemy.dialects.postgresql.base import PGDialect

from researchai.corpus.models import Language, SourceType, Topic
from researchai.textutil import MAX_KEYWORD_TERMS, keyword_terms, light_stem
from researchai.tools.errors import ErrorCode, ToolFailure
from researchai.tools.search import SearchFilters, reciprocal_rank_fusion, to_tsquery_text


def test_keyword_terms_drop_stop_words_and_stem() -> None:
    assert keyword_terms("How many public charging points are there?") == [
        "public",
        "charg",
        "point",
    ]
    assert keyword_terms("ელექტრომობილები საქართველოში") == ["ელექტრომობილ", "საქართველო"]


@pytest.mark.parametrize(
    ("token", "stem"),
    [("taxes", "tax"), ("ev", "ev"), ("bus", "bus"), ("ები", "ები"), ("დატენვა", "დატენვ")],
)
def test_light_stem_keeps_a_minimum_stem(token: str, stem: str) -> None:
    assert light_stem(token) == stem


@pytest.mark.parametrize(
    "forms",
    [
        ("ელექტრომობილი", "ელექტრომობილის", "ელექტრომობილები", "ელექტრომობილებს"),
        ("დატენვა", "დატენვის", "დატენვაზე"),
        ("charging", "charger", "chargers"),
    ],
)
def test_prefix_terms_match_other_inflections(forms: tuple[str, ...]) -> None:
    for form in forms:
        (term,) = keyword_terms(form)
        for other in forms:
            if len(other) >= len(term) + 2 or other == form:
                assert other.startswith(term), (form, term, other)


def test_keyword_terms_strip_tsquery_syntax() -> None:
    terms = keyword_terms("tax & (VAT | !excise) :* 'x' <-> rate")
    assert terms == ["tax", "vat", "excise", "rate"]
    assert all(term.isalnum() for term in terms)


def test_keyword_terms_dedupe_casefold_and_limit() -> None:
    assert keyword_terms("EV ev Ev") == ["ev"]
    many = " ".join(f"w{i}x" for i in range(100))  # short tokens: no stemming
    assert len(keyword_terms(many)) == MAX_KEYWORD_TERMS


def test_keyword_terms_can_be_empty() -> None:
    assert keyword_terms("what is the a ?") == []


def test_tsquery_text() -> None:
    assert to_tsquery_text(["vat", "exci"]) == "vat:* | exci:*"


@given(st.text(max_size=300))
def test_keyword_terms_are_always_safe(text: str) -> None:
    for term in keyword_terms(text):
        assert term
        assert all(ch.isalnum() for ch in term)


def test_rrf_rewards_agreement_and_breaks_ties_by_id() -> None:
    fused = reciprocal_rank_fusion([["a", "b", "c"], ["b", "d"]], k=60)
    assert [item for item, _ in fused] == ["b", "a", "d", "c"]
    assert fused[0][1] == pytest.approx(1 / 62 + 1 / 61)
    # identical scores -> ordered by id
    assert [i for i, _ in reciprocal_rank_fusion([["z"], ["y"]])] == ["y", "z"]


def test_rrf_empty() -> None:
    assert reciprocal_rank_fusion([]) == []


def test_filters_reject_inverted_date_range() -> None:
    with pytest.raises(ToolFailure) as excinfo:
        SearchFilters(published_from=date(2026, 1, 1), published_to=date(2025, 1, 1))
    assert excinfo.value.code is ErrorCode.INVALID_ARGUMENT


def _sql(filters: SearchFilters) -> str:
    compiled = filters.clause().compile(
        dialect=PGDialect(),  # type: ignore[no-untyped-call]
        compile_kwargs={"literal_binds": True},
    )
    return str(compiled)


def test_filter_clause_contains_every_condition() -> None:
    sql = _sql(
        SearchFilters(
            language=Language.KA,
            topic=Topic.TAX,
            published_from=date(2024, 1, 1),
            published_to=date(2026, 1, 1),
            source_types=(SourceType.GOVERNMENT, SourceType.NEWS),
        )
    )
    for fragment in ("language = 'ka'", "topic = 'tax'", ">= '2024-01-01'", "<= '2026-01-01'"):
        assert fragment in sql
    assert "IN ('government', 'news')" in sql


def test_empty_filter_clause_is_true() -> None:
    assert _sql(SearchFilters()).strip().lower() == "true"
