"""Citation verification: deterministic checks first, entailment second."""

from __future__ import annotations

from datetime import date

import pytest

from researchai.agent.budget import Limits, RunBudget
from researchai.agent.citations import (
    check_citation,
    fetch_bodies,
    is_verbatim,
    lexical_support,
    missing_numbers,
    sentence_numbers,
    verify_report,
)
from researchai.agent.graph import _evidence_from_hits
from researchai.agent.llm import LLMError
from researchai.agent.models import (
    Calculation,
    Citation,
    CitationReport,
    Report,
    ReportSection,
    ReportSentence,
    ResearchPlan,
    SubQuestion,
)
from researchai.corpus.models import Language
from researchai.mcp_client.client import ToolCallError
from researchai.tools.errors import ErrorCode
from tests.agent_fakes import CHARGING, CONSUMPTION, TARIFF, FakeTools, ScriptedLLM, always

PLAN = ResearchPlan(
    sub_questions=[
        SubQuestion(id="sq1", text="Chargers?", queries=["q"]),
        SubQuestion(id="sq2", text="Costs?", queries=["q"]),
    ],
    planned_by="rules",
)
EVIDENCE = _evidence_from_hits(
    PLAN,
    [
        {"sub_question_id": "sq1", "query": "q", "hit": CHARGING},
        {"sub_question_id": "sq2", "query": "q", "hit": TARIFF},
        {"sub_question_id": "sq2", "query": "q", "hit": CONSUMPTION},
    ],
)
BY_DOC = {e.doc_id: e for e in EVIDENCE}


def cite(doc_id: str, quote: str, *, anchored: bool = True) -> Citation:
    item = BY_DOC[doc_id]
    offset = item.text.find(quote)
    start = item.char_start + offset if anchored and offset >= 0 else None
    return Citation(
        evidence_id=item.id,
        doc_id=doc_id,
        chunk_id=item.chunk_id,
        quote=quote,
        char_start=start,
        char_end=start + len(quote) if start is not None else None,
    )


CHARGING_QUOTE = "Public charging points: 512."
TARIFF_QUOTE = "0.292 GEL per kWh"
LEAF_QUOTE = "16.8 kWh per 100 km"


def report(*sections: list[ReportSentence], summary: list[ReportSentence] | None = None) -> Report:
    return Report(
        language=Language.EN,
        title="t",
        summary=summary or [],
        sections=[
            ReportSection(heading=f"h{i}", sub_question_id=f"sq{i + 1}", sentences=sentences)
            for i, sentences in enumerate(sections)
        ],
        generated_by="llm",
    )


def claim(text: str, *citations: Citation, calcs: list[str] | None = None) -> ReportSentence:
    return ReportSentence(
        text=text, kind="claim", citations=list(citations), calculation_ids=calcs or []
    )


def budget() -> RunBudget:
    return RunBudget(Limits(10, 2, 40_000, 5, 3))


async def run(
    rep: Report,
    *,
    llm: ScriptedLLM | None = None,
    calcs: list[Calculation] | None = None,
    question: str = "How much?",
) -> tuple[Report, CitationReport]:
    return await verify_report(
        rep,
        EVIDENCE,
        calcs or [],
        question,
        tools=FakeTools(),
        llm=llm,
        budget=budget(),
        max_output_tokens=500,
    )


# ----------------------------------------------------------------- pieces


def test_citation_statuses() -> None:
    body = "x" * 100 + "Public charging points: 512. Of which DC"
    bodies: dict[str, str | None] = {"ev-en-0013": body, "ev-en-0099": None}
    exact = Citation(evidence_id="E1", doc_id="ev-en-0013", chunk_id="c", quote=CHARGING_QUOTE,
                     char_start=100, char_end=128)  # fmt: skip
    assert check_citation(exact, bodies) == "verified"
    loose = exact.model_copy(update={"quote": "public   CHARGING points: 512."})
    assert check_citation(loose, bodies) == "verified"  # normalised comparison
    assert check_citation(exact.model_copy(update={"quote": "513"}), bodies) == "span_mismatch"
    assert check_citation(exact.model_copy(update={"char_start": None}), bodies) == "unanchored"
    assert (
        check_citation(exact.model_copy(update={"doc_id": "ev-en-0099"}), bodies)
        == "missing_document"
    )
    assert (
        check_citation(exact.model_copy(update={"doc_id": "ev-en-0098"}), bodies)
        == "missing_document"
    )


def test_sentence_numbers_and_tolerance() -> None:
    assert sentence_numbers("Prices of 21,000-26,000 GEL in 2026 and 4.91 GEL.") == [
        21000,
        26000,
        4.91,
    ]
    assert missing_numbers("It costs 4.91 GEL.", {4.9056}) == []
    assert missing_numbers("It costs 5.2 GEL.", {4.9056}) == [5.2]


def test_verbatim_detection_ignores_table_pipes_and_case() -> None:
    table_row = cite("ev-en-0013", CHARGING_QUOTE).model_copy(
        update={"quote": "| 0-101 kWh | 0.239 |"}
    )
    row = ReportSentence(
        text="Sources disagree: x reports \u201c0-101 kWh | 0.239\u201d.",
        kind="conflict",
        citations=[table_row],
    )
    assert is_verbatim(row)
    paraphrase = claim("There are 512 chargers.", cite("ev-en-0013", CHARGING_QUOTE))
    assert not is_verbatim(paraphrase)


@pytest.mark.parametrize(
    ("text", "level"),
    [
        ("Public charging points number 512.", "supported"),
        ("Public charging points number 512 in Batumi harbour.", "partial"),
        ("Hydrogen stations are planned.", None),
        ("It is 512.", "supported"),  # nothing but numbers/generic words to check
    ],
)
def test_lexical_support(text: str, level: str | None) -> None:
    assert lexical_support(text, [CHARGING.text]) == level


async def test_fetch_bodies_missing_vs_fatal() -> None:
    bodies = await fetch_bodies(["ev-en-0013", "ev-en-0999"], FakeTools())
    assert bodies["ev-en-0999"] is None
    assert bodies["ev-en-0013"] is not None
    with pytest.raises(ToolCallError):
        await fetch_bodies(["ev-en-0013"], FakeTools(fetch_error=ErrorCode.INTERNAL))


# ----------------------------------------------------------------- whole reports


async def test_verbatim_claims_pass_deterministically() -> None:
    cleaned, summary = await run(
        report([claim(CHARGING_QUOTE, cite("ev-en-0013", CHARGING_QUOTE))])
    )
    assert cleaned.sections[0].sentences[0].support == "verbatim"
    assert (summary.precision, summary.verbatim, summary.entailment) == (1.0, 1, "none")


async def test_misattributed_and_unsupported_numbers_are_removed() -> None:
    wrong_source = claim("The tariff is 0.292 GEL per kWh.", cite("ev-en-0013", CHARGING_QUOTE))
    invented = claim("There are 9999 chargers.", cite("ev-en-0013", CHARGING_QUOTE))
    cleaned, summary = await run(report([wrong_source, invented]))
    assert (summary.misattributed, summary.unsupported) == (1, 1)
    assert {r.status for r in summary.removed} == {"misattributed", "unsupported"}
    (only,) = cleaned.sections[0].sentences
    assert only.kind == "insufficient_evidence"  # the emptied section says so
    assert "sq1" in cleaned.insufficient
    assert summary.precision == 0.0


async def test_bad_citations_are_dropped_and_good_ones_kept() -> None:
    mixed = claim(
        CHARGING_QUOTE,
        cite("ev-en-0013", CHARGING_QUOTE),
        Citation(
            evidence_id="E9", doc_id="ev-en-0999", chunk_id="c", quote="x", char_start=1, char_end=2
        ),
    )
    unanchored = claim("Charging points: 512.", cite("ev-en-0013", CHARGING_QUOTE, anchored=False))
    cleaned, summary = await run(report([mixed, unanchored]))
    kept = cleaned.sections[0].sentences
    assert len(kept) == 1
    assert [c.doc_id for c in kept[0].citations] == ["ev-en-0013"]
    assert (summary.citations_total, summary.citations_verified) == (3, 1)
    assert summary.removed[0].reason == "no cited span could be verified"
    assert summary.precision == pytest.approx(1 / 3, abs=1e-3)


async def test_calculations_and_question_numbers_count_as_support() -> None:
    calc = Calculation(id="C1", description="cost", expression="16.8 * 0.292", result=4.9056,
                       evidence_ids=["E2", "E3"])  # fmt: skip
    computed = claim(
        "Charging a Leaf at home costs 4.91 GEL per 100 km.",
        cite("ev-en-0029", TARIFF_QUOTE),
        cite("ev-en-0058", LEAF_QUOTE),
        calcs=["C1"],
    )
    cleaned, summary = await run(report([], [computed]), calcs=[calc], question="Cost per 100 km?")
    assert summary.unsupported == 0
    assert cleaned.sections[1].sentences[0].support in ("supported", "partial")
    no_calc = computed.model_copy(update={"calculation_ids": []})
    _, again = await run(report([], [no_calc]), question="Cost per 100 km?")
    assert again.unsupported == 1  # 4.91 appears nowhere without the cited calculation


async def test_llm_entailment_labels_and_fallbacks() -> None:
    paraphrase_ok = claim(
        "Georgia has 512 public chargers in total.", cite("ev-en-0013", CHARGING_QUOTE)
    )
    overstated = claim(
        "Georgia has the most chargers in the region, 512.", cite("ev-en-0013", CHARGING_QUOTE)
    )
    skipped = claim("There are 512 public charging points.", cite("ev-en-0013", CHARGING_QUOTE))
    rep = report([paraphrase_ok, overstated, skipped])
    llm = ScriptedLLM({"Judgements": [always({"judgements": [
        {"id": "S0.0", "label": "partial", "reason": "'in total' not stated"},
        {"id": "S0.1", "label": "not_supported", "reason": "no regional comparison"},
        {"id": "S9.9", "label": "supported", "reason": "unknown id ignored"},
    ]})]})  # fmt: skip
    cleaned, summary = await run(rep, llm=llm)
    assert summary.entailment == "llm"
    assert [c.status for c in summary.flagged] == ["partial"]
    assert summary.removed[0].reason == "no regional comparison"
    assert summary.removed[0].method == "llm"
    kept = cleaned.sections[0].sentences
    assert [s.support for s in kept] == ["partial", "supported"]  # last one: lexical fallback
    assert kept[0].text == paraphrase_ok.text


async def test_llm_failure_falls_back_to_lexical() -> None:
    rep = report([claim("Public charging points number 512.", cite("ev-en-0013", CHARGING_QUOTE))])
    _, summary = await run(
        rep, llm=ScriptedLLM({"Judgements": [always(LLMError("transport", "down"))]})
    )
    assert summary.entailment == "lexical"
    assert summary.supported == 1


async def test_sentences_without_citations_are_not_checked() -> None:
    note = ReportSentence(text="This report covers 2 sub-questions.", kind="note")
    cleaned, summary = await run(report([], summary=[note]))
    assert cleaned.summary == [note]
    assert (summary.sentences_checked, summary.precision) == (0, 1.0)


def test_dates_are_plain() -> None:
    assert isinstance(date(2026, 1, 1), date)
