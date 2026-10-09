from __future__ import annotations

from researchai.agent.graph import _evidence_from_hits
from researchai.agent.models import (
    Calculation,
    QuestionKind,
    QuestionProfile,
    ResearchPlan,
    SubQuestion,
)
from researchai.agent.synthesis import (
    LLMReport,
    abstention_report,
    anchor_quote,
    extractive_report,
    extractive_section,
    fit_evidence,
    from_llm_report,
)
from researchai.corpus.models import Language, Topic
from tests.agent_fakes import CHARGING, CONSUMPTION, TARIFF, VAT, hit

PLAN = ResearchPlan(
    sub_questions=[
        SubQuestion(id="sq1", text="How many public charging points?", queries=["charging points"]),
        SubQuestion(id="sq2", text="What is the tariff?", queries=["tariff kWh"]),
        SubQuestion(id="sq3", text="What about hydrogen?", queries=["hydrogen"]),
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
PROFILE = QuestionProfile(
    language=Language.EN, topic=Topic.EV, kind=QuestionKind.FACTUAL, aspects=[]
)


def test_anchor_exact_and_normalized() -> None:
    charging = EVIDENCE[0]
    assert anchor_quote(charging, "Public charging points: 512.") == (100, 128)
    start, end = anchor_quote(charging, "  public   CHARGING points: 512 ") or (0, 0)
    assert CHARGING.text[start - 100 : end - 100] == "Public charging points: 512"
    assert anchor_quote(charging, "points: 513") is None
    assert anchor_quote(charging, "   ") is None


def test_anchor_folds_typographic_quotes_and_dashes() -> None:
    typographic = hit("ev-en-0099#c0", "It\u2019s 0\u2013101 kWh.")  # curly apostrophe, en dash
    evidence = _evidence_from_hits(
        PLAN, [{"sub_question_id": "sq1", "query": "q", "hit": typographic}]
    )[0]
    assert anchor_quote(evidence, "It's 0-101 kWh") == (0, 14)


def test_extractive_section_copies_sentences_verbatim_with_exact_spans() -> None:
    sentences = extractive_section(
        "How many public charging points?", ["charging points"], EVIDENCE[:1]
    )
    assert len(sentences) == 1
    sentence = sentences[0]
    citation = sentence.citations[0]
    assert sentence.text == "Public charging points: 512."
    assert (citation.char_start, citation.char_end) == (100, 128)
    assert citation.quote == sentence.text


def test_extractive_report_marks_missing_evidence_and_recommendation_limits() -> None:
    profile = PROFILE.model_copy(update={"kind": QuestionKind.RECOMMENDATION})
    report = extractive_report("Question?", profile, PLAN, EVIDENCE)
    assert report.generated_by == "rules"
    assert report.insufficient == ["sq3"]
    kinds = [s.kind for s in report.summary]
    assert kinds == ["note", "insufficient_evidence", "note"]
    assert report.sections[2].sentences[0].kind == "insufficient_evidence"
    assert all(c.anchored for c in report.citations())


def test_extractive_report_in_georgian() -> None:
    profile = PROFILE.model_copy(update={"language": Language.KA})
    report = extractive_report("კითხვა?", profile, PLAN, EVIDENCE)
    assert report.title.startswith("კვლევის ანგარიში")


def test_abstention_report() -> None:
    report = abstention_report("q", Language.EN, PLAN)
    assert report.insufficient == ["sq1", "sq2", "sq3"]
    assert [s.kind for s in report.sentences()] == ["insufficient_evidence"]


def _llm_report(
    sections: list[dict[str, object]], summary: list[dict[str, object]] | None = None
) -> LLMReport:
    return LLMReport.model_validate(
        {"title": " Report ", "summary": summary or [], "sections": sections}
    )


def test_from_llm_report_anchors_and_drops_unsupported_claims() -> None:
    raw = _llm_report(
        [
            {
                "sub_question_id": "sq1",
                "heading": "Chargers",
                "sentences": [
                    {
                        "text": "There are 512 public charging points.",
                        "kind": "claim",
                        "citations": [
                            {"evidence_id": "E1", "quote": "Public charging points: 512"}
                        ],
                    },
                    {
                        "text": "There are 9,000 chargers.",
                        "kind": "claim",
                        "citations": [{"evidence_id": "E7", "quote": "9,000"}],
                    },
                    {
                        "text": "Paraphrased.",
                        "kind": "claim",
                        "citations": [{"evidence_id": "E1", "quote": "not in the passage"}],
                    },
                ],
            },
            {"sub_question_id": "sq9", "heading": "Unknown", "sentences": []},
            {"sub_question_id": "sq1", "heading": "Duplicate", "sentences": []},
        ],
        summary=[
            {"text": "Home charging costs C1.", "kind": "note", "calculation_ids": ["C1", "C9"]}
        ],
    )
    calculation = Calculation(
        id="C1", description="d", expression="1+1", result=2, evidence_ids=["E2"]
    )
    report, stats = from_llm_report(raw, PROFILE, PLAN, EVIDENCE, [calculation])
    assert report.generated_by == "llm"
    assert report.title == "Report"
    chargers = report.sections[0]
    assert [s.text for s in chargers.sentences] == [
        "There are 512 public charging points.",
        "Paraphrased.",
    ]
    assert chargers.sentences[0].citations[0].anchored
    assert not chargers.sentences[1].citations[0].anchored  # kept, flagged for the verifier
    assert (stats.citations, stats.anchored, stats.unknown_evidence, stats.dropped_claims) == (
        2,
        1,
        1,
        1,
    )
    assert report.summary[0].calculation_ids == ["C1"]
    assert stats.unknown_calculations == 1
    # sq2 and sq3 had no supported claims -> marked insufficient with a sentence
    assert report.insufficient == ["sq2", "sq3"]
    assert report.sections[1].sentences[-1].kind == "insufficient_evidence"


def test_fit_evidence_keeps_one_passage_per_sub_question() -> None:
    kept, dropped = fit_evidence(EVIDENCE, token_allowance=10)
    assert dropped == 1
    assert {e.sub_question_id for e in kept} == {"sq1", "sq2"}
    everything, none = fit_evidence(EVIDENCE, token_allowance=100_000)
    assert (len(everything), none) == (3, 0)


def test_vat_fixture_is_a_long_sentence() -> None:
    assert len(VAT.text) > 25


def test_extractive_skips_short_lines_tables_and_off_topic_sentences() -> None:
    noisy = hit(
        "ev-en-0013#c3",
        "| Operator | Points |\n|---|---|\nShort one.\nThe weather in Batumi was pleasant this "
        "summer.\nVoltLine operates 190 public charging points in Georgia.",
    )
    evidence = _evidence_from_hits(PLAN, [{"sub_question_id": "sq1", "query": "q", "hit": noisy}])
    sentences = extractive_section(
        "How many public charging points?", ["charging points"], evidence
    )
    assert [s.text for s in sentences] == [
        "VoltLine operates 190 public charging points in Georgia."
    ]


def test_from_llm_report_keeps_the_models_own_insufficiency_sentence() -> None:
    raw = _llm_report(
        [{"sub_question_id": "sq3", "heading": "Hydrogen", "sentences": [
            {"text": "The corpus says nothing about hydrogen.", "kind": "insufficient_evidence"}]}]
    )  # fmt: skip
    report, _ = from_llm_report(raw, PROFILE, PLAN, EVIDENCE, [])
    hydrogen = report.sections[2]
    assert [s.text for s in hydrogen.sentences] == ["The corpus says nothing about hydrogen."]


def test_one_incidental_shared_word_is_not_relevance() -> None:
    sentence = hit("ev-en-0084#c0", "Experts say battery electric cars have already won the race.")
    evidence = _evidence_from_hits(
        PLAN, [{"sub_question_id": "sq1", "query": "q", "hit": sentence}]
    )
    assert extractive_section("Who won the 1998 football world cup?", [], evidence) == []
    assert extractive_section("Who won?", [], evidence)  # a one-term question needs one match
