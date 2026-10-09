"""Report guarantees: conflicts are always stated, superseded values never shown as current."""

from __future__ import annotations

from datetime import date

from researchai.agent.claims import extract_claims
from researchai.agent.conflicts import rule_conflicts
from researchai.agent.graph import _evidence_from_hits
from researchai.agent.models import (
    QuestionKind,
    QuestionProfile,
    Report,
    ReportSection,
    ReportSentence,
    ResearchPlan,
    SubQuestion,
)
from researchai.agent.prompts import writer_user
from researchai.agent.synthesis import extractive_report
from researchai.agent.verification import (
    Findings,
    conflict_sentence,
    ensure_conflicts_reported,
    missing_sentence,
)
from researchai.corpus.models import Language, SourceType, Topic
from tests.agent_fakes import hit

PLAN = ResearchPlan(
    sub_questions=[
        SubQuestion(id="sq1", text="How many public charging points?", queries=["charging points"]),
        SubQuestion(id="sq2", text="What about hydrogen?", queries=["hydrogen"]),
    ],
    planned_by="rules",
)
PROFILE = QuestionProfile(
    language=Language.EN, topic=Topic.EV, kind=QuestionKind.FACTUAL, aspects=[]
)
NEW = hit(
    "ev-en-0013#c1",
    "Public charging points registry total: 512 public charging points.",
    published=date(2026, 7, 15),
)
OLD = hit(
    "ev-en-0014#c0",
    "The public charging points registry total was 385 public charging points.",
    published=date(2025, 2, 12),
)
EVIDENCE = _evidence_from_hits(
    PLAN, [{"sub_question_id": "sq1", "query": "q", "hit": h} for h in (NEW, OLD)]
)
CLAIMS = extract_claims(EVIDENCE, {"ev-en-0013": 0.9, "ev-en-0014": 0.9})
CONFLICTS = rule_conflicts(CLAIMS)
FINDINGS = Findings(claims=tuple(CLAIMS), conflicts=tuple(CONFLICTS))


def test_fixture_has_one_outdated_conflict() -> None:
    (conflict,) = CONFLICTS
    assert conflict.kind == "outdated"


def test_findings_helpers() -> None:
    assert FINDINGS.insufficient(PLAN, EVIDENCE) == ["sq2"]  # no verdicts: by coverage
    old_claim = next(c for c in CLAIMS if c.doc_id == "ev-en-0014")
    assert FINDINGS.superseded() == {("ev-en-0014", old_claim.char_start)}
    assert [c.id for c in FINDINGS.conflicts_for("sq1")] == [CONFLICTS[0].id]
    assert FINDINGS.conflicts_for("sq2") == []
    assert Findings().quality() == {}


def test_conflict_sentence_cites_both_sides_exactly() -> None:
    claims = {c.id: c for c in CLAIMS}
    by_evidence = {e.id: e for e in EVIDENCE}
    sentence = conflict_sentence(CONFLICTS[0], claims, by_evidence, Language.EN)
    assert sentence.kind == "conflict"
    assert sentence.text.startswith("Sources disagree: Publisher (2026-07-15) reports")
    assert sentence.text.endswith("The newer figure is used in this report.")
    assert [c.doc_id for c in sentence.citations] == ["ev-en-0013", "ev-en-0014"]
    assert all(c.anchored for c in sentence.citations)
    georgian = conflict_sentence(CONFLICTS[0], claims, by_evidence, Language.KA)
    assert georgian.text.startswith("წყაროები ერთმანეთს ეწინააღმდეგება")


def test_missing_sentence() -> None:
    sentence = missing_sentence(["Japan", "Model S"], Language.EN)
    assert sentence.kind == "insufficient_evidence"
    assert "Japan, Model S" in sentence.text


def test_offline_report_hides_superseded_values_and_states_the_conflict() -> None:
    report = extractive_report("How many?", PROFILE, PLAN, EVIDENCE, FINDINGS)
    section = report.sections[0]
    claims = [s for s in section.sentences if s.kind == "claim"]
    assert [s.citations[0].doc_id for s in claims] == ["ev-en-0013"]  # 385 never shown as a fact
    conflicts = [s for s in section.sentences if s.kind == "conflict"]
    assert len(conflicts) == 1
    missing = Findings(missing_entities=("Japan",))
    with_missing = extractive_report("How many?", PROFILE, PLAN, EVIDENCE, missing)
    assert any("Japan" in s.text for s in with_missing.summary)


def _report(sentences: list[ReportSentence], sub_question: str = "sq1") -> Report:
    return Report(
        language=Language.EN,
        title="t",
        summary=[],
        sections=[ReportSection(heading="h", sub_question_id=sub_question, sentences=sentences)],
        generated_by="llm",
    )


def test_missing_conflicts_are_appended_where_they_belong() -> None:
    report, added = ensure_conflicts_reported(_report([]), FINDINGS, EVIDENCE)
    assert added == 1
    assert report.sections[0].sentences[-1].kind == "conflict"
    # Already reported (cites both sides): left alone.
    again, none = ensure_conflicts_reported(report, FINDINGS, EVIDENCE)
    assert (again, none) == (report, 0)
    # The conflict's sub-question has no section: it goes to the summary.
    elsewhere, added = ensure_conflicts_reported(_report([], "sq2"), FINDINGS, EVIDENCE)
    assert added == 1
    assert elsewhere.summary[-1].kind == "conflict"


def test_a_one_sided_conflict_sentence_does_not_count() -> None:
    claims = {c.id: c for c in CLAIMS}
    full = conflict_sentence(CONFLICTS[0], claims, {e.id: e for e in EVIDENCE}, Language.EN)
    one_sided = full.model_copy(update={"citations": full.citations[:1]})
    _, added = ensure_conflicts_reported(_report([one_sided]), FINDINGS, EVIDENCE)
    assert added == 1


def test_writer_prompt_lists_conflicts_quality_and_missing_names() -> None:
    prompt = writer_user(
        "q",
        PLAN,
        EVIDENCE,
        [],
        insufficient=["sq2"],
        conflicts=CONFLICTS,
        claims=CLAIMS,
        missing=["Japan"],
        quality={"ev-en-0013": (0.9, 1.0)},
    )
    assert "X1 (outdated): newer/preferred E1:" in prompt
    assert 'reliability="0.90" freshness="1.00"' in prompt
    assert "never mentions: Japan" in prompt
    assert "without sufficient evidence: sq2" in prompt


def test_source_type_fixture() -> None:
    assert NEW.source_type is SourceType.GOVERNMENT
