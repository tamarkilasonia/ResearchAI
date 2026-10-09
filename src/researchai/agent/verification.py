"""Verification results shared by both report writers.

Whatever writes the report, the same guarantees hold:

* every detected conflict is stated in the report, citing both sides with exact
  spans (added from a template if the LLM leaves one out);
* values marked as superseded by a newer source are not presented as current;
* sub-questions judged insufficient say so, and missing key entities are named.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from researchai.agent.models import (
    Citation,
    Claim,
    Conflict,
    Evidence,
    Report,
    ReportSentence,
    ResearchPlan,
    SourceAssessment,
    SufficiencyVerdict,
)
from researchai.corpus.models import Language

TEXT = {
    Language.EN: {
        "conflict": "Sources disagree: {new_pub} ({new_date}) reports “{new}”, while "
        "{old_pub} ({old_date}) reports “{old}”.",
        "outdated": " The newer figure is used in this report.",
        "missing": "Insufficient evidence: the corpus contains no information about {items}.",
    },
    Language.KA: {
        "conflict": "წყაროები ერთმანეთს ეწინააღმდეგება: {new_pub} ({new_date}) წერს: "
        "„{new}“, ხოლო {old_pub} ({old_date}) წერს: „{old}“.",
        "outdated": " ანგარიშში უფრო ახალი მონაცემი გამოიყენება.",
        "missing": "არასაკმარისი მონაცემები: კორპუსში არ არის ინფორმაცია შემდეგზე: {items}.",
    },
}


@dataclass(frozen=True)
class Findings:
    assessments: tuple[SourceAssessment, ...] = ()
    claims: tuple[Claim, ...] = ()
    conflicts: tuple[Conflict, ...] = ()
    verdicts: tuple[SufficiencyVerdict, ...] = ()
    missing_entities: tuple[str, ...] = ()

    def insufficient(self, plan: ResearchPlan, evidence: Sequence[Evidence]) -> list[str]:
        """Sub-question ids without sufficient evidence (by verdict, else by coverage)."""
        if self.verdicts:
            return [v.sub_question_id for v in self.verdicts if not v.sufficient]
        covered = {e.sub_question_id for e in evidence}
        return [sq.id for sq in plan.sub_questions if sq.id not in covered]

    def superseded(self) -> set[tuple[str, int]]:
        """``(doc_id, char_start)`` of claims an outdated conflict marks as superseded."""
        by_id = {c.id: c for c in self.claims}
        return {
            (by_id[i].doc_id, by_id[i].char_start)
            for conflict in self.conflicts
            if conflict.kind == "outdated"
            for i in conflict.others
            if i in by_id
        }

    def quality(self) -> dict[str, tuple[float, float]]:
        """doc_id -> (reliability, freshness)."""
        return {a.doc_id: (a.reliability, a.freshness) for a in self.assessments}

    def conflicts_for(self, sub_question_id: str) -> list[Conflict]:
        by_id = {c.id: c for c in self.claims}
        return [
            conflict
            for conflict in self.conflicts
            if conflict.current[0] in by_id
            and by_id[conflict.current[0]].sub_question_id == sub_question_id
        ]


def claim_citation(claim: Claim, evidence: dict[str, Evidence]) -> Citation:
    item = evidence[claim.evidence_id]
    return Citation(
        evidence_id=claim.evidence_id,
        doc_id=claim.doc_id,
        chunk_id=item.chunk_id,
        quote=claim.text,
        char_start=claim.char_start,
        char_end=claim.char_end,
    )


def conflict_sentence(
    conflict: Conflict,
    claims: dict[str, Claim],
    evidence: dict[str, Evidence],
    language: Language,
) -> ReportSentence:
    new, old = claims[conflict.current[0]], claims[conflict.others[0]]
    text = TEXT[language]["conflict"].format(
        new_pub=evidence[new.evidence_id].publisher,
        new_date=new.published.isoformat(),
        new=new.text.strip("| "),
        old_pub=evidence[old.evidence_id].publisher,
        old_date=old.published.isoformat(),
        old=old.text.strip("| "),
    )
    if conflict.kind == "outdated":
        text += TEXT[language]["outdated"]
    return ReportSentence(
        text=text,
        kind="conflict",
        citations=[claim_citation(new, evidence), claim_citation(old, evidence)],
    )


def missing_sentence(entities: Sequence[str], language: Language) -> ReportSentence:
    return ReportSentence(
        text=TEXT[language]["missing"].format(items=", ".join(entities)),
        kind="insufficient_evidence",
    )


def _reported(conflict: Conflict, report: Report, claims: dict[str, Claim]) -> bool:
    """A conflict sentence cites evidence from both sides of the conflict."""
    current = {claims[i].evidence_id for i in conflict.current if i in claims}
    others = {claims[i].evidence_id for i in conflict.others if i in claims}
    for sentence in report.sentences():
        if sentence.kind != "conflict":
            continue
        cited = {c.evidence_id for c in sentence.citations}
        if cited & current and cited & others:
            return True
    return False


def ensure_conflicts_reported(
    report: Report, findings: Findings, evidence: Sequence[Evidence]
) -> tuple[Report, int]:
    """Append a templated conflict sentence for every conflict the report omits."""
    claims = {c.id: c for c in findings.claims}
    by_evidence = {e.id: e for e in evidence}
    sections = list(report.sections)
    summary = list(report.summary)
    added = 0
    for conflict in findings.conflicts:
        if _reported(conflict, report, claims):
            continue
        sentence = conflict_sentence(conflict, claims, by_evidence, report.language)
        target = claims[conflict.current[0]].sub_question_id
        for index, section in enumerate(sections):
            if section.sub_question_id == target:
                sections[index] = section.model_copy(
                    update={"sentences": [*section.sentences, sentence]}
                )
                break
        else:
            summary.append(sentence)
        added += 1
    if not added:
        return report, 0
    return report.model_copy(update={"sections": sections, "summary": summary}), added
