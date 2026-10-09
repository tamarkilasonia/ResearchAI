"""Verification results shared by both report writers.

Whatever writes the report, the same guarantees hold:

* every detected conflict *relevant to the question* is stated in the report,
  citing both sides with exact spans (added from a template if the LLM leaves it
  out); irrelevant ones stay in the audit result only;
* values marked as superseded by a newer source are not presented as current:
  the extractive writer never picks them, and LLM sentences whose citations all
  point at superseded spans are dropped;
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
from researchai.textutil import DOMAIN_COMMON, keyword_terms

MIN_SHARED_TERMS = 2

TEXT = {
    Language.EN: {
        "conflict": "Sources disagree: {new_pub} ({new_date}) reports “{new}”, while "
        "{old_pub} ({old_date}) reports “{old}”.",
        "outdated": " The newer figure is used in this report.",
        "missing": "Insufficient evidence: the corpus contains no information about {items}.",
        "insufficient": "Insufficient evidence: the corpus does not contain information on this "
        "point.",
    },
    Language.KA: {
        "conflict": "წყაროები ერთმანეთს ეწინააღმდეგება: {new_pub} ({new_date}) წერს: "
        "„{new}“, ხოლო {old_pub} ({old_date}) წერს: „{old}“.",
        "outdated": " ანგარიშში უფრო ახალი მონაცემი გამოიყენება.",
        "missing": "არასაკმარისი მონაცემები: კორპუსში არ არის ინფორმაცია შემდეგზე: {items}.",
        "insufficient": "არასაკმარისი მონაცემები: კორპუსში ამ საკითხზე ინფორმაცია არ არის.",
    },
}


@dataclass(frozen=True)
class Findings:
    assessments: tuple[SourceAssessment, ...] = ()
    claims: tuple[Claim, ...] = ()
    conflicts: tuple[Conflict, ...] = ()
    verdicts: tuple[SufficiencyVerdict, ...] = ()
    missing_entities: tuple[str, ...] = ()
    # sub-question id -> terms of the question, the sub-question and its queries.
    # Empty: every conflict counts as relevant.
    relevance_terms: tuple[tuple[str, frozenset[str]], ...] = ()

    def reportable(self, conflict: Conflict) -> bool:
        """Whether a conflict concerns what was asked (else: audit trail only)."""
        claims = {c.id: c for c in self.claims}
        current = claims.get(conflict.current[0])
        terms = dict(self.relevance_terms).get(current.sub_question_id) if current else None
        if terms is None:
            return True
        terms = terms - DOMAIN_COMMON  # "car", "electric", "Georgia" do not make it relevant
        if not terms:
            return False
        needed = min(MIN_SHARED_TERMS, len(terms))
        return any(
            len(terms & set(keyword_terms(claims[i].text))) >= needed
            for i in (*conflict.current, *conflict.others)
            if i in claims
        )

    def reportable_conflicts(self) -> list[Conflict]:
        return [c for c in self.conflicts if self.reportable(c)]

    def superseded_ranges(self) -> dict[str, list[tuple[int, int]]]:
        """doc_id -> character ranges of claims superseded by a newer source."""
        claims = {c.id: c for c in self.claims}
        ranges: dict[str, list[tuple[int, int]]] = {}
        for conflict in self.conflicts:
            if conflict.kind != "outdated":
                continue
            for claim_id in conflict.others:
                if claim_id in claims:
                    claim = claims[claim_id]
                    ranges.setdefault(claim.doc_id, []).append((claim.char_start, claim.char_end))
        return ranges

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
            for conflict in self.reportable_conflicts()
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


def insufficient_sentence(language: Language) -> ReportSentence:
    return ReportSentence(text=TEXT[language]["insufficient"], kind="insufficient_evidence")


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
    for conflict in findings.reportable_conflicts():
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


def _overlaps(citation: Citation, ranges: dict[str, list[tuple[int, int]]]) -> bool:
    if citation.char_start is None or citation.char_end is None:
        return False
    return any(
        citation.char_start < end and start < citation.char_end
        for start, end in ranges.get(citation.doc_id, [])
    )


def drop_superseded_claims(report: Report, findings: Findings) -> tuple[Report, int]:
    """Remove claims/recommendations whose every citation is a superseded value."""
    ranges = findings.superseded_ranges()
    if not ranges:
        return report, 0
    dropped = 0

    def keep(sentence: ReportSentence) -> bool:
        nonlocal dropped
        stale = (
            sentence.kind in ("claim", "recommendation")
            and bool(sentence.citations)
            and all(_overlaps(c, ranges) for c in sentence.citations)
        )
        dropped += stale
        return not stale

    sections = [
        section.model_copy(update={"sentences": [s for s in section.sentences if keep(s)]})
        for section in report.sections
    ]
    summary = [s for s in report.summary if keep(s)]
    trimmed = report.model_copy(update={"sections": sections, "summary": summary})
    return fill_empty_sections(trimmed), dropped


def fill_empty_sections(report: Report) -> Report:
    """A sub-question section left without supported sentences says insufficient evidence."""
    supported = ("claim", "conflict", "recommendation")
    sections = []
    insufficient = list(report.insufficient)
    for section in report.sections:
        sentences = list(section.sentences)
        if section.sub_question_id and not any(s.kind in supported for s in sentences):
            if not any(s.kind == "insufficient_evidence" for s in sentences):
                sentences.append(insufficient_sentence(report.language))
            if section.sub_question_id not in insufficient:
                insufficient.append(section.sub_question_id)
        sections.append(section.model_copy(update={"sentences": sentences}))
    return report.model_copy(update={"sections": sections, "insufficient": insufficient})
