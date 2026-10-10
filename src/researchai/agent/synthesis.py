"""Report writing.

Offline (rules)
    Extractive: for each sub-question, the best-matching sentences of its
    evidence are copied verbatim as claims, each cited with its exact span.
    The baseline never rewrites or combines facts, so it cannot invent any; it
    also cannot reason, compute or recommend, which the report says plainly.

LLM
    The model writes sentences with ``(evidence id, quote)`` citations. Each
    quote is anchored to exact offsets in the cited passage; citations to
    unknown evidence are discarded and a claim left without any citation is
    dropped (and counted) instead of being shown unsupported.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from researchai.agent.anchoring import anchor_quote
from researchai.agent.budget import RunBudget
from researchai.agent.llm import LLMClient, estimate_tokens, structured_call
from researchai.agent.models import (
    Calculation,
    Citation,
    Evidence,
    QuestionKind,
    QuestionProfile,
    Report,
    ReportSection,
    ReportSentence,
    ResearchPlan,
    SentenceKind,
)
from researchai.agent.prompts import WRITER_SYSTEM, writer_user
from researchai.agent.verification import (
    Findings,
    conflict_sentence,
    drop_superseded_claims,
    ensure_conflicts_reported,
    insufficient_sentence,
    missing_sentence,
)
from researchai.corpus.models import Language
from researchai.safety.documents import instruction_spans, overlaps
from researchai.textutil import keyword_terms, sentence_spans

SENTENCES_PER_SECTION = 2
MIN_SENTENCE_CHARS = 25
# A sentence must share this many content terms with its sub-question (fewer if the
# sub-question itself has fewer), so one incidental word ("won", "world") is not enough.
MIN_TERM_OVERLAP = 2

TEXT = {
    Language.EN: {
        "title": "Research report: {question}",
        "scope": "This report answers {n} sub-questions using {m} passages from {d} documents "
        "of the frozen research corpus.",
        "insufficient_summary": "Evidence was insufficient for: {items}.",
        "no_recommendation": "This baseline does not form a recommendation; weigh the cited "
        "points above against your own situation.",
        "abstain": "Insufficient evidence: the corpus does not contain information to answer "
        "this question.",
    },
    Language.KA: {
        "title": "კვლევის ანგარიში: {question}",
        "scope": "ანგარიში პასუხობს {n} ქვეკითხვას კვლევის კორპუსის {d} დოკუმენტის {m} "
        "ფრაგმენტზე დაყრდნობით.",
        "insufficient_summary": "მონაცემები არასაკმარისი იყო შემდეგ საკითხებზე: {items}.",
        "no_recommendation": "ეს საბაზისო ვერსია რეკომენდაციას არ აყალიბებს; შეაფასეთ ზემოთ "
        "მოყვანილი ფაქტები თქვენი სიტუაციის მიხედვით.",
        "abstain": "არასაკმარისი მონაცემები: კორპუსში ამ კითხვაზე პასუხის გასაცემი ინფორმაცია "
        "არ არის.",
    },
}

_TABLE_RULE = re.compile(r"^\|?[\s:|-]+\|?$")


def make_citation(evidence: Evidence, quote: str) -> Citation:
    span = anchor_quote(evidence, quote)
    return Citation(
        evidence_id=evidence.id,
        doc_id=evidence.doc_id,
        chunk_id=evidence.chunk_id,
        quote=quote.strip(),
        char_start=span[0] if span else None,
        char_end=span[1] if span else None,
    )


# ----------------------------------------------------------------- extractive (rules)


@dataclass(order=True)
class _Candidate:
    score: float
    tiebreak: tuple[str, int] = field(compare=True)
    evidence: Evidence = field(compare=False)
    start: int = field(compare=False)
    end: int = field(compare=False)


def _candidates(
    evidence: Evidence,
    terms: set[str],
    skip: set[tuple[str, int]],
    quality: dict[str, tuple[float, float]],
) -> list[_Candidate]:
    reliability, freshness = quality.get(evidence.doc_id, (0.5, 0.5))
    found = []
    hidden = instruction_spans(evidence.text)
    for start, end in sentence_spans(evidence.text):
        sentence = evidence.text[start:end]
        if len(sentence) < MIN_SENTENCE_CHARS or _TABLE_RULE.match(sentence):
            continue
        if overlaps(start, end, hidden):
            continue  # instruction-like text is never quoted
        if sentence.startswith("|") and not any(ch.isdigit() for ch in sentence):
            continue  # a table header row states nothing
        if (evidence.doc_id, evidence.char_start + start) in skip:
            continue  # a value a newer source supersedes; it appears only in a conflict
        overlap = len(terms & set(keyword_terms(sentence)))
        if overlap < min(MIN_TERM_OVERLAP, len(terms)):
            continue
        has_number = any(ch.isdigit() for ch in sentence)
        # Higher score first; ties broken by evidence id then position (deterministic).
        score = overlap + 0.5 * has_number + 0.1 * evidence.score + reliability + 0.5 * freshness
        found.append(_Candidate(-score, (evidence.id, start), evidence, start, end))
    return found


def extractive_section(
    sub_question_text: str,
    queries: Sequence[str],
    evidence: Sequence[Evidence],
    findings: Findings | None = None,
) -> list[ReportSentence]:
    findings = findings or Findings()
    terms = set(keyword_terms(" ".join([sub_question_text, *queries])))
    skip, quality = findings.superseded(), findings.quality()
    ranked = sorted(c for item in evidence for c in _candidates(item, terms, skip, quality))
    chosen: list[_Candidate] = []
    used: set[str] = set()
    for candidate in ranked:
        if candidate.evidence.id in used:
            continue  # one sentence per passage, for diversity of sources
        chosen.append(candidate)
        used.add(candidate.evidence.id)
        if len(chosen) == SENTENCES_PER_SECTION:
            break
    return [
        ReportSentence(
            text=c.evidence.text[c.start : c.end],
            kind="claim",
            citations=[
                Citation(
                    evidence_id=c.evidence.id,
                    doc_id=c.evidence.doc_id,
                    chunk_id=c.evidence.chunk_id,
                    quote=c.evidence.text[c.start : c.end],
                    char_start=c.evidence.char_start + c.start,
                    char_end=c.evidence.char_start + c.end,
                )
            ],
        )
        for c in chosen
    ]


def _scope_sentence(
    language: Language, plan: ResearchPlan, evidence: Sequence[Evidence]
) -> ReportSentence:
    text = TEXT[language]["scope"].format(
        n=len(plan.sub_questions), m=len(evidence), d=len({e.doc_id for e in evidence})
    )
    return ReportSentence(text=text, kind="note")


def _insufficient(language: Language) -> ReportSentence:
    return insufficient_sentence(language)


def extractive_report(
    question: str,
    profile: QuestionProfile,
    plan: ResearchPlan,
    evidence: Sequence[Evidence],
    findings: Findings | None = None,
) -> Report:
    findings = findings or Findings()
    language = profile.language
    judged_insufficient = set(findings.insufficient(plan, evidence))
    claims = {c.id: c for c in findings.claims}
    by_evidence = {e.id: e for e in evidence}
    sections: list[ReportSection] = []
    insufficient: list[str] = []
    for sq in plan.sub_questions:
        own = [e for e in evidence if e.sub_question_id == sq.id]
        sentences = (
            extractive_section(sq.text, sq.queries, own, findings)
            if own and sq.id not in judged_insufficient
            else []
        )
        if not sentences:
            insufficient.append(sq.id)
            sentences = [_insufficient(language)]
        sentences += [
            conflict_sentence(conflict, claims, by_evidence, language)
            for conflict in findings.conflicts_for(sq.id)
        ]
        sections.append(ReportSection(heading=sq.text, sub_question_id=sq.id, sentences=sentences))
    summary = [_scope_sentence(language, plan, evidence)]
    if findings.missing_entities:
        summary.append(missing_sentence(findings.missing_entities, language))
    if insufficient:
        items = "; ".join(sq.text for sq in plan.sub_questions if sq.id in insufficient)
        summary.append(
            ReportSentence(
                text=TEXT[language]["insufficient_summary"].format(items=items),
                kind="insufficient_evidence",
            )
        )
    if profile.kind is QuestionKind.RECOMMENDATION:
        summary.append(ReportSentence(text=TEXT[language]["no_recommendation"], kind="note"))
    return Report(
        language=language,
        title=TEXT[language]["title"].format(question=question),
        summary=summary,
        sections=sections,
        insufficient=insufficient,
        generated_by="rules",
    )


def abstention_report(
    question: str, language: Language, plan: ResearchPlan, missing: Sequence[str] = ()
) -> Report:
    summary = [ReportSentence(text=TEXT[language]["abstain"], kind="insufficient_evidence")]
    if missing:
        summary.append(missing_sentence(missing, language))
    return Report(
        language=language,
        title=TEXT[language]["title"].format(question=question),
        summary=summary,
        sections=[],
        insufficient=[sq.id for sq in plan.sub_questions],
        generated_by="rules",
    )


# ----------------------------------------------------------------- LLM


class _LLMCitation(BaseModel):
    model_config = ConfigDict(extra="forbid")
    evidence_id: str
    quote: str = Field(min_length=1, max_length=600)


class _LLMSentence(BaseModel):
    model_config = ConfigDict(extra="forbid")
    text: str = Field(min_length=1, max_length=800)
    kind: SentenceKind
    citations: list[_LLMCitation] = Field(default_factory=list, max_length=4)
    calculation_ids: list[str] = Field(default_factory=list, max_length=4)


class _LLMSection(BaseModel):
    model_config = ConfigDict(extra="forbid")
    sub_question_id: str
    heading: str = Field(min_length=1, max_length=200)
    sentences: list[_LLMSentence] = Field(max_length=8)


class LLMReport(BaseModel):
    model_config = ConfigDict(extra="forbid")
    title: str = Field(min_length=1, max_length=200)
    summary: list[_LLMSentence] = Field(max_length=6)
    sections: list[_LLMSection] = Field(max_length=10)


@dataclass
class AnchorStats:
    citations: int = 0
    anchored: int = 0
    unknown_evidence: int = 0
    dropped_claims: int = 0
    unknown_calculations: int = 0
    templated_insufficient: int = 0
    templated_conflicts: int = 0
    superseded_dropped: int = 0


def _convert_sentence(
    raw: _LLMSentence,
    evidence: dict[str, Evidence],
    calculation_ids: set[str],
    stats: AnchorStats,
    language: Language,
) -> ReportSentence | None:
    if raw.kind == "insufficient_evidence":
        # Fixed wording: the model's own text could smuggle in an uncited claim
        # ("... does not affect range, as there is no data") under this kind.
        stats.templated_insufficient += 1
        return _insufficient(language)
    citations = []
    for cite in raw.citations:
        item = evidence.get(cite.evidence_id)
        if item is None:
            stats.unknown_evidence += 1
            continue
        citation = make_citation(item, cite.quote)
        stats.citations += 1
        stats.anchored += citation.anchored
        citations.append(citation)
    calcs = [c for c in raw.calculation_ids if c in calculation_ids]
    stats.unknown_calculations += len(raw.calculation_ids) - len(calcs)
    kind: SentenceKind = raw.kind
    if kind in ("claim", "conflict") and not citations:
        stats.dropped_claims += 1
        return None
    return ReportSentence(
        text=raw.text.strip(), kind=kind, citations=citations, calculation_ids=calcs
    )


def from_llm_report(
    raw: LLMReport,
    profile: QuestionProfile,
    plan: ResearchPlan,
    evidence: Sequence[Evidence],
    calculations: Sequence[Calculation],
) -> tuple[Report, AnchorStats]:
    by_id = {item.id: item for item in evidence}
    calc_ids = {c.id for c in calculations}
    stats = AnchorStats()

    def convert(sentences: Sequence[_LLMSentence]) -> list[ReportSentence]:
        converted = (
            _convert_sentence(s, by_id, calc_ids, stats, profile.language) for s in sentences
        )
        return [s for s in converted if s is not None]

    known = {sq.id: sq for sq in plan.sub_questions}
    sections: dict[str, ReportSection] = {}
    for raw_section in raw.sections:
        if raw_section.sub_question_id not in known or raw_section.sub_question_id in sections:
            continue
        sections[raw_section.sub_question_id] = ReportSection(
            heading=raw_section.heading.strip(),
            sub_question_id=raw_section.sub_question_id,
            sentences=convert(raw_section.sentences),
        )
    # Every sub-question gets a section; one with nothing supported says so.
    insufficient: list[str] = []
    ordered: list[ReportSection] = []
    for sq in plan.sub_questions:
        section = sections.get(sq.id) or ReportSection(
            heading=sq.text, sub_question_id=sq.id, sentences=[]
        )
        if not any(s.kind in ("claim", "conflict") for s in section.sentences):
            insufficient.append(sq.id)
            if not any(s.kind == "insufficient_evidence" for s in section.sentences):
                section = section.model_copy(
                    update={"sentences": [*section.sentences, _insufficient(profile.language)]}
                )
        ordered.append(section)
    report = Report(
        language=profile.language,
        title=raw.title.strip(),
        summary=convert(raw.summary),
        sections=ordered,
        insufficient=insufficient,
        generated_by="llm",
    )
    return report, stats


def fit_evidence(evidence: Sequence[Evidence], token_allowance: int) -> tuple[list[Evidence], int]:
    """Drop the lowest-scored passages (keeping one per sub-question) until they fit."""
    kept = list(evidence)
    dropped = 0
    while sum(estimate_tokens(e.text) + 60 for e in kept) > token_allowance:
        counts: dict[str, int] = {}
        for item in kept:
            counts[item.sub_question_id] = counts.get(item.sub_question_id, 0) + 1
        removable = [e for e in kept if counts[e.sub_question_id] > 1]
        if not removable:
            break
        victim = min(removable, key=lambda e: (e.score, e.id))
        kept.remove(victim)
        dropped += 1
    return kept, dropped


async def llm_report(
    question: str,
    profile: QuestionProfile,
    plan: ResearchPlan,
    evidence: Sequence[Evidence],
    calculations: Sequence[Calculation],
    findings: Findings,
    *,
    llm: LLMClient,
    budget: RunBudget,
    max_output_tokens: int,
    max_evidence_tokens: int,
) -> tuple[Report, AnchorStats, int]:
    """Returns the report, anchoring statistics and how many passages were trimmed."""
    fits_context = llm.context_tokens - max_output_tokens - 2500  # instructions, schema, question
    allowance = min(fits_context, max_evidence_tokens)
    fitted, trimmed = fit_evidence(evidence, allowance)
    language_name: Literal["English", "Georgian"] = (
        "Georgian" if profile.language is Language.KA else "English"
    )
    result = await structured_call(
        llm,
        budget,
        system=WRITER_SYSTEM.replace("{language}", language_name),
        user=writer_user(
            question,
            plan,
            fitted,
            calculations,
            insufficient=findings.insufficient(plan, evidence),
            conflicts=findings.reportable_conflicts(),
            claims=findings.claims,
            missing=findings.missing_entities,
            quality=findings.quality(),
        ),
        schema=LLMReport,
        max_tokens=max_output_tokens,
    )
    report, stats = from_llm_report(result.value, profile, plan, fitted, calculations)
    report, stats.superseded_dropped = drop_superseded_claims(report, findings)
    report, stats.templated_conflicts = ensure_conflicts_reported(report, findings, evidence)
    if findings.missing_entities:
        report = report.model_copy(
            update={
                "summary": [
                    missing_sentence(findings.missing_entities, profile.language),
                    *report.summary,
                ]
            }
        )
    return report, stats, trimmed
