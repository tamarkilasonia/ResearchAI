"""Citation verification: deterministic checks first, entailment second.

For every cited sentence of a report (claims, conflicts, recommendations):

1. **Spans.** Each cited document is fetched once through MCP and every cited
   span must equal its quote (whitespace and case normalised) and must not
   overlap instruction-like text (untrusted, see :mod:`researchai.safety`).
   Citations that fail are removed from the sentence; a sentence left without a
   verified citation is *unsupported*.
2. **Numbers.** Every number in the sentence must appear in its cited spans, in
   a cited calculation result, or in the question itself. A number found only in
   some *other* passage makes the sentence *misattributed*; found nowhere, it is
   *unsupported*.
3. **Entailment**, only for sentences that are not verbatim quotes: an LLM judges
   supported / partial / not supported in one batched call; offline (or if the
   call fails) lexical coverage of the sentence's content words by the cited
   spans decides. A judgement can lower support, never rescue a deterministic
   failure.

Unsupported and misattributed sentences are removed before the user sees the
report; partially supported ones are kept and flagged. The per-report summary
(citation precision, counts, removed and flagged sentences) goes into the result.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import date
from typing import Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field

from researchai.agent.analysis import evidence_numbers
from researchai.agent.anchoring import normalized
from researchai.agent.budget import BudgetExceeded, RunBudget
from researchai.agent.claims import facts_in
from researchai.agent.llm import LLMClient, LLMError, structured_call
from researchai.agent.models import (
    Calculation,
    Citation,
    CitationCheck,
    CitationReport,
    CitationStatus,
    Evidence,
    Report,
    ReportSentence,
    SentenceCheck,
    SentenceStatus,
    SupportLevel,
)
from researchai.agent.prompts import ENTAILMENT_SYSTEM, entailment_user
from researchai.agent.verification import fill_empty_sections
from researchai.mcp_client.client import ToolCallError
from researchai.safety.documents import instruction_spans, overlaps
from researchai.textutil import DOMAIN_COMMON, keyword_terms
from researchai.tools.errors import ErrorCode
from researchai.tools.schemas import DocumentResponse

CHECKED_KINDS = ("claim", "conflict", "recommendation")
NUMBER_TOLERANCE = 0.006  # relative; 4.91 matches a computed 4.9056
SUPPORTED_COVERAGE = 0.7
PARTIAL_COVERAGE = 0.4
_NO_DATE = date(2000, 1, 1)


class DocumentFetcher(Protocol):
    async def fetch_document(
        self, doc_id: str, *, start: int | None = None, end: int | None = None
    ) -> DocumentResponse: ...


async def fetch_bodies(doc_ids: Sequence[str], tools: DocumentFetcher) -> dict[str, str | None]:
    """doc_id -> full text, or None if the document does not exist."""
    bodies: dict[str, str | None] = {}
    for doc_id in dict.fromkeys(doc_ids):
        try:
            bodies[doc_id] = (await tools.fetch_document(doc_id)).text
        except ToolCallError as exc:
            if exc.code is not ErrorCode.NOT_FOUND and exc.code is not ErrorCode.INVALID_ARGUMENT:
                raise
            bodies[doc_id] = None
    return bodies


def check_citation(citation: Citation, bodies: dict[str, str | None]) -> CitationStatus:
    body = bodies.get(citation.doc_id)
    if body is None:
        return "missing_document"
    if citation.char_start is None or citation.char_end is None:
        return "unanchored"
    span = body[citation.char_start : citation.char_end]
    if normalized(span) != normalized(citation.quote):
        return "span_mismatch"
    if overlaps(citation.char_start, citation.char_end, instruction_spans(body)):
        return "instruction_text"  # untrusted instructions never support a statement
    return "verified"


def sentence_numbers(text: str) -> list[float]:
    """Numbers stated in a sentence (dates excluded; a range contributes both ends)."""
    values: list[float] = []
    for fact in facts_in(text, 0, _NO_DATE):
        values.append(fact.value)
        if fact.high is not None:
            values.append(fact.high)
    return values


def _matches(value: float, available: set[float]) -> bool:
    return any(abs(value - a) <= NUMBER_TOLERANCE * max(1.0, abs(a)) for a in available)


def missing_numbers(text: str, available: set[float]) -> list[float]:
    return [v for v in sentence_numbers(text) if not _matches(v, available)]


def is_verbatim(sentence: ReportSentence) -> bool:
    """The sentence is, or quotes, every cited span word for word."""
    text = normalized(sentence.text)
    return all(normalized(c.quote) in text for c in sentence.citations)


def lexical_support(text: str, spans: Sequence[str]) -> SupportLevel | None:
    """Coverage of the sentence's content words by the cited spans (None: not supported)."""
    terms = {
        t for t in keyword_terms(text) if t not in DOMAIN_COMMON and not any(c.isdigit() for c in t)
    }
    if not terms:
        return "supported"
    span_terms = set(keyword_terms(" ".join(spans)))
    coverage = len(terms & span_terms) / len(terms)
    if coverage >= SUPPORTED_COVERAGE:
        return "supported"
    if coverage >= PARTIAL_COVERAGE:
        return "partial"
    return None


SEVERITY = {"supported": 0, "partial": 1, "not_supported": 2}


def _judgement_key(raw: str) -> str:
    """Models echo ids as shown ("[S0.1]") or reformat them ("s0.1 "): normalise."""
    return raw.strip().strip("[]").strip().upper()


class _Judgement(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str
    label: Literal["supported", "partial", "not_supported"]
    reason: str = Field(max_length=300)


class Judgements(BaseModel):
    model_config = ConfigDict(extra="forbid")
    judgements: list[_Judgement] = Field(max_length=40)


@dataclass
class _Pending:
    """A sentence that passed the deterministic checks and awaits entailment."""

    key: str
    sentence: ReportSentence
    sub_question_id: str | None
    spans: list[str]
    checks: list[CitationCheck]
    verified: list[Citation]


@dataclass
class _Outcome:
    keep: dict[str, ReportSentence] = field(default_factory=dict)
    checks: dict[str, SentenceCheck] = field(default_factory=dict)


def _check(
    pending: _Pending | ReportSentence,
    status: SentenceStatus,
    method: Literal["deterministic", "llm", "lexical"],
    reason: str,
    sub_question_id: str | None = None,
    citations: Sequence[CitationCheck] = (),
) -> SentenceCheck:
    sentence = pending.sentence if isinstance(pending, _Pending) else pending
    return SentenceCheck(
        text=sentence.text,
        kind=sentence.kind,
        sub_question_id=pending.sub_question_id
        if isinstance(pending, _Pending)
        else sub_question_id,
        status=status,
        method=method,
        reason=reason,
        citations=list(pending.checks if isinstance(pending, _Pending) else citations),
    )


Method = Literal["llm", "lexical"]


async def _entail(
    pending: Sequence[_Pending],
    calculations: Sequence[Calculation],
    llm: LLMClient | None,
    budget: RunBudget,
    max_output_tokens: int,
) -> tuple[dict[str, tuple[str, str, Method]], Literal["llm", "lexical", "none"]]:
    """key -> (label, reason, method); labels: supported / partial / not_supported."""
    if not pending:
        return {}, "none"
    # Identical statements with identical spans (a summary repeating a section) are
    # judged once; models tend to skip exact duplicates.
    groups: dict[tuple[str, tuple[str, ...]], list[_Pending]] = {}
    for item in pending:
        signature = (normalized(item.sentence.text), tuple(normalized(s) for s in item.spans))
        groups.setdefault(signature, []).append(item)
    representatives = [members[0] for members in groups.values()]
    verdicts: dict[str, tuple[str, str, Method]] = {}
    if llm is not None:
        try:
            result = await structured_call(
                llm,
                budget,
                system=ENTAILMENT_SYSTEM,
                user=entailment_user(
                    [(p.key, p.sentence.text, p.spans) for p in representatives], calculations
                ),
                schema=Judgements,
                max_tokens=min(max_output_tokens, 1500),
            )
            keys = {p.key for p in representatives}
            for judgement in result.value.judgements:
                key = _judgement_key(judgement.id)
                if key not in keys:
                    continue
                # A model may judge the same statement twice; keep the most severe label.
                previous = verdicts.get(key)
                if previous is None or SEVERITY[judgement.label] > SEVERITY[previous[0]]:
                    verdicts[key] = (judgement.label, judgement.reason, "llm")
        except (LLMError, BudgetExceeded):
            verdicts = {}
        for members in groups.values():
            if members[0].key in verdicts:
                for member in members[1:]:
                    verdicts[member.key] = verdicts[members[0].key]
    for item in pending:
        if item.key not in verdicts:  # offline, failed call, or an item the model skipped
            level = lexical_support(item.sentence.text, item.spans)
            verdicts[item.key] = (
                level or "not_supported",
                "lexical coverage of cited spans",
                "lexical",
            )
    overall: Literal["llm", "lexical"] = (
        "llm" if any(m == "llm" for _, _, m in verdicts.values()) else "lexical"
    )
    return verdicts, overall


async def verify_report(
    report: Report,
    evidence: Sequence[Evidence],
    calculations: Sequence[Calculation],
    question: str,
    *,
    tools: DocumentFetcher,
    llm: LLMClient | None,
    budget: RunBudget,
    max_output_tokens: int,
) -> tuple[Report, CitationReport]:
    cited = [
        (key, section_id, sentence)
        for key, section_id, sentence in _sentences(report)
        if sentence.kind in CHECKED_KINDS and sentence.citations
    ]
    bodies = await fetch_bodies([c.doc_id for _, _, s in cited for c in s.citations], tools)
    calc_by_id = {c.id: c for c in calculations}
    question_numbers = set(sentence_numbers(question))
    all_evidence_numbers = (
        set().union(*(evidence_numbers(e.text) for e in evidence)) if evidence else set()
    )

    outcome = _Outcome()
    pending: list[_Pending] = []
    citations_total = citations_verified = 0
    for key, section_id, sentence in cited:
        checks = [
            CitationCheck(
                evidence_id=c.evidence_id, doc_id=c.doc_id, status=check_citation(c, bodies)
            )
            for c in sentence.citations
        ]
        citations_total += len(checks)
        verified = [
            c for c, k in zip(sentence.citations, checks, strict=True) if k.status == "verified"
        ]
        citations_verified += len(verified)
        if not verified:
            outcome.checks[key] = _check(
                sentence, "unsupported", "deterministic", "no cited span could be verified",
                section_id, checks,
            )  # fmt: skip
            continue
        spans = [c.quote for c in verified]
        cited_calcs = [calc_by_id[i] for i in sentence.calculation_ids if i in calc_by_id]
        available = set().union(*(evidence_numbers(s) for s in spans)) | question_numbers
        for calc in cited_calcs:
            available |= {calc.result, round(calc.result, 2), round(calc.result, 1)}
        missing = missing_numbers(sentence.text, available)
        if missing:
            elsewhere = all(_matches(v, all_evidence_numbers) for v in missing)
            status: SentenceStatus = "misattributed" if elsewhere else "unsupported"
            where = "another passage" if elsewhere else "any passage"
            reason = (
                f"numbers {[f'{v:g}' for v in missing]} are not in the cited spans but in {where}"
            )
            outcome.checks[key] = _check(
                sentence, status, "deterministic", reason, section_id, checks
            )
            continue
        trimmed = sentence.model_copy(update={"citations": verified})
        if is_verbatim(trimmed):
            outcome.keep[key] = trimmed.model_copy(update={"support": "verbatim"})
            outcome.checks[key] = _check(
                trimmed, "verbatim", "deterministic", "quotes its cited spans", section_id, checks
            )
            continue
        pending.append(_Pending(key, trimmed, section_id, spans, checks, verified))

    verdicts, method = await _entail(pending, calculations, llm, budget, max_output_tokens)
    for item in pending:
        label, reason, how = verdicts[item.key]
        if label == "not_supported":
            outcome.checks[item.key] = _check(item, "unsupported", how, reason)
            continue
        level: SupportLevel = "supported" if label == "supported" else "partial"
        outcome.keep[item.key] = item.sentence.model_copy(update={"support": level})
        outcome.checks[item.key] = _check(item, level, how, reason)

    cleaned = _apply(report, outcome, {key for key, _, _ in cited})
    results = list(outcome.checks.values())
    precise = sum(
        sum(1 for c in check.citations if c.status == "verified")
        for check in results
        if check.status in ("verbatim", "supported")
    )
    summary = CitationReport(
        sentences_checked=len(cited),
        citations_total=citations_total,
        citations_verified=citations_verified,
        precision=round(precise / citations_total, 4) if citations_total else 1.0,
        verbatim=sum(c.status == "verbatim" for c in results),
        supported=sum(c.status == "supported" for c in results),
        partial=sum(c.status == "partial" for c in results),
        unsupported=sum(c.status == "unsupported" for c in results),
        misattributed=sum(c.status == "misattributed" for c in results),
        entailment=method,
        removed=[c for c in results if c.status in ("unsupported", "misattributed")],
        flagged=[c for c in results if c.status == "partial"],
    )
    return cleaned, summary


def _sentences(report: Report) -> list[tuple[str, str | None, ReportSentence]]:
    found: list[tuple[str, str | None, ReportSentence]] = [
        (f"S{i}", None, s) for i, s in enumerate(report.summary)
    ]
    for j, section in enumerate(report.sections):
        found += [
            (f"S{j}.{i}", section.sub_question_id, s) for i, s in enumerate(section.sentences)
        ]
    return found


def _apply(report: Report, outcome: _Outcome, checked: set[str]) -> Report:
    """Keep verified sentences (with their verified citations), drop the rest."""

    def resolve(key: str, sentence: ReportSentence) -> ReportSentence | None:
        if key not in checked:
            return sentence
        return outcome.keep.get(key)

    summary = [s for i, x in enumerate(report.summary) if (s := resolve(f"S{i}", x)) is not None]
    sections = [
        section.model_copy(
            update={
                "sentences": [
                    s
                    for i, x in enumerate(section.sentences)
                    if (s := resolve(f"S{j}.{i}", x)) is not None
                ]
            }
        )
        for j, section in enumerate(report.sections)
    ]
    return fill_empty_sections(report.model_copy(update={"summary": summary, "sections": sections}))
