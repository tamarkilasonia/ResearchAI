"""Typed data flowing through the research graph.

Every factual statement in a :class:`Report` is a ``claim`` sentence with at
least one :class:`Citation` pointing at a document and (once anchored) an exact
character span. Sentences that carry no new facts (notes, statements of
missing evidence) are typed as such, so a scorer never has to guess.
"""

from __future__ import annotations

from datetime import date
from enum import StrEnum
from typing import Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from researchai.corpus.models import Language, SourceType, Topic
from researchai.safety.models import SafetyReport


class _Model(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class QuestionKind(StrEnum):
    FACTUAL = "factual"
    NUMERIC = "numeric"
    COMPARISON = "comparison"
    RECOMMENDATION = "recommendation"


class RunStatus(StrEnum):
    ANSWERED = "answered"
    ABSTAINED = "abstained"  # no usable evidence at all
    REJECTED = "rejected"  # input failed validation (empty, too long)
    REFUSED = "refused"  # the input guard refused it (prompt injection, harmful request)
    FAILED = "failed"  # infrastructure error; never a fabricated answer


class QuestionProfile(_Model):
    language: Language
    topic: Topic | None
    kind: QuestionKind
    aspects: list[str]


class SubQuestion(_Model):
    id: str = Field(pattern=r"^sq\d+$")
    text: str = Field(min_length=3, max_length=300)
    queries: list[str] = Field(min_length=1, max_length=3)
    aspect: str | None = None


class ResearchPlan(_Model):
    sub_questions: list[SubQuestion] = Field(min_length=1)
    planned_by: Literal["rules", "llm"]

    @model_validator(mode="after")
    def _unique_ids(self) -> Self:
        ids = [sq.id for sq in self.sub_questions]
        if len(ids) != len(set(ids)):
            raise ValueError("sub-question ids must be unique")
        return self


class Evidence(_Model):
    """One retrieved passage. ``id`` (E1, E2, ...) is how prompts and citations refer to it."""

    id: str = Field(pattern=r"^E\d+$")
    sub_question_id: str
    query: str
    chunk_id: str
    doc_id: str
    title: str
    section: str
    text: str
    char_start: int
    char_end: int
    published: date
    language: Language
    source_type: SourceType
    publisher: str
    score: float


SourceFlag = Literal["low_reliability", "stale", "user_generated"]
ConflictKind = Literal["outdated", "contradiction"]


class SourceAssessment(_Model):
    """How much to trust a source, decided by rules (source type and age)."""

    doc_id: str
    source_type: SourceType
    published: date
    age_days: int
    reliability: float = Field(ge=0, le=1)
    freshness: float = Field(ge=0, le=1)
    flags: list[SourceFlag] = Field(default_factory=list)


class NumericFact(_Model):
    value: float
    high: float | None = Field(default=None, description="Upper bound when the text gives a range")
    unit: str | None
    as_of_year: int
    as_of_month: int | None
    char_start: int
    char_end: int
    subject: list[str] = Field(default_factory=list, description="Content terms of its clause")
    labels: list[str] = Field(
        default_factory=list,
        description="Threshold/range numbers qualifying the fact ('101' in 'up to 101 kWh')",
    )

    @property
    def period(self) -> tuple[int, int]:
        return (self.as_of_year, self.as_of_month or 0)


class Claim(_Model):
    """One sentence of evidence stating something checkable (numbers, or an LLM-flagged fact)."""

    id: str = Field(pattern=r"^K\d+$")
    evidence_id: str
    sub_question_id: str
    doc_id: str
    text: str
    char_start: int
    char_end: int
    facts: list[NumericFact] = Field(default_factory=list)
    subject: list[str] = Field(default_factory=list)
    published: date
    reliability: float = Field(ge=0, le=1)


class Conflict(_Model):
    """Sources that disagree about the same quantity or fact.

    ``outdated``: an older value superseded by a newer source (``current`` holds
    the newest claims). ``contradiction``: sources from the same period disagree.
    """

    id: str = Field(pattern=r"^X\d+$")
    kind: ConflictKind
    current: list[str] = Field(min_length=1, description="Claim ids of the newest value(s)")
    others: list[str] = Field(min_length=1, description="Claim ids that disagree with them")
    unit: str | None = None
    detected_by: Literal["rules", "llm"]
    summary: str


class SufficiencyVerdict(_Model):
    sub_question_id: str
    sufficient: bool
    relevant_evidence: list[str] = Field(default_factory=list)
    reason: str


class Calculation(_Model):
    """Arithmetic run through the MCP ``calculate`` tool on numbers from evidence."""

    id: str = Field(pattern=r"^C\d+$")
    description: str
    expression: str
    result: float
    evidence_ids: list[str] = Field(min_length=1)


class Citation(_Model):
    evidence_id: str
    doc_id: str
    chunk_id: str
    quote: str = Field(min_length=1)
    # Absolute offsets in the document body; None when the quote could not be
    # located verbatim in the cited passage (the citation verifier flags these).
    char_start: int | None = None
    char_end: int | None = None

    @property
    def anchored(self) -> bool:
        return self.char_start is not None and self.char_end is not None


SentenceKind = Literal["claim", "recommendation", "insufficient_evidence", "conflict", "note"]
# How a kept sentence is supported by its citations (set by the citation verifier).
SupportLevel = Literal["verbatim", "supported", "partial"]


class ReportSentence(_Model):
    text: str = Field(min_length=1)
    kind: SentenceKind
    citations: list[Citation] = Field(default_factory=list)
    calculation_ids: list[str] = Field(default_factory=list)
    support: SupportLevel | None = None

    @model_validator(mode="after")
    def _claims_cite(self) -> Self:
        if self.kind in ("claim", "conflict") and not self.citations:
            raise ValueError(f"a {self.kind} sentence needs at least one citation")
        return self


class ReportSection(_Model):
    heading: str
    sub_question_id: str | None
    sentences: list[ReportSentence]


class Report(_Model):
    language: Language
    title: str
    summary: list[ReportSentence]
    sections: list[ReportSection]
    insufficient: list[str] = Field(
        default_factory=list, description="Sub-question ids without usable evidence"
    )
    generated_by: Literal["rules", "llm"]

    def sentences(self) -> list[ReportSentence]:
        return [*self.summary, *(s for section in self.sections for s in section.sentences)]

    def citations(self) -> list[Citation]:
        return [c for sentence in self.sentences() for c in sentence.citations]


CitationStatus = Literal[
    "verified", "unanchored", "span_mismatch", "missing_document", "instruction_text"
]
SentenceStatus = Literal["verbatim", "supported", "partial", "unsupported", "misattributed"]


class CitationCheck(_Model):
    evidence_id: str
    doc_id: str
    status: CitationStatus


class SentenceCheck(_Model):
    text: str
    kind: SentenceKind
    sub_question_id: str | None
    status: SentenceStatus
    method: Literal["deterministic", "llm", "lexical"]
    reason: str
    citations: list[CitationCheck] = Field(default_factory=list)


class CitationReport(_Model):
    """Per-report citation verification (deterministic first, then entailment)."""

    sentences_checked: int
    citations_total: int
    citations_verified: int = Field(description="Cited span re-fetched and equal to the quote")
    precision: float = Field(
        ge=0,
        le=1,
        description="Share of citations whose span is verified AND supports the sentence",
    )
    verbatim: int
    supported: int
    partial: int
    unsupported: int
    misattributed: int
    entailment: Literal["llm", "lexical", "none"]
    removed: list[SentenceCheck] = Field(default_factory=list)
    flagged: list[SentenceCheck] = Field(default_factory=list)


class TraceEvent(_Model):
    node: str
    ms: float
    detail: dict[str, Any] = Field(default_factory=dict)


class Usage(_Model):
    searches: int = 0
    tool_calls: int = 0
    llm_calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    iterations: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


class ToolCall(_Model):
    tool: str
    arguments: dict[str, Any]
    ok: bool
    error_code: str | None
    latency_ms: float


class ResearchResult(_Model):
    run_id: str
    question: str
    status: RunStatus
    mode: Literal["offline", "llm"]
    profile: QuestionProfile | None = None
    plan: ResearchPlan | None = None
    evidence: list[Evidence] = Field(default_factory=list)
    assessments: list[SourceAssessment] = Field(default_factory=list)
    claims: list[Claim] = Field(default_factory=list)
    conflicts: list[Conflict] = Field(default_factory=list)
    sufficiency: list[SufficiencyVerdict] = Field(default_factory=list)
    missing_entities: list[str] = Field(default_factory=list)
    calculations: list[Calculation] = Field(default_factory=list)
    report: Report | None = None
    citation_report: CitationReport | None = None
    reason: str | None = Field(
        default=None, description="Why a run was rejected, refused or failed"
    )
    safety: SafetyReport = Field(default_factory=SafetyReport)
    trace: list[TraceEvent] = Field(default_factory=list)
    tool_calls: list[ToolCall] = Field(default_factory=list)
    usage: Usage = Field(default_factory=Usage)
    budget_exhausted: list[str] = Field(default_factory=list)
    total_ms: float = 0.0
