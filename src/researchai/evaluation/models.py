"""Labelled evaluation cases and the registry of traps planted in the corpus."""

from __future__ import annotations

from enum import StrEnum
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from researchai.corpus.models import Language, Topic
from researchai.safety.models import GuardCategory

DOC_ID = r"^(ev|tax)-(en|ka)-\d{4}$"
Split = Literal["dev", "holdout"]
ToolName = Literal["search_corpus", "fetch_document", "calculate", "get_source_metadata"]
TrapKind = Literal["contradiction", "outdated", "low_reliability", "injection"]


class _Model(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class Category(StrEnum):
    FACTUAL = "factual"
    COMPARISON = "comparison"
    RECOMMENDATION = "recommendation"
    NUMERIC = "numeric"
    CONFLICT = "conflict"
    OUTDATED = "outdated"
    UNANSWERABLE = "unanswerable"
    INJECTION = "injection"
    HARMFUL = "harmful"
    GEORGIAN = "georgian"
    MULTI_HOP = "multi_hop"


class Behavior(StrEnum):
    ANSWER = "answer"
    FLAG_CONFLICT = "flag_conflict"  # answer, and state the disagreement with both sides
    ABSTAIN = "abstain"  # the corpus does not contain the answer
    REFUSE = "refuse"  # the input guard must refuse; no tool call


class Trap(_Model):
    """A deliberately planted case inside the corpus (ground truth, invisible to the agent)."""

    id: str = Field(pattern=r"^TR\d{2}$")
    kind: TrapKind
    topic: Topic
    language: Language
    since: Literal["1.0.0", "1.1.0"] = Field(description="Corpus version that planted it")
    subject: str = Field(min_length=5)
    planted: list[str] = Field(min_length=1, description="Trap documents (wrong, old or injected)")
    reference: list[str] = Field(
        default_factory=list, description="Documents with the reliable or current value"
    )
    planted_values: list[str] = Field(default_factory=list)
    reference_values: list[str] = Field(default_factory=list)
    payload: str | None = Field(default=None, description="Injected instruction, verbatim")
    style: Literal["explicit", "paraphrased"] | None = None
    expected: str = Field(min_length=10, description="What a correct agent does")

    @model_validator(mode="after")
    def _consistent(self) -> Self:
        if self.kind == "injection":
            if self.payload is None or self.style is None or self.reference:
                raise ValueError(f"{self.id}: an injection has a payload and a style, no reference")
        elif not (self.reference and self.planted_values and self.reference_values):
            raise ValueError(f"{self.id}: a {self.kind} trap needs reference docs and values")
        elif self.payload is not None or self.style is not None:
            raise ValueError(f"{self.id}: only injections have a payload or style")
        return self


class Fact(_Model):
    """A reference fact; every value must appear in at least one of its sources."""

    statement: str = Field(min_length=5)
    values: list[str] = Field(min_length=1)
    sources: list[str] = Field(min_length=1)


class ExpectedCalculation(_Model):
    expression: str = Field(min_length=3)
    result: float
    sources: list[str] = Field(min_length=1, description="Documents stating the operands")


class Case(_Model):
    id: str = Field(pattern=r"^[a-z_]+-\d{3}$")
    category: Category
    language: Language
    topic: Topic | None = None
    split: Split
    question: str = Field(min_length=10, max_length=600)
    behavior: Behavior
    key_points: list[str] = Field(default_factory=list)
    facts: list[Fact] = Field(default_factory=list)
    outdated_values: list[str] = Field(
        default_factory=list, description="Values that must not be presented as current"
    )
    forbidden: list[str] = Field(
        default_factory=list, description="Text that must never appear in the report"
    )
    expected_tools: list[ToolName] = Field(default_factory=list)
    calculation: ExpectedCalculation | None = None
    traps: list[str] = Field(default_factory=list)
    missing: list[str] = Field(
        default_factory=list, description="What the corpus lacks (abstention cases)"
    )
    refusal_category: GuardCategory | None = None
    notes: str = ""

    @property
    def group(self) -> str:
        """Cases about the same trap share a split."""
        return self.traps[0] if self.traps else self.id

    @property
    def expected_sources(self) -> list[str]:
        return list(dict.fromkeys(doc for fact in self.facts for doc in fact.sources))
