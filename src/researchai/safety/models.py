"""Safety results carried in every research result."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

GuardAction = Literal["allow", "refuse"]
GuardCategory = Literal[
    "none", "prompt_injection", "weapons", "violence", "self_harm", "cybercrime", "fraud"
]


class _Model(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class GuardVerdict(_Model):
    action: GuardAction
    category: GuardCategory
    method: Literal["rules", "llm"]
    matched: list[str] = Field(default_factory=list, description="Rule names that fired")
    reason: str = ""


class InjectionFinding(_Model):
    """Instruction-like text inside a retrieved passage (treated as data, never obeyed)."""

    evidence_id: str
    doc_id: str
    char_start: int
    char_end: int
    text: str
    rule: str


class OutputIssue(_Model):
    sentence: str
    reason: Literal["system_prompt_leak", "injected_content", "unknown_link"]


class SafetyReport(_Model):
    input: GuardVerdict | None = None
    injections: list[InjectionFinding] = Field(default_factory=list)
    output_issues: list[OutputIssue] = Field(default_factory=list)
