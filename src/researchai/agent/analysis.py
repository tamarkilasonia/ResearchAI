"""Calculations over evidence numbers, executed by the MCP ``calculate`` tool.

The LLM may *propose* arithmetic, but a deterministic check decides whether it
runs: every operand must occur in the evidence it cites (allowing English
``1,800.5`` and Georgian ``1 800,5`` notations) or be a unit constant such as
100 or 12. Nothing is computed from numbers the model made up.
"""

from __future__ import annotations

import ast
import re
from collections.abc import Mapping, Sequence
from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field

from researchai.agent.budget import BudgetExceeded, RunBudget
from researchai.agent.llm import LLMClient, LLMError, structured_call
from researchai.agent.models import Calculation, Evidence
from researchai.agent.prompts import ANALYST_SYSTEM, analyst_user
from researchai.mcp_client.client import ToolCallError
from researchai.tools.schemas import CalculationResponse

MAX_CALCULATIONS = 4
UNIT_CONSTANTS = frozenset({1.0, 2.0, 10.0, 12.0, 100.0, 1000.0})

# Digits with optional group/decimal separators: 1,800 / 1 800 / 0.55 / 0,55 / 86 900.
_NUMBER = re.compile(r"\d+(?:[ ,.]\d+)*")


class CalculatorTool(Protocol):
    async def calculate(self, expression: str) -> CalculationResponse: ...


class _ProposedCalculation(BaseModel):
    model_config = ConfigDict(extra="forbid")
    description: str = Field(min_length=3, max_length=200)
    expression: str = Field(min_length=1, max_length=200)
    evidence_ids: list[str] = Field(min_length=1, max_length=4)


class ProposedCalculations(BaseModel):
    model_config = ConfigDict(extra="forbid")
    calculations: list[_ProposedCalculation] = Field(max_length=MAX_CALCULATIONS)


def _readings(token: str) -> set[float]:
    """Every value a number token can denote under EN or KA conventions.

    ``1,800`` -> {1800, 1.8}; ``0,55`` -> {55, 0.55}; ``1.800,5`` -> {1800.5, 1.8005};
    ``86 900`` -> {86900, 86, 900}; ``0.292`` -> {0.292}. A dot alone is always a
    decimal point, so ``16.8`` never reads as 168.
    """
    compact = token.replace(" ", "")
    if "," in compact and "." in compact:
        candidates = [compact.replace(",", ""), compact.replace(".", "").replace(",", ".")]
    elif "," in compact:
        candidates = [compact.replace(",", ""), compact.replace(",", ".")]
    else:
        candidates = [compact]
    values: set[float] = set()
    for candidate in candidates:
        try:
            values.add(round(float(candidate), 6))
        except ValueError:
            continue
    if " " in token:
        for part in token.split(" "):
            values |= _readings(part)
    return values


def evidence_numbers(text: str) -> set[float]:
    values: set[float] = set()
    for match in _NUMBER.finditer(text):
        values |= _readings(match.group())
    return values


def operands(expression: str) -> list[float]:
    """Numeric literals of an arithmetic expression (raises LLMError if not arithmetic)."""
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        raise LLMError("schema", f"not arithmetic: {expression!r}") from exc
    numbers: list[float] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant):
            if isinstance(node.value, bool) or not isinstance(node.value, int | float):
                raise LLMError("schema", f"non-numeric constant in {expression!r}")
            numbers.append(round(float(node.value), 6))
        elif isinstance(node, ast.Name | ast.Call | ast.Attribute):
            raise LLMError("schema", f"names and calls are not allowed: {expression!r}")
    return numbers


def check_grounded(proposal: _ProposedCalculation, evidence: Mapping[str, Evidence]) -> str | None:
    """None if every operand is grounded in the cited evidence, else the reason."""
    unknown = [e for e in proposal.evidence_ids if e not in evidence]
    if unknown:
        return f"unknown evidence ids {unknown}"
    available = set().union(*(evidence_numbers(evidence[e].text) for e in proposal.evidence_ids))
    try:
        values = operands(proposal.expression)
    except LLMError as exc:
        return str(exc)
    missing = [v for v in values if v not in available and v not in UNIT_CONSTANTS]
    if missing:
        return f"operands {missing} not found in {proposal.evidence_ids}"
    return None


async def run_calculations(
    question: str,
    evidence: Sequence[Evidence],
    *,
    llm: LLMClient,
    budget: RunBudget,
    tools: CalculatorTool,
    max_output_tokens: int,
) -> tuple[list[Calculation], list[str]]:
    """Propose, check and execute calculations. Returns results and notes for the trace."""
    notes: list[str] = []
    try:
        proposed = await structured_call(
            llm,
            budget,
            system=ANALYST_SYSTEM,
            user=analyst_user(question, evidence),
            schema=ProposedCalculations,
            max_tokens=min(max_output_tokens, 800),
        )
    except (LLMError, BudgetExceeded) as exc:
        return [], [f"no calculations: {type(exc).__name__}: {exc}"]

    by_id = {item.id: item for item in evidence}
    results: list[Calculation] = []
    for proposal in proposed.value.calculations:
        if reason := check_grounded(proposal, by_id):
            notes.append(f"rejected {proposal.expression!r}: {reason}")
            continue
        try:
            response = await tools.calculate(proposal.expression)
        except ToolCallError as exc:
            notes.append(f"calculate failed for {proposal.expression!r}: {exc.code.value}")
            continue
        results.append(
            Calculation(
                id=f"C{len(results) + 1}",
                description=proposal.description,
                expression=proposal.expression,
                result=response.result,
                evidence_ids=proposal.evidence_ids,
            )
        )
    return results, notes
