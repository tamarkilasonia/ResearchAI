"""Conflict and outdated-value detection.

Rules (always)
    Numeric facts from different documents are *comparable* when they have the
    same unit (unit-less counts only when >= 10), the same threshold labels, the
    same discriminators (city, car model, network, qualifiers such as DC/AC,
    new/used, petrol/diesel) and sufficiently overlapping subjects. Only
    *directly* comparable facts are related (no transitive chaining). The newest
    fact is the reference; older differing values are ``outdated`` (superseded)
    and contemporaneous differing values are a ``contradiction``. Each reference
    yields at most one record of each kind, so a time series is one record, not
    pairwise noise. Forecasts (values dated after their source was published,
    such as "450 chargers by 2030") are never treated as current values.

LLM (when available)
    Disagreements without numbers ("VAT applies" vs "exempt from VAT") are
    invisible to the rules. The model may flag passage pairs with verbatim
    quotes; quotes must anchor in the passages and the documents must differ.
    The kind is still decided by rules, from publication dates.
"""

from __future__ import annotations

from collections.abc import Sequence

from pydantic import BaseModel, ConfigDict, Field

from researchai.agent.anchoring import anchor_quote
from researchai.agent.budget import BudgetExceeded, RunBudget
from researchai.agent.llm import LLMClient, LLMError, structured_call
from researchai.agent.models import Claim, Conflict, ConflictKind, Evidence, NumericFact
from researchai.agent.prompts import CONFLICT_SYSTEM, conflict_user
from researchai.textutil import DOMAIN_COMMON

MIN_SUBJECT_OVERLAP = 0.5
MIN_SHARED_TERMS = 2
MIN_UNITLESS_VALUE = 10
SAME_VALUE_TOLERANCE = 0.02
MAX_CONFLICTS = 12

# Terms that make two otherwise similar statements about different things.
DISCRIMINATORS = frozenset(
    {
        # places
        "tbilisi", "batumi", "kutaisi", "rustavi", "gori", "poti", "gudauri", "stepantsminda",
        "telavi", "zugdidi", "borjomi", "kobuleti", "khashuri", "zestaponi", "samtredia",
        "senaki", "mtskheta", "japan", "china", "europe", "turkey", "armenia",
        "თბილის", "ბათუმ", "ქუთაის", "რუსთავ", "გუდაურ", "ზესტაფონ", "ხაშურ",
        # vehicles and brands
        "leaf", "tesla", "bolt", "ioniq", "niro", "atto", "dolphin", "seal", "byd", "prius",
        "camry", "golf", "chevrolet", "nissan", "hyundai", "kia", "volkswagen", "toyota",
        # charging networks
        "voltline", "chargege", "eroute", "mtsvane",
        # qualifiers
        "dc", "ac", "fast", "home", "public", "new", "used", "hybrid", "hybrids", "diesel",
        "petrol", "premium", "winter", "summer", "heat", "pump", "micro", "vat", "excise",
        "regions", "kartli", "grid",
    }
)  # fmt: skip


def _discriminators(terms: Sequence[str]) -> frozenset[str]:
    return frozenset(t for t in terms if t in DISCRIMINATORS)


def comparable(a: NumericFact, b: NumericFact) -> bool:
    if a.unit != b.unit:
        return False
    if a.unit is None and min(a.value, b.value) < MIN_UNITLESS_VALUE:
        return False
    if a.labels != b.labels:
        return False
    if _discriminators(a.subject) != _discriminators(b.subject):
        return False
    left, right = set(a.subject) - DOMAIN_COMMON, set(b.subject) - DOMAIN_COMMON
    if not left or not right:
        return False
    shared = len(left & right)
    if shared < MIN_SHARED_TERMS:
        return False
    return shared / min(len(left), len(right)) >= MIN_SUBJECT_OVERLAP


def consistent(a: NumericFact, b: NumericFact) -> bool:
    """Same value (2% tolerance), or overlapping ranges, or a point inside a range."""
    a_low, a_high = a.value, a.high if a.high is not None else a.value
    b_low, b_high = b.value, b.high if b.high is not None else b.value
    if a.high is None and b.high is None:
        scale = max(abs(a.value), abs(b.value)) or 1.0
        return abs(a.value - b.value) / scale <= SAME_VALUE_TOLERANCE
    return a_low <= b_high and b_low <= a_high


def _show(fact: NumericFact) -> str:
    value = f"{fact.value:g}" if fact.high is None else f"{fact.value:g}-{fact.high:g}"
    unit = f" {fact.unit}" if fact.unit else ""
    month = f"-{fact.as_of_month:02d}" if fact.as_of_month else ""
    return f"{value}{unit} ({fact.as_of_year}{month})"


def _same_period(a: NumericFact, b: NumericFact) -> bool:
    if a.as_of_year != b.as_of_year:
        return False
    return not a.as_of_month or not b.as_of_month or a.as_of_month == b.as_of_month


def _is_forecast(claim: Claim, fact: NumericFact) -> bool:
    """A value dated after its source was published ("450 chargers by 2030")."""
    return fact.as_of_year > claim.published.year


def rule_conflicts(claims: Sequence[Claim]) -> list[Conflict]:
    """Conflicts between *directly* comparable facts (no transitive chaining).

    Facts are visited newest first (then most reliable). Each unvisited fact
    becomes the reference for all directly comparable facts from other
    documents; those agreeing with it, older and disagreeing (outdated), or
    contemporaneous and disagreeing (contradiction) are then marked as handled.
    """
    facts = [
        (claim, fact) for claim in claims for fact in claim.facts if not _is_forecast(claim, fact)
    ]
    order = sorted(
        range(len(facts)),
        key=lambda i: (
            -facts[i][1].as_of_year,
            -(facts[i][1].as_of_month or 0),
            -facts[i][0].reliability,
            facts[i][0].id,
        ),
    )
    handled: set[int] = set()
    conflicts: list[Conflict] = []
    for i in order:
        if i in handled:
            continue
        ref_claim, ref_fact = facts[i]
        partners = [
            j
            for j in order
            if j != i
            and j not in handled
            and facts[j][0].doc_id != ref_claim.doc_id
            and comparable(ref_fact, facts[j][1])
        ]
        if not partners:
            continue
        handled.add(i)
        handled.update(partners)
        disagree = [j for j in partners if not consistent(facts[j][1], ref_fact)]
        contemporaneous = [j for j in disagree if _same_period(facts[j][1], ref_fact)]
        older = [j for j in disagree if j not in contemporaneous]
        if contemporaneous:
            conflicts.append(_record("contradiction", [i], contemporaneous, facts, ref_fact.unit))
        if older:
            conflicts.append(_record("outdated", [i], older, facts, ref_fact.unit))
    conflicts.sort(key=lambda c: (c.kind != "contradiction", c.summary))
    return [
        c.model_copy(update={"id": f"X{n}"}) for n, c in enumerate(conflicts[:MAX_CONFLICTS], 1)
    ]


def _record(
    kind: ConflictKind,
    current: list[int],
    others: list[int],
    facts: list[tuple[Claim, NumericFact]],
    unit: str | None,
) -> Conflict:
    current_ids = list(dict.fromkeys(facts[i][0].id for i in current))
    other_ids = list(dict.fromkeys(facts[i][0].id for i in others))
    now = "; ".join(f"{_show(facts[i][1])} in {facts[i][0].doc_id}" for i in current)
    then = "; ".join(f"{_show(facts[i][1])} in {facts[i][0].doc_id}" for i in others)
    verb = "superseded" if kind == "outdated" else "contradicted"
    return Conflict(
        id="X0",
        kind=kind,
        current=current_ids,
        others=other_ids,
        unit=unit,
        detected_by="rules",
        summary=f"{now} {verb} {then}",
    )


# ----------------------------------------------------------------- LLM pass


class _Disagreement(BaseModel):
    model_config = ConfigDict(extra="forbid")
    evidence_a: str
    quote_a: str = Field(min_length=3, max_length=500)
    evidence_b: str
    quote_b: str = Field(min_length=3, max_length=500)
    explanation: str = Field(max_length=300)


class Disagreements(BaseModel):
    model_config = ConfigDict(extra="forbid")
    disagreements: list[_Disagreement] = Field(max_length=6)


def _claim_for(item: Evidence, quote: str, claim_id: str, reliability: float) -> Claim | None:
    span = anchor_quote(item, quote)
    if span is None:
        return None
    start, end = span
    return Claim(
        id=claim_id,
        evidence_id=item.id,
        sub_question_id=item.sub_question_id,
        doc_id=item.doc_id,
        text=item.text[start - item.char_start : end - item.char_start],
        char_start=start,
        char_end=end,
        published=item.published,
        reliability=reliability,
    )


def _covered(conflicts: Sequence[Conflict], claims: Sequence[Claim], a: str, b: str) -> bool:
    """Whether a rule conflict already relates evidence ``a`` and ``b``."""
    evidence_of = {c.id: c.evidence_id for c in claims}
    for conflict in conflicts:
        involved = {evidence_of.get(i) for i in (*conflict.current, *conflict.others)}
        if {a, b} <= involved:
            return True
    return False


async def llm_conflicts(
    question: str,
    evidence: Sequence[Evidence],
    claims: Sequence[Claim],
    existing: Sequence[Conflict],
    reliability: dict[str, float],
    *,
    llm: LLMClient,
    budget: RunBudget,
    max_output_tokens: int,
) -> tuple[list[Claim], list[Conflict], list[str]]:
    """New claims and conflicts found by the model, plus notes for the trace."""
    try:
        result = await structured_call(
            llm,
            budget,
            system=CONFLICT_SYSTEM,
            user=conflict_user(question, evidence),
            schema=Disagreements,
            max_tokens=min(max_output_tokens, 800),
        )
    except (LLMError, BudgetExceeded) as exc:
        return [], [], [f"no LLM conflict check: {type(exc).__name__}: {exc}"]

    by_id = {item.id: item for item in evidence}
    new_claims: list[Claim] = []
    new_conflicts: list[Conflict] = []
    notes: list[str] = []
    next_claim = len(claims) + 1
    for item in result.value.disagreements:
        a, b = by_id.get(item.evidence_a), by_id.get(item.evidence_b)
        if a is None or b is None or a.doc_id == b.doc_id:
            notes.append(f"ignored {item.evidence_a}/{item.evidence_b}: unknown or same document")
            continue
        if _covered([*existing, *new_conflicts], [*claims, *new_claims], a.id, b.id):
            continue
        claim_a = _claim_for(a, item.quote_a, f"K{next_claim}", reliability.get(a.doc_id, 0.5))
        claim_b = _claim_for(b, item.quote_b, f"K{next_claim + 1}", reliability.get(b.doc_id, 0.5))
        if claim_a is None or claim_b is None:
            notes.append(f"ignored {a.id}/{b.id}: quote not found in passage")
            continue
        next_claim += 2
        new_claims += [claim_a, claim_b]
        newer, older = sorted((claim_a, claim_b), key=lambda c: (c.published, c.reliability))[::-1]
        kind: ConflictKind = "outdated" if newer.published != older.published else "contradiction"
        new_conflicts.append(
            Conflict(
                id=f"X{len(existing) + len(new_conflicts) + 1}",
                kind=kind,
                current=[newer.id],
                others=[older.id],
                detected_by="llm",
                summary=f"{newer.doc_id} ({newer.published}) vs {older.doc_id} "
                f"({older.published}): {item.explanation}",
            )
        )
    return new_claims, new_conflicts, notes
