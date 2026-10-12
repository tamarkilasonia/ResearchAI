"""Agreement between the LLM judge and human labels, per criterion.

Cohen's kappa corrects raw agreement for the agreement expected by chance from
each rater's label distribution: ``(p_o - p_e) / (1 - p_e)``. With ordinal labels
(0 < 1 < 2) the quadratically weighted kappa also credits near misses. Kappa is
undefined when chance agreement is total (both raters always give one label);
it is then reported as ``None``. Interpretation follows Landis and Koch (1977);
below 0.4 the judge is reported as unreliable for that criterion.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence

from pydantic import BaseModel, ConfigDict

LABELS = (0, 1, 2)
RELIABLE_KAPPA = 0.4


class _Model(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


def cohen_kappa(
    a: Sequence[int], b: Sequence[int], *, labels: Sequence[int] = LABELS, weighted: bool = False
) -> float | None:
    """Cohen's kappa of two raters over the same items; quadratic weights if ``weighted``."""
    if len(a) != len(b):
        raise ValueError("both raters must label the same items")
    n = len(a)
    if n == 0:
        return None
    k = len(labels)
    index = {label: i for i, label in enumerate(labels)}

    def weight(i: int, j: int) -> float:
        if not weighted:
            return 0.0 if i == j else 1.0
        return (i - j) ** 2 / (k - 1) ** 2

    observed = [[0.0] * k for _ in range(k)]
    for x, y in zip(a, b, strict=True):
        observed[index[x]][index[y]] += 1 / n
    rows = [sum(observed[i]) for i in range(k)]
    cols = [sum(observed[i][j] for i in range(k)) for j in range(k)]
    disagreement = sum(weight(i, j) * observed[i][j] for i in range(k) for j in range(k))
    expected = sum(weight(i, j) * rows[i] * cols[j] for i in range(k) for j in range(k))
    if expected == 0:
        return None
    return round(1 - disagreement / expected, 4)


def interpret(kappa: float | None) -> str:
    if kappa is None:
        return "undefined"
    for bound, label in ((0.0, "poor"), (0.2, "slight"), (0.4, "fair"), (0.6, "moderate"),
                         (0.8, "substantial")):  # fmt: skip
        if kappa <= bound:
            return label
    return "almost perfect"


class CriterionAgreement(_Model):
    criterion: str
    n: int
    raw_agreement: float | None
    kappa: float | None
    weighted_kappa: float | None
    interpretation: str
    reliable: bool
    human_mean: float | None
    judge_mean: float | None
    confusion: list[list[int]]  # rows: human label 0..2, columns: judge label 0..2


def agreement(criterion: str, pairs: Sequence[tuple[int, int]]) -> CriterionAgreement:
    """``pairs`` are (human, judge) labels of the items both labelled."""
    human = [h for h, _ in pairs]
    judge = [j for _, j in pairs]
    counts = Counter(pairs)
    kappa = cohen_kappa(human, judge)
    n = len(pairs)
    return CriterionAgreement(
        criterion=criterion,
        n=n,
        raw_agreement=round(sum(h == j for h, j in pairs) / n, 4) if n else None,
        kappa=kappa,
        weighted_kappa=cohen_kappa(human, judge, weighted=True),
        interpretation=interpret(kappa),
        reliable=kappa is not None and kappa >= RELIABLE_KAPPA,
        human_mean=round(sum(human) / n, 3) if n else None,
        judge_mean=round(sum(judge) / n, 3) if n else None,
        confusion=[[counts[(h, j)] for j in LABELS] for h in LABELS],
    )
