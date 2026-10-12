"""Human labels for calibrating the LLM judge.

``sample_items`` draws a seeded sample of judgeable reports from a run, spread
across categories, and writes ``items.jsonl``. ``label_session`` shows one item
at a time and saves every answer immediately (``labels-<labeler>.jsonl``), so a
session can stop and resume at any point. Neither the judge's labels nor any
automatic score are ever shown to the labeller.
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from researchai.agent.models import ResearchResult
from researchai.evaluation.agreement import CriterionAgreement, agreement
from researchai.evaluation.judge import CRITERIA, Judgement, judgeable, report_for_review
from researchai.evaluation.models import Behavior, Case

ITEMS_FILE = "items.jsonl"
CURRENT_FILE = "CURRENT_ITEM.md"

RUBRIC = {
    "completeness": (
        "Completeness - are the key points answered?",
        ["0  none or almost none",
         "1  some key points missing, or only buried (e.g. in a quoted 'sources disagree')",
         "2  every key point stated clearly as part of the answer"],
    ),
    "relevance": (
        "Relevance - does it stay on the question?",
        ["0  mostly about something else",
         "1  answers it, but with substantial off-topic sections",
         "2  answers what was asked; little or no off-topic content"],
    ),
    "quality": (
        "Quality - could a reader rely on and act on it?",
        ["0  confusing, contradicts itself, or misleading",
         "1  useful, but with real problems (repetition, structure, one misleading statement)",
         "2  clear, consistent, direct; conflicts and 'insufficient evidence' used correctly"],
    ),
}  # fmt: skip


class _Model(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class LabelItem(_Model):
    item: int
    case_id: str
    category: str
    language: str
    question: str
    key_points: list[str]
    report: str


class HumanLabel(_Model):
    item: int
    case_id: str
    labeler: str
    labels: dict[str, int | None] = Field(description="criterion -> 0..2, None when skipped")
    unsure: list[str] = Field(default_factory=list)
    note: str = ""
    labelled_at: datetime


# ----------------------------------------------------------------- sample


def quotas(categories: Sequence[str], size: int) -> dict[str, int]:
    """Items per category: round-robin over the sorted categories until ``size``."""
    ordered = sorted(set(categories))
    counts = dict.fromkeys(ordered, 0)
    for n in range(size):
        if not ordered:
            break
        counts[ordered[n % len(ordered)]] += 1
    return counts


def sample_items(
    cases: Sequence[Case],
    results: dict[str, ResearchResult],
    *,
    size: int,
    seed: int,
    existing: Sequence[LabelItem] = (),
) -> tuple[list[LabelItem], list[str]]:
    """Seeded sample of judgeable reports, spread over the answerable categories.

    Each category gets a fixed quota and walks its cases in a seeded order,
    taking judgeable reports and skipping the rest. A case that has no result
    yet stops its category, so the sample can be drawn while a run is still
    going: labelling can start on finished categories and later additions never
    change what was already chosen. Returns all items (``existing`` first,
    newcomers appended) and the categories still waiting for results.
    """
    answerable = [c for c in cases if c.behavior in (Behavior.ANSWER, Behavior.FLAG_CONFLICT)]
    by_category: dict[str, list[Case]] = defaultdict(list)
    for case in answerable:
        by_category[case.category.value].append(case)
    taken = {item.case_id for item in existing}
    items = list(existing)
    waiting: list[str] = []
    for category, quota in quotas(list(by_category), size).items():
        members = sorted(
            by_category[category],
            key=lambda c: hashlib.sha256(f"{seed}:{c.id}".encode()).hexdigest(),
        )
        picked = 0
        for case in members:
            if picked == quota:
                break
            if case.id not in results:
                waiting.append(category)
                break
            if not judgeable(case, results[case.id]):
                continue
            picked += 1
            if case.id not in taken:
                items.append(
                    LabelItem(
                        item=len(items) + 1,
                        case_id=case.id,
                        category=category,
                        language=case.language.value,
                        question=case.question,
                        key_points=case.key_points,
                        report=report_for_review(results[case.id]),
                    )
                )
    return items, waiting


def write_items(directory: Path, items: Sequence[LabelItem], meta: dict[str, object]) -> None:
    """Write (or extend) a sample; items already written are never changed."""
    directory.mkdir(parents=True, exist_ok=True)
    existing = read_items(directory) if (directory / ITEMS_FILE).exists() else []
    if items[: len(existing)] != existing:
        raise ValueError(f"{directory} holds a different sample; use a new folder")
    (directory / "meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    (directory / ITEMS_FILE).write_text(
        "".join(item.model_dump_json() + "\n" for item in items), encoding="utf-8"
    )


def read_items(directory: Path) -> list[LabelItem]:
    lines = (directory / ITEMS_FILE).read_text(encoding="utf-8").splitlines()
    return [LabelItem.model_validate_json(line) for line in lines if line.strip()]


def labels_file(directory: Path, labeler: str) -> Path:
    return directory / f"labels-{labeler}.jsonl"


def read_labels(directory: Path, labeler: str) -> dict[int, HumanLabel]:
    file = labels_file(directory, labeler)
    if not file.exists():
        return {}
    labels = [HumanLabel.model_validate_json(line) for line in file.read_text("utf-8").splitlines()]
    return {label.item: label for label in labels}


# ----------------------------------------------------------------- session


def render_item(item: LabelItem, total: int) -> str:
    points = "\n".join(f"- {point}" for point in item.key_points)
    return (
        f"# Item {item.item} of {total} ({item.category}, {item.language})\n\n"
        f"**Question:** {item.question}\n\n**Key points:**\n{points}\n\n---\n\n{item.report}"
    )


def _parse(answer: str) -> tuple[int | None, bool] | str:
    """'0'..'2' (optional 'u' = unsure), 's' = skip, 'q' = quit; anything else is invalid."""
    answer = answer.strip().lower()
    if answer in ("s", "q"):
        return answer
    unsure = answer.endswith("u")
    digits = answer.removesuffix("u").strip()
    if digits in ("0", "1", "2"):
        return int(digits), unsure
    return "invalid"


def label_session(
    directory: Path,
    labeler: str,
    *,
    ask: Callable[[str], str] | None = None,
    show: Callable[[str], None] | None = None,
) -> int:
    """Label the remaining items; returns how many were labelled in this session."""
    ask = ask or input
    show = show or print
    items = read_items(directory)
    done = read_labels(directory, labeler)
    todo = [item for item in items if item.item not in done]
    show(f"{len(done)} of {len(items)} items labelled; {len(todo)} to go. "
         "Answer 0, 1 or 2 (add u if unsure, e.g. 1u), s to skip, q to quit.")  # fmt: skip
    labelled = 0
    for item in todo:
        text = render_item(item, len(items))
        (directory / CURRENT_FILE).write_text(text, encoding="utf-8")
        show("\n" + "=" * 78 + "\n" + text + "\n" + "=" * 78)
        labels: dict[str, int | None] = {}
        unsure: list[str] = []
        for criterion in CRITERIA:
            title, levels = RUBRIC[criterion]
            while True:
                answer = _parse(ask(f"\n{title}\n  " + "\n  ".join(levels) + "\n> "))
                if answer == "q":
                    show(f"Saved. {len(done) + labelled} of {len(items)} items labelled.")
                    return labelled
                if answer == "invalid":
                    show("Please answer 0, 1 or 2 (optionally with u), s or q.")
                    continue
                if answer == "s":
                    labels[criterion] = None
                else:
                    assert not isinstance(answer, str)  # noqa: S101 - narrowed above
                    labels[criterion] = answer[0]
                    if answer[1]:
                        unsure.append(criterion)
                break
        note = ask("Note (optional, Enter to skip)\n> ").strip()
        label = HumanLabel(
            item=item.item,
            case_id=item.case_id,
            labeler=labeler,
            labels=labels,
            unsure=unsure,
            note=note,
            labelled_at=datetime.now(UTC),
        )
        with labels_file(directory, labeler).open("a", encoding="utf-8") as handle:
            handle.write(label.model_dump_json() + "\n")
        labelled += 1
    (directory / CURRENT_FILE).unlink(missing_ok=True)
    meta_file = directory / "meta.json"
    complete = (
        json.loads(meta_file.read_text("utf-8")).get("complete", True)
        if meta_file.exists()
        else True
    )
    if complete:
        show(f"All {len(items)} items labelled. Thank you!")
    else:
        show(f"All {len(items)} items available so far are labelled. More are added when the "
             "run finishes; run the same command again then.")  # fmt: skip
    return labelled


# ----------------------------------------------------------------- agreement


def compare(
    items: Sequence[LabelItem],
    human: dict[int, HumanLabel],
    judge: dict[str, Judgement],
) -> list[CriterionAgreement]:
    """Per-criterion agreement over items both the human and the judge labelled."""
    results = []
    for criterion in CRITERIA:
        pairs = [
            (h, j)
            for item in items
            if item.item in human
            and item.case_id in judge
            and (h := human[item.item].labels.get(criterion)) is not None
            and (j := judge[item.case_id].labels.get(criterion)) is not None
        ]
        results.append(agreement(criterion, pairs))
    return results


def render_agreement(rows: Sequence[CriterionAgreement], *, labeler: str, model: str) -> str:
    lines = [
        f"# Judge calibration: {model} vs {labeler}",
        "",
        "| Criterion | Items | Raw agreement | Cohen's kappa | Weighted kappa | Reading | "
        "Human mean | Judge mean |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:

        def show(value: float | None) -> str:
            return "n/a" if value is None else f"{value:.2f}"

        verdict = r.interpretation + ("" if r.reliable else " (unreliable)")
        lines.append(
            f"| {r.criterion} | {r.n} | {show(r.raw_agreement)} | {show(r.kappa)} | "
            f"{show(r.weighted_kappa)} | {verdict} | {show(r.human_mean)} | {show(r.judge_mean)} |"
        )
    lines += ["", "Confusion matrices (rows: human 0/1/2, columns: judge 0/1/2):", ""]
    for r in rows:
        lines.append(f"- {r.criterion}: {r.confusion}")
    return "\n".join(lines) + "\n"
