"""Side-by-side results tables for the README and the evaluation docs."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from researchai.agent.models import RunStatus
from researchai.evaluation import __main__ as cli
from researchai.evaluation.models import Behavior, Category
from researchai.evaluation.report import comparison_table
from researchai.evaluation.runner import (
    JudgeCriterion,
    JudgeSection,
    RunConfig,
    RunStore,
    build_report,
)
from researchai.evaluation.scoring import CaseScore


def _report(split: str, model: str | None, judge: JudgeSection | None = None) -> object:
    now = datetime(2026, 10, 12, tzinfo=UTC)
    config = RunConfig(
        run_id=f"{split}-{model}", dataset="default", dataset_version="1.0.0",
        split=split, mode="llm" if model else "offline", model=model,  # type: ignore[arg-type]
        embedder="e", corpus_version="1.1.0", manifest_sha256="0" * 64, git_commit=None,
        started_at=now, finished_at=now, case_ids=["factual-001"],
    )  # fmt: skip
    score = CaseScore(
        case_id="factual-001", category=Category.FACTUAL, language="en", split=split,
        question="Q?", expected=Behavior.ANSWER, status=RunStatus.ANSWERED, behavior_ok=True,
        facts_total=2, facts_stated=1, total_ms=4000.0, tokens=900, passed=True,
        mode="llm" if model else "offline",
    )  # fmt: skip
    report = build_report(config, [(score, None)])  # type: ignore[list-item]
    return report.model_copy(update={"judge": judge})


def test_runs_side_by_side_with_judge_and_kappa(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    judge = JudgeSection(model="llama3.1:8b", calibrated_on=35, criteria={
        "completeness": JudgeCriterion(n=60, mean=1.25, kappa=0.52, reliable=True),
        "relevance": JudgeCriterion(n=60, mean=1.5),
    })  # fmt: skip
    offline, llm = _report("dev", None), _report("dev", "qwen2.5:7b", judge)
    table = comparison_table([offline, llm])  # type: ignore[list-item]
    assert table.startswith("| Metric | dev · offline (1) | dev · qwen2.5:7b (1) |")
    assert "| Reference facts stated | 50% (1/2) | 50% (1/2) |" in table
    assert "| Latency p50 / p95 | 4.0 s / 4.0 s | 4.0 s / 4.0 s |" in table
    assert "| Judge: completeness | n/a | 1.25 / 2 (κ 0.52) |" in table
    assert "| Judge: relevance | n/a | 1.50 / 2 (not calibrated) |" in table
    assert "| Judge: quality | n/a | n/a |" in table

    paths = []
    for report in (offline, llm):
        store = RunStore.create(tmp_path, report.config)  # type: ignore[attr-defined]
        store.write_report(report)  # type: ignore[arg-type]
        paths.append(str(store.path))
    assert cli.main(["tables", *paths]) == 0
    assert capsys.readouterr().out == table
