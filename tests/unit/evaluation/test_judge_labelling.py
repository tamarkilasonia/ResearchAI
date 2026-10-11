"""LLM judge, human labelling and Cohen's kappa."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from hypothesis import given
from hypothesis import strategies as st

from researchai.agent.llm import LLMError
from researchai.agent.models import (
    Report,
    ReportSection,
    ReportSentence,
    ResearchResult,
    RunStatus,
)
from researchai.corpus.models import Language
from researchai.evaluation import __main__ as cli
from researchai.evaluation.agreement import agreement, cohen_kappa, interpret
from researchai.evaluation.dataset import Dataset, load_dataset
from researchai.evaluation.judge import (
    Judgement,
    build_judge,
    judge_case,
    judge_run,
    judge_summary,
    judge_user,
    judgeable,
    read_judgements,
    report_for_review,
)
from researchai.evaluation.labelling import (
    CURRENT_FILE,
    HumanLabel,
    compare,
    label_session,
    read_items,
    read_labels,
    render_agreement,
    sample_items,
    write_items,
)
from researchai.evaluation.runner import RunConfig, RunStore, judge_section
from tests.agent_fakes import ScriptedLLM, always
from tests.support import make_settings

ROOT = Path(__file__).resolve().parents[3]
EVAL = ROOT / "eval"
GOOD = {
    "completeness": 2,
    "completeness_reason": "all points",
    "relevance": 1,
    "relevance_reason": "some drift",
    "quality": 0,
    "quality_reason": "contradicts itself",
}


@pytest.fixture(scope="module")
def dataset() -> Dataset:
    return load_dataset("default", EVAL)


def answered(case_id: str, text: str = "Public charging points: 512.") -> ResearchResult:
    report = Report(
        language=Language.EN,
        title=f"Report for {case_id}",
        summary=[ReportSentence(text=text, kind="note")],
        sections=[ReportSection(heading="Count", sub_question_id="sq1",
                                sentences=[ReportSentence(text="More detail.", kind="note")])],
        generated_by="llm",
    )  # fmt: skip
    return ResearchResult(run_id=case_id, question="q", status=RunStatus.ANSWERED, mode="llm",
                          report=report)  # fmt: skip


# ----------------------------------------------------------------- kappa


def test_kappa_matches_hand_computed_values() -> None:
    assert cohen_kappa([0, 0, 1, 1], [0, 1, 1, 1]) == 0.5
    assert cohen_kappa([0, 1, 2, 2], [0, 2, 2, 1]) == 0.2
    assert cohen_kappa([0, 1, 2, 2], [0, 2, 2, 1], weighted=True) == 0.6364
    assert cohen_kappa([0, 1, 2], [0, 1, 2]) == 1.0
    assert cohen_kappa([1, 1, 1], [1, 1, 1]) is None  # chance agreement is total
    assert cohen_kappa([], []) is None
    with pytest.raises(ValueError, match="same items"):
        cohen_kappa([0], [0, 1])


@given(st.lists(st.tuples(st.sampled_from([0, 1, 2]), st.sampled_from([0, 1, 2])), min_size=1))
def test_kappa_is_symmetric_and_at_most_one(pairs: list[tuple[int, int]]) -> None:
    a, b = [x for x, _ in pairs], [y for _, y in pairs]
    for weighted in (False, True):
        kappa = cohen_kappa(a, b, weighted=weighted)
        assert kappa == cohen_kappa(b, a, weighted=weighted)
        assert kappa is None or kappa <= 1.0
    if len(set(a)) > 1:
        assert cohen_kappa(a, a) == 1.0


def test_interpretation_and_agreement_row() -> None:
    assert [interpret(k) for k in (None, -0.1, 0.1, 0.3, 0.5, 0.7, 0.9)] == [
        "undefined", "poor", "slight", "fair", "moderate", "substantial", "almost perfect"
    ]  # fmt: skip
    row = agreement("quality", [(0, 0), (1, 1), (2, 1), (2, 2)])
    assert (row.n, row.raw_agreement, row.human_mean, row.judge_mean) == (4, 0.75, 1.25, 1.0)
    assert row.confusion == [[1, 0, 0], [0, 1, 0], [0, 1, 1]]
    assert row.reliable
    empty = agreement("quality", [])
    assert (empty.kappa, empty.raw_agreement, empty.reliable) == (None, None, False)


# ----------------------------------------------------------------- judge


def test_what_the_judge_sees(dataset: Dataset) -> None:
    case = next(c for c in dataset.cases if c.id == "factual-001")
    result = answered("factual-001")
    review = report_for_review(result)
    assert "Status:" not in review  # no automatic score
    assert review.startswith("# Report for factual-001")
    prompt = judge_user(case, review)
    assert "Question:\nHow many public charging points" in prompt
    assert "- 512 public charging points in the June 2026 registry" in prompt
    assert prompt.endswith("</report>")
    assert judgeable(case, result)
    assert not judgeable(case, result.model_copy(update={"status": RunStatus.FAILED}))
    harmful = next(c for c in dataset.cases if c.id == "harmful-001")
    assert not judgeable(harmful, result)


def test_the_judge_must_not_be_the_generator() -> None:
    settings = make_settings(judge_model="llama3.1:8b", llm_model="qwen2.5:7b")
    with pytest.raises(ValueError, match="must not grade its own writing"):
        build_judge(settings, generator="llama3.1:8b")
    assert build_judge(settings, generator="qwen2.5:7b").model == "llama3.1:8b"
    assert build_judge(settings, generator="llama3.1:8b", allow_same_model=True)


async def test_judging_is_resumable_and_retries_failures(dataset: Dataset, tmp_path: Path) -> None:
    cases = [c for c in dataset.cases if c.id in ("factual-001", "factual-003", "harmful-001")]
    results = {c.id: answered(c.id) for c in cases}
    failing = ScriptedLLM({"JudgeOutput": [always(LLMError("transport", "down"))]})
    seen: list[str] = []
    judged = await judge_run(tmp_path, failing, cases, results,
                             on_judgement=lambda n, j: seen.append(f"{n}:{j.case_id}"))  # fmt: skip
    assert judged == {}
    assert seen == ["1:factual-001", "2:factual-003"]  # harmful cases are not judgeable
    working = ScriptedLLM({"JudgeOutput": [always(GOOD)]})
    judged = await judge_run(tmp_path, working, cases, results)
    assert set(judged) == {"factual-001", "factual-003"}
    assert judged["factual-001"].labels == {"completeness": 2, "relevance": 1, "quality": 0}
    assert judged["factual-001"].reasons["quality"] == "contradicts itself"
    assert len(working.requests) == 2
    again = await judge_run(tmp_path, working, cases, results)
    assert len(working.requests) == 2  # nothing left to judge
    assert set(again) == set(read_judgements(tmp_path))
    lines = (tmp_path / "judgements.jsonl").read_text().splitlines()
    assert len(lines) == 4  # two failures kept for the record, two judgements
    summary = judge_summary(judged)
    assert summary["completeness"] == {"n": 2, "mean": 2.0}
    assert read_judgements(tmp_path / "missing") == {}


async def test_a_failed_judgement_records_the_error(dataset: Dataset) -> None:
    case = next(c for c in dataset.cases if c.id == "factual-001")
    llm = ScriptedLLM({"JudgeOutput": [always({"completeness": 5})]})
    judgement = await judge_case(llm, case, answered(case.id))
    assert judgement.labels == {"completeness": None, "relevance": None, "quality": None}
    assert judgement.error is not None
    assert judgement.error.startswith("schema")


# ----------------------------------------------------------------- labelling


def _dev_answers(dataset: Dataset) -> tuple[list[Any], dict[str, ResearchResult]]:
    cases = dataset.split("dev")
    return cases, {c.id: answered(c.id) for c in cases}


def test_sample_is_seeded_and_spread_over_categories(dataset: Dataset) -> None:
    cases, results = _dev_answers(dataset)
    items = sample_items(cases, results, size=20, seed=11)
    assert len(items) == 20
    assert [i.item for i in items] == list(range(1, 21))
    categories = {i.category for i in items}
    assert "harmful" not in categories
    assert "unanswerable" not in categories
    assert len(categories) == 9  # every answerable category
    assert items == sample_items(cases, results, size=20, seed=11)
    assert items != sample_items(cases, results, size=20, seed=12)
    assert len(sample_items(cases, results, size=500, seed=1)) == 77  # all judgeable dev cases


def test_a_labelling_session_saves_every_answer_and_resumes(
    dataset: Dataset, tmp_path: Path
) -> None:
    cases, results = _dev_answers(dataset)
    items = sample_items(cases, results, size=3, seed=11)
    folder = tmp_path / "labels"
    write_items(folder, items, {"run_id": "r"})
    assert read_items(folder) == items
    shown: list[str] = []
    # item 1: invalid, then 2; unsure 1; skip quality; a note. item 2: quit at once.
    answers = iter(["7", "2", "1u", "s", "short and clear", "q"])
    count = label_session(folder, "tamar", ask=lambda _: next(answers), show=shown.append)
    assert count == 1
    assert any("Please answer 0, 1 or 2" in s for s in shown)
    assert (folder / CURRENT_FILE).read_text().startswith(f"# Item 2 of 3 ({items[1].category}")
    saved = read_labels(folder, "tamar")
    assert list(saved) == [1]
    assert saved[1].labels == {"completeness": 2, "relevance": 1, "quality": None}
    assert (saved[1].unsure, saved[1].note) == (["relevance"], "short and clear")

    answers = iter(["0", "0", "0", "", "2", "2", "2u", ""])
    assert label_session(folder, "tamar", ask=lambda _: next(answers), show=shown.append) == 2
    assert set(read_labels(folder, "tamar")) == {1, 2, 3}
    assert not (folder / CURRENT_FILE).exists()
    assert read_labels(folder, "someone-else") == {}
    with pytest.raises(FileExistsError):
        write_items(folder, items, {})


def test_compare_uses_only_items_both_labelled(dataset: Dataset) -> None:
    cases, results = _dev_answers(dataset)
    items = sample_items(cases, results, size=4, seed=11)
    now = datetime.now(UTC)

    def human(item: int, labels: dict[str, int | None]) -> HumanLabel:
        return HumanLabel(item=item, case_id=items[item - 1].case_id, labeler="h",
                          labels=labels, labelled_at=now)  # fmt: skip

    def judge(case_id: str, value: int) -> Judgement:
        return Judgement(case_id=case_id, model="judge", prompt_version="v",
                         labels={"completeness": value, "relevance": value, "quality": value},
                         judged_at=now)  # fmt: skip

    humans = {1: human(1, {"completeness": 2, "relevance": 2, "quality": None}),
              2: human(2, {"completeness": 0, "relevance": 1, "quality": 1}),
              3: human(3, {"completeness": 1, "relevance": 1, "quality": 1})}  # fmt: skip
    judges = {
        items[i].case_id: judge(items[i].case_id, value) for i, value in ((0, 2), (1, 0), (3, 1))
    }
    rows = {r.criterion: r for r in compare(items, humans, judges)}
    assert (rows["completeness"].n, rows["completeness"].kappa) == (2, 1.0)
    assert rows["quality"].n == 1  # item 1 skipped by the human, item 3 not judged
    text = render_agreement(list(rows.values()), labeler="h", model="judge")
    assert "| completeness | 2 | 1.00 | 1.00 |" in text
    assert "(unreliable)" in text  # quality: kappa undefined with one item
    assert "Confusion matrices" in text


# ----------------------------------------------------------------- run report and CLI


def _llm_run(tmp_path: Path, dataset: Dataset, *, split: str = "dev", mode: str = "llm") -> Path:
    cases = dataset.split(split)[:30]  # type: ignore[arg-type]
    now = datetime(2026, 10, 12, tzinfo=UTC)
    config = RunConfig(
        run_id=f"run-{split}-{mode}", dataset="default", dataset_version="1.0.0",
        split=split, mode=mode, model="qwen2.5:7b", embedder="e",  # type: ignore[arg-type]
        corpus_version="1.1.0", manifest_sha256="0" * 64, git_commit=None,
        started_at=now, finished_at=now, case_ids=[c.id for c in cases],
    )  # fmt: skip
    store = RunStore.create(tmp_path, config)
    for case in cases:
        store.append(case.id, answered(case.id))
    return store.path


def _args(*rest: str) -> list[str]:
    return ["--eval-dir", str(EVAL), "--corpus-dir", str(ROOT / "corpus"), *rest]


def test_cli_judge_sample_label_and_agreement(
    tmp_path: Path, dataset: Dataset, monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:  # fmt: skip
    run = _llm_run(tmp_path, dataset)
    assert judge_section(run) is None
    judge = JudgeLLM({"JudgeOutput": [always(GOOD)]})
    monkeypatch.setattr(cli, "build_judge", lambda *a, **k: judge)
    settings = make_settings(judge_model="llama3.1:8b")
    assert cli.main(_args("judge", str(run)), settings) == 0
    out = capsys.readouterr().out
    assert "report(s) judged by judge-model" in out
    section = judge_section(run)
    assert section is not None
    assert (section.model, section.calibrated_on) == ("judge-model", 0)
    assert section.criteria["completeness"].reliable is None
    assert "Not calibrated yet" in (run / "report.md").read_text()

    labels = tmp_path / "labels"
    assert cli.main(_args("label-sample", str(run), "--out", str(labels), "--size", "6")) == 0
    assert "6 item(s) written" in capsys.readouterr().out
    items = read_items(labels)
    answers = iter(["2", "1", "0", ""] * len(items))
    monkeypatch.setattr("builtins.input", lambda _: next(answers))
    assert cli.main(_args("label", str(labels), "--labeler", "tamar")) == 0
    assert len(read_labels(labels, "tamar")) == 6

    assert cli.main(_args("agreement", str(labels), str(run), "--labeler", "tamar"), settings) == 0
    out = capsys.readouterr().out
    assert "# Judge calibration: judge-model vs tamar" in out
    calibration = json.loads((run / "calibration.json").read_text())
    assert calibration["items"] == 6
    assert (labels / "agreement.md").exists()
    section = judge_section(run)
    assert section is not None
    assert section.calibrated_on == 6
    assert section.criteria["completeness"].kappa is None  # both raters always said 2
    assert "Calibrated on 6 human-labelled items" in (run / "report.md").read_text()


class JudgeLLM(ScriptedLLM):
    @property
    def model(self) -> str:
        return "judge-model"


def test_cli_guards(tmp_path: Path, dataset: Dataset, capsys: pytest.CaptureFixture[str]) -> None:
    offline = _llm_run(tmp_path, dataset, mode="offline")
    assert cli.main(_args("label-sample", str(offline), "--out", str(tmp_path / "x"))) == 1
    assert "label LLM-mode reports" in capsys.readouterr().err
    holdout = _llm_run(tmp_path, dataset, split="holdout")
    assert cli.main(_args("label-sample", str(holdout), "--out", str(tmp_path / "y"))) == 1
    assert "the holdout stays unseen" in capsys.readouterr().err
    run = _llm_run(tmp_path / "same", dataset)
    same = make_settings(judge_model="qwen2.5:7b")
    assert cli.main(_args("judge", str(run)), same) == 1
    assert "must not grade its own writing" in capsys.readouterr().err
    assert cli.main(_args("agreement", str(tmp_path / "x"), str(run))) == 1
