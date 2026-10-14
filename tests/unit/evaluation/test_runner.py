"""Metrics, run reports, the runner and the evaluation command line (no database)."""

from __future__ import annotations

import json
import shutil
import subprocess
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from researchai.agent.models import ResearchResult, RunStatus, Usage
from researchai.corpus.loader import Corpus, load_corpus
from researchai.evaluation import __main__ as cli
from researchai.evaluation.dataset import Dataset, load_dataset
from researchai.evaluation.metrics import Metric, aggregate, percentile
from researchai.evaluation.models import Behavior, Category
from researchai.evaluation.report import render_report
from researchai.evaluation.retrieval import case_recall, distinct, reciprocal_rank, summarize
from researchai.evaluation.runner import (
    EvaluationError,
    RunConfig,
    RunStore,
    build_report,
    new_run_id,
    pending_cases,
    rescore,
    run_cases,
    score_store,
    select_cases,
)
from researchai.evaluation.scoring import CaseScore, TrapOutcome
from researchai.safety.models import GuardVerdict, SafetyReport
from tests.support import make_settings

ROOT = Path(__file__).resolve().parents[3]
EVAL = ROOT / "eval"


@pytest.fixture(scope="module")
def corpus() -> Corpus:
    return load_corpus(ROOT / "corpus")


@pytest.fixture(scope="module")
def dataset() -> Dataset:
    return load_dataset("default", EVAL)


def score(**fields: Any) -> CaseScore:
    base: dict[str, Any] = {
        "case_id": "factual-001",
        "category": Category.FACTUAL,
        "language": "en",
        "split": "dev",
        "question": "Q?",
        "expected": Behavior.ANSWER,
        "status": RunStatus.ANSWERED,
        "behavior_ok": True,
        "passed": True,
    }
    return CaseScore(**(base | fields))


# ----------------------------------------------------------------- metrics


def test_metric_formatting_and_empty_denominators() -> None:
    assert str(Metric.of(3, 4)) == "75% (3/4)"
    assert Metric.of(0, 0).value is None
    assert str(Metric.of(0, 0)) == "n/a"
    assert percentile([], 50) is None
    assert percentile([5.0, 1.0, 3.0, 2.0, 4.0], 50) == 3.0
    assert percentile([5.0, 1.0, 3.0, 2.0, 4.0], 95) == 5.0


def test_aggregate_counts_each_metric_over_its_own_cases() -> None:
    scores = [
        score(facts_total=2, facts_stated=1, claims=4, grounded_claims=4, supported_claims=3,
              hallucinated_claims=1, citations=5, verified_citations=5, precise_citations=4,
              reference_sources_cited=1, reference_sources_retrieved=2, tools_ok=True,
              topic_ok=True, total_ms=2000, tokens=1000, llm_calls=4, searches=6),
        score(case_id="numeric-001", category=Category.NUMERIC, facts_total=1, facts_stated=1,
              calculation_ok=False, passed=False, tools_ok=False, total_ms=4000, tokens=3000,
              llm_calls=5, searches=4),
        score(case_id="harmful-001", category=Category.HARMFUL, expected=Behavior.REFUSE,
              status=RunStatus.REFUSED, refusal_category_ok=True, tools_ok=True, total_ms=1),
        score(case_id="unanswerable-001", category=Category.UNANSWERABLE,
              expected=Behavior.ABSTAIN, status=RunStatus.ANSWERED, behavior_ok=False,
              passed=False, language="ka", total_ms=1000),
        score(case_id="conflict-001", category=Category.CONFLICT, expected=Behavior.FLAG_CONFLICT,
              status=RunStatus.REFUSED, behavior_ok=False, passed=False, rule_contradictions=2,
              trap_contradictions=1,
              traps=[TrapOutcome(trap="TR01", kind="contradiction", retrieved=True,
                                 detected=True, reported=False),
                     TrapOutcome(trap="TR10", kind="low_reliability", retrieved=True,
                                 detected=False, adopted=True),
                     TrapOutcome(trap="TR20", kind="injection", retrieved=True, detected=True),
                     TrapOutcome(trap="TR24", kind="injection", retrieved=False,
                                 detected=False, adopted=True)]),
        score(case_id="outdated-001", category=Category.OUTDATED, outdated_as_current=["0.248"],
              prompt_leak=True, passed=False, llm_fallbacks=["planner"], mode="llm"),
    ]  # fmt: skip
    m = aggregate(scores, cost_per_1k_tokens=0.5)
    assert m.cases == 6
    assert str(m.passed) == "33% (2/6)"
    assert (m.fact_recall.numerator, m.fact_recall.denominator) == (2, 3)
    assert str(m.citation_recall) == "75% (3/4)"
    assert str(m.citation_precision) == "80% (4/5)"
    assert str(m.hallucination_rate) == "25% (1/4)"
    assert str(m.retrieval_recall) == "67% (2/3)"
    assert str(m.tool_arguments) == "50% (1/2)"
    assert str(m.calculations) == "0% (0/1)"
    assert (str(m.conflicts_reported), str(m.conflicts_detected)) == ("0% (0/1)", "100% (1/1)")
    assert str(m.conflict_precision) == "50% (1/2)"
    assert (str(m.low_reliability_resisted), str(m.low_reliability_flagged)) == (
        "0% (0/1)", "0% (0/1)"
    )  # fmt: skip
    assert str(m.outdated_avoided) == "0% (0/1)"
    assert str(m.abstained_when_expected) == "0% (0/1)"
    assert str(m.false_abstentions) == "0% (0/4)"
    assert str(m.refused_when_expected) == "100% (1/1)"
    assert str(m.refusal_category) == "100% (1/1)"
    assert str(m.false_refusals) == "20% (1/5)"
    assert str(m.injection_resisted) == "50% (1/2)"
    assert str(m.injection_detected) == "100% (1/1)"
    assert str(m.prompt_leaks) == "17% (1/6)"
    assert str(m.llm_fallbacks) == "100% (1/1)"  # of researched LLM-mode cases only
    # Latency and tokens exclude refused runs (they never research anything).
    assert (m.latency_p50_ms, m.latency_p95_ms) == (1000.0, 4000.0)
    assert m.mean_tokens == 1000.0
    assert m.cost_per_report == 0.5
    assert set(m.by_language) == {"en", "ka"}
    assert str(m.by_category["factual"].fact_recall) == "50% (1/2)"


# ----------------------------------------------------------------- runner and report


def _result(question: str, run_id: str) -> ResearchResult:
    if "imported Tesla" in question:
        return ResearchResult(
            run_id=run_id, question=question, status=RunStatus.REFUSED, mode="offline",
            safety=SafetyReport(input=GuardVerdict(action="refuse", category="fraud",
                                                   method="rules")),
        )  # fmt: skip
    return ResearchResult(
        run_id=run_id,
        question=question,
        status=RunStatus.ABSTAINED,
        mode="offline",
        usage=Usage(searches=2),
        total_ms=25.0,
    )


def _config(case_ids: list[str]) -> RunConfig:
    now = datetime(2026, 10, 10, 12, 0, tzinfo=UTC)
    return RunConfig(
        run_id=new_run_id("dev", "offline", now), dataset="default", dataset_version="1.0.0",
        split="dev", mode="offline", model=None, embedder="hash:v1:1024", search_mode="hybrid",
        corpus_version="1.1.0", manifest_sha256="0" * 64, git_commit="abc1234",
        started_at=now, finished_at=now, case_ids=case_ids,
    )  # fmt: skip


async def test_run_store_score_and_rescore(
    tmp_path: Path, dataset: Dataset, corpus: Corpus
) -> None:
    cases = select_cases(dataset, "dev", ["harmful-001", "unanswerable-001", "factual-001"])
    config = _config([c.id for c in cases])
    store = RunStore.create(tmp_path, config)
    seen: list[str] = []

    async def run(question: str, run_id: str) -> ResearchResult:
        assert run_id.startswith("eval-")
        return _result(question, run_id)

    def save(done: int, s: CaseScore, result: ResearchResult) -> None:
        store.append(s.case_id, result)
        seen.append(f"{done}:{s.case_id}")

    outcomes = await run_cases(cases, run, dataset, corpus, concurrency=2, on_result=save)
    assert len(seen) == 3
    assert [s.case_id for s, _ in outcomes] == [c.id for c in cases]  # input order
    assert config.run_id == "20261010T120000Z-dev-offline"
    assert set(store.results()) == {c.id for c in cases}  # saved before any report exists

    report = score_store(store, config, dataset, corpus)
    by_id = {s.case_id: s for s in report.scores}
    assert by_id["harmful-001"].passed
    assert by_id["unanswerable-001"].passed
    assert by_id["factual-001"].reasons == ["expected answer, got abstained"]
    files = {p.name for p in store.path.iterdir()}
    assert files == {"config.json", "summary.json", "results.jsonl", "report.md"}
    markdown = (store.path / "report.md").read_text(encoding="utf-8")
    assert "## Failed cases (1 of 3)" in markdown
    assert "**Why it failed:** expected answer, got abstained" in markdown
    assert "search mode `hybrid`" in markdown
    with pytest.raises(FileExistsError):
        RunStore.create(tmp_path, config)

    again = rescore(store.path, dataset, corpus, cost_per_1k_tokens=1.0)
    assert [s.passed for s in again.scores] == [s.passed for s in report.scores]
    assert (
        json.loads((store.path / "summary.json").read_text())["config"]["git_commit"] == "abc1234"
    )


async def test_newer_results_replace_failed_ones_and_only_pending_cases_rerun(
    tmp_path: Path, dataset: Dataset, corpus: Corpus
) -> None:
    cases = select_cases(dataset, "dev", ["unanswerable-001", "unanswerable-003", "factual-001"])
    store = RunStore.create(tmp_path, _config([c.id for c in cases]))
    failed = ResearchResult(run_id="x", question="q", status=RunStatus.FAILED, mode="offline")
    store.append("unanswerable-001", _result("q", "a"))
    store.append("unanswerable-003", failed)
    assert [c.id for c in pending_cases(cases, store.results())] == [
        "factual-001", "unanswerable-003"
    ]  # fmt: skip
    store.append("unanswerable-003", _result("q", "b"))
    assert store.results()["unanswerable-003"].status is RunStatus.ABSTAINED
    report = score_store(store, store.config(), dataset, corpus)
    assert [s.case_id for s in report.scores] == ["unanswerable-001", "unanswerable-003"]


async def test_a_failed_case_can_stop_the_run(dataset: Dataset, corpus: Corpus) -> None:
    cases = select_cases(dataset, "dev", limit=4)
    calls: list[str] = []

    async def run(question: str, run_id: str) -> ResearchResult:
        calls.append(question)
        return ResearchResult(run_id=run_id, question=question, status=RunStatus.FAILED,
                              mode="offline", reason="tool backend unavailable")  # fmt: skip

    async def backend_down(result: ResearchResult) -> bool:
        return True

    outcomes = await run_cases(cases, run, dataset, corpus, should_stop=backend_down)
    assert len(calls) == len(outcomes) == 1  # the other three never started

    async def backend_up(result: ResearchResult) -> bool:
        return False

    outcomes = await run_cases(cases, run, dataset, corpus, should_stop=backend_up)
    assert len(outcomes) == 4


async def test_archived_runs_can_be_rescored(
    tmp_path: Path, dataset: Dataset, corpus: Corpus
) -> None:
    import gzip

    store = RunStore.create(tmp_path, _config(["unanswerable-001"]))
    store.append("unanswerable-001", _result("q", "r"))
    plain = store.path / "results.jsonl"
    (store.path / "results.jsonl.gz").write_bytes(gzip.compress(plain.read_bytes()))
    plain.unlink()
    report = rescore(store.path, dataset, corpus)
    assert [s.case_id for s in report.scores] == ["unanswerable-001"]


async def test_rescore_rejects_results_of_unknown_cases(
    tmp_path: Path, dataset: Dataset, corpus: Corpus
) -> None:
    store = RunStore.create(tmp_path, _config(["unanswerable-001"]))
    store.append("gone-001", _result("q", "r"))
    with pytest.raises(EvaluationError, match="gone-001"):
        rescore(store.path, dataset, corpus)


def test_case_selection(dataset: Dataset) -> None:
    assert len(select_cases(dataset, "dev", limit=5)) == 5
    assert len(select_cases(dataset, "holdout")) == 42
    with pytest.raises(EvaluationError, match="factual-002"):
        select_cases(dataset, "dev", ["factual-002"])  # a holdout case


def test_report_without_traps_or_failures() -> None:
    report = build_report(_config(["factual-001"]), [])
    report = report.model_copy(update={"scores": [score()]})
    markdown = render_report(report)
    assert "## Traps" not in markdown
    assert "## Failed cases (0 of 1)" in markdown
    assert "| Cost | latency p50 / p95 | n/a / n/a |" in markdown


# ----------------------------------------------------------------- retrieval


def test_retrieval_recall_and_rank(dataset: Dataset) -> None:
    case = next(c for c in dataset.cases if c.id == "comparison-001")  # one source for 2 facts
    other = next(c for c in dataset.cases if c.id == "factual-004")
    ranked = distinct(["ev-en-0001", "ev-en-0057", "ev-en-0001", "ev-en-0008"])
    assert ranked == ["ev-en-0001", "ev-en-0057", "ev-en-0008"]
    assert case_recall(case, ranked, 1) == 0.0
    assert case_recall(case, ranked, 2) == 1.0
    assert reciprocal_rank(case, ranked) == 0.5
    assert reciprocal_rank(case, ["ev-en-0001"]) == 0.0
    summary = summarize(
        [case, other], {"vector": {case.id: ranked, other.id: ["ev-en-0008"]}}, [1, 3]
    )
    assert summary == {"vector": {"recall@1": 0.5, "recall@3": 1.0, "mrr": 0.75}}


# ----------------------------------------------------------------- command line


def _args(*rest: str) -> list[str]:
    return ["--eval-dir", str(EVAL), "--corpus-dir", str(ROOT / "corpus"), *rest]


def test_cli_run_prints_progress_and_the_report_path(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    calls: dict[str, Any] = {}

    async def fake_evaluate(settings: Any, dataset: Dataset, corpus: Corpus, **kw: Any) -> Any:
        calls.update(kw)
        kw["on_result"](1, score())
        return build_report(_config(["factual-001"]), []), tmp_path / "run"

    monkeypatch.setattr(cli, "evaluate", fake_evaluate)
    assert cli.main(_args("run", "--case", "factual-001", "--limit", "1"), make_settings()) == 0
    out = capsys.readouterr().out
    assert "[  1/1] factual-001" in out
    assert f"report: {tmp_path / 'run' / 'report.md'}" in out
    assert (calls["split"], calls["mode"], calls["case_ids"], calls["limit"]) == (
        "dev", "offline", ["factual-001"], 1
    )  # fmt: skip


def test_cli_guards_the_holdout_and_reports_errors(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    assert cli.main(_args("run", "--split", "holdout"), make_settings()) == 1
    assert "holdout runs need --holdout" in capsys.readouterr().err

    async def failing(*args: Any, **kw: Any) -> Any:
        raise EvaluationError("the corpus is not ingested")

    monkeypatch.setattr(cli, "evaluate", failing)
    assert cli.main(_args("run", "--split", "holdout", "--holdout"), make_settings()) == 1
    assert "not ingested" in capsys.readouterr().err


def test_cli_retrieval_compares_modes(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    class Hit:
        def __init__(self, doc_id: str) -> None:
            self.doc_id = doc_id

    class Response:
        hits = (Hit("ev-en-0013"), Hit("ev-en-0013"), Hit("tax-en-0001"))

    class FakeSearch:
        def __init__(self, sessions: Any, embedder: Any) -> None:
            pass

        async def search(self, query: str, *, top_k: int, mode: str) -> Response:
            assert top_k == 15
            return Response()

    class Embedder:
        name = "fake"

    @asynccontextmanager
    async def backend(settings: Any, corpus: Corpus) -> AsyncIterator[Any]:
        class Backend:
            sessions = None
            embedder = Embedder()

        yield Backend()

    monkeypatch.setattr(cli, "open_backend", backend)
    monkeypatch.setattr(cli, "CorpusSearch", FakeSearch)
    settings = make_settings(eval_output_dir=tmp_path)
    assert cli.main(_args("retrieval", "--k", "1", "5"), settings) == 0
    out = capsys.readouterr().out
    assert '"recall@5"' in out
    written = json.loads(next(tmp_path.glob("*-dev-retrieval.json")).read_text())
    assert set(written["all"]) == {"vector", "keyword", "hybrid"}
    assert written["embedder"] == "fake"
    assert set(written["by_language"]) == {"en", "ka"}


def test_cli_rescore(
    tmp_path: Path, dataset: Dataset, corpus: Corpus, capsys: pytest.CaptureFixture[str]
) -> None:
    path = RunStore.create(tmp_path, _config([])).path
    assert cli.main(_args("rescore", str(path)), make_settings()) == 0
    assert "passed n/a" in capsys.readouterr().out


def test_git_commit(monkeypatch: pytest.MonkeyPatch) -> None:
    commit = cli.git_commit()
    assert commit is None or len(commit.split("+")[0]) >= 7
    monkeypatch.setattr(shutil, "which", lambda _: None)
    assert cli.git_commit() is None
    monkeypatch.setattr(shutil, "which", lambda _: "git")

    def broken(*args: Any, **kwargs: Any) -> Any:
        raise subprocess.CalledProcessError(128, "git")

    monkeypatch.setattr(subprocess, "run", broken)
    assert cli.git_commit() is None


def test_cli_check_applies_thresholds(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    report = build_report(_config(["factual-001"]), [])
    report = report.model_copy(
        update={"metrics": aggregate([score(), score(case_id="x-001", passed=False)])}
    )
    store = RunStore.create(tmp_path, report.config)
    store.write_report(report)
    path = store.path
    assert cli.main(["check", str(path), "--min", "passed=0.5", "--max", "prompt_leaks=0"]) == 0
    assert capsys.readouterr().out.strip() == "ok"
    assert cli.main(["check", str(path), "--min", "passed=0.9", "--max", "calculations=0"]) == 1
    captured = capsys.readouterr()
    assert "FAIL passed = 0.5 (min 0.9)" in captured.err
    assert "1 threshold(s) broken" in captured.out  # calculations: nothing to measure
    assert cli.main(["check", str(path), "--max", "latency_p50_ms=1"]) == 2
    assert "unknown metric" in capsys.readouterr().err
