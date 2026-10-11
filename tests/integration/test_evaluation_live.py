"""Evaluation against the real stack: PostgreSQL + pgvector, MCP tools, the offline agent."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import pytest
import pytest_asyncio

from researchai.agent.models import ResearchResult, RunStatus
from researchai.api.evaluations import EvaluationRequest, dataset_runner
from researchai.config import Settings
from researchai.corpus.loader import Corpus, load_corpus
from researchai.evaluation.dataset import Dataset, DatasetError, load_dataset
from researchai.evaluation.live import evaluate, open_backend
from researchai.evaluation.runner import EvaluationError, RunStore
from tests.integration.conftest import TEST_DIM, scratch_database_url, serving_real_corpus
from tests.support import make_settings

pytestmark = pytest.mark.asyncio(loop_scope="module")

ROOT = Path(__file__).resolve().parents[2]
DATABASE = scratch_database_url("_test_eval")
CASES = ["factual-001", "harmful-009", "unanswerable-001", "conflict-001"]


@pytest_asyncio.fixture(scope="module", loop_scope="module")
async def mcp_url() -> AsyncIterator[str]:
    async with serving_real_corpus(DATABASE) as url:
        yield url


@pytest.fixture(scope="module")
def dataset() -> Dataset:
    return load_dataset("default", ROOT / "eval")


@pytest.fixture(scope="module")
def corpus() -> Corpus:
    return load_corpus(ROOT / "corpus")


def _settings(tmp_path: Path, **overrides: Any) -> Settings:
    fields: dict[str, Any] = {
        "database_url": DATABASE, "embedding_dim": TEST_DIM, "eval_output_dir": tmp_path
    }  # fmt: skip
    return make_settings(**(fields | overrides))


async def test_offline_run_in_process(
    mcp_url: str, dataset: Dataset, corpus: Corpus, tmp_path: Path
) -> None:
    seen: list[str] = []
    report, path = await evaluate(
        _settings(tmp_path),
        dataset, corpus, split="dev", mode="offline", case_ids=CASES,
        git_commit="test", on_result=lambda _, s: seen.append(s.case_id),
    )  # fmt: skip
    assert sorted(seen) == sorted(CASES)
    assert report.config.embedder == f"hash:v1:{TEST_DIM}"
    assert report.config.search_mode == "hybrid"
    assert report.metrics.cases == 4
    by_id = {s.case_id: s for s in report.scores}
    assert by_id["harmful-009"].passed  # refused by rules, no tool call
    assert by_id["unanswerable-001"].status.value == "abstained"
    assert by_id["factual-001"].fact_recall == 1.0
    assert by_id["factual-001"].hallucinated_claims == 0
    assert [t.trap for t in by_id["conflict-001"].traps] == ["TR01"]
    lines = (path / "results.jsonl").read_text().splitlines()
    assert [json.loads(line)["case_id"] for line in lines] == [s.case_id for s in report.scores]
    assert (path / "report.md").read_text().startswith("# Evaluation run")


async def test_the_api_runner_evaluates_a_split_through_mcp_over_http(
    mcp_url: str, tmp_path: Path
) -> None:
    settings = _settings(tmp_path, mcp_server_url=mcp_url, corpus_dir=ROOT / "corpus",
                         eval_dir=ROOT / "eval")  # fmt: skip
    summary = await dataset_runner(settings)(EvaluationRequest(split="dev"))
    assert summary["cases"] == 93
    assert summary["metrics"]["refused_when_expected"]["denominator"] == 8
    assert summary["metrics"]["prompt_leaks"]["numerator"] == 0
    assert Path(summary["path"]).joinpath("summary.json").exists()
    with pytest.raises(DatasetError):
        await dataset_runner(settings)(EvaluationRequest(dataset="missing"))


async def test_runs_refuse_a_stale_index_or_a_missing_model(
    mcp_url: str, dataset: Dataset, corpus: Corpus, tmp_path: Path
) -> None:
    with pytest.raises(EvaluationError, match="re-run ingestion"):
        await evaluate(
            _settings(tmp_path, embedding_dim=32),
            dataset, corpus, split="dev", mode="offline", case_ids=CASES[:1],
        )  # fmt: skip
    with pytest.raises(EvaluationError, match="LLM_PROVIDER"):
        await evaluate(
            _settings(tmp_path),
            dataset, corpus, split="dev", mode="llm", case_ids=CASES[:1],
        )  # fmt: skip
    with pytest.raises(EvaluationError, match="no cases selected"):
        await evaluate(
            _settings(tmp_path),
            dataset, corpus, split="dev", mode="offline", limit=0,
        )  # fmt: skip


async def test_an_interrupted_run_resumes_only_missing_and_failed_cases(
    mcp_url: str, dataset: Dataset, corpus: Corpus, tmp_path: Path
) -> None:
    settings = _settings(tmp_path)
    _, path = await evaluate(
        settings, dataset, corpus, split="dev", mode="offline", case_ids=CASES[:2]
    )
    store = RunStore(path)
    store.append(  # pretend the second case hit an outage
        CASES[1], ResearchResult(run_id="x", question="q", status=RunStatus.FAILED,
                                 mode="offline", reason="tool backend unavailable"),
    )  # fmt: skip
    config = store.config()
    store.write_config(config.model_copy(update={"case_ids": CASES}))  # and never got further
    reran: list[str] = []
    resumed, same = await evaluate(
        settings, dataset, corpus, split="holdout", mode="llm", resume=path,
        git_commit="def5678", on_result=lambda _, s: reran.append(s.case_id),
    )  # fmt: skip
    assert same == path
    assert sorted(reran) == sorted(CASES[1:])  # not the case that already succeeded
    assert (resumed.config.split, resumed.config.mode) == ("dev", "offline")  # from the run
    assert [s.case_id for s in resumed.scores] == CASES
    assert resumed.config.resumed[0].endswith("@def5678")
    assert all(s.status is not RunStatus.FAILED for s in resumed.scores)


async def test_backend_health(mcp_url: str, corpus: Corpus, tmp_path: Path) -> None:
    async with open_backend(_settings(tmp_path), corpus, mcp_url=mcp_url) as backend:
        assert await backend.healthy()
    async with open_backend(
        _settings(tmp_path), corpus, mcp_url="http://127.0.0.1:9/mcp"
    ) as backend:
        assert not await backend.healthy()
