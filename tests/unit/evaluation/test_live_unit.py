"""The per-case guard of live evaluations, without a database."""

from __future__ import annotations

import asyncio
from pathlib import Path
from types import TracebackType
from typing import Any

import pytest

from researchai.agent.models import ResearchResult, RunStatus
from researchai.evaluation import __main__ as cli
from researchai.evaluation import live
from researchai.evaluation.runner import EvaluationError, RunConfig, RunStore
from tests.support import make_settings

ROOT = Path(__file__).resolve().parents[3]


class HangingTools:
    """Opening the session never completes (what froze the first overnight run)."""

    def __init__(self, target: Any, timeout: float) -> None:
        pass

    async def __aenter__(self) -> HangingTools:
        await asyncio.Event().wait()
        return self  # pragma: no cover

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        return None


class BrokenTools(HangingTools):
    async def __aenter__(self) -> BrokenTools:
        raise ConnectionError("server gone")


async def test_a_hanging_tool_session_becomes_a_failed_case(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(live, "ResearchTools", HangingTools)
    monkeypatch.setattr(live, "SESSION_MARGIN_SECONDS", 0.05)
    run = live.agent_runner(make_settings(run_timeout_seconds=0.05), lambda: "x", None)
    result = await run("How many chargers?", "r1")
    assert (result.status, result.mode) == (RunStatus.FAILED, "offline")
    assert result.reason == "case exceeded 0.1 s including its tool session"


async def test_a_broken_tool_session_becomes_a_failed_case(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(live, "ResearchTools", BrokenTools)
    run = live.agent_runner(make_settings(), lambda: "x", object())  # type: ignore[arg-type]
    result = await run("How many chargers?", "r2")
    assert (result.status, result.mode) == (RunStatus.FAILED, "llm")
    assert result.reason == "tool session failed: ConnectionError"


def test_cli_resume_reports_what_is_left(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    config = RunConfig.model_validate_json(
        RunConfig(
            run_id="r", dataset="default", dataset_version="1.0.0", split="holdout",
            mode="offline", model=None, embedder="e", corpus_version="1.1.0",
            manifest_sha256="0", git_commit=None, started_at="2026-10-11T00:00:00Z",  # type: ignore[arg-type]
            finished_at="2026-10-11T00:00:00Z",  # type: ignore[arg-type]
            case_ids=["factual-002", "factual-004"],
        ).model_dump_json()
    )  # fmt: skip
    store = RunStore.create(tmp_path, config)
    store.append(
        "factual-002",
        ResearchResult(run_id="x", question="q", status=RunStatus.ANSWERED, mode="offline"),
    )
    seen: dict[str, Any] = {}

    async def fake_evaluate(*args: Any, **kw: Any) -> Any:
        seen.update(kw)
        raise EvaluationError("stop here")

    monkeypatch.setattr(cli, "evaluate", fake_evaluate)
    base = ["--eval-dir", str(ROOT / "eval"), "--corpus-dir", str(ROOT / "corpus"), "run"]
    assert cli.main([*base, "--resume", str(store.path)], make_settings()) == 1
    assert "resuming a holdout run needs --holdout" in capsys.readouterr().err
    assert cli.main([*base, "--resume", str(store.path), "--holdout"], make_settings()) == 1
    assert f"resuming {store.path}: 1 case(s) to run" in capsys.readouterr().out
    assert seen["resume"] == store.path
