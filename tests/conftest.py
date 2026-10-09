"""Suite-wide pytest hooks.

No silent skips: a skipped test fails the run unless ``ALLOW_SKIPS=1`` is set.
Tests that need infrastructure are deselected with a marker
(``-m "not integration"``), which is visible in the summary, instead of
skipping themselves when a service is missing.
"""

from __future__ import annotations

import os

import pytest

from researchai.config import Settings

pytest_plugins = ["pytester"]

_skipped: list[str] = []


@pytest.fixture(autouse=True)
def _isolated_settings_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep the developer's shell environment out of every test."""
    for name in Settings.model_fields:
        monkeypatch.delenv(name.upper(), raising=False)


def pytest_runtest_logreport(report: pytest.TestReport) -> None:
    if report.skipped:
        _skipped.append(report.nodeid)


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    if _skipped and os.environ.get("ALLOW_SKIPS") != "1" and exitstatus == 0:
        session.exitstatus = pytest.ExitCode.TESTS_FAILED


def pytest_terminal_summary(terminalreporter: pytest.TerminalReporter) -> None:
    if _skipped and os.environ.get("ALLOW_SKIPS") != "1":
        terminalreporter.write_sep("=", "skipped tests are not allowed", red=True)
        for nodeid in _skipped:
            terminalreporter.write_line(f"SKIPPED {nodeid}")
