"""The suite-wide conftest turns any skipped test into a failed run."""

from __future__ import annotations

from pathlib import Path

import pytest

CONFTEST = Path(__file__).resolve().parents[1] / "conftest.py"


@pytest.fixture
def skipping_project(pytester: pytest.Pytester) -> pytest.Pytester:
    pytester.makeconftest(CONFTEST.read_text(encoding="utf-8"))
    pytester.makepyfile(
        """
        import pytest

        def test_runs():
            pass

        @pytest.mark.skip(reason="service missing")
        def test_quietly_skipped():
            pass
        """
    )
    return pytester


def test_skip_fails_the_run(
    skipping_project: pytest.Pytester, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("ALLOW_SKIPS", raising=False)
    result = skipping_project.runpytest_subprocess("-p", "no:randomly")
    assert result.ret == pytest.ExitCode.TESTS_FAILED
    result.stdout.fnmatch_lines(
        ["*skipped tests are not allowed*", "SKIPPED *test_quietly_skipped"]
    )


def test_skip_allowed_when_opted_in(
    skipping_project: pytest.Pytester, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ALLOW_SKIPS", "1")
    result = skipping_project.runpytest_subprocess("-p", "no:randomly")
    assert result.ret == pytest.ExitCode.OK
