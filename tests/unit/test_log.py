from __future__ import annotations

import json
import logging
import sys

import pytest

from researchai.log import (
    JsonFormatter,
    TextFormatter,
    bind_run_id,
    configure_logging,
    current_run_id,
)
from tests.support import make_settings


def _record(msg: str = "hello", **extra: object) -> logging.LogRecord:
    record = logging.LogRecord("researchai.test", logging.INFO, __file__, 1, msg, None, None)
    for key, value in extra.items():
        setattr(record, key, value)
    return record


def test_json_formatter_basic_fields() -> None:
    payload = json.loads(JsonFormatter().format(_record()))
    assert payload["level"] == "INFO"
    assert payload["logger"] == "researchai.test"
    assert payload["msg"] == "hello"
    assert "run_id" not in payload
    assert payload["ts"].endswith("+00:00")


def test_json_formatter_keeps_georgian_text_readable() -> None:
    line = JsonFormatter().format(_record("ელექტრომობილი"))
    assert "ელექტრომობილი" in line


def test_json_formatter_includes_extras_and_run_id() -> None:
    with bind_run_id("run-42"):
        payload = json.loads(JsonFormatter().format(_record(node="planner", tokens=17)))
    assert payload["run_id"] == "run-42"
    assert payload["node"] == "planner"
    assert payload["tokens"] == 17


def test_json_formatter_includes_exception() -> None:
    try:
        raise ValueError("boom")
    except ValueError:
        record = logging.LogRecord("x", logging.ERROR, __file__, 1, "failed", None, sys.exc_info())
    payload = json.loads(JsonFormatter().format(record))
    assert "ValueError: boom" in payload["exc"]


def test_run_id_is_scoped() -> None:
    assert current_run_id() is None
    with bind_run_id("a"):
        assert current_run_id() == "a"
        with bind_run_id("b"):
            assert current_run_id() == "b"
        assert current_run_id() == "a"
    assert current_run_id() is None


def test_text_formatter_appends_run_id() -> None:
    formatter = TextFormatter()
    assert "run_id" not in formatter.format(_record())
    with bind_run_id("r1"):
        assert formatter.format(_record()).endswith("[run_id=r1]")


@pytest.mark.parametrize(("fmt", "cls"), [("json", JsonFormatter), ("text", TextFormatter)])
def test_configure_logging_installs_one_handler(fmt: str, cls: type[logging.Formatter]) -> None:
    root = logging.getLogger()
    saved_handlers, saved_level = root.handlers[:], root.level
    try:
        configure_logging(make_settings(log_format=fmt, log_level="WARNING"))
        configure_logging(make_settings(log_format=fmt, log_level="WARNING"))
        assert len(root.handlers) == 1
        assert isinstance(root.handlers[0].formatter, cls)
        assert root.level == logging.WARNING
    finally:
        root.handlers[:] = saved_handlers
        root.setLevel(saved_level)
