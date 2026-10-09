"""CLI argument handling. Real database runs are in tests/integration."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from researchai.corpus import __main__ as corpus_cli
from researchai.corpus.ingest import IngestReport
from researchai.db import __main__ as db_cli
from researchai.db import migrate
from tests.support import EN_BODY, make_settings, write_corpus, write_document


def run(argv: list[str], root: Path) -> int:
    return corpus_cli.main(["--dir", str(root), *argv], settings=make_settings())


def test_validate_ok(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert run(["validate"], write_corpus(tmp_path)) == 0
    assert "ok: 2 documents" in capsys.readouterr().out


def test_validate_reports_problems(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    root = write_corpus(tmp_path)
    write_document(root, "ev-ka-0002", body="English body in a Georgian document.")
    assert run(["validate"], root) == 1
    assert "declared 'ka'" in capsys.readouterr().err


def test_stats(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert run(["stats"], write_corpus(tmp_path)) == 0
    stats = json.loads(capsys.readouterr().out)
    assert stats["documents"] == 2
    assert stats["by_language"] == {"en": 1, "ka": 1}
    assert stats["by_topic_language"] == {"ev/en": 1, "ev/ka": 1}
    assert stats["synthetic"] == 2
    assert stats["chunks"] >= 2


def test_manifest_write_then_check(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    root = write_corpus(tmp_path)
    assert run(["manifest", "--write"], root) == 0
    assert run(["manifest", "--check"], root) == 0
    write_document(root, "ev-en-0001", body=EN_BODY + "\nEdited.")
    assert run(["manifest", "--check"], root) == 1
    assert "stale" in capsys.readouterr().out


def test_manifest_requires_mode(tmp_path: Path) -> None:
    with pytest.raises(SystemExit):
        run(["manifest"], write_corpus(tmp_path))


def test_ingest_wires_settings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    captured: dict[str, Any] = {}

    async def fake_ingest(corpus: Any, embedder: Any, sessions: Any, **kwargs: Any) -> IngestReport:
        captured.update(kwargs, embedder=embedder.name, docs=len(corpus.documents))
        return IngestReport("ingested", "0.1.0", "abc", embedder.name, 2, 4)

    monkeypatch.setattr(corpus_cli, "ingest", fake_ingest)
    settings = make_settings(embedding_dim=64, embedding_batch_size=7)
    assert (
        corpus_cli.main(["--dir", str(write_corpus(tmp_path)), "ingest", "--force"], settings) == 0
    )
    assert captured == {"batch_size": 7, "force": True, "embedder": "hash:v1:64", "docs": 2}
    assert json.loads(capsys.readouterr().out)["status"] == "ingested"


@pytest.mark.parametrize(
    ("argv", "expected"),
    [
        (["upgrade"], ("upgrade", "head")),
        (["upgrade", "0001"], ("upgrade", "0001")),
        (["downgrade"], ("downgrade", "base")),
    ],
)
def test_db_cli_dispatch(
    argv: list[str], expected: tuple[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[tuple[str, str, int]] = []

    async def fake_upgrade(engine: Any, *, embedding_dim: int, revision: str) -> None:
        calls.append(("upgrade", revision, embedding_dim))

    async def fake_downgrade(engine: Any, *, embedding_dim: int, revision: str) -> None:
        calls.append(("downgrade", revision, embedding_dim))

    monkeypatch.setattr(migrate, "upgrade", fake_upgrade)
    monkeypatch.setattr(migrate, "downgrade", fake_downgrade)
    assert db_cli.main(argv, settings=make_settings(embedding_dim=64)) == 0
    assert calls == [(*expected, 64)]
