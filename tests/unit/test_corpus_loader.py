from __future__ import annotations

import unicodedata
from pathlib import Path

import pytest

from researchai.corpus.loader import (
    CorpusError,
    georgian_share,
    load_corpus,
    parse_document,
    read_manifest,
    write_manifest,
)
from tests.support import EN_BODY, meta_fields, render_document, write_corpus, write_document


def test_parse_document_roundtrip() -> None:
    document = parse_document(render_document(meta_fields(), EN_BODY))
    assert document.id == "ev-en-0001"
    assert document.body == EN_BODY.strip("\n")


def test_parse_normalizes_bom_crlf_and_nfc() -> None:
    decomposed = unicodedata.normalize("NFD", "Café prices")
    text = "﻿" + render_document(meta_fields(), f"{decomposed}\n\nLine two").replace("\n", "\r\n")
    document = parse_document(text)
    assert "\r" not in document.body
    assert document.body.startswith(unicodedata.normalize("NFC", "Café"))


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("no front matter", "opening"),
        ("+++\nid = 'x'\n", "closing"),
        ("+++\n+++\n", "validation error"),
    ],
)
def test_parse_document_errors(text: str, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        parse_document(text)


def test_parse_rejects_empty_body() -> None:
    with pytest.raises(ValueError, match="empty"):
        parse_document(render_document(meta_fields(), "\n  \n"))


def test_load_corpus(tmp_path: Path) -> None:
    corpus = load_corpus(write_corpus(tmp_path))
    assert corpus.version == "0.1.0"
    assert [d.id for d in corpus.documents] == ["ev-en-0001", "ev-ka-0001"]
    assert corpus.get("ev-ka-0001").meta.language.value == "ka"
    with pytest.raises(KeyError):
        corpus.get("ev-en-9999")


def test_load_corpus_collects_all_problems(tmp_path: Path) -> None:
    root = write_corpus(tmp_path)
    # wrong directory
    misplaced = tmp_path / "ev" / "en" / "ev-en-0002.md"
    misplaced.write_text(render_document(meta_fields("ev-ka-0002"), "ქართული ტექსტი"), "utf-8")
    # duplicate id in another file name
    (tmp_path / "ev" / "en" / "copy.md").write_text(
        render_document(meta_fields(), EN_BODY), "utf-8"
    )
    # English declared, Georgian body
    write_document(root, "ev-en-0003", body="ეს ტექსტი ქართულია და არა ინგლისური.")
    # Georgian declared, English body
    write_document(root, "ev-ka-0004", body="This text is English, not Georgian.")
    # snapshot after corpus snapshot
    write_document(root, "ev-en-0005", snapshot="2026-12-01", published="2026-10-01")
    # unparsable
    (tmp_path / "ev" / "en" / "ev-en-0006.md").write_text("+++\nid = \n+++\nbody", "utf-8")

    with pytest.raises(CorpusError) as excinfo:
        load_corpus(root)
    text = str(excinfo.value)
    assert "should be at ev/ka/ev-ka-0002.md" in text
    assert "duplicate id ev-en-0001" in text
    assert "declared 'en'" in text
    assert "declared 'ka'" in text
    assert "after corpus snapshot" in text
    assert "ev-en-0006.md" in text
    assert len(excinfo.value.problems) >= 7


def test_load_corpus_requires_config(tmp_path: Path) -> None:
    with pytest.raises(CorpusError, match=r"corpus\.toml"):
        load_corpus(tmp_path)


def test_load_corpus_rejects_non_date_snapshot(tmp_path: Path) -> None:
    (tmp_path / "corpus.toml").write_text('version = "1"\nsnapshot = "soon"\n', "utf-8")
    with pytest.raises(CorpusError, match="TOML date"):
        load_corpus(tmp_path)


def test_load_corpus_requires_documents(tmp_path: Path) -> None:
    (tmp_path / "corpus.toml").write_text('version = "1"\nsnapshot = 2026-09-01\n', "utf-8")
    with pytest.raises(CorpusError, match="no documents"):
        load_corpus(tmp_path)


def test_manifest_is_deterministic_and_tracks_changes(tmp_path: Path) -> None:
    root = write_corpus(tmp_path)
    first = load_corpus(root).manifest()
    assert first == load_corpus(root).manifest()
    assert first["document_count"] == 2
    write_document(root, "ev-en-0001", body=EN_BODY + "\nOne more sentence.")
    assert load_corpus(root).manifest()["sha256"] != first["sha256"]


def test_manifest_roundtrip(tmp_path: Path) -> None:
    root = write_corpus(tmp_path)
    corpus = load_corpus(root)
    path = write_manifest(root, corpus)
    assert path.name == "manifest.json"
    assert read_manifest(root) == corpus.manifest()


@pytest.mark.parametrize(
    ("text", "expected"),
    [("", 0.0), ("123 !!", 0.0), ("abc", 0.0), ("აბგ", 1.0), ("აბ ab", 0.5)],
)
def test_georgian_share(text: str, expected: float) -> None:
    assert georgian_share(text) == pytest.approx(expected)
