"""The committed corpus is valid, frozen, and reproducible."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import pytest

from researchai.corpus.chunking import MAX_CHARS, chunk_document
from researchai.corpus.loader import Corpus, load_corpus, read_manifest
from scripts.generate_corpus_listings import generate, write_all

ROOT = Path(__file__).resolve().parents[2] / "corpus"


@pytest.fixture(scope="module")
def corpus() -> Corpus:
    return load_corpus(ROOT)


def test_manifest_matches_documents(corpus: Corpus) -> None:
    assert corpus.manifest() == read_manifest(ROOT), (
        "corpus changed: bump corpus.toml version and run `manifest --write`"
    )


def test_size_and_balance(corpus: Corpus) -> None:
    docs = corpus.documents
    assert 200 <= len(docs) <= 300
    counts = Counter((d.meta.topic.value, d.meta.language.value) for d in docs)
    assert set(counts) == {("ev", "en"), ("ev", "ka"), ("tax", "en"), ("tax", "ka")}
    assert counts["ev", "ka"] / (counts["ev", "en"] + counts["ev", "ka"]) >= 0.25
    assert counts["tax", "ka"] >= 10


def test_every_document_is_synthetic_and_says_so(corpus: Corpus) -> None:
    for document in corpus.documents:
        assert document.meta.synthetic
        assert ".invalid/" in document.meta.url


def test_titles_are_unique_per_language(corpus: Corpus) -> None:
    titles = Counter((d.meta.language, d.meta.title) for d in corpus.documents)
    duplicates = [title for title, n in titles.items() if n > 1]
    assert not duplicates


def test_every_document_chunks_within_limits(corpus: Corpus) -> None:
    for document in corpus.documents:
        chunks = chunk_document(document)
        assert chunks, document.id
        assert all(len(c.text) <= MAX_CHARS for c in chunks)


def test_generator_reproduces_committed_files(tmp_path: Path) -> None:
    written = write_all(tmp_path)
    assert len(written) == len(list(generate()))
    for path in written:
        committed = ROOT / path.relative_to(tmp_path)
        assert committed.read_text("utf-8") == path.read_text("utf-8"), committed.name
