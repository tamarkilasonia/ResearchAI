from __future__ import annotations

from typing import Any

import pytest
from pydantic import ValidationError

from researchai.corpus.models import Chunk, DocumentMeta, Language, SourceType, Topic
from tests.support import make_document, meta_fields


def test_valid_synthetic_meta() -> None:
    meta = DocumentMeta.model_validate(meta_fields("tax-ka-0042"))
    assert meta.topic is Topic.TAX
    assert meta.language is Language.KA
    assert meta.source_type is SourceType.NEWS


def test_valid_real_source_meta() -> None:
    meta = DocumentMeta.model_validate(
        meta_fields(url="https://example.org/a", synthetic=False, license="CC-BY-4.0")
    )
    assert not meta.synthetic


@pytest.mark.parametrize(
    "overrides",
    [
        {"topic": "tax"},
        {"language": "ka"},
        {"published": "2026-09-02"},
        {"url": "http://corpus.researchai.invalid/x"},
        {"url": "corpus.researchai.invalid/x"},
        {"url": "https://example.org/x"},  # synthetic must use .invalid
        {"license": "CC-BY-4.0"},  # synthetic must use license 'synthetic'
        {"synthetic": False},  # .invalid URL but not synthetic
        {"source_type": "rumour"},
        {"unexpected": "field"},
        {"id": "ev-de-0001"},
        {"id": "ev-en-1"},
        {"title": "ab"},
    ],
)
def test_invalid_meta(overrides: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        DocumentMeta.model_validate(meta_fields(**overrides))


def test_real_source_cannot_claim_synthetic_license() -> None:
    with pytest.raises(ValidationError):
        DocumentMeta.model_validate(
            meta_fields(url="https://example.org/a", synthetic=False, license="synthetic")
        )


def test_document_hash_tracks_body_and_meta() -> None:
    base = make_document()
    assert base.sha256 == make_document().sha256
    assert base.sha256 != make_document(body=base.body + " ").sha256
    assert base.sha256 != make_document(title="Another title").sha256
    assert base.id == "ev-en-0001"


def test_chunk_span_must_match_text() -> None:
    Chunk(id="d#c0", doc_id="d", ordinal=0, section="", text="abc", start=5, end=8)
    with pytest.raises(ValidationError):
        Chunk(id="d#c0", doc_id="d", ordinal=0, section="", text="abc", start=5, end=9)
