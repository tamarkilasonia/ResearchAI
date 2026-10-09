"""Shared test helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from researchai.config import Settings
from researchai.corpus.models import Document, DocumentMeta

EN_BODY = """# Charging at home

A 7.4 kW wallbox charges a 40 kWh battery overnight. Installation in Tbilisi
usually takes one day.

## Cost

Home charging costs less than public fast charging.
"""

KA_BODY = """# სახლში დატენვა

7.4 kW wallbox ელექტრომობილის ბატარეას ღამის განმავლობაში დატენის.

## ღირებულება

სახლში დატენვა უფრო იაფია, ვიდრე სწრაფი დატენვის სადგური.
"""


def make_settings(**overrides: Any) -> Settings:
    """Settings from defaults + overrides only; ignores any local .env file."""
    return Settings(_env_file=None, **overrides)


def meta_fields(doc_id: str = "ev-en-0001", **overrides: Any) -> dict[str, Any]:
    topic, language, _ = doc_id.split("-")
    fields: dict[str, Any] = {
        "id": doc_id,
        "title": f"Test document {doc_id}",
        "url": f"https://corpus.researchai.invalid/{topic}/{language}/{doc_id}",
        "published": "2025-05-01",
        "snapshot": "2026-09-01",
        "language": language,
        "topic": topic,
        "source_type": "news",
        "publisher": "Test Publisher",
        "synthetic": True,
        "license": "synthetic",
    }
    fields.update(overrides)
    return fields


def make_document(doc_id: str = "ev-en-0001", body: str | None = None, **meta: Any) -> Document:
    if body is None:
        body = KA_BODY if "-ka-" in doc_id else EN_BODY
    return Document(meta=DocumentMeta.model_validate(meta_fields(doc_id, **meta)), body=body)


def _toml_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str) and len(value) == 10 and value[4] == "-" and value[7] == "-":
        return value  # TOML date literal
    return '"' + str(value).replace("\\", "\\\\").replace('"', '\\"') + '"'


def render_document(fields: dict[str, Any], body: str) -> str:
    front = "\n".join(f"{key} = {_toml_value(value)}" for key, value in fields.items())
    return f"+++\n{front}\n+++\n\n{body}"


def write_document(
    root: Path, doc_id: str = "ev-en-0001", body: str | None = None, **meta: Any
) -> Path:
    fields = meta_fields(doc_id, **meta)
    if body is None:
        body = KA_BODY if fields["language"] == "ka" else EN_BODY
    path = root / str(fields["topic"]) / str(fields["language"]) / f"{doc_id}.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_document(fields, body), encoding="utf-8")
    return path


def write_corpus(root: Path, doc_ids: tuple[str, ...] = ("ev-en-0001", "ev-ka-0001")) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    (root / "corpus.toml").write_text('version = "0.1.0"\nsnapshot = 2026-09-01\n', "utf-8")
    for doc_id in doc_ids:
        write_document(root, doc_id)
    return root
