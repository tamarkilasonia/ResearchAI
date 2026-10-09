"""Load and validate the frozen corpus.

Layout::

    corpus/
      corpus.toml                  version and snapshot date
      manifest.json                hashes of every document (frozen check)
      <topic>/<language>/<id>.md   one document per file

Each document starts with TOML front matter between ``+++`` lines. Validation
collects every problem instead of stopping at the first, so a corpus author
sees all mistakes in one run.
"""

from __future__ import annotations

import hashlib
import json
import tomllib
import unicodedata
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from researchai.corpus.models import Document, DocumentMeta, Language

FRONT_MATTER_FENCE = "+++"
BOM = "\ufeff"
CORPUS_FILE = "corpus.toml"
MANIFEST_FILE = "manifest.json"

# Georgian Mkhedruli letters. Used to sanity-check the declared language.
_GEORGIAN = range(0x10D0, 0x1100)
MIN_GEORGIAN_SHARE_KA = 0.5
MAX_GEORGIAN_SHARE_EN = 0.05


class CorpusError(Exception):
    def __init__(self, problems: list[str]) -> None:
        self.problems = problems
        super().__init__(f"{len(problems)} corpus problem(s):\n" + "\n".join(problems))


@dataclass(frozen=True)
class Corpus:
    version: str
    snapshot: date
    documents: tuple[Document, ...]

    def get(self, doc_id: str) -> Document:
        for document in self.documents:
            if document.id == doc_id:
                return document
        raise KeyError(doc_id)

    def manifest(self) -> dict[str, Any]:
        docs = {d.id: d.sha256 for d in sorted(self.documents, key=lambda d: d.id)}
        digest = hashlib.sha256(json.dumps(docs, sort_keys=True).encode()).hexdigest()
        return {
            "version": self.version,
            "snapshot": self.snapshot.isoformat(),
            "document_count": len(docs),
            "sha256": digest,
            "documents": docs,
        }


def parse_document(text: str) -> Document:
    """Parse one file. Raises ``ValueError`` (incl. pydantic errors) on bad input.

    Text is normalised to NFC with LF line endings so that character offsets
    (chunk spans, citation spans) are stable across platforms and editors.
    """
    text = unicodedata.normalize("NFC", text.removeprefix(BOM).replace("\r\n", "\n"))
    lines = text.split("\n")
    if not lines or lines[0].strip() != FRONT_MATTER_FENCE:
        raise ValueError("missing opening +++ front matter fence")
    try:
        closing = next(
            i for i, line in enumerate(lines[1:], start=1) if line.strip() == FRONT_MATTER_FENCE
        )
    except StopIteration:
        raise ValueError("missing closing +++ front matter fence") from None
    meta = DocumentMeta.model_validate(tomllib.loads("\n".join(lines[1:closing])))
    body = "\n".join(lines[closing + 1 :]).strip("\n")
    if not body.strip():
        raise ValueError("document body is empty")
    return Document(meta=meta, body=body)


def georgian_share(text: str) -> float:
    letters = [ch for ch in text if ch.isalpha()]
    if not letters:
        return 0.0
    return sum(ord(ch) in _GEORGIAN for ch in letters) / len(letters)


def _check_language(document: Document) -> str | None:
    share = georgian_share(document.body)
    if document.meta.language is Language.KA and share < MIN_GEORGIAN_SHARE_KA:
        return f"declared 'ka' but only {share:.0%} of letters are Georgian"
    if document.meta.language is Language.EN and share > MAX_GEORGIAN_SHARE_EN:
        return f"declared 'en' but {share:.0%} of letters are Georgian"
    return None


def load_corpus(root: Path) -> Corpus:
    problems: list[str] = []
    try:
        config = tomllib.loads((root / CORPUS_FILE).read_text(encoding="utf-8"))
        version, snapshot = str(config["version"]), config["snapshot"]
        if not isinstance(snapshot, date):
            raise TypeError("snapshot must be a TOML date")
    except (OSError, KeyError, TypeError, tomllib.TOMLDecodeError) as exc:
        raise CorpusError([f"{root / CORPUS_FILE}: {exc}"]) from exc

    documents: list[Document] = []
    seen: dict[str, Path] = {}
    for path in sorted(root.glob("*/*/*.md")):
        rel = path.relative_to(root)
        try:
            document = parse_document(path.read_text(encoding="utf-8"))
        except (ValueError, ValidationError, tomllib.TOMLDecodeError) as exc:
            problems.append(f"{rel}: {exc}")
            continue
        meta = document.meta
        expected = Path(meta.topic.value, meta.language.value, f"{meta.id}.md")
        if rel != expected:
            problems.append(f"{rel}: should be at {expected}")
        if meta.id in seen:
            problems.append(f"{rel}: duplicate id {meta.id} (also {seen[meta.id]})")
        seen[meta.id] = rel
        if meta.snapshot > snapshot:
            problems.append(f"{rel}: snapshot {meta.snapshot} is after corpus snapshot {snapshot}")
        if (language_problem := _check_language(document)) is not None:
            problems.append(f"{rel}: {language_problem}")
        documents.append(document)

    if not documents and not problems:
        problems.append(f"{root}: no documents found")
    if problems:
        raise CorpusError(problems)
    return Corpus(version=version, snapshot=snapshot, documents=tuple(documents))


def read_manifest(root: Path) -> dict[str, Any]:
    data: dict[str, Any] = json.loads((root / MANIFEST_FILE).read_text(encoding="utf-8"))
    return data


def write_manifest(root: Path, corpus: Corpus) -> Path:
    path = root / MANIFEST_FILE
    path.write_text(json.dumps(corpus.manifest(), indent=2, sort_keys=True) + "\n", "utf-8")
    return path
