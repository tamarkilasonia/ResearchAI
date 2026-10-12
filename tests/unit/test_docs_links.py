"""Relative links in the README and the docs point at files that exist."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
DOCS = [ROOT / "README.md", *sorted((ROOT / "docs").glob("*.md")),
        ROOT / "eval" / "README.md", ROOT / "corpus" / "README.md"]  # fmt: skip
LINK = re.compile(r"\]\(([^)\s]+)\)")


@pytest.mark.parametrize("document", DOCS, ids=lambda p: str(p.relative_to(ROOT)))
def test_relative_links_resolve(document: Path) -> None:
    broken = []
    for target in LINK.findall(document.read_text(encoding="utf-8")):
        if target.startswith(("http://", "https://", "mailto:", "#")):
            continue
        path = target.split("#", 1)[0]
        if not (document.parent / path).exists():
            broken.append(target)
    assert broken == [], f"{document.name}: {broken}"
