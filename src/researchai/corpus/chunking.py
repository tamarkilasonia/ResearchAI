"""Section-aware chunking with exact character spans.

A chunk never crosses a Markdown heading. Inside a section, paragraphs are
packed greedily up to ``max_chars``; a paragraph that is too long is split at
sentence ends, and a sentence that is still too long is split at whitespace.

Every chunk is a contiguous slice ``body[start:end]`` of the document body, so
a citation can always be traced back to exact offsets in the source. Heading
lines are not part of any chunk; they are carried in ``Chunk.section``.
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from dataclasses import dataclass

from researchai.corpus.models import Chunk, Document

MAX_CHARS = 900

_HEADING = re.compile(r"^(#{1,6})[ \t]+(.+?)[ \t#]*$")
_PARAGRAPH_BREAK = re.compile(r"\n[ \t]*\n")
# Sentence end: terminal punctuation followed by whitespace. Georgian uses "." too.
_SENTENCE_END = re.compile(r"(?<=[.!?…])\s+")


@dataclass(frozen=True)
class Span:
    start: int
    end: int


@dataclass(frozen=True)
class Section:
    path: tuple[str, ...]
    start: int
    end: int


def split_sections(body: str) -> list[Section]:
    """Split a body into heading-delimited sections (heading lines excluded)."""
    sections: list[Section] = []
    stack: list[tuple[int, str]] = []
    path: tuple[str, ...] = ()
    start = 0
    offset = 0
    for line in body.splitlines(keepends=True):
        match = _HEADING.match(line.rstrip("\n"))
        if match:
            sections.append(Section(path, start, offset))
            level = len(match.group(1))
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, match.group(2).strip()))
            path = tuple(title for _, title in stack)
            start = offset + len(line)
        offset += len(line)
    sections.append(Section(path, start, len(body)))
    return [s for s in sections if body[s.start : s.end].strip()]


def _trim(body: str, span: Span) -> Span | None:
    text = body[span.start : span.end]
    stripped = text.strip()
    if not stripped:
        return None
    lead = len(text) - len(text.lstrip())
    return Span(span.start + lead, span.start + lead + len(stripped))


def _split_by(body: str, span: Span, pattern: re.Pattern[str]) -> list[Span]:
    pieces: list[Span] = []
    cursor = span.start
    for match in pattern.finditer(body, span.start, span.end):
        if (piece := _trim(body, Span(cursor, match.start()))) is not None:
            pieces.append(piece)
        cursor = match.end()
    if (piece := _trim(body, Span(cursor, span.end))) is not None:
        pieces.append(piece)
    return pieces


def _hard_split(body: str, span: Span, max_chars: int) -> Iterator[Span]:
    """Split at the last whitespace before ``max_chars``, or mid-word if none.

    ``span`` is trimmed, so every piece starts and ends on a non-space character.
    """
    start = span.start
    while span.end - start > max_chars:
        window = body[start : start + max_chars]
        cut = max(window.rfind(" "), window.rfind("\n"), window.rfind("\t"))
        end = start + cut if cut > 0 else start + max_chars
        yield Span(start, start + len(body[start:end].rstrip()))
        start = end
        while body[start].isspace():
            start += 1
    yield Span(start, span.end)


def _units(body: str, section: Section, max_chars: int) -> Iterator[Span]:
    """Smallest packable units of a section, each at most ``max_chars`` long."""
    for paragraph in _split_by(body, Span(section.start, section.end), _PARAGRAPH_BREAK):
        if paragraph.end - paragraph.start <= max_chars:
            yield paragraph
            continue
        for sentence in _split_by(body, paragraph, _SENTENCE_END):
            if sentence.end - sentence.start <= max_chars:
                yield sentence
            else:
                yield from _hard_split(body, sentence, max_chars)


def _pack(units: list[Span], max_chars: int) -> list[Span]:
    """Greedily merge consecutive units while the merged span fits."""
    packed: list[Span] = []
    for unit in units:
        if packed and unit.end - packed[-1].start <= max_chars:
            packed[-1] = Span(packed[-1].start, unit.end)
        else:
            packed.append(unit)
    return packed


def chunk_document(document: Document, *, max_chars: int = MAX_CHARS) -> list[Chunk]:
    if max_chars < 1:
        raise ValueError("max_chars must be at least 1")
    body = document.body
    chunks: list[Chunk] = []
    for section in split_sections(body):
        for span in _pack(list(_units(body, section, max_chars)), max_chars):
            ordinal = len(chunks)
            chunks.append(
                Chunk(
                    id=f"{document.id}#c{ordinal}",
                    doc_id=document.id,
                    ordinal=ordinal,
                    section=" > ".join(section.path),
                    text=body[span.start : span.end],
                    start=span.start,
                    end=span.end,
                )
            )
    return chunks
