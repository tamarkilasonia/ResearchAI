"""Pure text helpers shared by the corpus, the tools and the agent.

No I/O and no project imports, so both sides of the MCP boundary can use them.
"""

from __future__ import annotations

import re
import unicodedata

# Georgian Mkhedruli letters.
_GEORGIAN = range(0x10D0, 0x1100)
_TOKEN = re.compile(r"\w+", re.UNICODE)

MIN_TOKEN_CHARS = 2
MIN_STEM_CHARS = 3
MAX_KEYWORD_TERMS = 24

STOP_WORDS = frozenset(
    {
        # English
        "a", "an", "and", "are", "as", "at", "be", "by", "can", "do", "does", "for", "from",
        "has", "have", "how", "i", "in", "is", "it", "its", "me", "my", "of", "on", "or",
        "should", "than", "that", "the", "their", "there", "this", "to", "was", "what", "when",
        "where", "which", "who", "why", "will", "with", "would", "you", "your", "much", "many",
        "give", "tell", "about",
        # Georgian
        "და", "არის", "რა", "რომ", "თუ", "ან", "როგორ", "რამდენი", "რატომ", "სად", "როდის",
        "ეს", "ის", "მაქვს", "აქვს", "უნდა", "შეიძლება", "კი", "არა", "რომელი", "ვინ",
    }
)  # fmt: skip

# Words shared by most statements in this corpus; they say nothing about whether
# two statements concern the same quantity ("car", "electric", "Georgia", "year").
DOMAIN_COMMON = frozenset(
    {
        "electric", "car", "vehicl", "vehicle", "georgia", "georgian", "year", "month", "typical",
        "typically", "about", "around", "average", "also", "mid", "end", "first", "last", "report",
        "according", "passeng", "passenger", "market", "ელექტრომობილ", "საქართველო", "წლი",
    }
)  # fmt: skip

# Longest first. English inflections, and Georgian plural/case endings
# (plural -ებ-, genitive -ის, dative -ს, instrumental -ით, adverbial -ად,
# postpositions -ში/-ზე/-თვის/-გან, nominative -ი and stem vowels).
_EN_SUFFIXES = ("ingly", "ings", "ing", "ers", "ies", "es", "er", "ed", "s")
_KA_SUFFIXES = (
    "ებისთვის", "ებისგან", "ისთვის", "ებით", "ებში", "ებზე", "ებად", "ების", "ისგან",
    "ები", "ებს", "ში", "ზე", "ის", "ით", "ად", "ი", "ს", "ა", "ე", "ო",
)  # fmt: skip

# Inside a line, a sentence ends at terminal punctuation followed by whitespace.
_SENTENCE_END = re.compile(r"(?<=[.!?…])\s+")
_LINE = re.compile(r"[^\n]+")
_BULLET = re.compile(r"(?:[-*•]|\d+[.)])\s+")  # used with match(pos): anchored


def georgian_share(text: str) -> float:
    """Share of letters that are Georgian (0.0 when there are no letters)."""
    letters = [ch for ch in text if ch.isalpha()]
    if not letters:
        return 0.0
    return sum(ord(ch) in _GEORGIAN for ch in letters) / len(letters)


def light_stem(token: str) -> str:
    """Strip at most one known suffix, keeping at least ``MIN_STEM_CHARS``."""
    suffixes = _EN_SUFFIXES if token.isascii() else _KA_SUFFIXES
    for suffix in suffixes:
        if token.endswith(suffix) and len(token) - len(suffix) >= MIN_STEM_CHARS:
            return token[: -len(suffix)]
    return token


def keyword_terms(text: str) -> list[str]:
    """Distinct, stemmed, stop-word-free word tokens (alphanumeric only)."""
    normalized = unicodedata.normalize("NFKC", text).casefold()
    terms: list[str] = []
    for token in _TOKEN.findall(normalized):
        token = token.replace("_", "")
        if len(token) < MIN_TOKEN_CHARS or token in STOP_WORDS:
            continue
        stem = light_stem(token)
        if stem not in terms:
            terms.append(stem)
    return terms[:MAX_KEYWORD_TERMS]


def sentence_spans(text: str) -> list[tuple[int, int]]:
    """``(start, end)`` offsets of sentences, line by line, trimmed, bullets excluded.

    Every span satisfies ``text[start:end] == text[start:end].strip()`` and never
    crosses a line break, so list items and table rows stay separate.
    """
    spans: list[tuple[int, int]] = []
    for line in _LINE.finditer(text):
        raw = line.group()
        start = line.start() + len(raw) - len(raw.lstrip())
        end = line.start() + len(raw.rstrip())
        if start >= end:
            continue
        if bullet := _BULLET.match(text, start, end):
            start = bullet.end()
        cursor = start
        for match in _SENTENCE_END.finditer(text, start, end):
            spans.append((cursor, match.start()))
            cursor = match.end()
        # ``end`` is right-stripped, so text remains after the last break.
        spans.append((cursor, end))
    return spans
