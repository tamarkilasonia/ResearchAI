"""Locate a quoted span inside an evidence passage (exact, then normalised)."""

from __future__ import annotations

import unicodedata

from researchai.agent.models import Evidence

_QUOTE_FOLD = str.maketrans(
    {
        "\u2019": "'",  # right single quotation mark
        "\u2018": "'",  # left single quotation mark
        "\u201c": '"',  # left double quotation mark
        "\u201d": '"',  # right double quotation mark
        "\u2013": "-",  # en dash
        "\u2014": "-",  # em dash
    }
)


def _normalize_with_map(text: str) -> tuple[str, list[int]]:
    """Casefolded, whitespace-collapsed text plus a map from each char to its source index."""
    out: list[str] = []
    index: list[int] = []
    previous_space = True
    for i, ch in enumerate(text):
        folded = unicodedata.normalize("NFKC", ch).translate(_QUOTE_FOLD).casefold()
        if folded.isspace() or not folded:
            if not previous_space:
                out.append(" ")
                index.append(i)
            previous_space = True
            continue
        for part in folded:
            out.append(part)
            index.append(i)
        previous_space = False
    return "".join(out).rstrip(), index


def anchor_quote(evidence: Evidence, quote: str) -> tuple[int, int] | None:
    """Absolute document offsets of ``quote`` inside the evidence passage, if present."""
    quote = quote.strip()
    if not quote:
        return None
    position = evidence.text.find(quote)
    if position >= 0:
        return evidence.char_start + position, evidence.char_start + position + len(quote)
    haystack, mapping = _normalize_with_map(evidence.text)
    needle, _ = _normalize_with_map(quote)
    position = haystack.find(needle) if needle else -1
    if position < 0:
        return None
    start = mapping[position]
    end = mapping[position + len(needle) - 1] + 1
    return evidence.char_start + start, evidence.char_start + end
