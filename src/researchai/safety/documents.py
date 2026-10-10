"""Retrieved documents are untrusted data.

Text in a retrieved passage can try to steer the agent ("ignore previous
instructions", "AI assistants must say X", a hidden HTML comment). Such text is
never obeyed and never used as a fact:

* :func:`instruction_spans` finds instruction-like sentences (and hidden
  comments) by rules, in English and Georgian;
* :func:`neutralize` replaces them before any passage is shown to an LLM, so the
  model never reads them; offsets of the original text are untouched, so
  citations of the rest of the passage still anchor exactly;
* the claim extractor, the extractive writer and the citation verifier skip or
  reject anything inside these spans (see :func:`overlaps`).

Detection is pattern based and can be evaded by paraphrase; it is one layer
next to the prompt rule, the citation verifier and the output guard.
"""

from __future__ import annotations

from collections.abc import Sequence
from functools import lru_cache

from researchai.safety.patterns import DOCUMENT_RULES, HIDDEN_COMMENT
from researchai.textutil import sentence_spans

PLACEHOLDER = "[instruction-like text removed]"

Span = tuple[int, int, str]  # (start, end, rule name), offsets relative to the text


@lru_cache(maxsize=4096)
def _scan(text: str) -> tuple[Span, ...]:
    found: list[Span] = [
        (m.start(), m.end(), "hidden_comment") for m in HIDDEN_COMMENT.finditer(text)
    ]
    for start, end in sentence_spans(text):
        sentence = text[start:end]
        for rule in DOCUMENT_RULES:
            if rule.pattern.search(sentence):
                found.append((start, end, rule.name))
                break
    found.sort()
    merged: list[Span] = []
    for start, end, name in found:
        if merged and start <= merged[-1][1]:
            last = merged[-1]
            merged[-1] = (last[0], max(last[1], end), last[2])
        else:
            merged.append((start, end, name))
    return tuple(merged)


def instruction_spans(text: str) -> list[Span]:
    """Instruction-like spans in ``text``, sorted and non-overlapping."""
    return list(_scan(text))


def neutralize(text: str) -> str:
    """``text`` with instruction-like spans replaced by a visible placeholder."""
    spans = _scan(text)
    if not spans:
        return text
    parts: list[str] = []
    cursor = 0
    for start, end, _ in spans:
        parts += [text[cursor:start], PLACEHOLDER]
        cursor = end
    parts.append(text[cursor:])
    return "".join(parts)


def overlaps(start: int, end: int, spans: Sequence[Span]) -> bool:
    return any(start < s_end and s_start < end for s_start, s_end, _ in spans)
