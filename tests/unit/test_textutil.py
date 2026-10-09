from __future__ import annotations

from hypothesis import given
from hypothesis import strategies as st

from researchai.textutil import sentence_spans


def test_sentences_bullets_and_tables() -> None:
    text = "First one. Second!\n\n- Bullet item\n1. Numbered. Two\n| a | b |\nლარი 0.55. ბოლო"
    assert [text[s:e] for s, e in sentence_spans(text)] == [
        "First one.",
        "Second!",
        "Bullet item",
        "Numbered.",
        "Two",
        "| a | b |",
        "ლარი 0.55.",
        "ბოლო",
    ]


@given(st.text(alphabet="ab .!?\n-1)ქ\t", max_size=200))
def test_spans_are_trimmed_ordered_and_single_line(text: str) -> None:
    previous = 0
    for start, end in sentence_spans(text):
        piece = text[start:end]
        assert piece
        assert piece == piece.strip()
        assert "\n" not in piece
        assert start >= previous
        previous = end
