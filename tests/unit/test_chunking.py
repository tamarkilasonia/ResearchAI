from __future__ import annotations

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from researchai.corpus.chunking import chunk_document, split_sections
from tests.support import EN_BODY, KA_BODY, make_document


def test_sections_follow_heading_hierarchy() -> None:
    body = "Intro text.\n\n# A\n\nText A.\n\n## A1\n\nText A1.\n\n# B\n\nText B.\n### B deep\nDeep."
    paths = [section.path for section in split_sections(body)]
    assert paths == [(), ("A",), ("A", "A1"), ("B",), ("B", "B deep")]


def test_empty_sections_are_dropped() -> None:
    assert [s.path for s in split_sections("# A\n\n# B\ntext")] == [("B",)]


def test_chunks_are_exact_slices_with_sections() -> None:
    document = make_document(body=EN_BODY.strip())
    chunks = chunk_document(document)
    assert [c.section for c in chunks] == ["Charging at home", "Charging at home > Cost"]
    for chunk in chunks:
        assert document.body[chunk.start : chunk.end] == chunk.text
        assert not chunk.text.startswith("#")
    assert [c.id for c in chunks] == ["ev-en-0001#c0", "ev-en-0001#c1"]


def test_georgian_document_chunks() -> None:
    chunks = chunk_document(make_document("ev-ka-0001", body=KA_BODY.strip()))
    assert chunks[0].section == "სახლში დატენვა"
    assert "wallbox" in chunks[0].text


def test_paragraphs_are_packed_up_to_limit() -> None:
    body = "\n\n".join(f"Paragraph number {i} has some words." for i in range(10))
    chunks = chunk_document(make_document(body=body), max_chars=120)
    assert len(chunks) > 1
    assert all(len(c.text) <= 120 for c in chunks)
    assert "\n\n" in chunks[0].text  # several paragraphs in one chunk


def test_long_paragraph_splits_at_sentences() -> None:
    body = " ".join(f"Sentence {i} is here." for i in range(30))
    chunks = chunk_document(make_document(body=body), max_chars=100)
    assert all(c.text.endswith(".") for c in chunks)
    assert all(len(c.text) <= 100 for c in chunks)


def test_unbroken_text_is_hard_split() -> None:
    body = "x" * 250
    chunks = chunk_document(make_document(body=body), max_chars=100)
    assert [len(c.text) for c in chunks] == [100, 100, 50]


def test_long_sentence_splits_at_whitespace() -> None:
    body = " ".join(["word"] * 60)  # one 299-char sentence
    chunks = chunk_document(make_document(body=body), max_chars=50)
    assert all(len(c.text) <= 50 and not c.text.endswith(" ") for c in chunks)
    assert all(set(c.text.split()) == {"word"} for c in chunks)


def test_packing_is_greedy() -> None:
    body = ("a" * 60) + "\n\n" + ("b" * 50) + "\n\n" + "c" * 10
    assert len(chunk_document(make_document(body=body), max_chars=130)) == 1
    chunks = chunk_document(make_document(body=body), max_chars=100)
    assert [len(c.text) for c in chunks] == [60, 62]


def test_multiple_spaces_at_hard_split_boundary() -> None:
    body = "abc" + " " * 5 + "x" * 20
    chunks = chunk_document(make_document(body=body), max_chars=6)
    assert [c.text for c in chunks] == ["abc", "xxxxxx", "xxxxxx", "xxxxxx", "xx"]


@pytest.mark.parametrize("max_chars", [0, -5])
def test_invalid_limits(max_chars: int) -> None:
    with pytest.raises(ValueError, match="max_chars"):
        chunk_document(make_document(), max_chars=max_chars)


_line = st.one_of(
    st.text(alphabet="abc ქართ.!?\t", min_size=0, max_size=80),
    st.builds(lambda n, t: "#" * n + " " + t, st.integers(1, 3), st.text("xyზ ", max_size=10)),
    st.just(""),
)
_body = st.lists(_line, min_size=1, max_size=40).map("\n".join).filter(lambda b: b.strip() != "")


@settings(max_examples=300, suppress_health_check=[HealthCheck.filter_too_much])
@given(body=_body, max_chars=st.integers(1, 200))
def test_chunk_properties(body: str, max_chars: int) -> None:
    document = make_document(body=body)
    chunks = chunk_document(document, max_chars=max_chars)
    previous_end = 0
    for i, chunk in enumerate(chunks):
        assert chunk.ordinal == i
        assert document.body[chunk.start : chunk.end] == chunk.text  # exact slice
        assert chunk.text == chunk.text.strip()  # trimmed
        assert len(chunk.text) <= max_chars  # bounded
        assert chunk.start >= previous_end  # ordered, non-overlapping
        previous_end = chunk.end
    # Every non-whitespace character outside heading lines is covered by a chunk.
    covered: set[int] = set()
    for chunk in chunks:
        covered.update(range(chunk.start, chunk.end))
    for section in split_sections(document.body):
        for offset in range(section.start, section.end):
            if not document.body[offset].isspace():
                assert offset in covered
