from __future__ import annotations

import contextlib
import math

import pytest
from hypothesis import given
from hypothesis import strategies as st

from researchai.tools.calculator import MAX_MAGNITUDE, evaluate
from researchai.tools.errors import ErrorCode, ToolFailure


@pytest.mark.parametrize(
    ("expression", "expected"),
    [
        ("16.8 * 0.292", 4.9056),
        ("(26000 - 21000) / 21000 * 100", 23.80952380952381),
        ("2 ** 10", 1024.0),
        ("-3 + +4", 1.0),
        ("7 // 2 + 7 % 2", 4.0),
        ("min(4.91, 8.4, 14.28)", 4.91),
        ("max(1, 2)", 2.0),
        ("abs(-2.5)", 2.5),
        ("round(25.315, 2)", 25.32),
        ("round(4.6)", 5.0),
        ("1_000 * 3", 3000.0),
        ("  2*3  ", 6.0),
    ],
)
def test_valid_expressions(expression: str, expected: float) -> None:
    assert evaluate(expression) == pytest.approx(expected)


@pytest.mark.parametrize(
    ("expression", "code"),
    [
        ("", ErrorCode.INVALID_EXPRESSION),
        ("   ", ErrorCode.INVALID_EXPRESSION),
        ("1 +", ErrorCode.INVALID_EXPRESSION),
        ("__import__('os').system('ls')", ErrorCode.INVALID_EXPRESSION),
        ("x + 1", ErrorCode.INVALID_EXPRESSION),
        ("'a' * 3", ErrorCode.INVALID_EXPRESSION),
        ("True + 1", ErrorCode.INVALID_EXPRESSION),
        ("[1, 2]", ErrorCode.INVALID_EXPRESSION),
        ("(1).real", ErrorCode.INVALID_EXPRESSION),
        ("max(1)", ErrorCode.INVALID_EXPRESSION),
        ("round(1.5, 2.5)", ErrorCode.INVALID_EXPRESSION),
        ("round(1.5, 11)", ErrorCode.INVALID_EXPRESSION),
        ("round(x=1)", ErrorCode.INVALID_EXPRESSION),
        ("math.sqrt(4)", ErrorCode.INVALID_EXPRESSION),
        ("1 < 2", ErrorCode.INVALID_EXPRESSION),
        ("1e309", ErrorCode.INVALID_EXPRESSION),
        ("(-8) ** 0.5", ErrorCode.INVALID_EXPRESSION),
        ("1" + "+1" * 60, ErrorCode.INVALID_EXPRESSION),  # short, but > 100 syntax nodes
        ("1" * 201, ErrorCode.INVALID_EXPRESSION),
        ("1 / 0", ErrorCode.DIVISION_BY_ZERO),
        ("5 // (2 - 2)", ErrorCode.DIVISION_BY_ZERO),
        ("5 % 0", ErrorCode.DIVISION_BY_ZERO),
        ("2 ** 65", ErrorCode.NUMBER_TOO_LARGE),
        ("10 ** 16", ErrorCode.NUMBER_TOO_LARGE),
        ("1e16", ErrorCode.NUMBER_TOO_LARGE),
        ("1e300 ** 2", ErrorCode.NUMBER_TOO_LARGE),
        ("abs(-1e16)", ErrorCode.NUMBER_TOO_LARGE),
    ],
)
def test_rejected_expressions(expression: str, code: ErrorCode) -> None:
    with pytest.raises(ToolFailure) as excinfo:
        evaluate(expression)
    assert excinfo.value.code is code


def test_overflow_is_reported() -> None:
    with pytest.raises(ToolFailure) as excinfo:
        evaluate("9e14 ** 30")
    assert excinfo.value.code is ErrorCode.NUMBER_TOO_LARGE


_numbers = st.one_of(
    st.integers(min_value=-1000, max_value=1000).map(str),
    st.floats(min_value=0.001, max_value=1000, allow_nan=False).map(lambda f: f"{f:.3f}"),
)


def _expressions() -> st.SearchStrategy[str]:
    return st.recursive(
        _numbers,
        lambda inner: st.builds(
            lambda a, op, b: f"({a} {op} {b})", inner, st.sampled_from(["+", "-", "*"]), inner
        ),
        max_leaves=8,
    )


@given(_expressions())
def test_matches_python_arithmetic(expression: str) -> None:
    expected = float(eval(expression))  # noqa: S307 - generated digits and operators only
    if abs(expected) > MAX_MAGNITUDE:
        return
    assert math.isclose(evaluate(expression), expected, rel_tol=1e-9, abs_tol=1e-9)


@given(st.text(max_size=250))
def test_arbitrary_text_fails_only_with_tool_failures(text: str) -> None:
    with contextlib.suppress(ToolFailure):
        evaluate(text)
