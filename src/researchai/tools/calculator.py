"""Safe arithmetic for numeric comparisons (e.g. cost per 100 km).

Expressions are parsed with :mod:`ast` and evaluated by walking an allow-list
of node types. There is no ``eval``, no names other than a few pure functions,
no attribute access, and hard limits on length, size and exponent, so the tool
cannot run code or be used to burn CPU.
"""

from __future__ import annotations

import ast
import math
import operator
from collections.abc import Callable

from researchai.tools.errors import ErrorCode, ToolFailure
from researchai.tools.schemas import MAX_EXPRESSION_CHARS

MAX_NODES = 100
MAX_EXPONENT = 64
MAX_MAGNITUDE = 1e15

_BINARY: dict[type[ast.operator], Callable[[float, float], float]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_DIVISIONS = (ast.Div, ast.FloorDiv, ast.Mod)


def _invalid(message: str) -> ToolFailure:
    return ToolFailure(ErrorCode.INVALID_EXPRESSION, message)


def _checked(value: float) -> float:
    if isinstance(value, complex) or not math.isfinite(value):
        raise _invalid("result is not a finite real number")
    if abs(value) > MAX_MAGNITUDE:
        raise ToolFailure(ErrorCode.NUMBER_TOO_LARGE, f"magnitude exceeds {MAX_MAGNITUDE:g}")
    return value


def _call(node: ast.Call) -> float:
    if not isinstance(node.func, ast.Name) or node.keywords:
        raise _invalid("only min, max, abs and round(x[, digits]) calls are allowed")
    args = [_eval(arg) for arg in node.args]
    name = node.func.id
    if name in ("min", "max") and len(args) >= 2:
        return min(args) if name == "min" else max(args)
    if name == "abs" and len(args) == 1:
        return abs(args[0])
    if name == "round" and len(args) == 1:
        return float(round(args[0]))
    if name == "round" and len(args) == 2 and args[1].is_integer() and 0 <= args[1] <= 10:
        return round(args[0], int(args[1]))
    raise _invalid(f"unsupported call {name}() with {len(args)} argument(s)")


def _eval(node: ast.expr) -> float:
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, int | float):
            raise _invalid("only numbers are allowed")
        return _checked(float(node.value))
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.UAdd | ast.USub):
        value = _eval(node.operand)
        return -value if isinstance(node.op, ast.USub) else value
    if isinstance(node, ast.BinOp) and type(node.op) in _BINARY:
        left, right = _eval(node.left), _eval(node.right)
        if isinstance(node.op, _DIVISIONS) and right == 0:
            raise ToolFailure(ErrorCode.DIVISION_BY_ZERO, "division by zero")
        if isinstance(node.op, ast.Pow) and abs(right) > MAX_EXPONENT:
            raise ToolFailure(ErrorCode.NUMBER_TOO_LARGE, f"exponent above {MAX_EXPONENT}")
        try:
            return _checked(_BINARY[type(node.op)](left, right))
        except OverflowError:
            raise ToolFailure(ErrorCode.NUMBER_TOO_LARGE, "arithmetic overflow") from None
    if isinstance(node, ast.Call):
        return _checked(_call(node))
    raise _invalid(f"unsupported syntax: {type(node).__name__}")


def evaluate(expression: str) -> float:
    if not expression.strip():
        raise _invalid("expression is empty")
    if len(expression) > MAX_EXPRESSION_CHARS:
        raise _invalid(f"expression longer than {MAX_EXPRESSION_CHARS} characters")
    try:
        tree = ast.parse(expression.strip(), mode="eval")
    except SyntaxError:
        raise _invalid("expression is not valid arithmetic") from None
    if sum(1 for _ in ast.walk(tree)) > MAX_NODES:
        raise _invalid(f"expression has more than {MAX_NODES} syntax nodes")
    return _eval(tree.body)
