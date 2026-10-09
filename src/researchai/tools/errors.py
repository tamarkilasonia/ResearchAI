"""Stable tool error codes.

Codes are part of the tool contract: clients branch on them, tests pin them,
and they never change meaning. Messages are for humans and may change.
"""

from __future__ import annotations

import json
from enum import StrEnum


class ErrorCode(StrEnum):
    INVALID_ARGUMENT = "INVALID_ARGUMENT"
    NOT_FOUND = "NOT_FOUND"
    INVALID_EXPRESSION = "INVALID_EXPRESSION"
    DIVISION_BY_ZERO = "DIVISION_BY_ZERO"
    NUMBER_TOO_LARGE = "NUMBER_TOO_LARGE"
    INDEX_NOT_READY = "INDEX_NOT_READY"
    BACKEND_UNAVAILABLE = "BACKEND_UNAVAILABLE"
    UNKNOWN_TOOL = "UNKNOWN_TOOL"
    INTERNAL = "INTERNAL"


class ToolFailure(Exception):
    """An anticipated tool failure with a stable code."""

    def __init__(self, code: ErrorCode, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message

    def to_json(self) -> str:
        return json.dumps({"code": self.code.value, "message": self.message}, ensure_ascii=False)
