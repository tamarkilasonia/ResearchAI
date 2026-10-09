"""Import boundaries.

The agent must reach data only through MCP. Packages on the agent side may use
the tool *contract* (schemas, error codes, pure corpus models) but never the
database layer, the tool implementations or ingestion code.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[2] / "src"

AGENT_SIDE = ("researchai.agent", "researchai.mcp_client")
FORBIDDEN = (
    "researchai.db",
    "researchai.tools",
    "researchai.mcp_server",
    "researchai.corpus.ingest",
    "researchai.corpus.embeddings",
    "researchai.corpus.loader",
    "sqlalchemy",
    "asyncpg",
)
ALLOWED = ("researchai.tools.schemas", "researchai.tools.errors")


def _module_name(path: Path) -> str:
    parts = path.relative_to(SRC).with_suffix("").parts
    return ".".join(parts[:-1] if parts[-1] == "__init__" else parts)


def _imports(path: Path) -> set[str]:
    names: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text("utf-8"))):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            names.add(node.module)
            names.update(f"{node.module}.{alias.name}" for alias in node.names)
    return names


def _violates(name: str) -> bool:
    if any(name == a or name.startswith(f"{a}.") for a in ALLOWED):
        return False
    return any(name == f or name.startswith(f"{f}.") for f in FORBIDDEN)


def _agent_side_modules() -> list[Path]:
    return [path for path in sorted(SRC.rglob("*.py")) if _module_name(path).startswith(AGENT_SIDE)]


def test_agent_side_exists() -> None:
    assert _agent_side_modules(), "boundary test found nothing to check"


@pytest.mark.parametrize("path", _agent_side_modules(), ids=_module_name)
def test_agent_side_never_imports_data_access(path: Path) -> None:
    bad = sorted(name for name in _imports(path) if _violates(name))
    assert not bad, f"{_module_name(path)} imports {bad}; go through MCP instead"


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("researchai.db.models", True),
        ("researchai.tools.search", True),
        ("researchai.tools.schemas", False),
        ("researchai.tools.errors.ErrorCode", False),
        ("researchai.corpus.models", False),
        ("sqlalchemy.ext.asyncio", True),
        ("researchai.dbx", False),
    ],
)
def test_rule_matching(name: str, expected: bool) -> None:
    assert _violates(name) is expected
