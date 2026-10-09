"""`.env.example` must document every setting, and nothing that is not one."""

from __future__ import annotations

from pathlib import Path

from researchai.config import Settings
from tests.support import make_settings

ENV_EXAMPLE = Path(__file__).resolve().parents[2] / ".env.example"

# Read by docker compose, not by the application.
COMPOSE_ONLY = {
    "POSTGRES_USER",
    "POSTGRES_PASSWORD",
    "POSTGRES_DB",
    "LLM_BASE_URL_DOCKER",
    "EMBEDDING_BASE_URL_DOCKER",
    "DB_HOST_PORT",
    "API_HOST_PORT",
}


def _entries() -> dict[str, str]:
    """Map each KEY= line to the comment line directly above it ('' if none)."""
    entries: dict[str, str] = {}
    previous = ""
    for raw in ENV_EXAMPLE.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if line and not line.startswith("#") and "=" in line:
            entries[line.split("=", 1)[0]] = previous
        previous = line if line.startswith("#") else ""
    return entries


def test_every_setting_is_listed() -> None:
    expected = {name.upper() for name in Settings.model_fields}
    missing = expected - _entries().keys()
    assert not missing, f"add to .env.example: {sorted(missing)}"


def test_no_unknown_keys() -> None:
    known = {name.upper() for name in Settings.model_fields} | COMPOSE_ONLY
    unknown = _entries().keys() - known
    assert not unknown, f".env.example lists keys nothing reads: {sorted(unknown)}"


def test_every_key_has_a_comment() -> None:
    undocumented = [key for key, comment in _entries().items() if len(comment) < 10]
    assert not undocumented, f"add a comment above: {undocumented}"


def test_example_values_parse() -> None:
    values = {
        key.lower(): value
        for key, value in (
            line.split("=", 1)
            for line in ENV_EXAMPLE.read_text(encoding="utf-8").splitlines()
            if line and not line.startswith("#") and "=" in line
        )
        if key not in COMPOSE_ONLY
    }
    settings = make_settings(**values)
    assert settings.offline


def test_example_contains_no_real_secret() -> None:
    assert _entries().keys() >= {"LLM_API_KEY"}
    for line in ENV_EXAMPLE.read_text(encoding="utf-8").splitlines():
        if line.startswith("LLM_API_KEY="):
            assert line == "LLM_API_KEY="
