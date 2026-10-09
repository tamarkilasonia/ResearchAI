"""Shared test helpers."""

from __future__ import annotations

from typing import Any

from researchai.config import Settings


def make_settings(**overrides: Any) -> Settings:
    """Settings from defaults + overrides only; ignores any local .env file."""
    return Settings(_env_file=None, **overrides)
