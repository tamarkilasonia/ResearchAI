from __future__ import annotations

import pytest
from pydantic import ValidationError

from researchai.config import (
    AppEnv,
    EmbeddingProvider,
    LLMProvider,
    LogFormat,
    Settings,
    get_settings,
)
from tests.support import make_settings as make


def test_defaults_are_offline_and_deterministic() -> None:
    settings = make()
    assert settings.llm_provider is LLMProvider.NONE
    assert settings.offline
    assert settings.embedding_provider is EmbeddingProvider.HASH
    assert settings.app_env is AppEnv.DEV
    assert settings.log_format is LogFormat.JSON


def test_reads_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "ollama")
    monkeypatch.setenv("EMBEDDING_DIM", "768")
    monkeypatch.setenv("LOG_LEVEL", "DEBUG")
    settings = Settings(_env_file=None)
    assert settings.llm_provider is LLMProvider.OLLAMA
    assert not settings.offline
    assert settings.embedding_dim == 768
    assert settings.log_level == "DEBUG"


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("llm_provider", "gpt"),
        ("embedding_provider", "openai"),
        ("embedding_dim", "4"),
        ("embedding_dim", "99999"),
        ("llm_timeout_seconds", "0"),
        ("log_level", "TRACE"),
        ("database_url", "sqlite:///x.db"),
        ("database_url", "postgresql://u:p@h/db"),
        ("llm_base_url", "localhost:11434"),
    ],
)
def test_rejects_invalid_values(field: str, value: str) -> None:
    with pytest.raises(ValidationError):
        make(**{field: value})


def test_base_url_trailing_slash_is_stripped() -> None:
    assert make(llm_base_url="http://x:1/v1/").llm_base_url == "http://x:1/v1"


def test_api_key_is_never_rendered() -> None:
    settings = make(llm_api_key="sk-very-secret")
    assert "sk-very-secret" not in repr(settings)
    assert "sk-very-secret" not in str(settings.model_dump())
    assert settings.llm_api_key.get_secret_value() == "sk-very-secret"


def test_settings_are_immutable() -> None:
    settings = make()
    with pytest.raises(ValidationError):
        settings.llm_model = "other"  # type: ignore[misc]


def test_get_settings_is_cached() -> None:
    get_settings.cache_clear()
    try:
        assert get_settings() is get_settings()
    finally:
        get_settings.cache_clear()


def test_live_web_search_requires_url() -> None:
    with pytest.raises(ValidationError, match="LIVE_WEB_SEARCH_URL"):
        make(live_web_search_enabled="true")
    assert make(live_web_search_enabled="true", live_web_search_url="http://searx:8080")


def test_mcp_allowed_hosts_are_split() -> None:
    settings = make(mcp_allowed_hosts=" mcp-server:* , localhost:*,, ")
    assert settings.mcp_allowed_host_list == ["mcp-server:*", "localhost:*"]
    with pytest.raises(ValidationError):
        make(mcp_server_url="mcp-server:8765/mcp")
