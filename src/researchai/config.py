"""Application settings.

Every field is read from an environment variable of the same name in upper case
(``llm_provider`` -> ``LLM_PROVIDER``). Every field must be documented in
``.env.example``; ``tests/unit/test_env_example.py`` enforces that.
"""

from __future__ import annotations

from enum import StrEnum
from functools import lru_cache
from typing import Literal

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class AppEnv(StrEnum):
    DEV = "dev"
    TEST = "test"
    PROD = "prod"


class LogFormat(StrEnum):
    JSON = "json"
    TEXT = "text"


class LLMProvider(StrEnum):
    """Where completions come from.

    ``none`` is the offline mode: the agent runs on rules and templates and never
    opens a network connection, so tests and CI need no model and no API key.
    """

    NONE = "none"
    OLLAMA = "ollama"
    OPENAI_COMPATIBLE = "openai_compatible"


class EmbeddingProvider(StrEnum):
    """Where embeddings come from.

    ``hash`` is a deterministic, offline embedder (feature hashing). It is only
    meant for tests and CI; retrieval quality numbers are reported with ``ollama``.
    """

    HASH = "hash"
    OLLAMA = "ollama"


LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR"]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        frozen=True,
    )

    # --- runtime ---
    app_env: AppEnv = AppEnv.DEV
    log_level: LogLevel = "INFO"
    log_format: LogFormat = LogFormat.JSON

    # --- LLM ---
    llm_provider: LLMProvider = LLMProvider.NONE
    llm_base_url: str = "http://localhost:11434/v1"
    llm_model: str = "qwen2.5:7b"
    llm_api_key: SecretStr = SecretStr("")
    llm_timeout_seconds: float = Field(default=120.0, gt=0, le=600)
    judge_model: str = "llama3.1:8b"

    # --- embeddings ---
    embedding_provider: EmbeddingProvider = EmbeddingProvider.HASH
    embedding_model: str = "bge-m3"
    embedding_dim: int = Field(default=1024, ge=8, le=4096)

    # --- storage ---
    database_url: str = "postgresql+asyncpg://researchai:researchai@localhost:5434/researchai"

    @field_validator("database_url")
    @classmethod
    def _postgres_only(cls, value: str) -> str:
        if not value.startswith("postgresql+asyncpg://"):
            raise ValueError("DATABASE_URL must use the postgresql+asyncpg:// scheme")
        return value

    @field_validator("llm_base_url")
    @classmethod
    def _http_url(cls, value: str) -> str:
        if not value.startswith(("http://", "https://")):
            raise ValueError("LLM_BASE_URL must be an http(s) URL")
        return value.rstrip("/")

    @property
    def offline(self) -> bool:
        return self.llm_provider is LLMProvider.NONE


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Process-wide settings. Tests build ``Settings`` directly instead."""
    return Settings()
