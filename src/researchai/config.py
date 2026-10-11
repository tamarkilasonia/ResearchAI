"""Application settings.

Every field is read from an environment variable of the same name in upper case
(``llm_provider`` -> ``LLM_PROVIDER``). Every field must be documented in
``.env.example``; ``tests/unit/test_env_example.py`` enforces that.
"""

from __future__ import annotations

from enum import StrEnum
from functools import lru_cache
from pathlib import Path
from typing import Literal, Self

from pydantic import Field, SecretStr, field_validator, model_validator
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
    llm_timeout_seconds: float = Field(default=300.0, gt=0, le=1800)
    judge_model: str = "llama3.1:8b"
    llm_context_tokens: int = Field(default=16_384, ge=1024, le=1_000_000)
    llm_max_output_tokens: int = Field(default=2048, ge=128, le=32_768)
    llm_writer_evidence_tokens: int = Field(default=4000, ge=500, le=200_000)
    llm_seed: int = 42

    # --- agent budgets (per research run) ---
    max_searches: int = Field(default=12, ge=1, le=100)
    max_iterations: int = Field(default=2, ge=1, le=5)
    max_tokens_per_run: int = Field(default=40_000, ge=1000, le=1_000_000)
    max_subquestions: int = Field(default=6, ge=1, le=10)
    hits_per_subquestion: int = Field(default=4, ge=1, le=10)
    # auto: vector with a real embedding model, hybrid with the hash embedder (no
    # semantic signal). Chosen on the dev split: with bge-m3, vector beat hybrid
    # (recall@3 0.87 vs 0.78, 77 cases).
    search_mode: Literal["auto", "vector", "keyword", "hybrid"] = "auto"
    run_timeout_seconds: float = Field(default=600.0, gt=0, le=3600)

    # --- embeddings ---
    embedding_provider: EmbeddingProvider = EmbeddingProvider.HASH
    embedding_model: str = "bge-m3"
    embedding_base_url: str = "http://localhost:11434/v1"
    embedding_dim: int = Field(default=1024, ge=8, le=4096)
    embedding_batch_size: int = Field(default=32, ge=1, le=256)

    @property
    def agent_search_mode(self) -> Literal["vector", "keyword", "hybrid"]:
        if self.search_mode != "auto":
            return self.search_mode
        return "hybrid" if self.embedding_provider is EmbeddingProvider.HASH else "vector"

    # --- API ---
    api_max_concurrent_runs: int = Field(default=2, ge=1, le=32)
    api_wait_seconds: float = Field(default=600.0, gt=0, le=3600)

    # --- corpus ---
    corpus_dir: Path = Path("corpus")

    # --- evaluation ---
    eval_dir: Path = Path("eval")
    eval_output_dir: Path = Path("eval/runs")
    eval_cost_per_1k_tokens: float = Field(default=0.0, ge=0)

    # --- MCP tool server ---
    mcp_server_url: str = "http://localhost:8765/mcp"
    mcp_host: str = "127.0.0.1"
    mcp_port: int = Field(default=8765, ge=1, le=65535)
    mcp_allowed_hosts: str = "localhost:*,127.0.0.1:*"
    tool_timeout_seconds: float = Field(default=60.0, gt=0, le=600)
    live_web_search_enabled: bool = False
    live_web_search_url: str = ""

    # --- storage ---
    database_url: str = "postgresql+asyncpg://researchai:researchai@localhost:5434/researchai"

    @field_validator("database_url")
    @classmethod
    def _postgres_only(cls, value: str) -> str:
        if not value.startswith("postgresql+asyncpg://"):
            raise ValueError("DATABASE_URL must use the postgresql+asyncpg:// scheme")
        return value

    @field_validator("llm_base_url", "embedding_base_url", "mcp_server_url")
    @classmethod
    def _http_url(cls, value: str) -> str:
        if not value.startswith(("http://", "https://")):
            raise ValueError("base URLs must be http(s) URLs")
        return value.rstrip("/")

    @model_validator(mode="after")
    def _web_search_needs_url(self) -> Self:
        if self.live_web_search_enabled and not self.live_web_search_url.startswith(
            ("http://", "https://")
        ):
            raise ValueError("LIVE_WEB_SEARCH_ENABLED requires an http(s) LIVE_WEB_SEARCH_URL")
        return self

    @property
    def offline(self) -> bool:
        return self.llm_provider is LLMProvider.NONE

    @property
    def mcp_allowed_host_list(self) -> list[str]:
        return [host.strip() for host in self.mcp_allowed_hosts.split(",") if host.strip()]


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Process-wide settings. Tests build ``Settings`` directly instead."""
    return Settings()
