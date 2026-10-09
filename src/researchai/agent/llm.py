"""LLM access for the agent (OpenAI-compatible chat completions, Ollama by default).

Only structured output is used: every call asks for JSON matching a Pydantic
schema and the reply is validated. :func:`structured_call` adds the budget
check, a context-window check, one corrective retry and token accounting.
With ``LLM_PROVIDER=none`` there is no client at all and every node uses its
rule-based path.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any, Literal, Protocol

import httpx
from pydantic import BaseModel, ValidationError

from researchai.agent.budget import RunBudget
from researchai.config import LLMProvider, Settings

MAX_ATTEMPTS = 2  # one corrective retry

FailureKind = Literal["transport", "truncated", "invalid_json", "schema", "context"]


class LLMError(RuntimeError):
    def __init__(self, kind: FailureKind, message: str) -> None:
        super().__init__(f"{kind}: {message}")
        self.kind = kind


@dataclass(frozen=True)
class LLMUsage:
    prompt_tokens: int
    completion_tokens: int


@dataclass(frozen=True)
class Completion:
    content: str
    usage: LLMUsage


class LLMClient(Protocol):
    @property
    def model(self) -> str: ...

    @property
    def context_tokens(self) -> int: ...

    async def chat(
        self, messages: list[dict[str, str]], *, schema: dict[str, Any], max_tokens: int
    ) -> Completion: ...


def estimate_tokens(text: str) -> int:
    """Conservative token estimate: UTF-8 bytes / 3 (Georgian letters are 3 bytes)."""
    return math.ceil(len(text.encode("utf-8")) / 3)


class OpenAICompatibleLLM:
    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        api_key: str = "",
        timeout: float = 120.0,
        seed: int = 42,
        context_tokens: int = 16_384,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._url = f"{base_url.rstrip('/')}/chat/completions"
        self._model = model
        self._headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self._timeout = timeout
        self._seed = seed
        self._context_tokens = context_tokens
        self._transport = transport

    @property
    def model(self) -> str:
        return self._model

    @property
    def context_tokens(self) -> int:
        return self._context_tokens

    async def chat(
        self, messages: list[dict[str, str]], *, schema: dict[str, Any], max_tokens: int
    ) -> Completion:
        payload = {
            "model": self._model,
            "messages": messages,
            "temperature": 0,
            "seed": self._seed,
            "max_tokens": max_tokens,
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": "response", "schema": schema, "strict": True},
            },
        }
        try:
            async with httpx.AsyncClient(
                timeout=self._timeout, transport=self._transport
            ) as client:
                response = await client.post(self._url, json=payload, headers=self._headers)
                response.raise_for_status()
                body = response.json()
            choice = body["choices"][0]
            content = choice["message"]["content"] or ""
            usage = body.get("usage") or {}
        except (httpx.HTTPError, KeyError, IndexError, TypeError, ValueError) as exc:
            # str(ReadTimeout()) is empty, so always name the exception type.
            raise LLMError(
                "transport", f"chat completion failed: {type(exc).__name__}: {exc}"
            ) from exc
        if choice.get("finish_reason") == "length":
            raise LLMError("truncated", f"output hit max_tokens={max_tokens}")
        return Completion(
            content=content,
            usage=LLMUsage(
                prompt_tokens=int(usage.get("prompt_tokens", 0)),
                completion_tokens=int(usage.get("completion_tokens", 0)),
            ),
        )


def build_llm(settings: Settings) -> LLMClient | None:
    if settings.llm_provider is LLMProvider.NONE:
        return None
    return OpenAICompatibleLLM(
        base_url=settings.llm_base_url,
        model=settings.llm_model,
        api_key=settings.llm_api_key.get_secret_value(),
        timeout=settings.llm_timeout_seconds,
        seed=settings.llm_seed,
        context_tokens=settings.llm_context_tokens,
    )


@dataclass(frozen=True)
class StructuredResult[M: BaseModel]:
    value: M
    attempts: int


def _parse[M: BaseModel](content: str, schema: type[M]) -> M:
    try:
        data = json.loads(content)
    except ValueError as exc:
        raise LLMError("invalid_json", str(exc)) from exc
    try:
        return schema.model_validate(data)
    except ValidationError as exc:
        raise LLMError("schema", exc.errors(include_url=False).__repr__()[:500]) from exc


async def structured_call[M: BaseModel](
    llm: LLMClient,
    budget: RunBudget,
    *,
    system: str,
    user: str,
    schema: type[M],
    max_tokens: int,
) -> StructuredResult[M]:
    """One validated JSON completion, with at most one corrective retry.

    Raises :class:`~researchai.agent.budget.BudgetExceeded` before calling if the
    token budget could be exceeded, and :class:`LLMError` if the context window
    is too small or both attempts fail.
    """
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    json_schema = schema.model_json_schema()
    attempt = 0
    while True:
        attempt += 1
        estimate = sum(estimate_tokens(m["content"]) for m in messages) + estimate_tokens(
            json.dumps(json_schema)
        )
        if estimate + max_tokens > llm.context_tokens:
            raise LLMError("context", f"~{estimate} prompt tokens + {max_tokens} output")
        budget.check_tokens(estimate, max_tokens)
        completion = await llm.chat(messages, schema=json_schema, max_tokens=max_tokens)
        budget.record_tokens(completion.usage.prompt_tokens, completion.usage.completion_tokens)
        try:
            return StructuredResult(_parse(completion.content, schema), attempts=attempt)
        except LLMError as exc:
            if attempt >= MAX_ATTEMPTS:
                raise
            messages = [
                *messages,
                {"role": "assistant", "content": completion.content},
                {
                    "role": "user",
                    "content": f"That output was invalid ({exc}). Reply again with only "
                    "JSON that matches the schema exactly.",
                },
            ]
