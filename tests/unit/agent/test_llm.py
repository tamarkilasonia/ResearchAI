from __future__ import annotations

import json
from typing import Any

import httpx
import pytest
from pydantic import BaseModel

from researchai.agent.budget import BudgetExceeded, Limits, RunBudget
from researchai.agent.llm import (
    LLMError,
    OpenAICompatibleLLM,
    build_llm,
    estimate_tokens,
    structured_call,
)
from tests.agent_fakes import ScriptedLLM, always
from tests.support import make_settings


class Answer(BaseModel):
    value: int


def budget(max_tokens: int = 100_000) -> RunBudget:
    return RunBudget(Limits(10, 2, max_tokens, 5, 3))


def client(handler: Any) -> OpenAICompatibleLLM:
    return OpenAICompatibleLLM(
        base_url="http://ollama/v1/",
        model="qwen2.5:7b",
        api_key="k",
        transport=httpx.MockTransport(handler),
        context_tokens=4096,
    )


def ok(content: str, finish: str = "stop") -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "choices": [{"message": {"content": content}, "finish_reason": finish}],
            "usage": {"prompt_tokens": 11, "completion_tokens": 7},
        },
    )


async def test_chat_sends_deterministic_structured_request() -> None:
    seen: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen.update(json.loads(request.content), url=str(request.url))
        seen["auth"] = request.headers["authorization"]
        return ok('{"value": 1}')

    llm = client(handler)
    completion = await llm.chat(
        [{"role": "user", "content": "hi"}], schema={"type": "object"}, max_tokens=50
    )
    assert completion.content == '{"value": 1}'
    assert (completion.usage.prompt_tokens, completion.usage.completion_tokens) == (11, 7)
    assert seen["url"] == "http://ollama/v1/chat/completions"
    assert (seen["temperature"], seen["seed"], seen["max_tokens"]) == (0, 42, 50)
    assert seen["response_format"]["type"] == "json_schema"
    assert seen["auth"] == "Bearer k"
    assert (llm.model, llm.context_tokens) == ("qwen2.5:7b", 4096)


@pytest.mark.parametrize(
    ("response", "kind"),
    [
        (httpx.Response(500), "transport"),
        (httpx.Response(200, json={"choices": []}), "transport"),
        (ok('{"value": 1', finish="length"), "truncated"),
    ],
)
async def test_chat_failures(response: httpx.Response, kind: str) -> None:
    with pytest.raises(LLMError) as excinfo:
        await client(lambda _r: response).chat([], schema={}, max_tokens=5)
    assert excinfo.value.kind == kind


def test_build_llm() -> None:
    assert build_llm(make_settings()) is None
    llm = build_llm(make_settings(llm_provider="ollama", llm_model="m", llm_context_tokens=8192))
    assert llm is not None
    assert (llm.model, llm.context_tokens) == ("m", 8192)


def test_estimate_tokens_counts_georgian_as_heavier() -> None:
    assert estimate_tokens("abc") == 1
    assert estimate_tokens("აბგ") == 3


async def test_structured_call_success_records_tokens() -> None:
    llm = ScriptedLLM({"Answer": [always({"value": 3})]})
    run_budget = budget()
    result = await structured_call(
        llm, run_budget, system="s", user="u", schema=Answer, max_tokens=100
    )
    assert (result.value.value, result.attempts) == (3, 1)
    assert (run_budget.llm_calls, run_budget.tokens) == (1, 600)


@pytest.mark.parametrize("bad", ["not json", '{"value": "x"}', '{"other": 1}'])
async def test_structured_call_retries_once_with_feedback(bad: str) -> None:
    llm = ScriptedLLM({"Answer": [always(bad), always({"value": 9})]})
    result = await structured_call(
        llm, budget(), system="s", user="u", schema=Answer, max_tokens=10
    )
    assert (result.value.value, result.attempts) == (9, 2)
    retry_messages = llm.requests[1][1]
    assert retry_messages[-2] == {"role": "assistant", "content": bad}
    assert "invalid" in retry_messages[-1]["content"]


async def test_structured_call_gives_up_after_two_attempts() -> None:
    llm = ScriptedLLM({"Answer": [always("nope")]})
    with pytest.raises(LLMError) as excinfo:
        await structured_call(llm, budget(), system="s", user="u", schema=Answer, max_tokens=10)
    assert excinfo.value.kind == "invalid_json"
    assert len(llm.requests) == 2


async def test_structured_call_refuses_when_context_too_small() -> None:
    llm = ScriptedLLM({"Answer": [always({"value": 1})]}, context_tokens=200)
    with pytest.raises(LLMError) as excinfo:
        await structured_call(
            llm, budget(), system="s", user="x" * 600, schema=Answer, max_tokens=50
        )
    assert excinfo.value.kind == "context"
    assert llm.requests == []


async def test_structured_call_respects_token_budget() -> None:
    llm = ScriptedLLM({"Answer": [always({"value": 1})]})
    with pytest.raises(BudgetExceeded):
        await structured_call(
            llm, budget(1000), system="s", user="u", schema=Answer, max_tokens=1000
        )
    assert llm.requests == []
