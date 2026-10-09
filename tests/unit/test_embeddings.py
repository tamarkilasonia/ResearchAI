from __future__ import annotations

import json
import math

import httpx
import pytest

from researchai.corpus.embeddings import (
    EmbeddingError,
    HashEmbedder,
    OpenAICompatibleEmbedder,
    build_embedder,
)
from tests.support import make_settings


def cosine(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b, strict=True))


async def test_hash_embedder_is_deterministic_and_normalized() -> None:
    embedder = HashEmbedder(64)
    first, second = await embedder.embed(["Electric cars", "Electric cars"])
    assert first == second
    assert math.isclose(math.sqrt(sum(x * x for x in first)), 1.0)
    assert embedder.dim == 64
    assert embedder.name == "hash:v1:64"


async def test_hash_embedder_case_and_unicode_insensitive() -> None:
    embedder = HashEmbedder(64)
    fullwidth = "".join(chr(ord(ch) + 0xFEE0) for ch in "ELECTRIC")  # NFKC folds these
    assert embedder.embed_one(fullwidth) == embedder.embed_one("electric")


def test_hash_embedder_empty_text_is_zero_vector() -> None:
    assert HashEmbedder(16).embed_one("  ") == [0.0] * 16


def test_hash_embedder_rejects_tiny_dim() -> None:
    with pytest.raises(ValueError, match="dim"):
        HashEmbedder(4)


def test_hash_embedder_similarity_is_sensible() -> None:
    embedder = HashEmbedder(1024)
    query = embedder.embed_one("cost of charging an electric car")
    related = embedder.embed_one("charging costs for electric cars in Tbilisi")
    unrelated = embedder.embed_one("small business turnover tax threshold")
    assert cosine(query, related) > cosine(query, unrelated)


def test_hash_embedder_tolerates_georgian_inflection() -> None:
    embedder = HashEmbedder(1024)
    stem = embedder.embed_one("ელექტრომობილი")
    inflected = embedder.embed_one("ელექტრომობილის")
    unrelated = embedder.embed_one("გადასახადი")
    assert cosine(stem, inflected) > 0.5 > cosine(stem, unrelated)


def _transport(payload: object, status: int = 200) -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status, json=payload)

    return httpx.MockTransport(handler)


async def test_openai_embedder_success_orders_by_index_and_normalizes() -> None:
    seen: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["body"] = json.loads(request.content)
        seen["auth"] = request.headers.get("authorization")
        data = [{"index": 1, "embedding": [0.0, 2.0]}, {"index": 0, "embedding": [3.0, 4.0]}]
        return httpx.Response(200, json={"data": data})

    embedder = OpenAICompatibleEmbedder(
        base_url="http://ollama:11434/v1/",
        model="bge-m3",
        dim=2,
        api_key="k",
        transport=httpx.MockTransport(handler),
    )
    vectors = await embedder.embed(["a", "b"])
    assert vectors == [[0.6, 0.8], [0.0, 1.0]]
    assert seen == {
        "url": "http://ollama:11434/v1/embeddings",
        "body": {"model": "bge-m3", "input": ["a", "b"]},
        "auth": "Bearer k",
    }
    assert embedder.name == "openai_compatible:bge-m3"
    assert embedder.dim == 2


async def test_openai_embedder_sends_no_auth_without_key() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert "authorization" not in request.headers
        return httpx.Response(200, json={"data": [{"index": 0, "embedding": [1.0]}]})

    embedder = OpenAICompatibleEmbedder(
        base_url="http://x/v1", model="m", dim=1, transport=httpx.MockTransport(handler)
    )
    assert await embedder.embed(["a"]) == [[1.0]]


async def test_openai_embedder_empty_input_makes_no_request() -> None:
    def handler(request: httpx.Request) -> httpx.Response:  # pragma: no cover - must not run
        raise AssertionError("no request expected")

    embedder = OpenAICompatibleEmbedder(
        base_url="http://x/v1", model="m", dim=2, transport=httpx.MockTransport(handler)
    )
    assert await embedder.embed([]) == []


@pytest.mark.parametrize(
    ("payload", "status", "message"),
    [
        ({"error": "boom"}, 500, "failed"),
        ({"nope": []}, 200, "failed"),
        ({"data": "x"}, 200, "expected 1"),
        ({"data": []}, 200, "expected 1"),
        ({"data": [{"embedding": [1.0, 0.0]}]}, 200, "malformed"),
        ({"data": [{"index": 0, "embedding": [1.0, 0.0, 0.0]}]}, 200, "returned dim 3"),
    ],
)
async def test_openai_embedder_errors(payload: object, status: int, message: str) -> None:
    embedder = OpenAICompatibleEmbedder(
        base_url="http://x/v1", model="m", dim=2, transport=_transport(payload, status)
    )
    with pytest.raises(EmbeddingError, match=message):
        await embedder.embed(["a"])


async def test_openai_embedder_connection_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    embedder = OpenAICompatibleEmbedder(
        base_url="http://x/v1", model="m", dim=2, transport=httpx.MockTransport(handler)
    )
    with pytest.raises(EmbeddingError, match="refused"):
        await embedder.embed(["a"])


def test_build_embedder_by_provider() -> None:
    assert isinstance(build_embedder(make_settings(embedding_dim=32)), HashEmbedder)
    remote = build_embedder(
        make_settings(embedding_provider="ollama", embedding_model="bge-m3", llm_api_key="sk-x")
    )
    assert isinstance(remote, OpenAICompatibleEmbedder)
    assert remote.name == "openai_compatible:bge-m3"
    assert remote._headers == {}  # the LLM key never goes to the embedding host
