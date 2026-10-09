"""Text embedders.

``HashEmbedder`` is deterministic and offline (feature hashing over words and
character n-grams). It is good enough to exercise the pipeline in tests and CI,
and character n-grams give it some tolerance for Georgian inflection, but it is
not a semantic model; reported retrieval numbers use ``OpenAICompatibleEmbedder``
(Ollama serving bge-m3 by default).
"""

from __future__ import annotations

import hashlib
import math
import re
import unicodedata
from collections.abc import Sequence
from typing import Protocol

import httpx

from researchai.config import EmbeddingProvider, Settings

Vector = list[float]

_WORD = re.compile(r"\w+", re.UNICODE)


class EmbeddingError(RuntimeError):
    """The embedding backend failed or returned something unusable."""


class Embedder(Protocol):
    @property
    def name(self) -> str:
        """Stable identifier stored with each corpus build, e.g. ``ollama:bge-m3``."""
        ...

    @property
    def dim(self) -> int: ...

    async def embed(self, texts: Sequence[str]) -> list[Vector]: ...


def _l2_normalize(vector: Vector) -> Vector:
    norm = math.sqrt(sum(x * x for x in vector))
    return vector if norm == 0 else [x / norm for x in vector]


class HashEmbedder:
    """Signed feature hashing of word unigrams and character 3/4-grams."""

    def __init__(self, dim: int) -> None:
        if dim < 8:
            raise ValueError("dim must be at least 8")
        self._dim = dim

    @property
    def name(self) -> str:
        return f"hash:v1:{self._dim}"

    @property
    def dim(self) -> int:
        return self._dim

    def _features(self, text: str) -> list[tuple[str, float]]:
        normalized = unicodedata.normalize("NFKC", text).casefold()
        features: list[tuple[str, float]] = []
        for word in _WORD.findall(normalized):
            features.append((f"w:{word}", 1.0))
            padded = f"<{word}>"
            for n in (3, 4):
                features.extend(
                    (f"c{n}:{padded[i : i + n]}", 0.5) for i in range(len(padded) - n + 1)
                )
        return features

    def embed_one(self, text: str) -> Vector:
        vector = [0.0] * self._dim
        for feature, weight in self._features(text):
            digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
            index = int.from_bytes(digest[:4], "big") % self._dim
            vector[index] += weight if digest[4] & 1 else -weight
        return _l2_normalize(vector)

    async def embed(self, texts: Sequence[str]) -> list[Vector]:
        return [self.embed_one(text) for text in texts]


class OpenAICompatibleEmbedder:
    """Calls ``POST {base_url}/embeddings`` (OpenAI wire format; Ollama supports it)."""

    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        dim: int,
        api_key: str = "",
        timeout: float = 120.0,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._url = f"{base_url.rstrip('/')}/embeddings"
        self._model = model
        self._dim = dim
        self._headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self._timeout = timeout
        self._transport = transport

    @property
    def name(self) -> str:
        return f"openai_compatible:{self._model}"

    @property
    def dim(self) -> int:
        return self._dim

    async def embed(self, texts: Sequence[str]) -> list[Vector]:
        if not texts:
            return []
        try:
            async with httpx.AsyncClient(
                timeout=self._timeout, transport=self._transport
            ) as client:
                response = await client.post(
                    self._url,
                    json={"model": self._model, "input": list(texts)},
                    headers=self._headers,
                )
                response.raise_for_status()
                data = response.json()["data"]
        except (httpx.HTTPError, KeyError, TypeError, ValueError) as exc:
            raise EmbeddingError(f"embedding request to {self._url} failed: {exc}") from exc
        if not isinstance(data, list) or len(data) != len(texts):
            raise EmbeddingError(f"expected {len(texts)} embeddings, got a different payload")
        try:
            vectors = [item["embedding"] for item in sorted(data, key=lambda d: d["index"])]
        except (KeyError, TypeError) as exc:
            raise EmbeddingError(f"malformed embedding item: {exc}") from exc
        for vector in vectors:
            if len(vector) != self._dim:
                raise EmbeddingError(
                    f"model {self._model!r} returned dim {len(vector)}, "
                    f"EMBEDDING_DIM is {self._dim}"
                )
        return [_l2_normalize([float(x) for x in vector]) for vector in vectors]


def build_embedder(settings: Settings) -> Embedder:
    if settings.embedding_provider is EmbeddingProvider.HASH:
        return HashEmbedder(settings.embedding_dim)
    return OpenAICompatibleEmbedder(
        base_url=settings.embedding_base_url,
        model=settings.embedding_model,
        dim=settings.embedding_dim,
        # The LLM key is deliberately not forwarded: EMBEDDING_BASE_URL may be another host.
        timeout=settings.llm_timeout_seconds,
    )
