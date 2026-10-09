"""Hybrid corpus search: pgvector cosine + PostgreSQL full-text, fused with RRF.

Keyword leg
    The query is reduced to word tokens (no tsquery syntax can be injected),
    stop words are dropped, and each token is cut back to a stem by stripping
    one common suffix (English inflections; Georgian case and plural endings).
    Stems become prefix terms, so ``charg:*`` matches charging/charger/chargers
    and ``ელექტრომობილ:*`` matches ელექტრომობილი/ელექტრომობილის/ელექტრომობილები.
    Terms are OR-ed, so a long natural-language question still matches. This is
    a deliberately small stemmer, not a linguistic one.

Fusion
    Reciprocal rank fusion, ``score = sum(1 / (k + rank))`` with ``k = 60``.
    Ties break on chunk id, so identical inputs always give identical output.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import date

from sqlalchemy import ColumnElement, and_, func, select, true
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from researchai.corpus.embeddings import Embedder
from researchai.corpus.models import Language, SourceType, Topic
from researchai.db.models import ChunkRow, CorpusBuildRow, DocumentRow
from researchai.tools.errors import ErrorCode, ToolFailure
from researchai.tools.schemas import (
    MAX_QUERY_CHARS,
    MAX_TOP_K,
    SearchHit,
    SearchMode,
    SearchResponse,
)

RRF_K = 60
CANDIDATES_PER_LEG = 50
MIN_TOKEN_CHARS = 2
MIN_STEM_CHARS = 3
MAX_KEYWORD_TERMS = 24

# Longest first. English inflections, and Georgian plural/case endings
# (plural -ებ-, genitive -ის, dative -ს, instrumental -ით, adverbial -ად,
# postpositions -ში/-ზე/-თვის/-გან, nominative -ი and stem vowels).
_EN_SUFFIXES = ("ingly", "ings", "ing", "ers", "ies", "es", "er", "ed", "s")
_KA_SUFFIXES = (
    "ებისთვის",
    "ებისგან",
    "ისთვის",
    "ებით",
    "ებში",
    "ებზე",
    "ებად",
    "ების",
    "ისგან",
    "ები",
    "ებს",
    "ში",
    "ზე",
    "ის",
    "ით",
    "ად",
    "ი",
    "ს",
    "ა",
    "ე",
    "ო",
)

_TOKEN = re.compile(r"\w+", re.UNICODE)

STOP_WORDS = frozenset(
    {
        # English
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "can",
        "do",
        "does",
        "for",
        "from",
        "has",
        "have",
        "how",
        "i",
        "in",
        "is",
        "it",
        "its",
        "me",
        "my",
        "of",
        "on",
        "or",
        "should",
        "than",
        "that",
        "the",
        "their",
        "there",
        "this",
        "to",
        "was",
        "what",
        "when",
        "where",
        "which",
        "who",
        "why",
        "will",
        "with",
        "would",
        "you",
        "your",
        "much",
        "many",
        "give",
        "tell",
        "about",
        # Georgian
        "და",
        "არის",
        "რა",
        "რომ",
        "თუ",
        "ან",
        "როგორ",
        "რამდენი",
        "რატომ",
        "სად",
        "როდის",
        "ეს",
        "ის",
        "მაქვს",
        "აქვს",
        "უნდა",
        "შეიძლება",
        "კი",
        "არა",
        "რომელი",
        "ვინ",
    }
)


@dataclass(frozen=True)
class SearchFilters:
    language: Language | None = None
    topic: Topic | None = None
    published_from: date | None = None
    published_to: date | None = None
    source_types: tuple[SourceType, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if self.published_from and self.published_to and self.published_from > self.published_to:
            raise ToolFailure(ErrorCode.INVALID_ARGUMENT, "published_from is after published_to")

    def clause(self) -> ColumnElement[bool]:
        conditions: list[ColumnElement[bool]] = []
        if self.language is not None:
            conditions.append(DocumentRow.language == self.language.value)
        if self.topic is not None:
            conditions.append(DocumentRow.topic == self.topic.value)
        if self.published_from is not None:
            conditions.append(DocumentRow.published >= self.published_from)
        if self.published_to is not None:
            conditions.append(DocumentRow.published <= self.published_to)
        if self.source_types:
            conditions.append(DocumentRow.source_type.in_([s.value for s in self.source_types]))
        return and_(true(), *conditions)


def light_stem(token: str) -> str:
    """Strip at most one known suffix, keeping at least ``MIN_STEM_CHARS``."""
    suffixes = _EN_SUFFIXES if token.isascii() else _KA_SUFFIXES
    for suffix in suffixes:
        if token.endswith(suffix) and len(token) - len(suffix) >= MIN_STEM_CHARS:
            return token[: -len(suffix)]
    return token


def keyword_terms(query: str) -> list[str]:
    """Sanitised, de-duplicated prefix terms for ``to_tsquery('simple', ...)``."""
    normalized = unicodedata.normalize("NFKC", query).casefold()
    terms: list[str] = []
    for token in _TOKEN.findall(normalized):
        token = token.replace("_", "")
        if len(token) < MIN_TOKEN_CHARS or token in STOP_WORDS:
            continue
        stem = light_stem(token)
        if stem not in terms:
            terms.append(stem)
    return terms[:MAX_KEYWORD_TERMS]


def to_tsquery_text(terms: Sequence[str]) -> str:
    return " | ".join(f"{term}:*" for term in terms)


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[str]], *, k: int = RRF_K
) -> list[tuple[str, float]]:
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, item in enumerate(ranking, start=1):
            scores[item] = scores.get(item, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda pair: (-pair[1], pair[0]))


class CorpusSearch:
    def __init__(self, sessions: async_sessionmaker[AsyncSession], embedder: Embedder) -> None:
        self._sessions = sessions
        self._embedder = embedder

    async def _check_index(self, session: AsyncSession) -> None:
        """Refuse to search an index built with a different embedder.

        Checked on every search, not cached: the corpus can be re-ingested while
        the server runs, and vectors from another model of the same dimension
        would otherwise produce plausible-looking nonsense.
        """
        build = (
            await session.scalars(
                select(CorpusBuildRow).order_by(CorpusBuildRow.id.desc()).limit(1)
            )
        ).first()
        if build is None:
            raise ToolFailure(ErrorCode.INDEX_NOT_READY, "the corpus has not been ingested")
        if build.embedder != self._embedder.name:
            raise ToolFailure(
                ErrorCode.INDEX_NOT_READY,
                f"index was built with {build.embedder}, search uses {self._embedder.name}",
            )

    async def _vector_ranking(
        self, session: AsyncSession, query: str, filters: SearchFilters
    ) -> list[str]:
        (vector,) = await self._embedder.embed([query])
        rows = await session.scalars(
            select(ChunkRow.id)
            .join(DocumentRow)
            .where(filters.clause())
            .order_by(ChunkRow.embedding.cosine_distance(vector), ChunkRow.id)
            .limit(CANDIDATES_PER_LEG)
        )
        return list(rows.all())

    async def _keyword_ranking(
        self, session: AsyncSession, query: str, filters: SearchFilters
    ) -> list[str]:
        terms = keyword_terms(query)
        if not terms:
            return []
        tsquery = func.to_tsquery("simple", to_tsquery_text(terms))
        rank = func.ts_rank_cd(ChunkRow.tsv, tsquery)
        rows = await session.scalars(
            select(ChunkRow.id)
            .join(DocumentRow)
            .where(filters.clause(), ChunkRow.tsv.op("@@")(tsquery))
            .order_by(rank.desc(), ChunkRow.id)
            .limit(CANDIDATES_PER_LEG)
        )
        return list(rows.all())

    async def search(
        self,
        query: str,
        *,
        top_k: int = 8,
        mode: SearchMode = "hybrid",
        filters: SearchFilters | None = None,
    ) -> SearchResponse:
        query = query.strip()
        if not query or len(query) > MAX_QUERY_CHARS:
            raise ToolFailure(
                ErrorCode.INVALID_ARGUMENT, f"query must be 1-{MAX_QUERY_CHARS} characters"
            )
        if not 1 <= top_k <= MAX_TOP_K:
            raise ToolFailure(ErrorCode.INVALID_ARGUMENT, f"top_k must be 1-{MAX_TOP_K}")
        filters = filters or SearchFilters()

        async with self._sessions() as session:
            # One snapshot for the index check and both legs, even if a re-ingest commits.
            await session.connection(execution_options={"isolation_level": "REPEATABLE READ"})
            await self._check_index(session)
            vector = (
                await self._vector_ranking(session, query, filters) if mode != "keyword" else []
            )
            keyword = (
                await self._keyword_ranking(session, query, filters) if mode != "vector" else []
            )
            fused = reciprocal_rank_fusion([r for r in (vector, keyword) if r])[:top_k]
            result = await session.execute(
                select(ChunkRow, DocumentRow)
                .join(DocumentRow)
                .where(ChunkRow.id.in_([chunk_id for chunk_id, _ in fused]))
            )
            rows = {chunk.id: (chunk, document) for chunk, document in result.all()}

        vector_rank = {chunk_id: i for i, chunk_id in enumerate(vector, start=1)}
        keyword_rank = {chunk_id: i for i, chunk_id in enumerate(keyword, start=1)}
        hits = [
            SearchHit(
                chunk_id=chunk_id,
                doc_id=document.id,
                title=document.title,
                section=chunk.section,
                text=chunk.text,
                char_start=chunk.char_start,
                char_end=chunk.char_end,
                score=round(score, 6),
                vector_rank=vector_rank.get(chunk_id),
                keyword_rank=keyword_rank.get(chunk_id),
                published=document.published,
                language=Language(document.language),
                topic=Topic(document.topic),
                source_type=SourceType(document.source_type),
                publisher=document.publisher,
            )
            for chunk_id, score in fused
            for chunk, document in (rows[chunk_id],)
        ]
        return SearchResponse(query=query, mode=mode, hits=hits)
