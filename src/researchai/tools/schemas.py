"""Tool input limits and output models (shared by server and client)."""

from __future__ import annotations

from datetime import date
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from researchai.corpus.models import Language, SourceType, Topic

DOC_ID_PATTERN = r"^[a-z]+-(en|ka)-\d{4}$"
MAX_QUERY_CHARS = 500
MAX_TOP_K = 20
MAX_METADATA_IDS = 20
MAX_EXPRESSION_CHARS = 200
MAX_WEB_RESULTS = 10

SearchMode = Literal["hybrid", "vector", "keyword"]

# Retrieved text is data, never instructions. Every response that carries
# document or web content says so explicitly.
ContentTrust = Literal["untrusted"]


class _Out(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class SearchHit(_Out):
    chunk_id: str
    doc_id: str
    title: str
    section: str
    text: str
    char_start: int
    char_end: int
    score: float = Field(description="Reciprocal-rank-fusion score; higher is better")
    vector_rank: int | None = Field(description="1-based rank in the vector leg, if retrieved")
    keyword_rank: int | None = Field(description="1-based rank in the keyword leg, if retrieved")
    published: date
    language: Language
    topic: Topic
    source_type: SourceType
    publisher: str


class SearchResponse(_Out):
    query: str
    mode: Literal["hybrid", "vector", "keyword"]
    hits: list[SearchHit]
    content_trust: ContentTrust = "untrusted"


class SourceMetadata(_Out):
    doc_id: str
    title: str
    url: str
    publisher: str
    source_type: SourceType
    published: date
    snapshot: date
    language: Language
    topic: Topic
    synthetic: bool


class DocumentResponse(_Out):
    metadata: SourceMetadata
    text: str
    char_start: int
    char_end: int
    length: int = Field(description="Length of the full document body in characters")
    content_trust: ContentTrust = "untrusted"


class MetadataResponse(_Out):
    sources: list[SourceMetadata]
    missing: list[str]


class CalculationResponse(_Out):
    expression: str
    result: float


class WebResult(_Out):
    title: str
    url: str
    snippet: str


class WebSearchResponse(_Out):
    query: str
    results: list[WebResult]
    source: Literal["live_web"] = "live_web"
    content_trust: ContentTrust = "untrusted"
