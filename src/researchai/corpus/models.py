"""Corpus document and chunk models."""

from __future__ import annotations

import hashlib
import json
from datetime import date
from enum import StrEnum
from typing import Self
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, Field, model_validator

SYNTHETIC_LICENSE = "synthetic"
SYNTHETIC_HOST_SUFFIX = ".invalid"


class Language(StrEnum):
    EN = "en"
    KA = "ka"


class Topic(StrEnum):
    EV = "ev"  # electric cars in Georgia
    TAX = "tax"  # small-business (Individual Entrepreneur) tax regime in Georgia


class SourceType(StrEnum):
    GOVERNMENT = "government"
    REGULATOR = "regulator"
    STATISTICS_OFFICE = "statistics_office"
    UTILITY = "utility"
    NEWS = "news"
    INDUSTRY_REPORT = "industry_report"
    ACADEMIC = "academic"
    NGO = "ngo"
    COMPANY = "company"  # operator pages, e.g. a charging network
    DEALER = "dealer"
    BLOG = "blog"
    FORUM = "forum"


class DocumentMeta(BaseModel):
    """Front matter of one corpus file.

    Synthetic documents are written for this project. They carry
    ``synthetic = true``, ``license = "synthetic"`` and a URL on the reserved
    ``.invalid`` TLD so they can never be mistaken for a real web page.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str = Field(pattern=r"^[a-z]+-(en|ka)-\d{4}$")
    title: str = Field(min_length=3, max_length=300)
    url: str
    published: date
    snapshot: date
    language: Language
    topic: Topic
    source_type: SourceType
    publisher: str = Field(min_length=2, max_length=200)
    synthetic: bool
    license: str = Field(min_length=2, max_length=100)

    @model_validator(mode="after")
    def _consistent(self) -> Self:
        topic, language, _ = self.id.split("-")
        if topic != self.topic.value or language != self.language.value:
            raise ValueError(f"id {self.id!r} does not match topic/language")
        if self.published > self.snapshot:
            raise ValueError("published date is after snapshot date")
        parts = urlsplit(self.url)
        if parts.scheme != "https" or not parts.hostname:
            raise ValueError("url must be an absolute https URL")
        is_invalid_host = parts.hostname.endswith(SYNTHETIC_HOST_SUFFIX)
        if self.synthetic != is_invalid_host:
            raise ValueError("synthetic documents (and only they) use a .invalid URL")
        if self.synthetic != (self.license == SYNTHETIC_LICENSE):
            raise ValueError("synthetic documents (and only they) use license 'synthetic'")
        return self


class Document(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    meta: DocumentMeta
    body: str = Field(min_length=1)

    @property
    def id(self) -> str:
        return self.meta.id

    @property
    def sha256(self) -> str:
        """Content hash over metadata and body; changes if either changes."""
        canonical = json.dumps(
            {"meta": self.meta.model_dump(mode="json"), "body": self.body},
            sort_keys=True,
            ensure_ascii=False,
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class Chunk(BaseModel):
    """A contiguous slice ``body[start:end]`` of one document."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str
    doc_id: str
    ordinal: int = Field(ge=0)
    section: str
    text: str = Field(min_length=1)
    start: int = Field(ge=0)
    end: int

    @model_validator(mode="after")
    def _span(self) -> Self:
        if self.end - self.start != len(self.text):
            raise ValueError("chunk span length does not match text length")
        return self
