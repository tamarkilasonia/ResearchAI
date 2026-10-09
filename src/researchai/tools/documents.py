"""Document access: full text (or a span) and source metadata."""

from __future__ import annotations

import re
from collections.abc import Sequence

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from researchai.corpus.models import Language, SourceType, Topic
from researchai.db.models import DocumentRow
from researchai.tools.errors import ErrorCode, ToolFailure
from researchai.tools.schemas import (
    DOC_ID_PATTERN,
    MAX_METADATA_IDS,
    DocumentResponse,
    MetadataResponse,
    SourceMetadata,
)

_DOC_ID = re.compile(DOC_ID_PATTERN)


def _check_id(doc_id: str) -> None:
    if not _DOC_ID.match(doc_id):
        raise ToolFailure(ErrorCode.INVALID_ARGUMENT, f"malformed document id {doc_id!r}")


def metadata_of(row: DocumentRow) -> SourceMetadata:
    return SourceMetadata(
        doc_id=row.id,
        title=row.title,
        url=row.url,
        publisher=row.publisher,
        source_type=SourceType(row.source_type),
        published=row.published,
        snapshot=row.snapshot,
        language=Language(row.language),
        topic=Topic(row.topic),
        synthetic=row.synthetic,
    )


class DocumentStore:
    def __init__(self, sessions: async_sessionmaker[AsyncSession]) -> None:
        self._sessions = sessions

    async def fetch(
        self, doc_id: str, *, start: int | None = None, end: int | None = None
    ) -> DocumentResponse:
        _check_id(doc_id)
        async with self._sessions() as session:
            row = await session.get(DocumentRow, doc_id)
        if row is None:
            raise ToolFailure(ErrorCode.NOT_FOUND, f"no document {doc_id}")
        length = len(row.body)
        span_start = 0 if start is None else start
        span_end = length if end is None else end
        if not 0 <= span_start < span_end <= length:
            raise ToolFailure(
                ErrorCode.INVALID_ARGUMENT,
                f"span [{span_start}, {span_end}) is outside the document (length {length})",
            )
        return DocumentResponse(
            metadata=metadata_of(row),
            text=row.body[span_start:span_end],
            char_start=span_start,
            char_end=span_end,
            length=length,
        )

    async def metadata(self, doc_ids: Sequence[str]) -> MetadataResponse:
        if not 1 <= len(doc_ids) <= MAX_METADATA_IDS:
            raise ToolFailure(ErrorCode.INVALID_ARGUMENT, f"pass 1-{MAX_METADATA_IDS} document ids")
        wanted = list(dict.fromkeys(doc_ids))
        for doc_id in wanted:
            _check_id(doc_id)
        async with self._sessions() as session:
            rows = {
                row.id: row
                for row in (
                    await session.scalars(select(DocumentRow).where(DocumentRow.id.in_(wanted)))
                ).all()
            }
        return MetadataResponse(
            sources=[metadata_of(rows[i]) for i in wanted if i in rows],
            missing=[i for i in wanted if i not in rows],
        )
