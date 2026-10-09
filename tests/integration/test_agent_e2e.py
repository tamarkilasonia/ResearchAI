"""The offline agent over real MCP/HTTP against PostgreSQL and the real corpus."""

from __future__ import annotations

from collections.abc import AsyncIterator

import pytest
import pytest_asyncio

from researchai.agent.models import RunStatus
from researchai.agent.render import render_markdown
from researchai.agent.runner import ResearchAgent
from researchai.mcp_client.client import ResearchTools
from tests.integration.conftest import scratch_database_url, serving_real_corpus
from tests.support import make_settings

pytestmark = pytest.mark.asyncio(loop_scope="module")


@pytest_asyncio.fixture(scope="module", loop_scope="module")
async def mcp_url() -> AsyncIterator[str]:
    async with serving_real_corpus(scratch_database_url("_test_agent")) as url:
        yield url


async def _run(url: str, question: str) -> tuple[RunStatus, str, list[str]]:
    async with ResearchTools(url) as tools:
        result = await ResearchAgent(tools=tools, llm=None, settings=make_settings()).run(question)
        # Every citation span must match the source document exactly.
        for citation in result.report.citations() if result.report else []:
            assert citation.char_start is not None
            assert citation.char_end is not None
            document = await tools.fetch_document(
                citation.doc_id, start=citation.char_start, end=citation.char_end
            )
            assert document.text == citation.quote
    docs = sorted({c.doc_id for c in result.report.citations()}) if result.report else []
    return result.status, render_markdown(result), docs


async def test_comparison_question_is_answered_with_verifiable_citations(mcp_url: str) -> None:
    status, markdown, docs = await _run(
        mcp_url,
        "Compare the advantages and disadvantages of electric cars in Georgia and give me a "
        "recommendation.",
    )
    assert status is RunStatus.ANSWERED
    assert len(docs) >= 4
    assert "## Sources" in markdown


async def test_georgian_question_gets_a_georgian_report(mcp_url: str) -> None:
    status, markdown, _ = await _run(mcp_url, "რა ღირს ელექტრომობილის დატენვა საქართველოში?")
    assert status is RunStatus.ANSWERED
    assert markdown.startswith("# კვლევის ანგარიში")
    assert "## წყაროები" in markdown


async def test_question_outside_the_corpus_abstains(mcp_url: str) -> None:
    status, markdown, docs = await _run(mcp_url, "Who won the 1998 football world cup?")
    assert status is RunStatus.ABSTAINED
    assert docs == []
    assert "Insufficient evidence" in markdown


async def test_superseded_registry_counts_are_flagged_on_the_real_corpus(mcp_url: str) -> None:
    async with ResearchTools(mcp_url) as tools:
        result = await ResearchAgent(tools=tools, llm=None, settings=make_settings()).run(
            "How many public charging points are listed in the charging registry?"
        )
    assert result.status is RunStatus.ANSWERED
    outdated = [c for c in result.conflicts if c.kind == "outdated"]
    assert outdated, [c.summary for c in result.conflicts]
    claims = {c.id: c for c in result.claims}
    for conflict in outdated:
        newest = max(claims[i].published for i in conflict.current)
        assert all(claims[i].published < newest for i in conflict.others)
    assert result.report is not None
    assert any(s.kind == "conflict" for s in result.report.sentences())


async def test_unknown_place_abstains_on_the_real_corpus(mcp_url: str) -> None:
    async with ResearchTools(mcp_url) as tools:
        result = await ResearchAgent(tools=tools, llm=None, settings=make_settings()).run(
            "What is the price of a Tesla Model S in Japan?"
        )
    assert result.status is RunStatus.ABSTAINED
    assert result.missing_entities == ["Japan"]


async def test_offline_citations_all_verify_on_the_real_corpus(mcp_url: str) -> None:
    async with ResearchTools(mcp_url) as tools:
        result = await ResearchAgent(tools=tools, llm=None, settings=make_settings()).run(
            "Compare the advantages and disadvantages of electric cars in Georgia."
        )
    report = result.citation_report
    assert report is not None
    assert report.citations_total > 0
    assert report.citations_verified == report.citations_total
    assert report.precision == 1.0
    assert report.removed == []
    assert report.unsupported == report.misattributed == 0
