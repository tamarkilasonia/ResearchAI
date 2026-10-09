from __future__ import annotations

import json
from datetime import date
from typing import Any

import pytest

from researchai.agent import __main__ as agent_cli
from researchai.agent.models import (
    Calculation,
    Citation,
    Evidence,
    Report,
    ReportSection,
    ReportSentence,
    ResearchResult,
    RunStatus,
)
from researchai.agent.render import render_markdown
from researchai.corpus.models import Language, SourceType
from researchai.mcp_client.client import ResearchTools
from researchai.mcp_server.server import ToolServices, build_server
from researchai.tools.schemas import DocumentResponse
from tests.agent_fakes import CHARGING
from tests.support import make_settings
from tests.unit.test_mcp_server import FakeDocuments

EVIDENCE = Evidence(
    id="E1", sub_question_id="sq1", query="q", chunk_id="ev-en-0013#c0", doc_id="ev-en-0013",
    title="Registry", section="", text="512 points", char_start=0, char_end=10,
    published=date(2026, 7, 15), language=Language.EN, source_type=SourceType.GOVERNMENT,
    publisher="Ministry", score=0.1,
)  # fmt: skip
ANCHORED = Citation(evidence_id="E1", doc_id="ev-en-0013", chunk_id="ev-en-0013#c0",
                    quote="512 points", char_start=0, char_end=10)  # fmt: skip
LOOSE = Citation(evidence_id="E1", doc_id="ev-en-0099", chunk_id="ev-en-0099#c0", quote="x")


def result(language: Language = Language.EN, **overrides: Any) -> ResearchResult:
    report = Report(
        language=language,
        title="Report",
        summary=[ReportSentence(text="Scope.", kind="note")],
        sections=[
            ReportSection(heading="Chargers", sub_question_id="sq1", sentences=[
                ReportSentence(text="There are 512 points.", kind="claim", citations=[ANCHORED]),
                ReportSentence(
                    text="Sources disagree.", kind="conflict", citations=[ANCHORED, LOOSE]
                ),
                ReportSentence(text="No data.", kind="insufficient_evidence"),
                ReportSentence(
                    text="Costs 4.9 GEL.",
                    kind="claim",
                    citations=[ANCHORED],
                    calculation_ids=["C1"],
                ),
            ])
        ],
        generated_by="llm",
    )  # fmt: skip
    fields: dict[str, Any] = {
        "run_id": "r", "question": "q", "status": RunStatus.ANSWERED, "mode": "llm",
        "report": report, "evidence": [EVIDENCE],
        "calculations": [Calculation(id="C1", description="cost", expression="16.8*0.292",
                                     result=4.9056, evidence_ids=["E1"])],
    }  # fmt: skip
    fields.update(overrides)
    return ResearchResult(**fields)


def test_markdown_numbers_sources_and_marks_kinds() -> None:
    markdown = render_markdown(result())
    assert "- There are 512 points. [1]" in markdown
    assert "**conflict:** Sources disagree. [1][2, quote not found in source]" in markdown
    assert "*insufficient evidence:* No data." in markdown
    assert "Costs 4.9 GEL. [1][C1]" in markdown
    assert "- [C1] cost: `16.8*0.292` = 4.9056" in markdown
    assert "1. `ev-en-0013`: Registry. Ministry, 2026-07-15 (government)" in markdown
    assert "2. `ev-en-0099`: ev-en-0099" in markdown


def test_markdown_in_georgian_and_without_report() -> None:
    assert "## წყაროები" in render_markdown(result(Language.KA))
    failed = result(status=RunStatus.FAILED, report=None, reason="backend down")
    assert render_markdown(failed) == "**failed**: backend down\n"


@pytest.mark.parametrize(("as_json", "marker"), [(False, "# Research report"), (True, '"status"')])
def test_cli_runs_against_mcp(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], as_json: bool, marker: str
) -> None:
    class Search:
        async def search(self, query: str, **_: Any) -> Any:
            from researchai.tools.schemas import SearchResponse

            return SearchResponse(query=query, mode="hybrid", hits=[CHARGING])

    class Documents(FakeDocuments):
        async def fetch(
            self, doc_id: str, *, start: int | None = None, end: int | None = None
        ) -> DocumentResponse:
            body = " " * CHARGING.char_start + CHARGING.text  # the passage at its real offset
            meta = (await super().fetch("ev-en-0013")).metadata
            return DocumentResponse(
                metadata=meta, text=body, char_start=0, char_end=len(body), length=len(body)
            )

    server = build_server(ToolServices(search=Search(), documents=Documents()))
    monkeypatch.setattr(agent_cli, "ResearchTools", lambda *_args: ResearchTools(server))
    argv = ["How many public charging points are there?"] + (["--json"] if as_json else [])
    assert agent_cli.main(argv, settings=make_settings()) == 0
    out = capsys.readouterr().out
    assert marker in out
    if as_json:
        assert json.loads(out)["status"] == "answered"


def test_markdown_shows_partial_support_and_citation_precision() -> None:
    from researchai.agent.models import CitationReport, SentenceCheck

    base = result()
    assert base.report is not None
    flagged_sentence = (
        base.report.sections[0].sentences[0].model_copy(update={"support": "partial"})
    )
    section = base.report.sections[0].model_copy(
        update={"sentences": [flagged_sentence, *base.report.sections[0].sentences[1:]]}
    )
    removed = SentenceCheck(text="x", kind="claim", sub_question_id="sq1", status="unsupported",
                            method="llm", reason="r")  # fmt: skip
    checks = CitationReport(
        sentences_checked=4, citations_total=5, citations_verified=4, precision=0.6, verbatim=1,
        supported=1, partial=1, unsupported=1, misattributed=0, entailment="llm", removed=[removed],
    )  # fmt: skip
    markdown = render_markdown(
        base.model_copy(
            update={
                "report": base.report.model_copy(update={"sections": [section]}),
                "citation_report": checks,
            }
        )
    )
    assert "There are 512 points. [1] *(partially supported by its sources)*" in markdown
    assert "citation precision 60% · unsupported sentences removed: 1" in markdown
