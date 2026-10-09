"""Render a research result as Markdown with numbered source references."""

from __future__ import annotations

from researchai.agent.models import Citation, ReportSentence, ResearchResult
from researchai.corpus.models import Language

LABELS = {
    Language.EN: {
        "sources": "Sources",
        "status": "Status",
        "calculations": "Calculations",
        "insufficient": "insufficient evidence",
        "conflict": "conflict",
        "unanchored": "quote not found in source",
    },
    Language.KA: {
        "sources": "წყაროები",
        "status": "სტატუსი",
        "calculations": "გამოთვლები",
        "insufficient": "არასაკმარისი მონაცემები",
        "conflict": "წინააღმდეგობა",
        "unanchored": "ციტატა წყაროში ვერ მოიძებნა",
    },
}


def render_markdown(result: ResearchResult) -> str:
    report = result.report
    if report is None:
        return f"**{result.status.value}**: {result.reason or 'no report'}\n"
    labels = LABELS[report.language]
    evidence = {e.id: e for e in result.evidence}
    numbers: dict[str, int] = {}

    def ref(citation: Citation) -> str:
        number = numbers.setdefault(citation.doc_id, len(numbers) + 1)
        return f"[{number}]" if citation.anchored else f"[{number}, {labels['unanchored']}]"

    def line(sentence: ReportSentence) -> str:
        refs = "".join(sorted({ref(c) for c in sentence.citations}))
        calcs = "".join(f"[{c}]" for c in sentence.calculation_ids)
        prefix = {
            "insufficient_evidence": f"*{labels['insufficient']}:* ",
            "conflict": f"**{labels['conflict']}:** ",
        }.get(sentence.kind, "")
        return f"{prefix}{sentence.text} {refs}{calcs}".rstrip()

    parts = [f"# {report.title}", ""]
    parts += [line(s) for s in report.summary]
    for section in report.sections:
        parts += ["", f"## {section.heading}", ""]
        parts += [f"- {line(s)}" for s in section.sentences]
    if result.calculations:
        parts += ["", f"## {labels['calculations']}", ""]
        parts += [
            f"- [{c.id}] {c.description}: `{c.expression}` = {c.result:g}"
            for c in result.calculations
        ]
    if numbers:
        parts += ["", f"## {labels['sources']}", ""]
        docs = {e.doc_id: e for e in evidence.values()}
        for doc_id, number in sorted(numbers.items(), key=lambda pair: pair[1]):
            item = docs.get(doc_id)
            detail = (
                f"{item.title}. {item.publisher}, {item.published.isoformat()} "
                f"({item.source_type.value})"
                if item
                else doc_id
            )
            parts.append(f"{number}. `{doc_id}`: {detail}")
    parts += ["", f"*{labels['status']}: {result.status.value} · {result.mode}*", ""]
    return "\n".join(parts)
