"""Output guard: the last check before a report reaches the user.

Removes report sentences (and replaces a title or heading) that

* repeat a run of words from any system prompt (``system_prompt_leak``),
* repeat instruction-like text found in a retrieved passage
  (``injected_content``), or
* contain a link that appears in none of the retrieved passages
  (``unknown_link``; links are how injected text exfiltrates data).

It runs after the citation verifier, so a sentence that survives here has
verified citations and none of the above.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence

from researchai.agent.models import Evidence, Report, ReportSentence
from researchai.agent.prompts import SYSTEM_PROMPTS
from researchai.agent.verification import fill_empty_sections
from researchai.corpus.models import Language
from researchai.safety.models import InjectionFinding, OutputIssue

PROMPT_NGRAM = 8  # words; long enough that ordinary report sentences never share one
INJECTION_NGRAM = 6
_WORD = re.compile(r"\w+")
_LINK = re.compile(r"https?://[^\s)\]>\"']+|\bwww\.[^\s)\]>\"']+", re.IGNORECASE)
_NEUTRAL_TITLE = {Language.EN: "Research report", Language.KA: "კვლევის ანგარიში"}


def _ngrams(text: str, n: int) -> set[tuple[str, ...]]:
    words = [w.lower() for w in _WORD.findall(text)]
    return {tuple(words[i : i + n]) for i in range(len(words) - n + 1)}


def _union(texts: Iterable[str], n: int) -> set[tuple[str, ...]]:
    found: set[tuple[str, ...]] = set()
    for text in texts:
        found |= _ngrams(text, n)
    return found


_PROMPT_NGRAMS = _union(SYSTEM_PROMPTS, PROMPT_NGRAM)


def links(text: str) -> list[str]:
    return [m.group().rstrip(".,;:") for m in _LINK.finditer(text)]


class OutputGuard:
    def __init__(
        self, evidence: Sequence[Evidence], injections: Sequence[InjectionFinding]
    ) -> None:
        self._known_links = {link for e in evidence for link in links(e.text)}
        self._injected = _union((i.text for i in injections), INJECTION_NGRAM)

    def issue(self, text: str) -> OutputIssue | None:
        if _ngrams(text, PROMPT_NGRAM) & _PROMPT_NGRAMS:
            return OutputIssue(sentence=text, reason="system_prompt_leak")
        if _ngrams(text, INJECTION_NGRAM) & self._injected:
            return OutputIssue(sentence=text, reason="injected_content")
        if any(link not in self._known_links for link in links(text)):
            return OutputIssue(sentence=text, reason="unknown_link")
        return None

    def apply(self, report: Report) -> tuple[Report, list[OutputIssue]]:
        issues: list[OutputIssue] = []

        def keep(sentences: Sequence[ReportSentence]) -> list[ReportSentence]:
            kept = []
            for sentence in sentences:
                if (found := self.issue(sentence.text)) is not None:
                    issues.append(found)
                else:
                    kept.append(sentence)
            return kept

        title = report.title
        if (found := self.issue(title)) is not None:
            issues.append(found)
            title = _NEUTRAL_TITLE[report.language]
        sections = []
        for number, section in enumerate(report.sections, start=1):
            heading = section.heading
            if (found := self.issue(heading)) is not None:
                issues.append(found)
                heading = f"{number}."
            sections.append(
                section.model_copy(
                    update={"heading": heading, "sentences": keep(section.sentences)}
                )
            )
        if not issues:
            return report, []
        cleaned = report.model_copy(
            update={"title": title, "summary": keep(report.summary), "sections": sections}
        )
        return fill_empty_sections(cleaned), issues
