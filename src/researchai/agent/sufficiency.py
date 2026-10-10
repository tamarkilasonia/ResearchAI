"""Is the evidence enough? Decided by rules, per sub-question and for key entities.

Per sub-question
    Sufficient when at least one *relevant* passage (sharing two or more content
    terms with the sub-question) comes from a source of reliability >= 0.5, or
    when relevant passages come from two or more different documents.

Key entities
    Named things in the question (capitalised words in English questions, Latin
    brand/model names in Georgian ones) must appear somewhere: first in the
    evidence, then via one keyword search of the corpus. A name the corpus never
    mentions ("Japan") means the question cannot be answered as asked, however
    much generic evidence was found.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Protocol

from researchai.agent.models import (
    Evidence,
    ResearchPlan,
    SourceAssessment,
    SubQuestion,
    SufficiencyVerdict,
)
from researchai.corpus.models import Language
from researchai.safety.documents import neutralize
from researchai.textutil import keyword_terms, text_terms
from researchai.tools.schemas import SearchMode, SearchResponse

MIN_RELIABILITY = 0.5
MIN_INDEPENDENT_DOCS = 2
MIN_SHARED_TERMS = 2
MAX_ENTITY_CHECKS = 3

_CAPITALISED = re.compile(r"\b[A-Z][A-Za-z0-9.\-]*[A-Za-z0-9]\b|\b[A-Z]\b")
_LATIN = re.compile(r"\b[A-Za-z][A-Za-z0-9.\-]*[A-Za-z0-9]\b")
# Capitalised words that are not entities (question words, pronouns, units).
NOT_ENTITIES = frozenset(
    {
        "I", "A", "An", "The", "What", "Which", "Who", "Whom", "Whose", "When", "Where", "Why",
        "How", "Is", "Are", "Was", "Were", "Do", "Does", "Did", "Can", "Could", "Should",
        "Would", "Will", "Compare", "Give", "Tell", "List", "Explain", "Please", "In", "For",
        "If", "And", "Or", "GEL", "USD", "EUR", "EV", "EVs", "VAT", "AC", "DC", "kWh", "km",
    }
)  # fmt: skip


class EntitySearch(Protocol):
    async def search_corpus(
        self, query: str, *, top_k: int = 8, mode: SearchMode = "hybrid"
    ) -> SearchResponse: ...


def relevant(item: Evidence, sub_question: SubQuestion) -> bool:
    terms = set(keyword_terms(" ".join([sub_question.text, *sub_question.queries])))
    shared = len(terms & text_terms(neutralize(item.text)))
    return shared >= min(MIN_SHARED_TERMS, len(terms))


def verdicts(
    plan: ResearchPlan, evidence: Sequence[Evidence], assessments: Sequence[SourceAssessment]
) -> list[SufficiencyVerdict]:
    reliability = {a.doc_id: a.reliability for a in assessments}
    results: list[SufficiencyVerdict] = []
    for sq in plan.sub_questions:
        hits = [e for e in evidence if e.sub_question_id == sq.id and relevant(e, sq)]
        trusted = [e for e in hits if reliability.get(e.doc_id, 0.0) >= MIN_RELIABILITY]
        documents = {e.doc_id for e in hits}
        if trusted:
            ok, reason = True, f"{len(trusted)} relevant passage(s) from reliable sources"
        elif len(documents) >= MIN_INDEPENDENT_DOCS:
            ok, reason = True, f"relevant passages from {len(documents)} independent documents"
        elif hits:
            ok, reason = False, "only one relevant low-reliability source"
        else:
            ok, reason = False, "no relevant passage"
        results.append(
            SufficiencyVerdict(
                sub_question_id=sq.id,
                sufficient=ok,
                relevant_evidence=[e.id for e in hits],
                reason=reason,
            )
        )
    return results


def key_entities(question: str, language: Language) -> list[str]:
    """Names the answer must be about, in question order, without duplicates."""
    if language is Language.KA:
        found = _LATIN.findall(question)
    else:
        words = question.split()
        first = words[0] if words else ""
        found = [w for w in _CAPITALISED.findall(question) if w != first.strip("?,.!")]
    entities: list[str] = []
    for word in found:
        if word in NOT_ENTITIES or len(word) < 2 or word.isdigit():
            continue
        if word not in entities:
            entities.append(word)
    return entities


def mentioned(entity: str, texts: Sequence[str]) -> bool:
    needle = entity.casefold()
    return any(needle in text.casefold() for text in texts)


async def missing_entities(
    entities: Sequence[str],
    evidence: Sequence[Evidence],
    tools: EntitySearch,
    searches_left: int,
) -> tuple[list[str], int]:
    """Entities absent from the evidence *and* the corpus, and the searches used.

    An entity is reported missing only when a keyword search confirms that no
    passage mentions it; without search budget left, it is given the benefit of
    the doubt.
    """
    texts = [e.text for e in evidence] + [e.title for e in evidence]
    missing: list[str] = []
    used = 0
    for entity in entities[:MAX_ENTITY_CHECKS]:
        if mentioned(entity, texts):
            continue
        if used >= searches_left:
            break
        response = await tools.search_corpus(entity, top_k=5, mode="keyword")
        used += 1
        if not mentioned(
            entity, [h.text for h in response.hits] + [h.title for h in response.hits]
        ):
            missing.append(entity)
    return missing, used
