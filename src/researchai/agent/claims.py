"""Rule-based extraction of atomic numeric claims from evidence.

A claim is one sentence (or table row) of a passage that states at least one
number. Each number becomes a :class:`NumericFact` with its unit, an optional
range upper bound, the period it refers to ("as of"), its exact offsets in the
source document, the *subject* of its clause and its threshold *labels*, so that
the conflict detector can compare like with like:

* dates are masked before numbers are read, so "210 in 2023 to 290" is no range;
* compound units consume their numbers ("kWh per 100 km" does not yield 100 km);
* the nearest date wins, ties go to the date after the number ("290 in 2024");
* relative years ("a year earlier", "წლის წინ") shift the as-of year back;
* threshold labels ("up to 101 kWh", "101-301 kWh") keep tariff tiers apart;
* table cells take their unit from the column header and their subject and
  labels from the section heading, the header and the row label.

It misses claims without numbers ("EVs are exempt from VAT"); the LLM pass in
``conflicts`` covers those when a model is available.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import date

from researchai.agent.models import Claim, Evidence, NumericFact
from researchai.corpus.models import SourceType
from researchai.safety.documents import instruction_spans, overlaps
from researchai.textutil import keyword_terms, sentence_spans

UNIT_GAP = 3  # whitespace allowed between a number and its unit
DASH = "\u2013"  # en dash, as in "15-25" ranges

MONTHS = {
    # English
    "january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6, "july": 7,
    "august": 8, "september": 9, "october": 10, "november": 11, "december": 12,
    # Georgian stems (cover case forms: ივნისი, ივნისის, ივნისში, ...)
    "იანვ": 1, "თებერვ": 2, "მარტ": 3, "აპრილ": 4, "მაის": 5, "ივნის": 6, "ივლის": 7,
    "აგვისტ": 8, "სექტემბ": 9, "ოქტომბ": 10, "ნოემბ": 11, "დეკემბ": 12,
}  # fmt: skip

UNITS: tuple[tuple[str, re.Pattern[str]], ...] = tuple(
    (unit, re.compile(pattern, re.IGNORECASE))
    for unit, pattern in (
        ("GEL/kWh", r"(?:gel|lari|ლარი)\s*(?:per|/|1)\s*(?:kwh|კვტ\.?სთ)(?:-ზე)?"),
        ("GEL/l", r"(?:gel|ლარი)\s*(?:per|/)\s*(?:litre|liter|l\b|ლიტრ)\w*"),
        ("kWh/100km", r"(?:kwh|კვტ\.?სთ)\s*(?:per|/)?\s*100\s*(?:km|კმ)(?:-ზე)?"),
        ("l/100km", r"(?:litres?|liters?|l)\s*(?:of petrol\s*)?(?:per|/)\s*100\s*(?:km|კმ)"),
        ("GEL/min", r"(?:gel|ლარი)\s*(?:per|/)\s*(?:minute|min\b|წუთ)\w*"),
        ("GEL", r"(?:gel\b|lari\b|ლარი|ლარ|₾)"),
        ("USD", r"(?:usd\b|us dollars?|აშშ დოლარი|დოლარი|\$)"),
        ("EUR", r"(?:eur\b|euros?\b|ევრო)"),
        ("%", r"(?:%|per ?cent\b|პროცენტ)"),
        ("kWh", r"(?:kwh\b|კვტ\.?სთ)"),
        ("kW", r"kw\b"),
        ("km", r"(?:km\b|კმ|kilomet)"),
        ("m", r"(?:metres?\b|meters?\b|მეტრ)"),
        ("°C", r"(?:degrees?\b|°c|გრადუს)"),
        ("days", r"(?:working days?|business days?|days?\b|სამუშაო დღე|დღე)"),
        ("years", r"(?:years?\b|წელი)"),
    )
)

_NUMBER = r"\d{1,3}(?:[ ,]\d{3})+(?:[.,]\d+)?|\d+(?:[.,]\d+)?"
_FACT = re.compile(rf"(?<![\w.])({_NUMBER})(?:\s*(?:-|{DASH}|to|until)\s*({_NUMBER}))?(?!\w)")
_YEAR = re.compile(r"(?<!\d)(199\d|20[0-4]\d)(?!\d)")
_FULL_DATE = re.compile(
    r"(?<!\d)(\d{1,2})\s+([A-Za-z]+)\s+(199\d|20[0-4]\d)"  # 1 January 2025
    r"|(199\d|20[0-4]\d)\s+წლის\s+(\d{1,2})\s+([ა-ჰ]+)"  # 2025 წლის 1 იანვრიდან
)
_MONTH_YEAR = re.compile(r"([A-Za-z]+)\s+(199\d|20[0-4]\d)|(199\d|20[0-4]\d)\s+წლის\s+([ა-ჰ]+)")
_ISO_DATE = re.compile(r"(199\d|20[0-4]\d)-(\d{2})-(\d{2})")
_PREVIOUS_YEAR = re.compile(
    r"a year earlier|year before|last year|previous year|წლის წინ|შარშან", re.IGNORECASE
)
# A comma between digits (28,400) is a thousands separator, not a clause break.
_CLAUSE_BREAK = re.compile(
    r"(?<!\d),|,(?!\d)|[;()]|\s(?:of which|while|whereas|but|compared with|versus)\s"
)
_LABEL_NUMBER = r"\d{1,3}(?:[ ,]\d{3})+|\d+"
# Numbers that qualify a fact rather than state it: range bounds and thresholds.
_LABEL = re.compile(
    rf"(?P<lo>{_LABEL_NUMBER})\s*(?:-|{DASH}|to)\s*(?P<hi>{_LABEL_NUMBER})"
    rf"|(?P<kw>up to|above|over|below|under|more than|less than)\s+(?P<n1>{_LABEL_NUMBER})"
    rf"|(?P<n2>{_LABEL_NUMBER})\s*(?:[^\s\d]+)?\s*(?P<kw2>-მდე|მდე|ზე მეტი)",
    re.IGNORECASE,
)
_AT_MOST = frozenset({"up to", "below", "under", "less than", "-მდე", "მდე"})
_TABLE_RULE = re.compile(r"^\|?[\s:|-]+\|?$")

# Pages of these publishers are about the publisher's own service (its tariffs, its
# chargers, its stock), so the publisher name qualifies every fact on the page.
SELF_DESCRIBING = frozenset({SourceType.UTILITY, SourceType.COMPANY, SourceType.DEALER})

UNIT_WORDS = frozenset(
    {"gel", "lari", "usd", "eur", "kwh", "kw", "km", "per", "percent", "litre", "liter",
     "ლარ", "ლარი", "კვტ", "სთ", "კმ", "პროცენტ", "აშშ", "დოლარ", "ევრო"}
)  # fmt: skip


def _to_float(token: str) -> float:
    """Parse a number token; raises ``ValueError`` if it is not one."""
    text = token.strip()
    if " " in text:  # space-grouped thousands; a comma left over is a decimal comma
        text = text.replace(" ", "").replace(",", ".")
    elif "," in text and "." in text:
        text = text.replace(",", "")
    elif "," in text:
        grouped = re.fullmatch(r"\d{1,3}(?:,\d{3})+", text)
        text = text.replace(",", "") if grouped else text.replace(",", ".")
    return float(text)


def parse_number(token: str) -> float | None:
    """EN ``1,800.5`` / ``1,800``; KA ``1 800,5`` / ``0,55``; plain ``16.8``; else None."""
    try:
        return _to_float(token)
    except ValueError:
        return None


def _month(word: str) -> int | None:
    lowered = word.casefold()
    for name, number in MONTHS.items():
        if lowered.startswith(name):
            return number
    return None


@dataclass(frozen=True)
class DateMention:
    start: int
    end: int
    year: int
    month: int | None


def date_mentions(text: str) -> list[DateMention]:
    """Every date-like mention, most specific first, non-overlapping."""
    found: list[DateMention] = []

    def add(start: int, end: int, year: int, month: int | None) -> None:
        if not any(start < d.end and d.start < end for d in found):
            found.append(DateMention(start, end, year, month))

    for m in _ISO_DATE.finditer(text):
        add(m.start(), m.end(), int(m.group(1)), int(m.group(2)))
    for m in _FULL_DATE.finditer(text):
        year, word = (m.group(3), m.group(2)) if m.group(3) else (m.group(4), m.group(6))
        if (month := _month(word)) is not None:
            add(m.start(), m.end(), int(year), month)
    for m in _MONTH_YEAR.finditer(text):
        word, year = (m.group(1), m.group(2)) if m.group(2) else (m.group(4), m.group(3))
        if (month := _month(word)) is not None:
            add(m.start(), m.end(), int(year), month)
    for m in _YEAR.finditer(text):
        add(m.start(), m.end(), int(m.group(1)), None)
    return found


def _unit_at(text: str, position: int) -> tuple[str | None, int]:
    """The unit right after ``position`` and the offset where it ends."""
    rest = text[position:]
    gap = len(rest) - len(rest.lstrip())
    if gap > UNIT_GAP:
        return None, position
    for unit, pattern in UNITS:
        if match := pattern.match(text, position + gap):
            return unit, match.end()
    return None, position


def unit_in(text: str) -> str | None:
    """The first unit mentioned anywhere in ``text`` (for table headers)."""
    for unit, pattern in UNITS:
        if pattern.search(text):
            return unit
    return None


def _clause_bounds(text: str, start: int, end: int) -> tuple[int, int]:
    """Bounds of the clause around ``text[start:end]`` (a colon does not end a clause)."""
    left = max((m.end() for m in _CLAUSE_BREAK.finditer(text, 0, start)), default=0)
    right = next((m.start() for m in _CLAUSE_BREAK.finditer(text, end)), len(text))
    return left, right


def subject_terms(text: str) -> list[str]:
    """Content terms: keywords minus numbers and unit words."""
    return [
        term
        for term in keyword_terms(text)
        if term not in UNIT_WORDS and not any(ch.isdigit() for ch in term)
    ]


def threshold_labels(text: str, exclude: tuple[int, int] | None = None) -> list[str]:
    """Directional threshold labels in ``text``: ``<=101``, ``>301``, ``101-301``.

    A range starting at 0 is the same as "up to" (``0-101`` == ``<=101``).
    Labels overlapping ``exclude`` (the fact's own number) are skipped.
    """

    def outside(group: str, match: re.Match[str]) -> bool:
        start, end = match.span(group)
        return start >= 0 and not (exclude is not None and start < exclude[1] and exclude[0] < end)

    labels: set[str] = set()
    for match in _LABEL.finditer(text):
        if match.group("lo") is not None and outside("lo", match) and outside("hi", match):
            low, high = _to_float(match.group("lo")), _to_float(match.group("hi"))
            labels.add(f"<={high:g}" if low == 0 else f"{low:g}-{high:g}")
            continue
        for number, keyword in (("n1", "kw"), ("n2", "kw2")):
            if match.group(number) is None or not outside(number, match):
                continue
            value = _to_float(match.group(number))
            if value > 0:
                direction = "<=" if match.group(keyword).casefold() in _AT_MOST else ">"
                labels.add(f"{direction}{value:g}")
    return sorted(labels)


def _masked(text: str, dates: Sequence[DateMention]) -> str:
    chars = list(text)
    for mention in dates:
        chars[mention.start : mention.end] = " " * (mention.end - mention.start)
    return "".join(chars)


def _nearest(dates: Sequence[DateMention], start: int, end: int) -> DateMention:
    """Nearest date; on a tie the one after the number wins ("290 in 2024")."""

    def gap(mention: DateMention) -> tuple[int, int]:
        if mention.start >= end:
            return (mention.start - end, 0)
        return (max(0, start - mention.end), 1)

    return min(dates, key=gap)


def facts_in(
    text: str,
    offset: int,
    published: date,
    *,
    context: str = "",
    context_labels: Sequence[str] = (),
    header_unit: str | None = None,
) -> list[NumericFact]:
    """Numeric facts of one sentence (or table cell).

    ``offset`` maps positions to the document; ``context`` adds subject terms
    (e.g. the section heading); ``context_labels`` adds threshold labels (e.g.
    from a table row label); ``header_unit`` applies when the number has none.
    """
    dates = date_mentions(text)
    masked = _masked(text, dates)
    facts: list[NumericFact] = []
    consumed_until = 0
    for match in _FACT.finditer(masked):
        if match.start() < consumed_until:
            continue  # part of the previous fact's unit ("per 100 km")
        low = _to_float(match.group(1))
        high = _to_float(match.group(2)) if match.group(2) else None
        unit, unit_end = _unit_at(masked, match.end())
        consumed_until = unit_end
        if high is not None and high < low:
            high = None
        left, right = _clause_bounds(masked, match.start(), match.end())
        clause = text[left:right]
        if dates:
            nearest = _nearest(dates, match.start(), match.end())
            year, month = nearest.year, nearest.month
        else:
            year, month = published.year, published.month
        if _PREVIOUS_YEAR.search(clause):
            year, month = year - 1, None
        own = (match.start() - left, match.end() - left)
        masked_clause = masked[left:right]  # dates masked: "2023 to 290" is no range
        facts.append(
            NumericFact(
                value=low,
                high=high,
                unit=unit or header_unit,
                as_of_year=year,
                as_of_month=month,
                char_start=offset + match.start(),
                char_end=offset + match.end(),
                subject=subject_terms(f"{context} {clause}"),
                labels=sorted({*context_labels, *threshold_labels(masked_clause, own)}),
            )
        )
    return facts


def _cells(row: str) -> list[tuple[int, int]]:
    """``(start, end)`` of every cell of a Markdown table row, trimmed.

    Empty cells are kept (as empty spans) so that cell positions stay aligned
    with the header's columns.
    """
    inner_start = 1 if row.startswith("|") else 0
    inner_end = len(row) - 1 if row.endswith("|") and len(row) > 1 else len(row)
    spans: list[tuple[int, int]] = []
    cursor = inner_start
    for part in row[inner_start:inner_end].split("|"):
        stripped = part.strip()
        start = cursor + (len(part) - len(part.lstrip()))
        spans.append((start, start + len(stripped)))
        cursor += len(part) + 1
    return spans


def _table_facts(
    item: Evidence, header: str, row: str, row_start: int, owner: str = ""
) -> list[NumericFact]:
    columns = [header[s:e] for s, e in _cells(header)]
    cells = _cells(row)
    facts: list[NumericFact] = []
    for index, (start, end) in enumerate(cells):
        column = columns[index] if index < len(columns) else ""
        label = " ".join(row[s:e] for j, (s, e) in enumerate(cells) if j != index)
        facts.extend(
            facts_in(
                row[start:end],
                item.char_start + row_start + start,
                item.published,
                context=f"{owner} {item.section} {column} {label}",
                context_labels=threshold_labels(label),
                header_unit=unit_in(column),
            )
        )
    return facts


def extract_claims(evidence: Iterable[Evidence], reliability: dict[str, float]) -> list[Claim]:
    claims: list[Claim] = []
    for item in evidence:
        header: str | None = None
        owner = item.publisher if item.source_type in SELF_DESCRIBING else ""
        context = f"{owner} {item.section}"
        hidden = instruction_spans(item.text)
        for start, end in sentence_spans(item.text):
            sentence = item.text[start:end]
            if overlaps(start, end, hidden):
                continue  # instruction-like text is untrusted data, never a claim
            if not sentence.startswith("|"):
                header = None
                facts = facts_in(sentence, item.char_start + start, item.published, context=context)
            elif _TABLE_RULE.match(sentence):
                continue
            elif header is None:
                header = sentence  # first row of a table block
                continue
            else:
                facts = _table_facts(item, header, sentence, start, owner)
            if not facts:
                continue
            claims.append(
                Claim(
                    id=f"K{len(claims) + 1}",
                    evidence_id=item.id,
                    sub_question_id=item.sub_question_id,
                    doc_id=item.doc_id,
                    text=sentence,
                    char_start=item.char_start + start,
                    char_end=item.char_start + end,
                    facts=facts,
                    subject=subject_terms(sentence),
                    published=item.published,
                    reliability=reliability.get(item.doc_id, 0.5),
                )
            )
    return claims
