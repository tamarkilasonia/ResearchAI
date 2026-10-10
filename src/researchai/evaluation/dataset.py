"""Loading and validating the evaluation dataset and the trap registry.

Layout (``eval/``)::

    traps.toml                         planted traps (ground truth, outside the corpus)
    datasets/<name>/dataset.toml       name, version, corpus version, holdout rule
    datasets/<name>/cases/<category>.toml
    datasets/<name>/holdout.lock       digest of the holdout cases

Validation is strict because every metric later depends on these labels: every
reference value must appear in a cited document, every trap value in its
documents, every injected payload verbatim in its document, and the holdout
split must equal a deterministic, hash-based assignment (nobody picks which
cases are held out). Editing a holdout case changes the lock digest, so it
shows up in review.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import tomllib
import unicodedata
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

from pydantic import Field, ValidationError

from researchai.agent.analysis import UNIT_CONSTANTS, evidence_numbers, operands
from researchai.agent.llm import LLMError
from researchai.agent.profile import GEORGIAN_QUESTION_SHARE
from researchai.corpus.loader import Corpus
from researchai.corpus.models import Language
from researchai.evaluation.models import (
    Behavior,
    Case,
    Category,
    Split,
    Trap,
    _Model,
)
from researchai.textutil import georgian_share
from researchai.tools.calculator import evaluate
from researchai.tools.errors import ToolFailure

EVAL_DIR = Path("eval")
REGISTRY_FILE = "traps.toml"
MIN_CASES, MAX_CASES = 100, 150
_CANONICAL_NUMBER = re.compile(r"^-?\d+(\.\d+)?%?$")
_RELATIVE_TOLERANCE = 1e-4


class DatasetError(Exception):
    def __init__(self, problems: Sequence[str]) -> None:
        self.problems = list(problems)
        super().__init__(f"{len(problems)} dataset problem(s):\n" + "\n".join(problems))


class DatasetInfo(_Model):
    name: str = Field(pattern=r"^[a-z0-9_-]{1,64}$")
    version: str
    corpus_version: str
    description: str
    holdout_share: float = Field(gt=0, lt=0.5)
    salt: str = Field(min_length=8)


@dataclass(frozen=True)
class Dataset:
    info: DatasetInfo
    cases: tuple[Case, ...]
    traps: tuple[Trap, ...]
    root: Path

    def split(self, name: Split) -> list[Case]:
        return [case for case in self.cases if case.split == name]

    def trap(self, trap_id: str) -> Trap:
        return next(trap for trap in self.traps if trap.id == trap_id)

    @property
    def lock_file(self) -> Path:
        return self.root / "holdout.lock"


def _read_toml(path: Path) -> dict[str, object]:
    try:
        return tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise DatasetError([f"{path}: {exc}"]) from exc


def _entries(path: Path, key: str) -> list[dict[str, object]]:
    raw = _read_toml(path).get(key, [])
    if not isinstance(raw, list):
        raise DatasetError([f"{path}: [[{key}]] must be an array of tables"])
    return raw


def load_registry(path: Path) -> list[Trap]:
    traps: list[Trap] = []
    problems: list[str] = []
    for entry in _entries(path, "trap"):
        try:
            traps.append(Trap.model_validate(entry))
        except ValidationError as exc:
            problems.append(f"{path}: trap {entry.get('id')}: {exc}")
    if problems:
        raise DatasetError(problems)
    return traps


def load_dataset(name: str = "default", eval_dir: Path = EVAL_DIR) -> Dataset:
    root = eval_dir / "datasets" / name
    info = DatasetInfo.model_validate(_read_toml(root / "dataset.toml"))
    cases: list[Case] = []
    problems: list[str] = []
    for path in sorted((root / "cases").glob("*.toml")):
        for entry in _entries(path, "case"):
            try:
                case = Case.model_validate(entry)
            except ValidationError as exc:
                problems.append(f"{path.name}: case {entry.get('id')}: {exc}")
                continue
            if case.category.value != path.stem:
                problems.append(f"{path.name}: {case.id} has category {case.category.value}")
            cases.append(case)
    if problems:
        raise DatasetError(problems)
    return Dataset(info, tuple(cases), tuple(load_registry(eval_dir / REGISTRY_FILE)), root)


# ----------------------------------------------------------------- split and lock


def assign_splits(cases: Iterable[Case], share: float, salt: str) -> dict[str, Split]:
    """Deterministic stratified holdout: per category, groups in hash order until ``share``.

    A group is the cases about one trap (or a single case), so a trap's cases
    never straddle the split.
    """
    by_category: dict[Category, dict[str, list[Case]]] = defaultdict(lambda: defaultdict(list))
    for case in cases:
        by_category[case.category][case.group].append(case)
    result: dict[str, Split] = {}
    for groups in by_category.values():
        size = sum(len(members) for members in groups.values())
        target = max(1, math.floor(share * size + 0.5))
        taken = 0
        for group in sorted(
            groups, key=lambda g: hashlib.sha256(f"{salt}:{g}".encode()).hexdigest()
        ):
            split: Split = "holdout" if taken < target else "dev"
            for case in groups[group]:
                result[case.id] = split
            if split == "holdout":
                taken += len(groups[group])
    return result


def holdout_digest(cases: Iterable[Case]) -> str:
    holdout = sorted((c for c in cases if c.split == "holdout"), key=lambda c: c.id)
    payload = json.dumps(
        [c.model_dump(mode="json") for c in holdout], sort_keys=True, ensure_ascii=False
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def read_lock(dataset: Dataset) -> dict[str, object]:
    return (
        json.loads(dataset.lock_file.read_text(encoding="utf-8"))
        if dataset.lock_file.exists()
        else {}
    )


def write_lock(dataset: Dataset) -> dict[str, object]:
    lock: dict[str, object] = {
        "dataset": dataset.info.name,
        "version": dataset.info.version,
        "holdout_cases": len(dataset.split("holdout")),
        "sha256": holdout_digest(dataset.cases),
    }
    dataset.lock_file.write_text(json.dumps(lock, indent=2) + "\n", encoding="utf-8")
    return lock


# ----------------------------------------------------------------- grounding


def appears(value: str, text: str) -> bool:
    """Canonical numbers ("700000", "0.239", "18%") match any EN/KA spelling.

    Anything else matches as case-insensitive text.
    """
    if _CANONICAL_NUMBER.match(value):
        number = round(float(value.rstrip("%")), 6)
        return number in evidence_numbers(text)
    return unicodedata.normalize("NFC", value).casefold() in text.casefold()


def _grounded(value: str, doc_ids: Sequence[str], corpus: Corpus) -> bool:
    return any(appears(value, corpus.get(doc).body) for doc in doc_ids if _exists(doc, corpus))


def _exists(doc_id: str, corpus: Corpus) -> bool:
    try:
        corpus.get(doc_id)
    except KeyError:
        return False
    return True


# ----------------------------------------------------------------- validation


def _check_traps(dataset: Dataset, corpus: Corpus) -> list[str]:
    problems: list[str] = []
    ids = [t.id for t in dataset.traps]
    problems += [f"duplicate trap id {i}" for i, n in Counter(ids).items() if n > 1]
    for trap in dataset.traps:
        for doc in [*trap.planted, *trap.reference]:
            if not _exists(doc, corpus):
                problems.append(f"{trap.id}: unknown document {doc}")
        for value in trap.planted_values:
            if not _grounded(value, trap.planted, corpus):
                problems.append(f"{trap.id}: planted value {value!r} not in {trap.planted}")
        for value in trap.reference_values:
            if not _grounded(value, trap.reference, corpus):
                problems.append(f"{trap.id}: reference value {value!r} not in {trap.reference}")
        if trap.payload and not any(
            _exists(d, corpus) and trap.payload in corpus.get(d).body for d in trap.planted
        ):
            problems.append(f"{trap.id}: payload not found verbatim in {trap.planted}")
    return problems


_CATEGORY_BEHAVIOR = {
    Category.HARMFUL: Behavior.REFUSE,
    Category.UNANSWERABLE: Behavior.ABSTAIN,
    Category.CONFLICT: Behavior.FLAG_CONFLICT,
}
_CATEGORY_TRAPS = {
    Category.CONFLICT: {"contradiction", "low_reliability"},
    Category.OUTDATED: {"outdated"},
    Category.INJECTION: {"injection"},
}


def _check_case(case: Case, traps: dict[str, Trap], corpus: Corpus) -> list[str]:
    p: list[str] = []
    where = case.id
    if not case.id.startswith(case.category.value + "-"):
        p.append(f"{where}: id must start with '{case.category.value}-'")
    expected_behavior = _CATEGORY_BEHAVIOR.get(case.category)
    if expected_behavior is not None and case.behavior is not expected_behavior:
        p.append(f"{where}: {case.category.value} cases must be {expected_behavior.value}")
    if case.behavior in (Behavior.REFUSE, Behavior.ABSTAIN) and case.category not in (
        Category.HARMFUL,
        Category.UNANSWERABLE,
        Category.GEORGIAN,
    ):
        p.append(f"{where}: only harmful/unanswerable/georgian cases refuse or abstain")
    georgian = georgian_share(case.question) >= GEORGIAN_QUESTION_SHARE
    if georgian != (case.language is Language.KA):
        p.append(f"{where}: language {case.language.value} does not match the question")
    if case.category is Category.GEORGIAN and case.language is not Language.KA:
        p.append(f"{where}: georgian cases are asked in Georgian")

    if case.behavior is Behavior.REFUSE:
        if case.refusal_category is None or case.refusal_category == "none":
            p.append(f"{where}: a refusal names its category")
        if case.facts or case.expected_tools:
            p.append(f"{where}: a refused request has no facts and makes no tool call")
    elif case.refusal_category is not None:
        p.append(f"{where}: refusal_category only for refusals")
    if case.behavior is Behavior.ABSTAIN:
        if not case.missing or case.facts:
            p.append(f"{where}: an abstention names what is missing and has no facts")
    elif case.missing:
        p.append(f"{where}: 'missing' only for abstentions")
    if case.behavior in (Behavior.ANSWER, Behavior.FLAG_CONFLICT) and (
        not case.facts or not case.key_points
    ):
        p.append(f"{where}: an answer needs reference facts and key points")
    if case.behavior is not Behavior.REFUSE and "search_corpus" not in case.expected_tools:
        p.append(f"{where}: every non-refused case searches the corpus")
    if ("calculate" in case.expected_tools) != (case.calculation is not None):
        p.append(f"{where}: 'calculate' is expected exactly when a calculation is given")
    if case.category is Category.NUMERIC and case.calculation is None:
        p.append(f"{where}: numeric cases include the expected calculation")

    for fact in case.facts:
        for doc in fact.sources:
            if not _exists(doc, corpus):
                p.append(f"{where}: unknown document {doc}")
        for value in fact.values:
            if not _grounded(value, fact.sources, corpus):
                p.append(f"{where}: value {value!r} not found in {fact.sources}")
    if (calc := case.calculation) is not None:
        try:
            result = evaluate(calc.expression)
            values = operands(calc.expression)
        except (LLMError, ToolFailure) as exc:
            p.append(f"{where}: bad expression {calc.expression!r}: {exc}")
        else:
            if not math.isclose(result, calc.result, rel_tol=_RELATIVE_TOLERANCE):
                p.append(f"{where}: {calc.expression} = {result:g}, not {calc.result:g}")
            available = evidence_numbers(case.question).union(
                *(evidence_numbers(corpus.get(d).body) for d in calc.sources if _exists(d, corpus))
            )
            missing = [v for v in values if v not in available and v not in UNIT_CONSTANTS]
            if missing:
                p.append(f"{where}: operands {missing} not in {calc.sources}")

    kinds = _CATEGORY_TRAPS.get(case.category)
    for trap_id in case.traps:
        if trap_id not in traps:
            p.append(f"{where}: unknown trap {trap_id}")
        elif kinds is not None and traps[trap_id].kind not in kinds:
            p.append(f"{where}: trap {trap_id} is not a {sorted(kinds)} trap")
    if kinds is not None and not case.traps:
        p.append(f"{where}: {case.category.value} cases reference a trap")
    case_traps = [traps[t] for t in case.traps if t in traps]
    for value in case.outdated_values:
        if not any(_grounded(value, t.planted, corpus) for t in case_traps):
            p.append(f"{where}: outdated value {value!r} is not in its traps' documents")
    if case.category is Category.OUTDATED and not case.outdated_values:
        p.append(f"{where}: outdated cases list the values that must not be current")
    if case.category is Category.INJECTION and not case.forbidden:
        p.append(f"{where}: injection cases list text that must never appear")
    return p


def validate(dataset: Dataset, corpus: Corpus) -> list[str]:
    problems: list[str] = []
    if dataset.info.corpus_version != corpus.version:
        problems.append(
            f"dataset targets corpus {dataset.info.corpus_version}, corpus is {corpus.version}"
        )
    problems += _check_traps(dataset, corpus)
    traps = {t.id: t for t in dataset.traps}
    ids = [c.id for c in dataset.cases]
    problems += [f"duplicate case id {i}" for i, n in Counter(ids).items() if n > 1]
    questions = [c.question for c in dataset.cases]
    problems += [f"duplicate question {q!r}" for q, n in Counter(questions).items() if n > 1]
    for case in dataset.cases:
        problems += _check_case(case, traps, corpus)
    used = {t for c in dataset.cases for t in c.traps}
    problems += [f"trap {t} is not used by any case" for t in sorted(set(traps) - used)]

    if not MIN_CASES <= len(dataset.cases) <= MAX_CASES:
        problems.append(f"{len(dataset.cases)} cases; expected {MIN_CASES}-{MAX_CASES}")
    expected = assign_splits(dataset.cases, dataset.info.holdout_share, dataset.info.salt)
    problems += [
        f"{c.id}: split is {c.split}, the hash rule says {expected[c.id]}"
        for c in dataset.cases
        if c.split != expected[c.id]
    ]
    for category in Category:
        splits = {c.split for c in dataset.cases if c.category is category}
        if splits != {"dev", "holdout"}:
            problems.append(f"category {category.value} must have dev and holdout cases")
    lock = read_lock(dataset)
    if lock.get("sha256") != holdout_digest(dataset.cases):
        problems.append(
            "holdout cases differ from holdout.lock: holdout labels are frozen; fix only "
            "labelling errors, document them in eval/CHANGELOG.md, then re-lock"
        )
    return problems
