"""Evaluation dataset command line.

python -m researchai.evaluation validate [--dataset NAME]   check cases, traps, split and lock
python -m researchai.evaluation stats [--dataset NAME]      counts by category, split, language
python -m researchai.evaluation split --write               set each case's split by the hash rule
python -m researchai.evaluation lock --write                freeze the current holdout cases
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from collections.abc import Sequence
from pathlib import Path

from researchai.corpus.loader import CorpusError, load_corpus
from researchai.evaluation.dataset import (
    EVAL_DIR,
    Dataset,
    DatasetError,
    assign_splits,
    load_dataset,
    validate,
    write_lock,
)


def stats(dataset: Dataset) -> dict[str, object]:
    cases = dataset.cases
    return {
        "dataset": dataset.info.name,
        "version": dataset.info.version,
        "cases": len(cases),
        "splits": dict(Counter(c.split for c in cases)),
        "categories": {
            category: dict(Counter(c.split for c in cases if c.category.value == category))
            for category in sorted({c.category.value for c in cases})
        },
        "languages": dict(Counter(c.language.value for c in cases)),
        "topics": dict(Counter(c.topic.value if c.topic else "none" for c in cases)),
        "behaviors": dict(Counter(c.behavior.value for c in cases)),
        "traps": dict(Counter(t.kind for t in dataset.traps)),
    }


def write_splits(dataset: Dataset) -> int:
    """Rewrite ``split = "..."`` in the case files to match the hash rule."""
    expected = assign_splits(dataset.cases, dataset.info.holdout_share, dataset.info.salt)
    changed = 0
    for path in sorted((dataset.root / "cases").glob("*.toml")):
        text = path.read_text(encoding="utf-8")

        def fix(match: re.Match[str]) -> str:
            nonlocal changed
            want = expected[match.group("id")]
            if match.group("split") != want:
                changed += 1
            return f'{match.group("head")}split = "{want}"'

        text = re.sub(
            r'(?P<head>id = "(?P<id>[a-z_]+-\d{3})"\n(?:(?!\[\[case\]\]).*\n)*?)'
            r'split = "(?P<split>dev|holdout)"',
            fix,
            text,
        )
        path.write_text(text, encoding="utf-8")
    return changed


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m researchai.evaluation")
    parser.add_argument("--dataset", default="default")
    parser.add_argument("--eval-dir", type=Path, default=EVAL_DIR)
    parser.add_argument("--corpus-dir", type=Path, default=Path("corpus"))
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("validate")
    sub.add_parser("stats")
    sub.add_parser("split").add_argument("--write", action="store_true", required=True)
    sub.add_parser("lock").add_argument("--write", action="store_true", required=True)
    args = parser.parse_args(argv)
    try:
        dataset = load_dataset(args.dataset, args.eval_dir)
        if args.command == "stats":
            print(json.dumps(stats(dataset), indent=2))
            return 0
        if args.command == "split":
            print(f"{write_splits(dataset)} case(s) changed split")
            return 0
        if args.command == "lock":
            print(json.dumps(write_lock(dataset), indent=2))
            return 0
        problems = validate(dataset, load_corpus(args.corpus_dir))
    except (DatasetError, CorpusError) as exc:
        print(exc, file=sys.stderr)
        return 1
    for problem in problems:
        print(problem, file=sys.stderr)
    if problems:
        return 1
    print(f"ok: {len(dataset.cases)} cases, {len(dataset.traps)} traps")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
