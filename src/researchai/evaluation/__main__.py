"""Evaluation dataset command line.

python -m researchai.evaluation validate [--dataset NAME]   check cases, traps, split and lock
python -m researchai.evaluation stats [--dataset NAME]      counts by category, split, language
python -m researchai.evaluation split --write               set each case's split by the hash rule
python -m researchai.evaluation lock --write                freeze the current holdout cases
python -m researchai.evaluation run --split dev --mode offline [--case ID ...] [--limit N]
                                                            run the agent, score, write a report
python -m researchai.evaluation retrieval --split dev       recall@k of each search mode
python -m researchai.evaluation rescore RUN_DIR             score a finished run again

Holdout runs need --holdout: holdout results are reported, never used for tuning.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import shutil
import subprocess
import sys
from collections import Counter
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import get_args

from researchai.config import Settings, get_settings
from researchai.corpus.loader import Corpus, CorpusError, load_corpus
from researchai.evaluation.dataset import (
    EVAL_DIR,
    Dataset,
    DatasetError,
    assign_splits,
    load_dataset,
    validate,
    write_lock,
)
from researchai.evaluation.live import evaluate, open_backend
from researchai.evaluation.models import Split
from researchai.evaluation.retrieval import distinct, summarize
from researchai.evaluation.runner import EvaluationError, rescore
from researchai.evaluation.scoring import CaseScore
from researchai.tools.errors import ToolFailure
from researchai.tools.schemas import MAX_TOP_K, SearchMode
from researchai.tools.search import CorpusSearch

SEARCH_MODES: tuple[SearchMode, ...] = get_args(SearchMode)


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


def git_commit() -> str | None:
    """Short commit hash, with ``+dirty`` if the tree has changes; None outside a repository."""
    git = shutil.which("git")
    if git is None:
        return None
    try:
        commit = subprocess.run(  # noqa: S603 - fixed arguments, resolved executable
            [git, "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
        dirty = subprocess.run(  # noqa: S603
            [git, "status", "--porcelain"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None
    return f"{commit}+dirty" if dirty else commit


def _progress(total: int) -> Callable[[int, CaseScore], None]:
    def show(done: int, score: CaseScore) -> None:
        verdict = "pass" if score.passed else "FAIL"
        print(
            f"[{done:>3}/{total}] {score.case_id:<20} {score.status.value:<9} {verdict} "
            f"{score.total_ms / 1000:6.1f}s",
            flush=True,
        )

    return show


async def run_command(
    args: argparse.Namespace, settings: Settings, dataset: Dataset, corpus: Corpus
) -> int:
    split: Split = args.split
    total = len(dataset.split(split)) if not args.case else len(args.case)
    if args.limit is not None:
        total = min(total, args.limit)
    report, path = await evaluate(
        settings,
        dataset,
        corpus,
        split=split,
        mode=args.mode,
        case_ids=args.case or (),
        limit=args.limit,
        concurrency=args.concurrency,
        git_commit=git_commit(),
        on_result=_progress(total),
    )
    m = report.metrics
    print(f"\npassed {m.passed}; behaviour {m.behavior}; fact recall {m.fact_recall}")
    print(f"report: {path / 'report.md'}")
    return 0


async def retrieval_command(
    args: argparse.Namespace, settings: Settings, dataset: Dataset, corpus: Corpus
) -> int:
    cases = [c for c in dataset.split(args.split) if c.facts]
    ks = sorted(set(args.k))
    top_k = min(MAX_TOP_K, 3 * ks[-1])
    ranked: dict[str, dict[str, list[str]]] = {}
    async with open_backend(settings, corpus) as backend:
        search = CorpusSearch(backend.sessions, backend.embedder)
        for mode in SEARCH_MODES:
            ranked[mode] = {}
            for case in cases:
                response = await search.search(case.question, top_k=top_k, mode=mode)
                ranked[mode][case.id] = distinct([hit.doc_id for hit in response.hits])
        embedder = backend.embedder.name
    result = {
        "split": args.split,
        "cases": len(cases),
        "embedder": embedder,
        "corpus_version": corpus.version,
        "all": summarize(cases, ranked, ks),
        "by_language": {
            lang: summarize([c for c in cases if c.language.value == lang], ranked, ks)
            for lang in sorted({c.language.value for c in cases})
        },
    }
    settings.eval_output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out = settings.eval_output_dir / f"{stamp}-{args.split}-retrieval.json"
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"written: {out}")
    return 0


def main(argv: Sequence[str] | None = None, settings: Settings | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m researchai.evaluation")
    parser.add_argument("--dataset", default="default")
    parser.add_argument("--eval-dir", type=Path, default=EVAL_DIR)
    parser.add_argument("--corpus-dir", type=Path, default=Path("corpus"))
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("validate")
    sub.add_parser("stats")
    sub.add_parser("split").add_argument("--write", action="store_true", required=True)
    sub.add_parser("lock").add_argument("--write", action="store_true", required=True)
    sub.add_parser("rescore").add_argument("run_dir", type=Path)
    for name in ("run", "retrieval"):
        command = sub.add_parser(name)
        command.add_argument("--split", choices=get_args(Split), default="dev")
        command.add_argument(
            "--holdout", action="store_true", help="confirm a holdout run (never tune on it)"
        )
        if name == "run":
            command.add_argument("--mode", choices=("offline", "llm"), default="offline")
            command.add_argument("--case", action="append", help="run only these case ids")
            command.add_argument("--limit", type=int)
            command.add_argument("--concurrency", type=int, default=1)
        else:
            command.add_argument("--k", type=int, nargs="+", default=[1, 3, 5])
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
        corpus = load_corpus(args.corpus_dir)
        problems = validate(dataset, corpus)
        if args.command == "rescore" and not problems:
            settings = settings or get_settings()
            report = rescore(
                args.run_dir, dataset, corpus, cost_per_1k_tokens=settings.eval_cost_per_1k_tokens
            )
            print(f"passed {report.metrics.passed}; report: {args.run_dir / 'report.md'}")
            return 0
        if args.command in ("run", "retrieval") and not problems:
            if args.split == "holdout" and not args.holdout:
                raise EvaluationError(
                    "holdout runs need --holdout; holdout results are reported, never tuned on"
                )
            settings = settings or get_settings()
            handler = run_command if args.command == "run" else retrieval_command
            return asyncio.run(handler(args, settings, dataset, corpus))
    except (DatasetError, CorpusError, EvaluationError, ToolFailure) as exc:
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
