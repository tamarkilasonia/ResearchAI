"""Corpus command line.

python -m researchai.corpus validate           check every document
python -m researchai.corpus stats              counts by topic, language, source type
python -m researchai.corpus manifest --check   fail if the frozen manifest is stale
python -m researchai.corpus manifest --write   regenerate corpus/manifest.json
python -m researchai.corpus ingest [--force]   load into PostgreSQL + pgvector
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from collections import Counter
from collections.abc import Sequence
from pathlib import Path

from researchai.config import Settings, get_settings
from researchai.corpus.embeddings import build_embedder
from researchai.corpus.ingest import ingest, plan_chunks
from researchai.corpus.loader import CorpusError, load_corpus, read_manifest, write_manifest
from researchai.db.session import create_engine, session_factory


def _stats(root: Path) -> dict[str, object]:
    corpus = load_corpus(root)
    chunks = [chunk for _, chunks in plan_chunks(corpus) for chunk in chunks]
    docs = corpus.documents
    return {
        "version": corpus.version,
        "documents": len(docs),
        "chunks": len(chunks),
        "by_topic": dict(Counter(d.meta.topic.value for d in docs)),
        "by_language": dict(Counter(d.meta.language.value for d in docs)),
        "by_topic_language": dict(
            Counter(f"{d.meta.topic.value}/{d.meta.language.value}" for d in docs)
        ),
        "by_source_type": dict(Counter(d.meta.source_type.value for d in docs)),
        "synthetic": sum(d.meta.synthetic for d in docs),
    }


async def _ingest(settings: Settings, root: Path, *, force: bool) -> int:
    corpus = load_corpus(root)
    engine = create_engine(settings)
    try:
        report = await ingest(
            corpus,
            build_embedder(settings),
            session_factory(engine),
            batch_size=settings.embedding_batch_size,
            force=force,
        )
    finally:
        await engine.dispose()
    print(json.dumps(report.__dict__, indent=2))
    return 0


def main(argv: Sequence[str] | None = None, settings: Settings | None = None) -> int:
    settings = settings or get_settings()
    parser = argparse.ArgumentParser(prog="python -m researchai.corpus")
    parser.add_argument("--dir", type=Path, default=settings.corpus_dir, help="corpus root")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("validate")
    sub.add_parser("stats")
    manifest = sub.add_parser("manifest")
    mode = manifest.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true")
    mode.add_argument("--write", action="store_true")
    ingest_cmd = sub.add_parser("ingest")
    ingest_cmd.add_argument("--force", action="store_true", help="re-ingest even if unchanged")
    args = parser.parse_args(argv)
    root: Path = args.dir

    try:
        if args.command == "validate":
            corpus = load_corpus(root)
            print(f"ok: {len(corpus.documents)} documents, corpus version {corpus.version}")
        elif args.command == "stats":
            print(json.dumps(_stats(root), indent=2, ensure_ascii=False))
        elif args.command == "manifest" and args.write:
            print(f"wrote {write_manifest(root, load_corpus(root))}")
        elif args.command == "manifest":
            if load_corpus(root).manifest() != read_manifest(root):
                print("manifest is stale: run `manifest --write` and bump the corpus version")
                return 1
            print("manifest matches corpus")
        else:
            return asyncio.run(_ingest(settings, root, force=args.force))
    except CorpusError as exc:
        print(exc, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
