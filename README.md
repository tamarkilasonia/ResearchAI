# ResearchAI

An evaluated, citation-verified research agent. It plans a research question,
searches a frozen bilingual (English/Georgian) corpus in parallel through MCP
tools, detects conflicting and outdated sources, and writes a report in which
every factual claim cites a document and a supporting span. When evidence is
missing it abstains instead of guessing.

Status: under construction (phase 2 of 13: corpus and ingestion).

## Development

Requires Python 3.12, [uv](https://docs.astral.sh/uv/) and Docker.

```bash
make install            # uv sync: .venv with runtime + dev dependencies
cp .env.example .env    # every setting is documented there
make db                 # Postgres 16 + pgvector only (needed by integration tests)
make check              # ruff, mypy --strict, unit + integration tests, 95% coverage gate
make corpus             # validate the corpus and check the frozen manifest
make up                 # full stack: db, init (migrate + ingest), API on 127.0.0.1:8100
curl localhost:8100/health
make down
```

## Corpus

A frozen, fully synthetic corpus of 228 documents in English and Georgian on
two topics: electric cars in Georgia, and Georgia's small-business tax regime.
All publishers are fictional and the numbers are not authoritative; see
[corpus/README.md](corpus/README.md). Ingestion chunks documents by section,
embeds them (offline hash embedder for tests/CI, `bge-m3` via Ollama for real
runs) and stores them in pgvector with a full-text index:

```bash
uv run python -m researchai.db upgrade
EMBEDDING_PROVIDER=ollama uv run python -m researchai.corpus ingest
```

The default `LLM_PROVIDER=none` runs fully offline, so tests and CI need no
model and no API key. Skipped tests fail the run (set `ALLOW_SKIPS=1` to opt
out locally); infrastructure tests are deselected with `-m integration` instead.

License: MIT
