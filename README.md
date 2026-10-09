# ResearchAI

An evaluated, citation-verified research agent. It plans a research question,
searches a frozen bilingual (English/Georgian) corpus in parallel through MCP
tools, detects conflicting and outdated sources, and writes a report in which
every factual claim cites a document and a supporting span. When evidence is
missing it abstains instead of guessing.

Status: under construction (phase 3 of 13: MCP tool server).

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

## Tools (MCP)

The agent reaches data only through an MCP server (streamable HTTP); a test
fails if agent-side code imports the database or tool implementations.

| Tool | What it does |
|---|---|
| `search_corpus` | Hybrid search: pgvector cosine + PostgreSQL full text, fused with reciprocal rank fusion. Returns passages with document ids and exact character spans; filters by language, topic, date and source type. |
| `fetch_document` | Full text or a `[start, end)` span of a document, with metadata. |
| `get_source_metadata` | Publisher, source type, dates, URL and language for up to 20 documents. |
| `calculate` | Safe arithmetic (AST allow-list, size limits), e.g. cost per 100 km. |
| `live_web_search` | Optional SearxNG search. Off by default and not even registered unless `LIVE_WEB_SEARCH_ENABLED=true`; never used in evaluation. |

Failures carry stable codes (`INVALID_ARGUMENT`, `NOT_FOUND`, `INDEX_NOT_READY`,
`BACKEND_UNAVAILABLE`, `DIVISION_BY_ZERO`, ...) and never a stack trace.
Retrieved text is marked `content_trust: "untrusted"`. Search refuses to run
against an index built with a different embedding model.

```bash
uv run python -m researchai.mcp_server   # serves http://127.0.0.1:8765/mcp
```

License: MIT
