# ResearchAI

An evaluated, citation-verified research agent. It plans a research question,
searches a frozen bilingual (English/Georgian) corpus in parallel through MCP
tools, detects conflicting and outdated sources, and writes a report in which
every factual claim cites a document and a supporting span. When evidence is
missing it abstains instead of guessing.

Status: under construction (phase 1 of 13: scaffold).

## Development

Requires Python 3.12, [uv](https://docs.astral.sh/uv/) and Docker.

```bash
make install            # uv sync: .venv with runtime + dev dependencies
cp .env.example .env    # every setting is documented there
make check              # ruff, mypy --strict, unit tests with 95% coverage gate
make up                 # Postgres 16 + pgvector and the API on 127.0.0.1:8100
curl localhost:8100/health
make down
```

The default `LLM_PROVIDER=none` runs fully offline, so tests and CI need no
model and no API key. Skipped tests fail the run (set `ALLOW_SKIPS=1` to opt
out locally); infrastructure tests are deselected with `-m integration` instead.

License: MIT
