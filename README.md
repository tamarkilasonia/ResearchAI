# ResearchAI

An evaluated, citation-verified research agent. It plans a research question,
searches a frozen bilingual (English/Georgian) corpus in parallel through MCP
tools, detects conflicting and outdated sources, and writes a report in which
every factual claim cites a document and a supporting span. When evidence is
missing it abstains instead of guessing.

Status: under construction (phase 6 of 13: citation verification).

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

## Agent (LangGraph)

```
validate_input -> input_guard -> profile -> planner
  -> search_worker x N (parallel)  <-----------------------------------+
  -> collect -> assess_sources -> extract_claims -> check_sufficiency  |
       (insufficient sub-questions and budget left: follow-up round) --+
  -> detect_conflicts -> analyze (calculations via MCP) -> synthesize
  -> verify_citations -> finalize
```

- **Rules first, LLM where rules are unsure.** Profiling (language, topic, question
  kind) is rule-based. With `LLM_PROVIDER=none` the planner uses an aspect catalogue
  and the writer copies the best-matching sentences verbatim, each cited with its
  exact span: an offline baseline that cannot invent facts, and also cannot reason,
  compute or recommend (its reports say so). With an LLM, plans and reports are
  validated JSON; any LLM failure falls back to the rule path and is recorded.
- **Citations are anchored.** Quotes the LLM cites are located in the cited passage
  (exact, then whitespace/case-insensitive); claims whose citations point at unknown
  evidence are dropped, never shown unsupported.
- **Calculations are grounded.** The LLM may propose arithmetic, but it only runs if
  every operand appears in the evidence it cites (English and Georgian number
  formats); the MCP `calculate` tool computes the result.
- **Budgets** (all configurable): searches per run, search rounds, LLM tokens
  (checked before every call), sub-questions, wall-clock timeout. Exhausted budgets
  degrade the run and are reported; they never crash it.
- **No answer without evidence.** Off-corpus questions abstain; a tool backend
  failure ends the run as `failed`.

### Verification (rules first, LLM where rules are unsure)

- **Source assessment:** reliability by source type (official > industry/NGO >
  news > dealer > blog > forum) and freshness at the corpus snapshot date (via
  the MCP `get_source_metadata` tool). Evidence is re-ranked so current official
  sources come first.
- **Claim extraction:** every number in the evidence becomes a fact with unit,
  range, as-of period, exact span, subject and directional threshold labels
  (`<=101 kWh`, `>500000 GEL`). Table cells take their unit from the header.
- **Conflict detection:** directly comparable facts from different documents
  (same unit, labels, places/models/qualifiers, overlapping subject) with
  different values become *outdated* (an older value superseded by a newer
  source) or *contradiction* (same period). Forecasts ("by 2030") never count
  as current values. With an LLM, disagreements without numbers ("VAT applies"
  vs "exempt") are flagged too, with verbatim quotes that must anchor.
- **Sufficiency:** a sub-question needs a relevant passage from a reliable
  source or two independent documents; otherwise a follow-up round runs (within
  budget) or the report says *insufficient evidence*. Names in the question the
  corpus never mentions ("Japan") are confirmed by a keyword search; a focused
  question about them abstains.
- **Guarantees in every report:** each detected conflict is stated with both
  sides cited (added from a template if the LLM leaves it out), and superseded
  values are never presented as current facts.

### Citation verification

Every cited sentence is checked before the user sees the report
(`verify_citations`, between `synthesize` and `finalize`):

1. **Spans:** each cited document is re-fetched through MCP and the cited span
   must equal the quote. Unverifiable citations are dropped; a sentence left
   without one is removed as *unsupported*.
2. **Numbers:** every number in the sentence must appear in its cited spans, a
   cited calculation or the question. Found only in another passage, the
   sentence is *misattributed*; found nowhere, *unsupported*. Both are removed.
3. **Entailment** (only for non-verbatim sentences): one batched LLM judgement
   (supported / partial / not supported), or lexical coverage offline. Partially
   supported sentences stay, marked as such.

Each result carries a `citation_report`: citation precision (citations whose
span is verified and supports the sentence), counts per verdict, and the
removed and flagged sentences with reasons.

```bash
uv run python -m researchai.mcp_server &                    # tools on :8765
uv run python -m researchai.agent "How many public charging points are there?"
LLM_PROVIDER=ollama uv run python -m researchai.agent --json "..."  # full trace
```

For Ollama, start the server with a context window at least `LLM_CONTEXT_TOKENS`
(`OLLAMA_CONTEXT_LENGTH=16384 ollama serve`); the default window is smaller and
would silently truncate prompts.

License: MIT
