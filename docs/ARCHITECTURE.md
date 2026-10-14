# Architecture

ResearchAI answers research questions from a frozen corpus, with every factual
sentence cited to a document span and checked before the user sees it. This
page explains how the pieces fit together and why they are built this way.

```
                         +--------------------------- FastAPI (api/) ----------------------------+
 POST /research  ------> | RunService (background, concurrency limit) -> PostgresRunRepository   |
 GET  /research/{id}     |   stores every run with its full result (JSONB audit trail)           |
 POST /evaluations/run   |   evaluation runner (evaluation/live.py)                               |
                         +-----------------------------------+-----------------------------------+
                                                             |
                                       ResearchAgent (agent/runner.py): budgets, timeout, trace
                                                             |
                                    LangGraph StateGraph (agent/graph.py), typed state
                                                             |  tools only through MCP
                                                             v
                        MCP client (mcp_client/) --- streamable HTTP --->  MCP server (mcp_server/)
                                                                            search_corpus, fetch_document,
                                                                            get_source_metadata, calculate,
                                                                            live_web_search (off)
                                                                                    |
                                                         PostgreSQL 16 + pgvector (exact vector search,
                                                         full-text GIN index, documents, chunks, runs)
```

## Components

| Package | Responsibility |
|---|---|
| `corpus/` | Frozen corpus: loading and validation, section-aware chunking with exact character spans, embeddings (bge-m3 via Ollama, or a hash embedder offline), ingestion with a manifest-hash freeze |
| `db/` | SQLAlchemy 2 async models, Alembic migrations, sessions (pre-ping, connect timeout) |
| `tools/` | Tool implementations: hybrid/vector/keyword search with RRF, document store, safe calculator, optional live web search |
| `mcp_server/`, `mcp_client/` | The only way the agent reaches data: validated inputs, stable error codes, a typed client with a call log |
| `agent/` | LangGraph agent: profile, planner, parallel search, source assessment, claims, conflicts, sufficiency, calculations, synthesis, citation verification |
| `safety/` | Input guard, untrusted-document handling, output guard |
| `api/` | FastAPI endpoints, background runs, uniform errors with request ids |
| `evaluation/` | Dataset and trap registry, scorers, metrics, crash-safe runner, reports, LLM judge, labelling, Cohen's kappa |

## The research graph

```
validate_input -> input_guard -> profile -> planner
  -> search_worker x N (parallel)  <-----------------------------------+
  -> collect -> assess_sources -> extract_claims -> check_sufficiency  |
       (insufficient sub-questions and budget left: follow-up round) --+
  -> detect_conflicts -> analyze -> synthesize -> verify_citations
  -> output_guard -> finalize
```

**Rules first, LLM where rules are unsure.** Profiling, planning fallbacks, source
reliability, numeric claim extraction, rule-based conflicts, sufficiency, the
offline writer and the deterministic citation checks are code. The LLM plans,
spots disagreements without numbers, proposes calculations (only grounded
operands run), writes the report, judges entailment and classifies unclear
requests. Every LLM step validates JSON against a schema and falls back to the
rules if the call fails; the trace records it.

**Budgets** (searches, search rounds, tokens, sub-questions, wall-clock time)
are checked before work is done, never after, and exhausted budgets degrade the
run instead of crashing it.

**Why a graph.** The fan-out of search workers, the conditional follow-up round
and the early exits (invalid input, refusal, abstention) are explicit edges,
each node is small and testable on its own, and the state is typed.

## Guarantees and how they are enforced

| Guarantee | Enforced by |
|---|---|
| Every factual sentence cites a document span | Report model rejects uncited claims; quotes are anchored to exact offsets; the citation verifier re-fetches every span through MCP and removes sentences without a verified one |
| Numbers come from the sources | Numbers in a sentence must appear in its cited spans, a cited calculation or the question; calculation operands must appear in the cited evidence |
| Conflicts are shown, not silently resolved | Rule and LLM conflict detection; templated "sources disagree" sentences added (and restored after verification) for every relevant conflict; outdated values dropped from claims; low-reliability sources cannot supersede reliable ones |
| "Insufficient evidence" instead of guessing | Sufficiency verdicts, entity checks, abstention, sections without support say so |
| Retrieved text is data, never instructions | Instruction-like text is detected, hidden from every prompt, never a claim, quote, operand or verified citation; the output guard removes leaks, repeated injections and unknown links |
| Harmful requests are refused without tool use | Input guard (rules, LLM classifier in LLM mode) and an early exit in the graph |
| Repeatable evaluation | Frozen corpus with a manifest hash, exact (non-ANN) vector search, temperature 0 and a fixed seed, runs refuse a stale index |

## Data

- **Corpus**: 246 synthetic documents (English and Georgian, two topics),
  Markdown with TOML front matter, NFC/LF normalised so offsets are stable;
  `manifest.json` freezes every document's hash.
- **Index**: one row per chunk with its exact span, an embedding (`vector(1024)`)
  and a `simple`-configuration `tsvector`; vector search is exact (no ANN index)
  so results never depend on index build order.
- **Runs**: every research run is stored with its full result (plan, evidence,
  claims, conflicts, report, citation checks, safety verdicts, trace, tool calls,
  token usage), the audit trail.

## Deployment

Docker Compose runs four services: `db` (pgvector), `init` (migrations and
ingestion, then exits), `mcp-server` (not published outside the Docker network)
and `backend` (FastAPI). Ports bind to 127.0.0.1. Configuration is environment
only; `.env.example` documents every setting and a test fails if one is missing.
