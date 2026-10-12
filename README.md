# ResearchAI

**A research agent that only says what its sources say, and proves it.**

Ask a research question. ResearchAI plans the research, searches its sources in
parallel, checks them against each other and writes a report in which **every
factual sentence cites a document span**. Before you see the report, every
citation is checked again: unsupported sentences are removed, conflicting
sources are shown side by side, and missing evidence is called missing.

Then it is **measured**: 135 labelled questions, 29 traps planted in the corpus,
a held-out test set, and an LLM judge whose agreement with human labels is
reported next to every score.

English and Georgian · LangGraph · MCP · PostgreSQL + pgvector · local models (Ollama)

---

## Example

> **Question:** How many public charging points does Batumi have?

> According to the Ministry of Transport and Mobility, Batumi has 61 public charging
> points. **[1]**
>
> **Sources disagree:** Ministry of Transport and Mobility (2026-07-15) reports
> "Batumi has 61 public charging points", while Adjara Today (2026-07-09) reports
> "As of June 2026 Batumi has 74 public charging points". **[1][2]**
>
> 1. `ev-en-0013` Public charging infrastructure registry: status June 2026. *Ministry of Transport and Mobility (government)*
> 2. `ev-en-0125` Batumi chargers: 74 and counting. *Adjara Today (news)*

Real output of the local model (qwen2.5:7b), shortened. The "74" comes from a
news item **planted in the test corpus** to contradict the official registry: the
agent states both figures with their sources instead of silently picking one.

## What makes it different

| | |
|---|---|
| **Cited down to the character** | Every claim points to an exact span of a document. A verifier re-fetches each span, checks every number in the sentence against it and asks a model whether the span supports the claim. Sentences that fail are removed. |
| **Conflicts are shown, not hidden** | Disagreeing sources are reported side by side; outdated values are not presented as current; a blog or forum cannot override an official source. |
| **Knows when it doesn't know** | Missing evidence is stated as "insufficient evidence", and it is built to decline questions its sources cannot answer rather than guess. |
| **Documents are untrusted** | Instructions it detects in retrieved text ("ignore previous instructions…") never reach the model and never become a claim, and the output is checked again for leaks and planted links. Harmful requests are refused before any tool is used. |
| **Measured honestly** | Deterministic scorers re-check every report against the corpus itself, an LLM judge is calibrated against human labels (Cohen's kappa), and a holdout set is run once, at the end. |

## Results

Dev split (93 questions). The held-out set (42 questions) is run once, at the end.

| | Offline (rules only, no model) | Local LLM (qwen2.5:7b) |
|---|---|---|
| Correct behaviour (answer / flag / decline / refuse) | 91% | *running* |
| Reference facts stated | 44% | *running* |
| Claims unsupported by any retrieved source | 0 of 314 | *running* |
| Planted contradictions reported with both sides | 2 of 6 | *running* |
| Harmful requests refused | 4 of 8 | *running* |
| Injected instructions followed | 0 of 7 | *running* |
| Latency per question | 0.2 s | 3 to 8 min on an Apple M4 |

Full tables, the judge's scores with their kappa, and every failed case with its
reason: [docs/EVALUATION.md](docs/EVALUATION.md).

## How it works

```
question -> input guard -> planner -> parallel searches (MCP tools)
         -> source reliability -> claims -> conflicts -> enough evidence?
              (if not, one more search round)
         -> calculations -> report with citations -> citation verifier
         -> output guard -> report
```

- **LangGraph** runs the research as a typed graph with budgets on searches,
  rounds, tokens and time.
- **Rules first, LLM where rules are unsure.** Every step works offline; the model
  plans, writes and judges, and every model output is validated.
- The agent reaches data **only through an MCP tool server** (hybrid search,
  documents, metadata, a safe calculator), backed by PostgreSQL + pgvector.

More: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Quick start

```bash
docker compose up -d                  # database, tool server, API (offline mode)
curl -s -X POST 'localhost:8100/research?wait=true' \
  -H 'content-type: application/json' \
  -d '{"question": "How many public charging points are there in Georgia?"}' | jq -r .report_markdown
```

With a local model: install [Ollama](https://ollama.com), run `ollama pull qwen2.5:7b` and
`ollama pull bge-m3`, then set `LLM_PROVIDER=ollama` and `EMBEDDING_PROVIDER=ollama`
(see `.env.example`).

Development: `make install`, `make check` (lint, types, tests), `make mutation`.

## Evaluation

```bash
uv run python -m researchai.evaluation run --split dev --mode offline
```

- **Dataset:** 135 questions in 11 categories (facts, comparisons,
  recommendations, calculations, conflicts, outdated data, unanswerable, prompt
  injection, harmful requests, Georgian, multi-hop), each with reference facts
  that are checked against the documents.
- **Traps:** 29 planted in a fully synthetic corpus of 246 documents.
- **Holdout:** 42 questions chosen by a hash rule and frozen by a digest.

[docs/EVALUATION.md](docs/EVALUATION.md) has the methodology.

## Testing

650+ tests (unit, property-based, integration against PostgreSQL and the MCP
server over HTTP, safety, evaluation pipeline), 99% coverage, mypy strict, and a
**mutation check**: 29 planted bugs (a disabled conflict detector, skipped
citation checks, a missing iteration cap…) must all be caught, in CI on every
push. [docs/TESTING.md](docs/TESTING.md)

Every component in detail: [docs/REFERENCE.md](docs/REFERENCE.md).

## Limitations

- **Same author:** I wrote the corpus, the questions and the traps, knowing how
  the agent works. An independent test set would be a stronger test.
- **Synthetic data:** the corpus is synthetic (fictional publishers, plausible
  numbers), so the results say nothing about real-world sources.
- **Small samples:** 135 questions give rough estimates; some categories rest on
  3 to 5 held-out cases.
- **Local model:** a 7B model on a laptop is slow (minutes per question) and
  weaker in Georgian than in English.
- **Paraphrased injections:** detection of injected instructions is pattern
  based; reworded ones can get past it (the later checks still apply).

## Project structure

```
src/researchai/
  agent/        LangGraph research graph, verification, citation checks
  safety/       input guard, untrusted documents, output guard
  mcp_server/   tool server (search, documents, metadata, calculator)
  api/          FastAPI: research runs, evaluations, health
  evaluation/   dataset, scorers, runner, LLM judge, labelling, kappa
corpus/         frozen synthetic corpus (246 documents)
eval/           labelled questions, trap registry, holdout lock
```

MIT license.
