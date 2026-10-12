# Evaluation

How ResearchAI is measured, what the numbers are, and how far to trust them.

## Setup

- **Corpus:** 246 synthetic documents (v1.1.0), English and Georgian, on electric
  cars in Georgia and Georgia's small-business tax regime, frozen by a manifest
  hash. Evaluation never touches the internet.
- **Traps (29):** 4 same-period contradictions, 5 low-reliability sources (forum,
  dealer, blog) contradicting reliable ones, 9 hidden prompt injections (7
  explicit, 2 paraphrased; English and Georgian) and 11 outdated values superseded
  by newer documents. Ground truth in [`eval/traps.toml`](../eval/traps.toml),
  outside the corpus, so the agent cannot read it.
- **Dataset (135 cases):** factual, comparison, recommendation, numeric, conflict,
  outdated, unanswerable, injection, harmful, Georgian and multi-hop questions.
  Each case has the expected behaviour (answer / flag the conflict / decline /
  refuse), key points, reference facts with their source documents, expected
  tools and, where relevant, the expected calculation, values that must not be
  presented as current and text that must never appear. A validator checks every
  label against the corpus: each reference value must appear in a cited document.
- **Split:** 93 dev and 42 holdout cases, assigned per category by a hash rule
  (no hand-picking) and frozen by a digest. Rules and prompts were only ever
  changed using dev results; the holdout was run once, with the final code.
- **Models:** generator qwen2.5:7b, judge llama3.1:8b (a different family), embeddings
  bge-m3, all local through Ollama on an Apple M4 with 16 GB; temperature 0 and a
  fixed seed. The offline baseline uses no model at all (rules and templates).

## Scoring

Deterministic scorers re-check every final report against the corpus files, not
through the agent's own verifier. The LLM judge is used only for what code
cannot decide (completeness, relevance, report quality) and is reported only
with its agreement with human labels. Metric definitions:
[eval/README.md](../eval/README.md#metrics).

## Retrieval: vector, hybrid or keyword

Decided on the dev split (77 cases with reference facts, bge-m3): share of
reference facts whose source document is among the top k results.

| Mode | recall@1 | recall@3 | recall@5 | MRR |
|---|---|---|---|---|
| vector | **0.62** | **0.87** | **0.89** | **0.82** |
| hybrid (RRF) | 0.51 | 0.78 | 0.89 | 0.73 |
| keyword | 0.40 | 0.61 | 0.69 | 0.59 |

Per case, vector beat hybrid on recall@3 in 14 cases and lost in 4 (sign test
p = 0.03), so the agent searches in vector mode with a real embedding model;
the offline hash embedder keeps hybrid.

## Results

<!-- results:start -->
*(generated from the run summaries with `python -m researchai.evaluation tables`)*
<!-- results:end -->

## Judge calibration

<!-- calibration:start -->
35 reports of the dev LLM run (seeded sample, 3-4 per answerable category, never
the holdout) were labelled by one person with the rubric in
[eval/labels/GUIDELINES.md](../eval/labels/GUIDELINES.md), before seeing any judge
label; the judge (llama3.1:8b) graded the same reports. Data and labels:
[eval/labels/dev-qwen](../eval/labels/dev-qwen/).

| Criterion | Items | Raw agreement | Cohen's kappa | Weighted kappa | Reading | Human mean | Judge mean |
|---|---|---|---|---|---|---|---|
| completeness | 35 | 0.57 | 0.24 | 0.45 | fair (unreliable) | 1.09 | 1.29 |
| relevance | 35 | 0.71 | 0.02 | 0.15 | slight (unreliable) | 1.71 | 1.91 |
| quality | 35 | 0.57 | 0.25 | 0.34 | fair (unreliable) | 0.74 | 0.86 |

**The judge is not reliable enough on any criterion** (all kappas below 0.4), so its
scores are reported but should not be trusted:

- It is **more lenient** than the human on every criterion (higher means); most
  disagreements are the judge giving 2 where the human gave 1 (completeness: 9 of
  35 items).
- **Relevance shows the kappa paradox:** both raters gave 2 to 25 of 35 reports,
  so 71% raw agreement is barely better than chance (kappa 0.02).
- **Completeness is closer** when near misses are credited (weighted kappa 0.45).
- The judge was **not tuned on these labels**: doing so would need fresh labels to
  measure it again. Next steps would be a stronger judge model, one prompt per
  criterion, and more labels from a second person.
<!-- calibration:end -->

How to read it: Cohen's kappa corrects raw agreement for chance; quadratic-weighted
kappa also credits near misses on the 0-2 scale. Below 0.4 a criterion is marked
unreliable, and its judge score should not be trusted.

## What the numbers say

<!-- findings:start -->
On the dev split (93 questions), the local LLM against the offline rules-only baseline:

- **Better at almost everything the brief asks for:**
  - cases passing every check: 38% → 55%;
  - reference facts stated: 44% → 51%;
  - planted contradictions shown with both sides: 2 → 4 of 6;
  - unanswerable questions declined: 4 → 8 of 8;
  - harmful requests refused: 4 → 8 of 8 (the classifier catches reworded requests the rules miss);
  - expected calculations made: 0 → 4 of 12.
- **Grounding holds:** no claim in either mode is unsupported by every retrieved
  passage, and 94% of the LLM's citations support their sentence.
- **Its costs:**
  - 9 of 77 answerable questions declined (over-cautious sufficiency);
  - one harmless Georgian question (pension contributions) refused as fraud;
  - in 12% of researched cases a model step failed and fell back to rules;
  - about 4.3 minutes per question (p50) on an Apple M4.
- **Weak spots:**
  - numeric questions (4 of 12 calculations);
  - recommendations and multi-hop questions (fact recall 24% and 38%);
  - outdated values (5 of 8 avoided).
- **Injections:** no injected instruction was followed (7 of 7). The low pass rate
  of the injection category comes from missing facts, not from injections.
- **Georgian:** on par with English (58% vs 54% of cases passing).
<!-- findings:end -->

## Reproducing

```bash
make db                                                   # pgvector on 127.0.0.1:5434
EMBEDDING_PROVIDER=ollama uv run python -m researchai.corpus ingest
uv run python -m researchai.evaluation validate
uv run python -m researchai.evaluation run --split dev --mode offline
LLM_PROVIDER=ollama EMBEDDING_PROVIDER=ollama \
  uv run python -m researchai.evaluation run --split dev --mode llm
uv run python -m researchai.evaluation judge eval/runs/<llm-run>
uv run python -m researchai.evaluation tables eval/runs/<run> eval/runs/<run>
```

Runs are crash-safe (`--resume`) and record the git commit, corpus hash,
embedder, search mode and model. Every full research result is kept in the
run's `results.jsonl`.

## Limitations

- **Same author:** the corpus, the questions and the traps were written by the
  developer of the agent, who knew its rules while writing them. An independent
  test set would be a stronger test.
- **Synthetic data:** results say nothing about real-world sources.
- **Small samples:** 135 cases give coarse estimates; holdout categories rest on
  3 to 5 cases.
- **One labeller:** human labels come from one person; the kappas measure
  agreement with that person, not with "the truth".
- **Rough text checks:** a few outdated values and forbidden phrases are words,
  which a deterministic check can only approximate.
- **Local model:** a local 7B model on a laptop; larger models would score
  differently, and latency is dominated by local generation.
