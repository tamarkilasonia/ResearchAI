# Testing

```bash
make check          # ruff, mypy --strict, unit + integration tests with coverage (gate 95%)
make test           # unit tests only (no database)
make db && make test-integration   # PostgreSQL + pgvector on 127.0.0.1:5434
make mutation       # plant known bugs; fail if the tests miss one
```

## What is tested

| Layer | How |
|---|---|
| Unit | Every module, without a database or model: corpus loading and chunking, search fusion, tools, MCP server and client, every graph node, LLM structured calls (scripted model), verification, citation verification, safety, API (fake services), evaluation dataset, scorers, metrics, runner, judge, labelling, kappa |
| Property-based (Hypothesis) | Sentence splitting, chunk spans, the calculator, search fusion, claim extraction, quote anchoring, document neutralization, budgets, number matching in English and Georgian, the holdout split, Cohen's kappa |
| Integration | Real PostgreSQL + pgvector: migrations, ingestion, search; the MCP server over HTTP; the offline agent end to end on the real corpus; the API with the database; live evaluation runs, resume and backend health |
| Safety | Injection and harmful-request suites (English and Georgian, with benign look-alikes), poisoned documents through the whole graph with a scripted LLM (prompts never contain injected text), output guard, refusals with no tool calls |
| Evaluation pipeline | Dataset validation against the corpus, a broken dataset that triggers every validation rule, scorers on hand-built reports citing real corpus offsets, CI's offline evaluation with thresholds |
| Mutation | 29 planted bugs (below) |

**Rules for the suite.** Tests run in random order (pytest-randomly); every
warning is an error; a skipped test fails the run (no silent skips; integration
tests are selected with `-m integration` instead of being skipped); offline by
default (`LLM_PROVIDER=none`, hash embedder), so CI needs no model and no key.

## Mutation testing

`scripts/mutation_check.py` copies the repository to a temporary directory and
plants one bug at a time, runs the unit tests against the mutated copy and
requires every mutant to be caught. A no-op control (a comment change) survives,
as it should; a unit test keeps every mutant matched to the code it changes.

| Mutant | Planted bug |
|---|---|
| conflict-detector | rule-based conflict detection finds nothing |
| low-reliability-supersedes | a forum or blog may supersede an official value |
| citation-spans | every cited span counts as verified |
| citation-numbers | numbers in a sentence are never checked against its citations |
| iteration-cap | the search-round cap is never enforced |
| search-budget | every requested search is granted |
| token-budget | the token budget is never checked before an LLM call |
| run-timeout | a research run has no time limit |
| abstention | the agent never abstains |
| superseded-values | outdated values are kept as current claims |
| refusal-routing | a refused request still goes on to research |
| input-guard | the input guard allows every request |
| document-neutralization | instruction-like text in documents reaches LLM prompts |
| output-guard | the output guard removes nothing |
| calculator-magnitude | the calculator accepts unbounded results |
| tool-errors | backend failures are swallowed instead of reported |
| holdout-rule | the hash-based holdout split puts everything in dev |
| hallucination-scorer | the scorer never counts a hallucinated claim |
| outdated-scorer | the scorer flags an old value even next to the newer source |
| calculation-grounding | LLM calculations run on operands not in the evidence |
| instruction-citations | a citation of injected instructions counts as verified |
| entailment | an entailment verdict of "not supported" removes nothing |
| conflict-restore | a conflict removed by the verifier is not restored |
| instruction-claims | injected instructions can become claims |
| instruction-quotes | the offline writer may quote injected instructions |
| sufficiency | any passage counts as relevant evidence |
| kappa-chance | Cohen's kappa without the chance correction |
| judge-sees-scores | the judge sees the report's automatic score line |
| label-holdout | human labels may be drawn from the holdout |

The first full run found one real gap: `instruction-quotes` survived because the
citation verifier behind the offline writer removed the quoted injection anyway,
so no test exercised the writer's own guard. A test now does.

## Continuous integration

GitHub Actions on every push and pull request, without secrets:

```
lint + typecheck -> unit tests ---------\
                 -> integration tests ---+-> coverage gate (95%) -> mutation check
                                                -> evaluation (offline, dev split) -> docker build
```

The evaluation job ingests the corpus into a fresh pgvector service, runs the
offline agent on the dev split and fails if a report leaks a system prompt, a
harmless question is refused, a claim is unsupported by every retrieved passage
or an injected instruction shows up in a report. The run's report is uploaded
as an artefact.
