# Frozen research corpus

Evaluation runs use only this corpus, so every run is repeatable. It covers two
topics in English and Georgian:

| Topic | Code | Purpose |
|---|---|---|
| Electric cars in Georgia: prices, charging, import rules, taxes, electricity costs, climate and range, ownership costs | `ev` | main topic |
| Georgia's small-business (Individual Entrepreneur, 1% turnover) tax regime | `tax` | smaller second topic, to test generalisation |

`python -m researchai.corpus stats` prints current counts.

## Everything here is synthetic

**All documents are synthetic.** They were written for this project and are
labelled as such in their metadata (`synthetic = true`, `license = "synthetic"`,
and a URL on the reserved `.invalid` top-level domain). The documents describe a
plausible, internally consistent world as of the snapshot date (2026-09-01),
but **the numbers are not authoritative facts about Georgia.** Do not use them
for real decisions.

- **Publishers are fictional** ("National Revenue Office", "VoltLine",
  "Georgia Auto Review", ...). No document is attributed to a real organisation.
  Real place names and real car models are used for realism.
- **Facts change over time on purpose.** Older documents state older values: the
  2022 electricity tariff, a VAT rule that changed in 2024, the pre-2018
  small-business threshold, and so on. Dates decide which value is current.
  This tests whether the agent notices outdated sources.
- **Formatting varies on purpose.** Some Georgian documents use a comma as the
  decimal separator (`0,55 ლარი`) and a space as the thousands separator
  (`86 900`), as real Georgian sources often do.
- **Planted traps come later.** Contradictions, unreliable outliers and hidden
  prompt injections are added in a later corpus version, together with a
  ground-truth registry (see `CHANGELOG.md`).

Synthetic documents avoid licensing problems entirely. If real sources are ever
added, they must carry `synthetic = false`, a real `https` URL and their actual
licence, and only content that licence permits (short excerpts otherwise).

## Layout

```
corpus/
  corpus.toml                    version and snapshot date
  manifest.json                  sha256 of every document (the freeze)
  <topic>/<language>/<id>.md     one document per file
```

Each file starts with TOML front matter between `+++` lines:

```toml
+++
id = "ev-en-0013"                      # <topic>-<language>-<4 digits>
title = "Public charging infrastructure registry: status June 2026"
url = "https://ministry-of-transport-and-mobility.invalid/ev-en-0013"
published = 2026-07-15
snapshot = 2026-09-01
language = "en"                        # en | ka
topic = "ev"                           # ev | tax
source_type = "government"             # see SourceType in src/researchai/corpus/models.py
publisher = "Ministry of Transport and Mobility"
synthetic = true
license = "synthetic"
+++
```

The body is Markdown. Headings define sections; chunks never cross a heading.
Text is normalised to NFC with LF line endings, so character offsets (chunk
spans and citation spans) are stable.

## Generated documents

Charging-station pages, used-car price-index pages and import statistics
bulletins are generated from explicit tables by
`scripts/generate_corpus_listings.py`. A test checks that the script still
reproduces the committed files exactly.

## Changing the corpus

1. Edit or add documents.
2. `uv run python -m researchai.corpus validate` (checks metadata, paths, ids and the declared language against the script actually used).
3. Bump `version` in `corpus.toml` and add an entry to `CHANGELOG.md`.
4. `uv run python -m researchai.corpus manifest --write`.

Tests fail if the manifest does not match the documents.
