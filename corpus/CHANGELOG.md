# Corpus changelog

## 1.1.0 (2026-10-10)

Planted traps for evaluation: 18 new synthetic documents (246 in total). Existing
documents are unchanged. Ground truth: `eval/traps.toml`.

- Contradictions (same period, different numbers): `ev-en-0125`, `ev-en-0126`,
  `tax-en-0026`, `ev-ka-0046`.
- Low-reliability sources (forum, dealer, blog) disagreeing with reliable ones:
  `tax-en-0027`, `ev-en-0127`, `ev-en-0128`, `ev-ka-0047`, `tax-ka-0017`.
- Hidden prompt injections, explicit and paraphrased, English and Georgian:
  `ev-en-0129`, `ev-en-0130`, `ev-en-0131`, `ev-en-0132`, `ev-en-0133`,
  `tax-en-0028`, `tax-en-0029`, `ev-ka-0048`, `tax-ka-0018`.
- The time-dependent values of 1.0.0 (tariffs, VAT, threshold, filing, parking,
  corridor gap, prices, charger counts) are registered as `outdated` traps.

## 1.0.0 (2026-10-09)

First frozen corpus: 228 synthetic documents.

- `ev`: 187 documents (124 English, 63 Georgian), including 57 generated listing pages.
- `tax`: 41 documents (25 English, 16 Georgian).
- Changes over time are intentional (tariffs, VAT rules, thresholds, prices), with dated documents on both sides.
- No planted traps yet.
