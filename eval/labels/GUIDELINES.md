# Labelling guide

Human labels calibrate the LLM judge. This guide is for the person labelling;
the judge's prompt uses the same rubric word for word
(`src/researchai/evaluation/judge.py`).

## Why we label

An LLM judge will score what code cannot check: **completeness**, **relevance**
and **quality** of research reports. Your labels are the reference: we compare
the judge with you (Cohen's kappa per criterion) and report the judge's scores
only together with that agreement, and say where it is unreliable.

## What you get

- About **35 reports** from an LLM-mode run of the dev split: a seeded random
  sample, spread across categories (never the holdout).
- For each report: the **question**, the case's **key points**, and the report as
  the user would see it.
- **3 labels per report** (about 105 in total, roughly 1.5-2 hours). You can stop
  at any time; every answer is saved immediately.

## How to label

```bash
uv run python -m researchai.evaluation label eval/labels/<sample> --labeler <your-name>
```

The tool shows one report at a time, then asks for the three criteria in turn:

| Key | Meaning |
|---|---|
| `0`, `1`, `2` | the label |
| `0u`, `1u`, `2u` | the label, marked *unsure* |
| `s` | skip this criterion (you cannot judge it) |
| `q` | stop; run the same command later to continue where you left off |

After the three criteria you may type a short note (or just press Enter). The
current report is also written to `CURRENT_ITEM.md` in the sample folder:
open it in VS Code's Markdown preview if that is easier to read than the
terminal.

## What you judge

Only what is on the page, against the question and the key points. **Do not
check sources or numbers yourself**: citations, numbers, hallucinations,
refusals and safety are already checked automatically.

| Criterion | 2 | 1 | 0 |
|---|---|---|---|
| **Completeness**: are the key points answered? | Every key point is stated clearly as part of the answer | Some key points are missing, or only buried (e.g. inside a quoted "sources disagree" sentence) | None or almost none |
| **Relevance**: does it stay on the question? | Answers what was asked; little or no off-topic content | Answers it, but with substantial off-topic sections | Mostly about something else |
| **Quality**: could a reader rely on and act on it? | Clear, consistent, direct answer; conflicts and "insufficient evidence" used correctly | Useful, but with real problems (repetition, awkward structure, one misleading statement) | Confusing, contradicts itself, or misleading |

## Rules

1. **Judge each criterion separately.** A report can be complete but badly written.
2. **Label first.** The tool hides the judge's labels and the automatic scores, so
   they cannot influence you.
3. **Unsure?** Choose the label that fits best and mark it *unsure*. If you
   really cannot judge an item, *skip* it.
4. **Georgian reports** are labelled the same way.
5. **No re-labelling to match the judge.** Disagreements are what we measure.

## Worked example (a real report from the dev run)

**Question:** "Compare the annual maintenance costs of electric and petrol cars in Georgia."
**Key points:** electric 280-350 GEL a year; petrol sedan 850-1,100 GEL a year; tyres excluded.

The report opens with petrol prices and energy costs, says *insufficient
evidence* under both maintenance sections, and gives the correct numbers only
inside a "Sources disagree" line that sets a vague statement ("maintenance is
cheaper too") against those numbers. That is not a real disagreement.

| Criterion | Label | Why |
|---|---|---|
| Completeness | **1** | All three key points appear, but only buried in a quote, while the sections claim there is no evidence |
| Relevance | **1** | Two sections and a claim are about energy and fuel costs, which were not asked |
| Quality | **0** | It contradicts itself ("insufficient evidence" next to the numbers) and invents a conflict |

This report *passed* the automatic checks because the numbers are present,
which is exactly why these three criteria need a judge and your labels.
