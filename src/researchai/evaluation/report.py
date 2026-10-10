"""Render a run report as Markdown: configuration, metrics, breakdowns, failed cases."""

from __future__ import annotations

from typing import TYPE_CHECKING

from researchai.evaluation.metrics import GroupMetrics, Metric, RunMetrics

if TYPE_CHECKING:
    from researchai.evaluation.runner import RunReport

METRIC_ROWS: list[tuple[str, str, str]] = [
    ("Outcome", "passed", "Cases passing every check"),
    ("Outcome", "behavior", "Expected behaviour (answer / flag / abstain / refuse)"),
    ("Facts", "fact_recall", "Reference facts stated (values found in the report)"),
    ("Grounding", "source_grounding", "Claims with a citation that matches the corpus"),
    ("Grounding", "citation_recall", "Claims fully supported by their own citations"),
    ("Grounding", "citation_precision", "Citations that support their sentence"),
    ("Grounding", "hallucination_rate", "Claims unsupported by any retrieved passage"),
    ("Sources", "reference_source_recall", "Reference facts whose source document is cited"),
    ("Sources", "retrieval_recall", "Reference facts whose source document was retrieved"),
    ("Tools", "tool_selection", "Expected tools called (none for refusals)"),
    ("Tools", "tool_arguments", "Topic filter and calculation results correct"),
    ("Tools", "calculations", "Numeric cases with the expected calculation"),
    ("Conflicts", "conflicts_reported", "Planted contradictions stated with both sides"),
    ("Conflicts", "conflicts_detected", "Planted contradictions found by the detector"),
    ("Conflicts", "conflict_precision", "Detected contradictions that are planted ones"),
    ("Conflicts", "low_reliability_resisted", "Low-reliability values not adopted"),
    ("Conflicts", "low_reliability_flagged", "Low-reliability disagreements stated"),
    ("Conflicts", "outdated_avoided", "Outdated cases answered without the old value as current"),
    ("Abstention", "abstained_when_expected", "Unanswerable questions abstained"),
    ("Abstention", "false_abstentions", "Answerable questions abstained (lower is better)"),
    ("Safety", "refused_when_expected", "Harmful requests refused, no tool call"),
    ("Safety", "refusal_category", "Refusals with the expected category"),
    ("Safety", "false_refusals", "Other questions refused (lower is better)"),
    ("Safety", "injection_resisted", "Injection cases without forbidden text"),
    ("Safety", "injection_detected", "Retrieved planted injections marked instruction-like"),
    ("Safety", "prompt_leaks", "Reports repeating system prompt text (lower is better)"),
]


def _metric(metrics: RunMetrics, name: str) -> Metric:
    value = getattr(metrics, name)
    assert isinstance(value, Metric)  # noqa: S101 - METRIC_ROWS only names Metric fields
    return value


def _number(value: float | None, unit: str = "") -> str:
    return "n/a" if value is None else f"{value:,.1f}{unit}"


def _groups(title: str, groups: dict[str, GroupMetrics]) -> list[str]:
    lines = [
        f"## By {title}",
        "",
        f"| {title.capitalize()} | Cases | Passed | Behaviour | Fact recall |",
    ]
    lines.append("|---|---|---|---|---|")
    for name, g in groups.items():
        lines.append(f"| {name} | {g.cases} | {g.passed} | {g.behavior} | {g.fact_recall} |")
    return [*lines, ""]


def render_report(report: RunReport) -> str:
    c, m = report.config, report.metrics
    duration = (c.finished_at - c.started_at).total_seconds()
    lines = [
        f"# Evaluation run {c.run_id}",
        "",
        "| Setting | Value |",
        "|---|---|",
        f"| Dataset | {c.dataset} {c.dataset_version}, split `{c.split}`, "
        f"{len(c.case_ids)} cases |",
        f"| Mode | {c.mode}{f' ({c.model})' if c.model else ''} |",
        f"| Corpus | {c.corpus_version} (`{c.manifest_sha256[:12]}`) |",
        f"| Retrieval | embedder `{c.embedder}`, search mode `{c.search_mode}` |",
        f"| Code | `{c.git_commit or 'unknown'}` |",
        f"| Started | {c.started_at.isoformat(timespec='seconds')} ({duration / 60:.1f} min) |",
        "",
        "Completeness, relevance and report quality need the LLM judge (calibrated against",
        "human labels); the numbers below are deterministic checks only.",
        "",
        "## Results",
        "",
        "| Area | Metric | Value | Meaning |",
        "|---|---|---|---|",
    ]
    for area, name, meaning in METRIC_ROWS:
        label = name.replace("_", " ")
        lines.append(f"| {area} | {label} | {_metric(m, name)} | {meaning} |")
    p50 = _number(None if m.latency_p50_ms is None else m.latency_p50_ms / 1000, " s")
    p95 = _number(None if m.latency_p95_ms is None else m.latency_p95_ms / 1000, " s")
    lines += [
        f"| Cost | latency p50 / p95 | {p50} / {p95} | Per researched case |",
        f"| Cost | tokens per report | {_number(m.mean_tokens)} | Prompt + completion |",
        f"| Cost | LLM calls / searches | {_number(m.mean_llm_calls)} / {_number(m.mean_searches)}"
        " | Per researched case |",
        f"| Cost | cost per report | {_number(m.cost_per_report)} | "
        "At the configured token price |",
        "",
    ]
    lines += _groups("category", m.by_category)
    lines += _groups("language", m.by_language)

    trap_rows = [(s.case_id, t) for s in report.scores for t in s.traps]
    if trap_rows:
        lines += [
            "## Traps",
            "",
            "| Case | Trap | Kind | Retrieved | Detected | Reported | Adopted |",
            "|---|---|---|---|---|---|---|",
        ]
        yes = {True: "yes", False: "-"}
        lines += [
            f"| {case} | {t.trap} | {t.kind} | {yes[t.retrieved]} | {yes[t.detected]} | "
            f"{yes[t.reported]} | {yes[t.adopted]} |"
            for case, t in trap_rows
        ]
        lines.append("")

    failed = [s for s in report.scores if not s.passed]
    lines += [f"## Failed cases ({len(failed)} of {len(report.scores)})", ""]
    for s in failed:
        lines += [
            f"### {s.case_id} ({s.category.value}, {s.language})",
            "",
            f"- **Question:** {s.question}",
            f"- **Expected:** {s.expected.value}"
            + (f"; missing facts: {'; '.join(s.missing_facts)}" if s.missing_facts else ""),
            f"- **Actual:** {s.status.value}: {s.excerpt}",
            f"- **Why it failed:** {'; '.join(s.reasons)}",
            "",
        ]
    return "\n".join(lines)
