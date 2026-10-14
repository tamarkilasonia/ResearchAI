"""Mutation check: plant known bugs and verify the test suite catches each one.

Each mutant is one exact change to one file (for example disabling the conflict
detector or removing the iteration cap). The repository is copied to a
temporary directory once; every mutant is applied there, the unit tests run
against the mutated copy (``PYTHONPATH`` puts the copy's ``src`` first), and the
file is restored. The working tree is never modified.

A mutant that survives (the tests still pass) is a gap in the tests.

    python -m scripts.mutation_check            # all mutants
    python -m scripts.mutation_check --list
    python -m scripts.mutation_check --only iteration-cap --only citation-spans
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
COPIED = (
    "src",
    "tests",
    "scripts",
    "corpus",
    "eval",
    "docs",
    "README.md",
    "pyproject.toml",
    ".env.example",
)
IGNORED = shutil.ignore_patterns("__pycache__", "*.pyc", ".pytest_cache", "runs", "mutants")
PYTEST = ("-m", "pytest", "tests/unit", "-x", "-q", "-p", "no:randomly", "-p", "no:cacheprovider")


@dataclass(frozen=True)
class Mutant:
    name: str
    description: str
    path: str
    old: str
    new: str


def _early_return(signature: str, value: str, indent: str = "    ") -> tuple[str, str]:
    return signature, f"{signature}\n{indent}return {value}  # MUTANT"


MUTANTS: tuple[Mutant, ...] = (
    Mutant("conflict-detector", "rule-based conflict detection finds nothing",
           "src/researchai/agent/conflicts.py",
           *_early_return("def rule_conflicts(claims: Sequence[Claim]) -> list[Conflict]:", "[]")),
    Mutant("low-reliability-supersedes", "a forum or blog may supersede an official value",
           "src/researchai/agent/conflicts.py",
           *_early_return("def can_supersede(newer_reliability: float, older_reliability: float) "
                          "-> bool:", "True")),
    Mutant("citation-spans", "every cited span counts as verified",
           "src/researchai/agent/citations.py",
           *_early_return("def check_citation(citation: Citation, bodies: dict[str, str | None]) "
                          "-> CitationStatus:", '"verified"')),
    Mutant("citation-numbers", "numbers in a sentence are never checked against its citations",
           "src/researchai/agent/citations.py",
           "    return [v for v in sentence_numbers(text) if not _matches(v, available)]",
           "    return []  # MUTANT"),
    Mutant("iteration-cap", "the search-round cap is never enforced",
           "src/researchai/agent/budget.py",
           '        if self.iterations >= self.limits.max_iterations:\n'
           '            raise self._exhaust("iterations", f"iteration cap {self.limits.max_iterations}")\n',
           ""),
    Mutant("search-budget", "every requested search is granted",
           "src/researchai/agent/budget.py",
           "        granted = min(wanted, self.searches_left)", "        granted = wanted  # MUTANT"),
    Mutant("token-budget", "the token budget is never checked before an LLM call",
           "src/researchai/agent/budget.py",
           "        if worst_case > self.limits.max_tokens:", "        if False:  # MUTANT"),
    Mutant("run-timeout", "a research run has no time limit",
           "src/researchai/agent/runner.py",
           "                    timeout=self._settings.run_timeout_seconds,",
           "                    timeout=None,  # MUTANT"),
    Mutant("abstention", "the agent never abstains, whatever the evidence",
           "src/researchai/agent/graph.py",
           *_early_return("def abstain_reason(state: ResearchState, findings: Findings) "
                          "-> str | None:", "None")),
    Mutant("superseded-values", "outdated values are kept as current claims",
           "src/researchai/agent/verification.py",
           *_early_return("def drop_superseded_claims(report: Report, findings: Findings) "
                          "-> tuple[Report, int]:", "report, 0")),
    Mutant("refusal-routing", "a refused request still goes on to research",
           "src/researchai/agent/graph.py",
           '    return "finalize" if state.get("status") is RunStatus.REFUSED else "profile"',
           '    return "profile"  # MUTANT'),
    Mutant("input-guard", "the input guard allows every request",
           "src/researchai/safety/input_guard.py",
           "    fired = matches(INJECTION_RULES, question) + matches(HARMFUL_RULES, question)",
           "    fired = []  # MUTANT"),
    Mutant("document-neutralization", "instruction-like text in documents reaches LLM prompts",
           "src/researchai/safety/documents.py",
           *_early_return("def neutralize(text: str) -> str:", "text")),
    Mutant("output-guard", "the output guard removes nothing",
           "src/researchai/safety/output_guard.py",
           *_early_return("    def apply(self, report: Report) -> tuple[Report, list[OutputIssue]]:",
                          "report, []", indent="        ")),
    Mutant("calculator-magnitude", "the calculator accepts unbounded results",
           "src/researchai/tools/calculator.py",
           "    if abs(value) > MAX_MAGNITUDE:", "    if False:  # MUTANT"),
    Mutant("tool-errors", "backend failures are swallowed instead of reported",
           "src/researchai/mcp_server/server.py",
           "        raise ToolError(unavailable.to_json()) from exc",
           "        return None  # type: ignore[return-value]  # MUTANT"),
    Mutant("holdout-rule", "the hash-based holdout split puts everything in dev",
           "src/researchai/evaluation/dataset.py",
           '            split: Split = "holdout" if taken < target else "dev"',
           '            split: Split = "dev"  # MUTANT'),
    Mutant("hallucination-scorer", "the scorer never counts a hallucinated claim",
           "src/researchai/evaluation/scoring.py",
           '        counts["hallucinated"] += hallucinated', '        counts["hallucinated"] += 0'),
    Mutant("outdated-scorer", "the scorer flags an old value even next to the newer source",
           "src/researchai/evaluation/scoring.py",
           "        if sentence.kind not in CLAIM_KINDS or {c.doc_id for c in sentence.citations} "
           "& reference:", "        if sentence.kind not in CLAIM_KINDS:  # MUTANT"),
    Mutant("calculation-grounding", "LLM calculations run on operands not in the evidence",
           "src/researchai/agent/analysis.py",
           *_early_return("def check_grounded(proposal: _ProposedCalculation, evidence: "
                          "Mapping[str, Evidence]) -> str | None:", "None")),
    Mutant("instruction-citations", "a citation of injected instructions counts as verified",
           "src/researchai/agent/citations.py",
           "    if overlaps(citation.char_start, citation.char_end, instruction_spans(body)):",
           "    if False:  # MUTANT"),
    Mutant("entailment", "an entailment verdict of 'not supported' removes nothing",
           "src/researchai/agent/citations.py",
           '        if label == "not_supported":', "        if False:  # MUTANT"),
    Mutant("conflict-restore", "a conflict removed by the verifier is not restored",
           "src/researchai/agent/verification.py",
           "    restored, added = ensure_conflicts_reported(report, findings, evidence)",
           "    return report, 0  # MUTANT"),
    Mutant("instruction-claims", "injected instructions can become claims",
           "src/researchai/agent/claims.py",
           "            if overlaps(start, end, hidden):", "            if False:  # MUTANT"),
    Mutant("instruction-quotes", "the offline writer may quote injected instructions",
           "src/researchai/agent/synthesis.py",
           "        if overlaps(start, end, hidden):", "        if False:  # MUTANT"),
    Mutant("sufficiency", "any passage counts as relevant evidence",
           "src/researchai/agent/sufficiency.py",
           "    return shared >= min(MIN_SHARED_TERMS, len(terms))", "    return True  # MUTANT"),
    Mutant("kappa-chance", "Cohen's kappa without the chance correction",
           "src/researchai/evaluation/agreement.py",
           "    return round(1 - disagreement / expected, 4)", "    return round(1 - disagreement, 4)"),
    Mutant("judge-sees-scores", "the judge sees the report's automatic score line",
           "src/researchai/evaluation/judge.py",
           '    return _STATUS_LINE.sub("", render_markdown(result).rstrip()).rstrip() + "\\n"',
           '    return render_markdown(result)  # MUTANT'),
    Mutant("label-holdout", "human labels may be drawn from the holdout",
           "src/researchai/evaluation/__main__.py",
           '    if config.split != "dev":', "    if False:  # MUTANT"),
)  # fmt: skip


@dataclass(frozen=True)
class Outcome:
    mutant: Mutant
    caught: bool
    seconds: float
    detail: str


def _run_tests(workdir: Path) -> subprocess.CompletedProcess[str]:
    env = os.environ | {
        "PYTHONPATH": f"{workdir / 'src'}{os.pathsep}{workdir}",
        "PYTHONDONTWRITEBYTECODE": "1",
        "LLM_PROVIDER": "none",
        "EMBEDDING_PROVIDER": "hash",
    }
    return subprocess.run(  # noqa: S603 - fixed arguments, this interpreter
        [sys.executable, *PYTEST], cwd=workdir, env=env, capture_output=True, text=True,
        check=False,
    )  # fmt: skip


def _last_line(output: str) -> str:
    lines = [line for line in output.strip().splitlines() if line.strip()]
    return lines[-1] if lines else ""


def check(mutants: Sequence[Mutant], workdir: Path) -> list[Outcome]:
    outcomes: list[Outcome] = []
    for mutant in mutants:
        target = workdir / mutant.path
        original = target.read_text(encoding="utf-8")
        if original.count(mutant.old) != 1:
            raise SystemExit(f"{mutant.name}: the code to mutate is not found exactly once in "
                             f"{mutant.path}; update the mutant")  # fmt: skip
        target.write_text(original.replace(mutant.old, mutant.new), encoding="utf-8")
        started = time.perf_counter()
        try:
            result = _run_tests(workdir)
        finally:
            target.write_text(original, encoding="utf-8")
        seconds = time.perf_counter() - started
        caught = result.returncode != 0
        outcomes.append(Outcome(mutant, caught, seconds, _last_line(result.stdout)))
        verdict = "caught  " if caught else "SURVIVED"
        print(f"{verdict} {mutant.name:<26} {seconds:5.1f}s  {mutant.description}", flush=True)
    return outcomes


def copy_repository(destination: Path) -> None:
    for name in COPIED:
        source = ROOT / name
        if source.is_dir():
            shutil.copytree(source, destination / name, ignore=IGNORED)
        else:
            shutil.copy2(source, destination / name)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m scripts.mutation_check")
    parser.add_argument("--only", action="append", default=[], help="mutant name (repeatable)")
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args(argv)
    if args.list:
        for mutant in MUTANTS:
            print(f"{mutant.name:<26} {mutant.description}")
        return 0
    unknown = set(args.only) - {m.name for m in MUTANTS}
    if unknown:
        print(f"unknown mutant(s): {sorted(unknown)}", file=sys.stderr)
        return 2
    selected = [m for m in MUTANTS if not args.only or m.name in args.only]
    with tempfile.TemporaryDirectory(prefix="researchai-mutants-") as tmp:
        workdir = Path(tmp)
        copy_repository(workdir)
        baseline = _run_tests(workdir)
        if baseline.returncode != 0:
            print("the unmutated tests fail; fix them first:\n" + baseline.stdout[-2000:],
                  file=sys.stderr)  # fmt: skip
            return 2
        outcomes = check(selected, workdir)
    survived = [o for o in outcomes if not o.caught]
    print(f"\n{len(outcomes) - len(survived)} of {len(outcomes)} mutants caught")
    for outcome in survived:
        print(f"SURVIVED: {outcome.mutant.name} ({outcome.mutant.description})", file=sys.stderr)
    return 1 if survived else 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
