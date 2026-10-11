"""The mutation check's planted bugs must still match the code they mutate."""

from __future__ import annotations

from pathlib import Path

import pytest

from scripts.mutation_check import MUTANTS, main

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("mutant", MUTANTS, ids=lambda m: m.name)
def test_every_mutant_applies_to_exactly_one_place(mutant: object) -> None:
    from scripts.mutation_check import Mutant

    assert isinstance(mutant, Mutant)
    source = (ROOT / mutant.path).read_text(encoding="utf-8")
    assert source.count(mutant.old) == 1, f"{mutant.name}: update the mutant"
    assert mutant.old != mutant.new


def test_names_are_unique_and_listed(capsys: pytest.CaptureFixture[str]) -> None:
    names = [m.name for m in MUTANTS]
    assert len(names) == len(set(names))
    assert main(["--list"]) == 0
    listing = capsys.readouterr().out
    assert all(name in listing for name in names)
    assert main(["--only", "no-such-mutant"]) == 2
