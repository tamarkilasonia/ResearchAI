"""Per-run budgets: searches, search rounds (iterations) and LLM tokens.

Budgets are enforced before the work is done, never after: a search that
would exceed the cap is not issued, and an LLM call whose worst case
(prompt estimate + max output) would exceed the token cap is not made. The
graph then degrades (fewer searches, rule-based fallback) and records which
budget ran out instead of failing the run.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from researchai.config import Settings


class BudgetExceeded(Exception):
    def __init__(self, budget: str, message: str) -> None:
        super().__init__(message)
        self.budget = budget


@dataclass(frozen=True)
class Limits:
    max_searches: int
    max_iterations: int
    max_tokens: int
    max_subquestions: int
    hits_per_subquestion: int

    @classmethod
    def from_settings(cls, settings: Settings) -> Limits:
        return cls(
            max_searches=settings.max_searches,
            max_iterations=settings.max_iterations,
            max_tokens=settings.max_tokens_per_run,
            max_subquestions=settings.max_subquestions,
            hits_per_subquestion=settings.hits_per_subquestion,
        )


@dataclass
class RunBudget:
    limits: Limits
    searches: int = 0
    iterations: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    llm_calls: int = 0
    exhausted: list[str] = field(default_factory=list)

    @property
    def tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    @property
    def searches_left(self) -> int:
        return max(0, self.limits.max_searches - self.searches)

    def _exhaust(self, budget: str, message: str) -> BudgetExceeded:
        if budget not in self.exhausted:
            self.exhausted.append(budget)
        return BudgetExceeded(budget, message)

    def reserve_searches(self, wanted: int) -> int:
        """Grant up to ``wanted`` searches; record exhaustion if fewer are left."""
        granted = min(wanted, self.searches_left)
        if granted < wanted:
            self._exhaust("searches", f"search budget allows {granted} of {wanted}")
        self.searches += granted
        return granted

    def start_iteration(self) -> None:
        if self.iterations >= self.limits.max_iterations:
            raise self._exhaust("iterations", f"iteration cap {self.limits.max_iterations}")
        self.iterations += 1

    def can_follow_up(self) -> bool:
        """Whether another search round is allowed; records the budget that says no."""
        if self.iterations >= self.limits.max_iterations:
            self._exhaust("iterations", "no search rounds left for missing evidence")
            return False
        if self.searches_left == 0:
            self._exhaust("searches", "no searches left for missing evidence")
            return False
        return True

    def check_tokens(self, estimated_prompt: int, max_output: int) -> None:
        worst_case = self.tokens + estimated_prompt + max_output
        if worst_case > self.limits.max_tokens:
            raise self._exhaust(
                "tokens", f"call could reach {worst_case} > {self.limits.max_tokens} tokens"
            )

    def record_tokens(self, prompt: int, completion: int) -> None:
        self.llm_calls += 1
        self.prompt_tokens += prompt
        self.completion_tokens += completion
