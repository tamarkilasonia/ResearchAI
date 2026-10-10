"""Input guard: refuse prompt-injection attempts and clearly harmful requests.

1. Rules (English and Georgian, :mod:`researchai.safety.patterns`) refuse clear
   attacks and clearly harmful requests.
2. A question plainly asking *about* legality or consequences ("is it legal
   to ...", "what happens if ...") is allowed without a model.
3. With an LLM configured, a small classifier call screens every other
   question: rules miss paraphrases ("get past customs with an undervalued
   invoice"), and one short call costs seconds against minutes of research.
   Offline, rules alone decide; weak signals ("loophole", "without paying")
   are recorded in the verdict for auditing.

If the classifier fails, the request is allowed: the rules found no harmful
intent, and everything the agent says still has to come from cited corpus
evidence. A refused request never reaches the tools.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from researchai.agent.budget import BudgetExceeded, RunBudget
from researchai.agent.llm import LLMClient, LLMError, structured_call
from researchai.agent.models import Report, ReportSentence
from researchai.agent.prompts import GUARD_SYSTEM, guard_user
from researchai.corpus.models import Language
from researchai.safety.models import GuardCategory, GuardVerdict
from researchai.safety.patterns import (
    HARMFUL_RULES,
    INJECTION_RULES,
    INQUIRY,
    WEAK_SIGNALS,
    matches,
)

GUARD_MAX_TOKENS = 256


class GuardDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")
    decision: Literal["allow", "refuse"]
    category: GuardCategory
    reason: str = Field(max_length=300)


async def check_question(
    question: str, *, llm: LLMClient | None, budget: RunBudget
) -> GuardVerdict:
    fired = matches(INJECTION_RULES, question) + matches(HARMFUL_RULES, question)
    if fired:
        return GuardVerdict(
            action="refuse",
            category=fired[0].category,
            method="rules",
            matched=[rule.name for rule in fired],
        )
    weak = sorted({m.group().lower() for m in WEAK_SIGNALS.finditer(question)})
    if INQUIRY.search(question):
        return GuardVerdict(
            action="allow",
            category="none",
            method="rules",
            matched=[f"weak:{w}" for w in weak],
            reason="asks about legality or consequences",
        )
    if llm is None:
        return GuardVerdict(
            action="allow",
            category="none",
            method="rules",
            matched=[f"weak:{w}" for w in weak],
            reason="no rule matched; no classifier in offline mode" if weak else "",
        )
    try:
        result = await structured_call(
            llm,
            budget,
            system=GUARD_SYSTEM,
            user=guard_user(question),
            schema=GuardDecision,
            max_tokens=GUARD_MAX_TOKENS,
        )
    except (LLMError, BudgetExceeded) as exc:
        return GuardVerdict(
            action="allow",
            category="none",
            method="rules",
            matched=[f"weak:{w}" for w in weak],
            reason=f"classifier unavailable ({type(exc).__name__}); rules found no harmful intent",
        )
    decision = result.value
    refuse = decision.decision == "refuse" and decision.category != "none"
    return GuardVerdict(
        action="refuse" if refuse else "allow",
        category=decision.category if refuse else "none",
        method="llm",
        matched=[f"weak:{w}" for w in weak],
        reason=decision.reason,
    )


_REFUSALS: dict[str, dict[Language, str]] = {
    "prompt_injection": {
        Language.EN: "This request asks the assistant to change or reveal its instructions, so "
        "it was not run. Ask a research question about electric cars in Georgia or the "
        "small-business tax regime instead.",
        Language.KA: "ეს მოთხოვნა ცდილობს ასისტენტის ინსტრუქციების შეცვლას ან გამჟღავნებას, "
        "ამიტომ არ შესრულდა. დასვით საკვლევი კითხვა საქართველოში ელექტრომობილების ან მცირე "
        "ბიზნესის საგადასახადო რეჟიმის შესახებ.",
    },
    "fraud": {
        Language.EN: "This request asks for help with fraud or breaking the law (for example "
        "hiding income, under-declaring a value or falsifying documents), so it was not run. "
        "I can research the legal rules instead, such as how the 1% regime or EV import "
        "taxes work.",
        Language.KA: "ეს მოთხოვნა ითხოვს დახმარებას თაღლითობაში ან კანონის დარღვევაში "
        "(მაგალითად, შემოსავლის დამალვა, ღირებულების შემცირება დეკლარაციაში ან დოკუმენტების "
        "გაყალბება), ამიტომ არ შესრულდა. შემიძლია კანონიერი წესების კვლევა, მაგალითად, როგორ "
        "მუშაობს 1%-იანი რეჟიმი ან ელექტრომობილების იმპორტის გადასახადები.",
    },
    "self_harm": {
        Language.EN: "This request was not run. If you are thinking about harming yourself, "
        "please reach out now: in Georgia call 112, or talk to someone you trust.",
        Language.KA: "ეს მოთხოვნა არ შესრულდა. თუ საკუთარი თავისთვის ზიანის მიყენებაზე "
        "ფიქრობთ, გთხოვთ, ახლავე მიმართოთ დახმარებას: დარეკეთ 112-ზე ან ესაუბრეთ ადამიანს, "
        "რომელსაც ენდობით.",
    },
    "harm": {
        Language.EN: "This request asks for help causing harm, so it was not run.",
        Language.KA: "ეს მოთხოვნა ითხოვს დახმარებას ზიანის მიყენებაში, ამიტომ არ შესრულდა.",
    },
}
_TITLES = {Language.EN: "Request refused", Language.KA: "მოთხოვნა უარყოფილია"}


def refusal_report(category: GuardCategory, language: Language) -> Report:
    key = category if category in _REFUSALS else "harm"
    return Report(
        language=language,
        title=_TITLES[language],
        summary=[ReportSentence(text=_REFUSALS[key][language], kind="note")],
        sections=[],
        generated_by="rules",
    )
