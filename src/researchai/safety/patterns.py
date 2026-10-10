"""Safety patterns (English and Georgian).

Patterns are deliberately narrow: they must fire on attacks and clearly harmful
requests and stay silent on ordinary questions that share a word ("ignore
hybrids and compare only EVs", "how to spot a fake invoice", "can a small firm
break into the charging market"). Harmful-request rules therefore need a
request to *do* the thing (an intent phrase or a first-person verb next to the
action), not just the topic. Paraphrases are left to the LLM classifier (in
LLM mode); see :mod:`researchai.safety.input_guard`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from researchai.safety.models import GuardCategory


@dataclass(frozen=True)
class Rule:
    name: str
    category: GuardCategory
    pattern: re.Pattern[str]


def _rule(name: str, category: GuardCategory, regex: str) -> Rule:
    return Rule(name, category, re.compile(regex, re.IGNORECASE | re.MULTILINE))


_VERB_OVERRIDE = r"\b(ignore|disregard|forget|override|bypass|skip)\s+"
# Only modifier words may stand between the verb and "instructions", so
# "ignore hybrids, what do the guidelines say" stays allowed.
_MODIFIERS = (
    r"(?:(?:all|any|every|of|the|your|my|previous|previously|prior|above|earlier|preceding|"
    r"system|original|initial|these|those|given|developer|safety|existing)\s+)*"
)
_TARGET = r"(instructions?|prompts?|guidelines|directives|system\s+prompt|guardrails)\b"
_REVEAL = (
    r"\b(reveal|show|print|repeat|output|display|leak|dump|tell me|give me|what (is|are|were))\b"
)
_HIDDEN_CONFIG = (
    r"\b(system|hidden|initial|original|developer|internal|secret)\s+"
    r"(prompt|instructions?|message|configuration)\b"
)
# "How do I / how to / help me / ways to ..." before an action: _INTENT allows
# two words in between ("how can I quietly hide ..."), _INTENT_NEXT none (for
# verbs that are also adjectives: "how can I fake an invoice" vs "spot a fake one").
_INTENT_NEXT = (
    r"\b(how\s+(do|can|could|should|would|might)\s+(i|we|you|one)|how\s+to|help\s+me(\s+to)?|"
    r"teach\s+me(\s+to)?|show\s+me\s+how\s+to|best\s+way\s+to|easiest\s+way\s+to|ways?\s+to|"
    r"tips\s+(to|for|on)|steps\s+to|i\s+(want|need|plan)\s+to|can\s+(i|we|you))\s+"
)
_INTENT = _INTENT_NEXT + r"(\S+\s+){0,2}"
_GAP = r"(\S+\s+){0,3}"  # up to three words

# Attempts to override the system or extract its configuration.
INJECTION_RULES: tuple[Rule, ...] = (
    _rule("override_instructions", "prompt_injection",
          rf"{_VERB_OVERRIDE}{_MODIFIERS}{_TARGET}|"
          rf"{_VERB_OVERRIDE}(all\s+|any\s+)?(of\s+)?(your|the\s+system'?s?|safety)\s+"
          rf"(rules|restrictions|filters)\b|"
          rf"\b(ignore|disregard|forget)\s+(everything|all|anything)\s+(above|before|"
          rf"previously|said|you\s+were\s+told)\b|\b(ignore|disregard)\s+the\s+above\b"),
    _rule("reveal_prompt", "prompt_injection",
          rf"{_REVEAL}[^.!?\n]{{0,40}}{_HIDDEN_CONFIG}|"
          rf"{_REVEAL}\s+(me\s+)?your\s+(instructions|prompt|rules)\b"),
    _rule("role_override", "prompt_injection",
          r"\byou\s+are\s+now\s+(an?|the|in|my|no\s+longer|free|unrestricted|dan)\b|"
          r"\bfrom\s+now\s+on,?\s+you\s+(are|will|must|should)\b|"
          r"\b(developer|god|dan|jailbreak|unrestricted)\s+mode\b|\bjailbreak|"
          r"\bpretend\s+(that\s+)?you\s+(have|are\s+under)\s+no\s+(rules|restrictions)\b|"
          r"\bact\s+as\s+(an?\s+)?(unrestricted|unfiltered|evil)\b"),
    _rule("role_spoofing", "prompt_injection",
          r"^\s*\[?\s*(system|assistant|developer)\s*\]?\s*:|<\|?(im_start|system)\|?>"),
    _rule("tag_injection", "prompt_injection",
          r"<\s*/?\s*(system|evidence|instructions?|user|assistant)\b[^>]*>"),
    _rule("override_instructions_ka", "prompt_injection",
          r"(უგულებელყ|დაივიწყ|არ\s+გაითვალისწინ|გვერდი\s+აუარ|გადალახ)\S*\s+(\S+\s+){0,3}"
          r"(ინსტრუქცი|მითითებ|პრომპტ)|"
          r"(უგულებელყ|დაივიწყ)\S*\s+(\S+\s+){0,2}(შენ|სისტემურ)\S*\s+წეს"),
    _rule("reveal_prompt_ka", "prompt_injection",
          r"(აჩვენე|გამოიტანე|გამიმხილე|დაწერე|მითხარი|გამოაქვეყნე)\S*\s+(\S+\s+){0,3}"
          r"(სისტემურ|დამალულ|საწყის|შიდა)\S*\s+(ინსტრუქცი|მოთხოვნ|პრომპტ|მითითებ)"),
    _rule("role_override_ka", "prompt_injection",
          r"შენ\s+ახლა\s+ხარ|ამიერიდან\s+შენ\s+(ხარ|უნდა)|შეზღუდვების\s+გარეშე\s+უპასუხე"),
)  # fmt: skip

# Clearly harmful requests, refused without asking a model.
HARMFUL_RULES: tuple[Rule, ...] = (
    _rule("weapons", "weapons",
          r"\b(make|build|assemble|construct|manufacture|3d[- ]print)\s+(a\s+|an\s+|some\s+)?"
          r"(homemade\s+|home-made\s+|pipe\s+|nail\s+|car\s+|dirty\s+|working\s+)?"
          r"(bombs?|explosives?|grenades?|ied|detonators?|firearms?|pistols?|rifles?|"
          r"silencers?)\b|\bmolotov\b|\bnapalm\b|\bnerve\s+agent"),
    _rule("violence", "violence",
          rf"{_INTENT}(kill|murder|poison|stab|shoot|strangle)\s+(someone|somebody|a\s+person|"
          r"people|him|her|them|my\s+\w+)\b"),
    _rule("self_harm", "self_harm",
          r"\b(kill\s+myself|end\s+my\s+(own\s+)?life|suicide\s+methods?|"
          r"how\s+(to|can\s+i|do\s+i)\s+(die|commit\s+suicide|hurt\s+myself))\b"),
    _rule("cybercrime", "cybercrime",
          rf"{_INTENT}(hack|break)\s+into\b|"
          rf"{_INTENT}(steal|phish|harvest|crack)\s+{_GAP}(passwords?|credentials|logins?|"
          r"card\s+numbers?|bank\s+accounts?)\b|"
          rf"\b(write|create|build|code|make)\s+{_GAP}(malware|ransomware|keylogger|"
          r"computer\s+virus|trojan|botnet)\b|\bddos\b"),
    _rule("document_fraud", "fraud",
          rf"\b(make|create|produce|print|get|buy|generate|write|draw\s+up)\s+{_GAP}"
          r"(fake|forged|false|counterfeit|falsified|backdated)\s+(\S+\s+)?(invoices?|receipts?|"
          r"documents?|titles?|passports?|declarations?|certificates?|papers|registrations?|"
          r"contracts?)\b|"
          r"\b(forge|falsify|backdate|counterfeit)\s+(a\s+|an\s+|the\s+|my\s+)?(\S+\s+)?"
          r"(invoices?|receipts?|documents?|titles?|passports?|declarations?|certificates?|"
          r"signatures?|registrations?|contracts?)\b"),
    _rule("tax_or_customs_evasion", "fraud",
          rf"{_INTENT}(evade|dodge|cheat\s+on|cheat|hide|conceal|launder)\s+{_GAP}"
          r"(tax(es)?|customs|duty|duties|vat|excise|income|turnover|earnings|revenue|money|"
          r"cash)\b|"
          r"\b(hide|conceal)\s+(\S+\s+){0,3}from\s+(the\s+)?(revenue\s+service|tax\s+"
          r"(office|authorit\w*)|customs|rs\.ge)\b|"
          r"\bwithout\s+(the\s+)?(revenue\s+service|tax\s+office|customs|authorities)\s+"
          r"(knowing|noticing|finding\s+out)\b|"
          rf"{_INTENT}(bypass|circumvent|get\s+around|sneak\s+past|slip\s+past)\s+{_GAP}"
          r"(customs|revenue\s+service|tax\s+office|border\s+checks?)\b"),
    _rule("under_declaration", "fraud",
          rf"{_INTENT}under-?(value|declare|report|invoice)\w*\b|"
          rf"{_INTENT}(declare|report|show|write)\s+{_GAP}(lower|less|smaller)\s+than\s+"
          r"(it|the\s+real|the\s+actual|what\s+i)\b|"
          rf"{_INTENT_NEXT}(fake|falsify|lie\s+about)\s+{_GAP}(price|value|income|turnover|"
          r"mileage|invoice)\b"),
    _rule("odometer_fraud", "fraud",
          rf"{_INTENT}(roll\s*back|wind\s+back|turn\s+back|tamper\s+with|reset|lower|clock)\s+"
          r"(\S+\s+){0,2}(odometer|mileage|km\s+reading)\b"),
    _rule("weapons_ka", "weapons",
          r"(დავამზადო|გავაკეთო|ავაწყო|შევქმნა|დავამზადოთ)\s+(\S+\s+){0,2}"
          r"(ბომბ|ასაფეთქებ|ყუმბარ|იარაღ|დეტონატორ)|"
          r"(ბომბ|ასაფეთქებ|ყუმბარ)\S*\s+(დამზადებ|აწყობ|გაკეთებ)"),
    _rule("fraud_ka", "fraud",
          r"(დავმალო|დავუმალო|დავმალოთ|დავუმალოთ|გადავმალო)\s+(\S+\s+){0,3}"
          r"(შემოსავ|ბრუნვ|გადასახად|თანხ|ფულ|დღგ)|"
          r"(შემოსავ|ბრუნვ|თანხ|ფულ)\S*\s+(\S+\s+){0,2}(დავმალო|დავუმალო|დავმალოთ|დავუმალოთ)|"
          r"(გავაყალბო|გავაყალბოთ|გავყალბო)|"
          r"(დავამზადო|გავაკეთო|ვიყიდო|შევქმნა|მოვამზადო|ვიშოვო)\s+(\S+\s+){0,1}ყალბ|"
          r"(შევამცირო|გადავახვიო|შევცვალო|დავწიო|ჩამოვწიო)\s+(\S+\s+){0,2}გარბენ|"
          r"გარბენ\S*\s+(\S+\s+){0,2}(შევამცირო|გადავახვიო|შევცვალო|დავწიო|ჩამოვწიო)|"
          r"(შევამცირო|დავწიო|ჩამოვწიო)\s+(\S+\s+){0,2}(ინვოის|დეკლარაცი)\S*\s+(\S+\s+){0,2}"
          r"(ღირებულებ|ფას)|"
          r"(საბაჟო|შემოსავლების\s+სამსახურ|საგადასახადო)\S*\s+გვერდის\s+ავლით"),
    _rule("self_harm_ka", "self_harm",
          r"(თავი\s+მოვიკლა|სიცოცხლე\s+მოვისწრაფო|თვითმკვლელობის\s+(გზ|მეთოდ))"),
)  # fmt: skip

# Wording worth noting in the audit trail but never refused on its own: "is
# there a loophole in the 1% regime" is a fair research question.
WEAK_SIGNALS = re.compile(
    r"\b(bypass|circumvent|evade|evasion|untraceable|cheat|loophole|without\s+paying|"
    r"avoid\s+paying|off\s+the\s+books|under\s+the\s+table|grey\s+import|"
    r"without\s+(anyone|them)\s+(knowing|noticing))\b|"
    r"გვერდის\s+ავლით|ისე\s+რომ\s+(არ|ვერ)\s+(გაიგ|შეამჩნი)|უკანონოდ|ჩუმად",
    re.IGNORECASE,
)

# Questions *about* legality or consequences ("is it legal to ...", "what happens
# if ...") are research questions: a weak signal in them does not need a classifier.
INQUIRY = re.compile(
    r"\b(is\s+it\s+(legal|illegal|lawful|allowed|permitted|against\s+the\s+law)|"
    r"what\s+happens\s+(if|when|to)|what\s+(are|is)\s+the\s+(penalt\w+|fines?|consequences?|"
    r"punishment|sanctions?)|are\s+there\s+(penalties|fines|sanctions))\b|"
    r"კანონიერია|ლეგალურია|დასაშვებია|რა\s+ხდება,?\s+(თუ|როცა)|"
    r"რა\s+(ჯარიმა|სასჯელი|სანქცია|პასუხისმგებლობა)",
    re.IGNORECASE,
)

# Instruction-like text inside retrieved documents: injection phrasing, text
# addressed to an AI, or text telling the reader what to answer. Applied
# sentence by sentence; hidden HTML comments are matched across lines.
DOCUMENT_RULES: tuple[Rule, ...] = (
    *INJECTION_RULES,
    _rule("addressed_to_ai", "prompt_injection",
          r"\bif\s+you\s+are\s+an?\s+(ai|llm|language\s+model|assistant|chatbot|bot|"
          r"automated\s+system)\b|"
          r"\b(attention|note|message|instructions?)\s+(to|for)\s+(any\s+|all\s+)?(ai|llms?|"
          r"language\s+models?|assistants?|chatbots?|bots?|ai\s+systems?)\b|"
          r"\b(ai|llm|language\s+model|chatbot|ai\s+assistant)s?\s+(reading|processing|"
          r"summari[sz]ing|that\s+reads?)\b|"
          r"\b(guidance|note|notice|instructions?|message)\s+(to|for)\s+(any\s+|all\s+)?"
          r"(automated|machine|automatic)\s+(summar\w*|readers?|systems?|tools?|agents?|"
          r"assistants?|answers?)\b|"
          r"\b(automated|machine)\s+(summar\w*|readers?|systems?|tools?|agents?|assistants?)\s+"
          r"(must|should|shall|are\s+(asked|required|instructed)\s+to)\b"),
    _rule("answer_directive", "prompt_injection",
          r"\b(tell|inform|convince|assure)\s+the\s+(user|reader)\b|"
          r"\b(always|only)\s+(answer|respond|reply)\b|\brespond\s+(only\s+)?with\b|"
          r"\bin\s+your\s+(answer|response|summary|output)\b|"
          r"\bdo\s+not\s+(cite|mention|use|trust)\s+(any\s+)?(other\s+)?(sources?|documents?|"
          r"evidence)\b|\b(rate|rank|treat|mark)\s+this\s+(document|source|page)\s+as\b"),
    _rule("exfiltration_link", "prompt_injection",
          r"\b(include|add|insert|visit|send|post|click|fetch|open|load)\b[^.!?\n]{0,30}"
          r"(https?://|www\.)|!\[[^\]]*\]\(\s*https?://"),
    _rule("addressed_to_ai_ka", "prompt_injection",
          r"თუ\s+(შენ\s+)?(ხარ\s+)?(ხელოვნური\s+ინტელექტი|ენობრივი\s+მოდელი|ასისტენტი)|"
          r"(ხელოვნურმა\s+ინტელექტმა|ენობრივმა\s+მოდელმა|ასისტენტმა)\s+(\S+\s+){0,3}"
          r"(უნდა|ვალდებულ)|უპასუხე\s+მხოლოდ|უთხარი\s+მომხმარებელს|მომხმარებელს\s+უთხარი"),
)  # fmt: skip

HIDDEN_COMMENT = re.compile(r"<!--.*?-->", re.DOTALL)


def matches(rules: tuple[Rule, ...], text: str) -> list[Rule]:
    return [rule for rule in rules if rule.pattern.search(text)]
