"""Rule-based question profiling: language, topic, kind and research aspects.

Rules first: these decisions are cheap, deterministic and testable, and they
drive the offline planner. The LLM planner receives the profile as a hint.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from researchai.agent.models import QuestionKind, QuestionProfile
from researchai.corpus.models import Language, Topic
from researchai.textutil import georgian_share, keyword_terms

GEORGIAN_QUESTION_SHARE = 0.3
MIN_FOCUSED_ASPECTS = 2


@dataclass(frozen=True)
class Aspect:
    key: str
    topic: Topic
    question_en: str
    question_ka: str
    query_en: str
    query_ka: str
    triggers: tuple[str, ...]  # stems; an aspect matches when a question term starts with one


ASPECTS: tuple[Aspect, ...] = (
    Aspect(
        "prices", Topic.EV,
        "What do electric cars cost to buy in Georgia, new and used?",
        "რა ღირს ელექტრომობილი საქართველოში, ახალი და მეორადი?",
        "electric car prices used and new Georgia",
        "ელექტრომობილის ფასი ახალი მეორადი",
        ("price", "pric", "cost", "cheap", "expensiv", "buy", "afford",
         "ფას", "ღირ", "იაფ", "ძვირ"),
    ),
    Aspect(
        "charging", Topic.EV,
        "How developed is public and home charging in Georgia?",
        "რამდენად განვითარებულია საჯარო და სახლის დატენვა საქართველოში?",
        "public charging points fast chargers home charging",
        "საჯარო დამტენი წერტილები სწრაფი დატენვა",
        ("charg", "charger", "station", "plug", "wallbox", "დამტენ", "დატენ", "სადგურ"),
    ),
    Aspect(
        "import_taxes", Topic.EV,
        "Which import taxes and fees apply to electric cars?",
        "რა გადასახადები და მოსაკრებლები ვრცელდება ელექტრომობილის იმპორტზე?",
        "electric car import excise VAT customs fee",
        "ელექტრომობილის იმპორტი აქციზი დღგ განბაჟება",
        ("tax", "excise", "vat", "custom", "import", "duty", "fee", "regist",
         "გადასახად", "აქციზ", "დღგ", "განბაჟ", "იმპორტ", "რეგისტრ"),
    ),
    Aspect(
        "energy_costs", Topic.EV,
        "How much does charging cost compared with petrol?",
        "რა ჯდება დატენვა ბენზინთან შედარებით?",
        "electricity tariff kWh charging cost petrol price",
        "ელექტროენერგიის ტარიფი დატენვის ფასი ბენზინი",
        ("electric", "tariff", "kwh", "petrol", "fuel", "gasolin", "energ", "100",
         "ტარიფ", "ელექტროენერგ", "ბენზინ", "საწვავ"),
    ),
    Aspect(
        "consumption", Topic.EV,
        "How much energy do electric cars use in real driving?",
        "რამდენ ენერგიას მოიხმარს ელექტრომობილი რეალურ პირობებში?",
        "real-world consumption kWh per 100 km",
        "მოხმარება კვტ.სთ 100 კმ-ზე",
        ("consum", "efficien", "kwh", "100", "მოხმარ"),
    ),
    Aspect(
        "climate_range", Topic.EV,
        "How do Georgia's climate and terrain affect range and batteries?",
        "როგორ მოქმედებს საქართველოს კლიმატი და რელიეფი სავალზე და ბატარეაზე?",
        "winter range loss heat battery degradation mountain pass",
        "ზამთარში სავალის კლება სიცხე ბატარეა",
        ("winter", "cold", "range", "heat", "climat", "mountain", "pass", "degrad", "summer",
         "ზამთ", "სავალ", "ცივ", "სიცხ", "კლიმატ", "ზაფხულ"),
    ),
    Aspect(
        "ownership", Topic.EV,
        "What are the maintenance, insurance and depreciation costs?",
        "რა ჯდება ტექმომსახურება, დაზღვევა და როგორ უფასურდება მანქანა?",
        "maintenance insurance depreciation battery replacement cost",
        "ტექმომსახურება დაზღვევა გაუფასურება ბატარეის შეცვლა",
        ("mainten", "insur", "depreci", "resale", "repair", "battery", "warrant", "ownership",
         "ტექმომსახ", "დაზღვ", "გაუფას", "შეკეთ", "ბატარე", "გარანტ"),
    ),
    Aspect(
        "sb_status", Topic.TAX,
        "What are the conditions and tax rate of small business status?",
        "რა პირობები და განაკვეთი აქვს მცირე ბიზნესის სტატუსს?",
        "small business status 1% turnover threshold 500,000",
        "მცირე ბიზნესის სტატუსი 1% ბრუნვა ზღვარი",
        ("small", "status", "turnover", "threshold", "limit", "rate", "მცირე", "სტატუს",
         "ბრუნვ", "ზღვარ", "განაკვეთ"),
    ),
    Aspect(
        "registration", Topic.TAX,
        "How does one register as an individual entrepreneur?",
        "როგორ დავრეგისტრირდე ინდივიდუალურ მეწარმედ?",
        "register individual entrepreneur fee Public Registry",
        "ინდივიდუალური მეწარმის რეგისტრაცია საფასური",
        ("regist", "entrepreneur", "foreign", "apply", "რეგისტრ", "მეწარმ", "უცხო"),
    ),
    Aspect(
        "vat", Topic.TAX,
        "When must a small business register for VAT?",
        "როდის უნდა დარეგისტრირდეს მცირე ბიზნესი დღგ-ზე?",
        "VAT registration threshold 100,000 GEL 12 months",
        "დღგ რეგისტრაციის ზღვარი 100 000 ლარი",
        ("vat", "დღგ"),
    ),
    Aspect(
        "filing", Topic.TAX,
        "What are the filing deadlines, pension contributions and penalties?",
        "რა ვადები, საპენსიო შენატანები და ჯარიმებია?",
        "monthly tax return deadline penalty pension contribution",
        "დეკლარაციის ვადა ჯარიმა საპენსიო შენატანი",
        ("deadlin", "return", "declar", "penalt", "fine", "pension", "late", "file", "filing",
         "ვად", "დეკლარაც", "ჯარიმ", "საპენსიო"),
    ),
    Aspect(
        "exclusions", Topic.TAX,
        "Which activities are excluded from small business status?",
        "რომელი საქმიანობებია გამორიცხული მცირე ბიზნესის სტატუსიდან?",
        "activities excluded from small business status consulting",
        "მცირე ბიზნესის სტატუსიდან გამორიცხული საქმიანობა კონსულტაცია",
        ("consult", "exclud", "eligib", "allow", "activit", "კონსულტ", "გამორიცხ", "საქმიანობ"),
    ),
)  # fmt: skip

DEFAULT_ASPECTS = {
    Topic.EV: ("prices", "charging", "import_taxes", "energy_costs", "climate_range", "ownership"),
    Topic.TAX: ("sb_status", "registration", "vat", "filing", "exclusions"),
}

_TOPIC_TRIGGERS = {
    Topic.EV: ("electr", "ev", "evs", "car", "vehicl", "tesla", "leaf", "bolt", "byd", "ioniq",
               "charg", "ელექტრომობ", "მანქან", "ავტომობ", "დამტენ", "დატენ"),
    Topic.TAX: ("entrepreneur", "business", "turnover", "freelanc", "მეწარმ", "ბიზნეს",
                "ბრუნვ", "ფრილანს"),
}  # fmt: skip

# Kind triggers match raw lower-cased words (question words are stop words for
# search, but they are exactly what signals the kind). Multi-word triggers match
# as phrases. Georgian "ღირს" means both "costs" and "is worth it", so only the
# phrase "ღირს თუ არა" ("is it worth it") signals a recommendation.
_KIND_TRIGGERS = {
    QuestionKind.RECOMMENDATION: ("recommend", "should", "worth", "advis", "choos", "გირჩევ",
                                  "ღირს თუ არა", "რჩევ", "ავირჩიო", "ვიყიდო"),
    QuestionKind.COMPARISON: ("compar", "versus", "vs", "advantag", "disadvantag", "pros", "cons",
                              "differ", "შეადარ", "შედარებ", "დადებით", "უარყოფით", "განსხვავ"),
    QuestionKind.NUMERIC: ("how much", "how many", "cost", "price", "calculat", "per", "რამდენ",
                           "ფას", "ღირს", "ღირდა", "გამოთვალ"),
}  # fmt: skip
_WORD = re.compile(r"\w+", re.UNICODE)


SHORT_TRIGGER_CHARS = 3


def _hit(term: str, trigger: str) -> bool:
    """Short triggers ("ev", "vat", "car") must match exactly; longer ones are prefixes."""
    if len(trigger) <= SHORT_TRIGGER_CHARS:
        return term == trigger
    return term.startswith(trigger)


def _matches(terms: list[str], triggers: tuple[str, ...]) -> bool:
    return any(_hit(term, trigger) for term in terms for trigger in triggers)


def detect_language(text: str) -> Language:
    return Language.KA if georgian_share(text) >= GEORGIAN_QUESTION_SHARE else Language.EN


def aspect(key: str) -> Aspect:
    return next(a for a in ASPECTS if a.key == key)


def profile_question(question: str) -> QuestionProfile:
    terms = keyword_terms(question)
    lowered = question.casefold()
    words = _WORD.findall(lowered)
    scores = {
        topic: sum(_matches([t], triggers) for t in terms)
        for topic, triggers in _TOPIC_TRIGGERS.items()
    }
    best = max(scores.values())
    topic = None if best == 0 else max(scores, key=lambda t: (scores[t], t is Topic.EV))

    kind = QuestionKind.FACTUAL
    for candidate in (QuestionKind.RECOMMENDATION, QuestionKind.COMPARISON, QuestionKind.NUMERIC):
        triggers = _KIND_TRIGGERS[candidate]
        phrases = tuple(t for t in triggers if " " in t)
        single = tuple(t for t in triggers if " " not in t)
        if _matches(words, single) or any(phrase in lowered for phrase in phrases):
            kind = candidate
            break

    aspects: list[str] = []
    if topic is not None:
        aspects = [a.key for a in ASPECTS if a.topic is topic and _matches(terms, a.triggers)]
        # Broad comparison/advice questions cover the whole topic; focused ones
        # ("should I register for VAT?") keep to the aspects they name.
        broad = kind in (QuestionKind.COMPARISON, QuestionKind.RECOMMENDATION)
        if broad and len(aspects) < MIN_FOCUSED_ASPECTS:
            aspects = list(DEFAULT_ASPECTS[topic])
    return QuestionProfile(
        language=detect_language(question), topic=topic, kind=kind, aspects=aspects
    )
