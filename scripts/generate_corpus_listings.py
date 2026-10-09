"""Generate the list-like part of the synthetic corpus from explicit tables.

Charging-station directory pages, used-car price-index pages and import
statistics bulletins are naturally templated in the real world, so they are
generated here instead of being written by hand. All data below is synthetic.

    uv run python scripts/generate_corpus_listings.py [--out corpus]

The output is committed; ``tests/unit/test_corpus_frozen.py`` checks that this
script still reproduces the committed files byte for byte.
"""

from __future__ import annotations

import argparse
import json
import re
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

SNAPSHOT = "2026-09-01"


@dataclass(frozen=True)
class GeneratedDoc:
    id: str
    title: str
    published: str
    source_type: str
    publisher: str
    body: str


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def render(doc: GeneratedDoc, publisher_slug: str | None = None) -> str:
    topic, language, _ = doc.id.split("-")
    host = publisher_slug or _slug(doc.publisher)
    front = [
        "+++",
        f"id = {json.dumps(doc.id)}",
        f"title = {json.dumps(doc.title, ensure_ascii=False)}",
        f'url = "https://{host}.invalid/{doc.id}"',
        f"published = {doc.published}",
        f"snapshot = {SNAPSHOT}",
        f'language = "{language}"',
        f'topic = "{topic}"',
        f'source_type = "{doc.source_type}"',
        f"publisher = {json.dumps(doc.publisher, ensure_ascii=False)}",
        "synthetic = true",
        'license = "synthetic"',
        "+++",
    ]
    return "\n".join(front) + "\n\n" + doc.body.strip() + "\n"


# ----------------------------------------------------------------- charging stations

# Network prices in GEL/kWh (AC, DC) and idle-fee note.
NETWORK_PRICES: dict[str, tuple[float, float | None, str]] = {
    "VoltLine": (0.55, 0.85, "0.30 GEL per minute after a 10-minute grace period"),
    "ChargeGE": (0.50, 0.79, "none"),
    "eRoute": (0.52, 0.89, "none"),
    "Mtsvane Charge": (0.49, 0.75, "none"),
    "Sea View Hotel Batumi": (0.60, None, "none"),
}


@dataclass(frozen=True)
class Station:
    doc_id: str
    name: str
    network: str
    city: str
    address: str
    connectors: tuple[tuple[str, int, int], ...]  # (type, kW, count)
    hours: str
    nearby: str
    verified: str
    note: str = ""


STATIONS_EN = (
    Station(
        "ev-en-0086",
        "VoltLine Saburtalo Hub",
        "VoltLine",
        "Tbilisi",
        "45 Vazha-Pshavela Avenue",
        (("CCS2", 150, 2), ("CHAdeMO", 50, 1), ("Type 2", 22, 2)),
        "24 hours",
        "supermarket, café",
        "2026-08-12",
    ),
    Station(
        "ev-en-0087",
        "VoltLine Vake Park",
        "VoltLine",
        "Tbilisi",
        "70 Chavchavadze Avenue",
        (("Type 2", 22, 4),),
        "07:00-23:00",
        "Vake Park",
        "2026-07-30",
    ),
    Station(
        "ev-en-0088",
        "VoltLine Didube",
        "VoltLine",
        "Tbilisi",
        "118 Tsereteli Avenue",
        (("CCS2", 60, 2), ("Type 2", 11, 2)),
        "24 hours",
        "bus station",
        "2026-08-03",
        "One CCS2 connector has been out of order since July 2026.",
    ),
    Station(
        "ev-en-0089",
        "VoltLine Gori Highway",
        "VoltLine",
        "Gori",
        "E60 highway, km 76 service area",
        (("CCS2", 120, 2), ("CHAdeMO", 50, 1)),
        "24 hours",
        "café, toilets",
        "2026-08-20",
    ),
    Station(
        "ev-en-0090",
        "VoltLine Batumi Boulevard",
        "VoltLine",
        "Batumi",
        "31 Ninoshvili Street",
        (("CCS2", 60, 2), ("Type 2", 22, 4)),
        "24 hours",
        "seafront boulevard",
        "2026-07-18",
    ),
    Station(
        "ev-en-0091",
        "ChargeGE Rustavi Mall",
        "ChargeGE",
        "Rustavi",
        "Rustavi Mall car park, Kostava Street",
        (("CCS2", 50, 2), ("Type 2", 22, 4)),
        "09:00-22:00",
        "shopping mall",
        "2026-08-05",
    ),
    Station(
        "ev-en-0092",
        "ChargeGE Gldani Market",
        "ChargeGE",
        "Tbilisi",
        "2 Khizanishvili Street",
        (("Type 2", 22, 6),),
        "24 hours",
        "market",
        "2026-06-28",
    ),
    Station(
        "ev-en-0093",
        "ChargeGE Batumi Airport",
        "ChargeGE",
        "Batumi",
        "Batumi Airport car park P1",
        (("CCS2", 50, 1), ("Type 2", 22, 2)),
        "24 hours",
        "airport terminal",
        "2026-08-14",
    ),
    Station(
        "ev-en-0094",
        "ChargeGE Kutaisi Centre",
        "ChargeGE",
        "Kutaisi",
        "15 Rustaveli Avenue",
        (("CCS2", 50, 1), ("Type 2", 22, 2)),
        "24 hours",
        "city centre",
        "2026-07-22",
    ),
    Station(
        "ev-en-0095",
        "eRoute Khashuri",
        "eRoute",
        "Khashuri",
        "E60 highway, km 118",
        (("CCS2", 120, 2), ("CHAdeMO", 50, 1), ("Type 2", 22, 1)),
        "24 hours",
        "café",
        "2026-08-09",
        "Westbound, this is the last fast charger before the Rikoti pass; the next one is in Zestaponi, 85 km away.",
    ),
    Station(
        "ev-en-0096",
        "eRoute Zestaponi",
        "eRoute",
        "Zestaponi",
        "E60 highway, km 203",
        (("CCS2", 120, 2), ("Type 2", 22, 1)),
        "24 hours",
        "restaurant",
        "2026-08-09",
        "Eastbound, this is the last fast charger before the Rikoti pass; the next one is in Khashuri, 85 km away.",
    ),
    Station(
        "ev-en-0097",
        "eRoute Samtredia",
        "eRoute",
        "Samtredia",
        "E60 highway, km 265",
        (("CCS2", 150, 2), ("CHAdeMO", 50, 1)),
        "24 hours",
        "fuel station shop",
        "2026-07-11",
    ),
    Station(
        "ev-en-0098",
        "eRoute Gudauri",
        "eRoute",
        "Gudauri",
        "Main ski lift car park",
        (("CCS2", 60, 1), ("Type 2", 22, 1)),
        "24 hours",
        "ski lifts, cafés",
        "2026-08-01",
        "Opened in December 2024 as the first DC fast charger on the Georgian Military Road above Pasanauri.",
    ),
    Station(
        "ev-en-0099",
        "eRoute Kobuleti",
        "eRoute",
        "Kobuleti",
        "Coastal road, northern entrance",
        (("CCS2", 120, 2),),
        "24 hours",
        "beach",
        "2026-07-27",
    ),
    Station(
        "ev-en-0100",
        "Mtsvane Charge Vake Business Centre",
        "Mtsvane Charge",
        "Tbilisi",
        "37 Chavchavadze Avenue",
        (("CCS2", 50, 1), ("Type 2", 22, 3)),
        "08:00-21:00",
        "offices, café",
        "2026-08-18",
    ),
    Station(
        "ev-en-0101",
        "Mtsvane Charge Ortachala",
        "Mtsvane Charge",
        "Tbilisi",
        "77 Gorgasali Street",
        (("Type 2", 22, 4),),
        "24 hours",
        "supermarket",
        "2026-06-30",
    ),
    Station(
        "ev-en-0102",
        "Mtsvane Charge Telavi",
        "Mtsvane Charge",
        "Telavi",
        "Hotel car park, Chavchavadze Street",
        (("Type 2", 22, 2),),
        "24 hours",
        "hotel, wine bars",
        "2026-07-08",
    ),
    Station(
        "ev-en-0103",
        "Sea View Hotel Batumi charger",
        "Sea View Hotel Batumi",
        "Batumi",
        "Hotel car park, Gogebashvili Street",
        (("Type 2", 11, 2),),
        "24 hours",
        "hotel",
        "2026-07-15",
        "Free for hotel guests; other drivers pay the listed price.",
    ),
    Station(
        "ev-en-0104",
        "VoltLine Mtskheta",
        "VoltLine",
        "Mtskheta",
        "E117 junction, Mtskheta bypass",
        (("CCS2", 120, 2), ("Type 2", 22, 2)),
        "24 hours",
        "café",
        "2026-08-11",
    ),
    Station(
        "ev-en-0105",
        "Mtsvane Charge Stepantsminda",
        "Mtsvane Charge",
        "Stepantsminda",
        "Village centre car park",
        (("Type 2", 22, 1),),
        "24 hours",
        "guesthouses",
        "2026-07-02",
        "AC only. The nearest DC fast charger is in Gudauri.",
    ),
)

STATIONS_KA = (
    Station(
        "ev-ka-0101",
        "VoltLine საბურთალო",
        "VoltLine",
        "თბილისი",
        "ვაჟა-ფშაველას გამზირი 45",
        (("CCS2", 150, 2), ("CHAdeMO", 50, 1), ("Type 2", 22, 2)),
        "24 საათი",
        "სუპერმარკეტი, კაფე",
        "2026-08-12",
    ),
    Station(
        "ev-ka-0102",
        "eRoute ხაშური",
        "eRoute",
        "ხაშური",
        "E60 მაგისტრალი, 118-ე კმ",
        (("CCS2", 120, 2), ("CHAdeMO", 50, 1), ("Type 2", 22, 1)),
        "24 საათი",
        "კაფე",
        "2026-08-09",
        "დასავლეთის მიმართულებით ეს არის ბოლო სწრაფი დამტენი რიკოთის უღელტეხილამდე; შემდეგი ზესტაფონშია, 85 კმ-ში.",
    ),
    Station(
        "ev-ka-0103",
        "eRoute ზესტაფონი",
        "eRoute",
        "ზესტაფონი",
        "E60 მაგისტრალი, 203-ე კმ",
        (("CCS2", 120, 2), ("Type 2", 22, 1)),
        "24 საათი",
        "რესტორანი",
        "2026-08-09",
    ),
    Station(
        "ev-ka-0104",
        "eRoute გუდაური",
        "eRoute",
        "გუდაური",
        "სათხილამურო საბაგიროს მთავარი პარკინგი",
        (("CCS2", 60, 1), ("Type 2", 22, 1)),
        "24 საათი",
        "საბაგირო, კაფეები",
        "2026-08-01",
        "გაიხსნა 2024 წლის დეკემბერში, როგორც პირველი სწრაფი დამტენი საქართველოს სამხედრო გზაზე ფასანაურის ზემოთ.",
    ),
    Station(
        "ev-ka-0105",
        "ChargeGE რუსთავი მოლი",
        "ChargeGE",
        "რუსთავი",
        "რუსთავი მოლის პარკინგი, კოსტავას ქუჩა",
        (("CCS2", 50, 2), ("Type 2", 22, 4)),
        "09:00-22:00",
        "სავაჭრო ცენტრი",
        "2026-08-05",
    ),
    Station(
        "ev-ka-0106",
        "VoltLine ისანი",
        "VoltLine",
        "თბილისი",
        "ნავთლუღის ქუჩა 10",
        (("CCS2", 60, 2), ("Type 2", 22, 2)),
        "24 საათი",
        "მეტროსადგური",
        "2026-07-19",
    ),
    Station(
        "ev-ka-0107",
        "ChargeGE ზუგდიდი",
        "ChargeGE",
        "ზუგდიდი",
        "რუსთაველის ქუჩა 50",
        (("CCS2", 50, 1), ("Type 2", 22, 2)),
        "24 საათი",
        "ქალაქის ცენტრი",
        "2026-07-25",
    ),
    Station(
        "ev-ka-0108",
        "მწვანე დატენვა ბორჯომი",
        "Mtsvane Charge",
        "ბორჯომი",
        "სასტუმროს პარკინგი, რუსთაველის ქუჩა",
        (("Type 2", 22, 2),),
        "24 საათი",
        "პარკი, სასტუმრო",
        "2026-06-24",
    ),
    Station(
        "ev-ka-0109",
        "VoltLine ფოთი",
        "VoltLine",
        "ფოთი",
        "პორტის მისასვლელი, დავით აღმაშენებლის ქუჩა",
        (("CCS2", 60, 1), ("Type 2", 22, 2)),
        "24 საათი",
        "პორტი",
        "2026-08-07",
    ),
    Station(
        "ev-ka-0110",
        "eRoute სენაკი",
        "eRoute",
        "სენაკი",
        "E60 მაგისტრალი, 301-ე კმ",
        (("CCS2", 120, 2),),
        "24 საათი",
        "საწვავის სადგური",
        "2026-07-29",
    ),
)

KA_NETWORK_PUBLISHER = {"Mtsvane Charge": "მწვანე დატენვა"}


def _ka_number(value: float) -> str:
    """Georgian convention: comma as decimal separator."""
    return f"{value:.2f}".replace(".", ",")


def station_en(s: Station) -> GeneratedDoc:
    ac, dc, idle = NETWORK_PRICES[s.network]
    has_dc = any(kind != "Type 2" for kind, _, _ in s.connectors)
    rows = "\n".join(f"| {kind} | {kw} kW | {count} |" for kind, kw, count in s.connectors)
    prices = [f"- AC charging: {ac:.2f} GEL per kWh"]
    if has_dc and dc is not None:
        prices.append(f"- DC fast charging: {dc:.2f} GEL per kWh")
    prices.append(f"- Idle fee: {idle}")
    note = f"\n\n## Status\n\n{s.note}" if s.note else ""
    body = f"""# {s.name}

Operator: {s.network}. Location: {s.address}, {s.city}. Opening hours: {s.hours}.

## Connectors

| Connector | Power | Count |
|---|---|---|
{rows}

## Prices

{chr(10).join(prices)}

## Nearby

{s.nearby}{note}

Information last verified on {s.verified}."""
    return GeneratedDoc(
        s.doc_id, f"Charging station: {s.name}", s.verified, "company", s.network, body
    )


def station_ka(s: Station) -> GeneratedDoc:
    ac, dc, _ = NETWORK_PRICES[s.network]
    has_dc = any(kind != "Type 2" for kind, _, _ in s.connectors)
    rows = "\n".join(f"| {kind} | {kw} kW | {count} |" for kind, kw, count in s.connectors)
    prices = [f"- AC დატენვა: {_ka_number(ac)} ლარი 1 კვტ.სთ-ზე"]
    if has_dc and dc is not None:
        prices.append(f"- სწრაფი DC დატენვა: {_ka_number(dc)} ლარი 1 კვტ.სთ-ზე")
    note = f"\n\n## შენიშვნა\n\n{s.note}" if s.note else ""
    publisher = KA_NETWORK_PUBLISHER.get(s.network, s.network)
    body = f"""# {s.name}

ოპერატორი: {publisher}. მისამართი: {s.address}, {s.city}. სამუშაო საათები: {s.hours}.

## კონექტორები

| კონექტორი | სიმძლავრე | რაოდენობა |
|---|---|---|
{rows}

## ფასები

{chr(10).join(prices)}

## ახლოს

{s.nearby}{note}

ინფორმაცია ბოლოს გადამოწმდა {s.verified}."""
    return GeneratedDoc(
        s.doc_id, f"დამტენი სადგური: {s.name}", s.verified, "company", publisher, body
    )


# ----------------------------------------------------------------- price index


@dataclass(frozen=True)
class PricePage:
    doc_id: str
    model: str
    years: str
    low: int
    high: int
    listings: int
    mileage: str
    note: str
    month: str  # e.g. "August 2026"
    published: str


PRICES_EN = (
    PricePage(
        "ev-en-0106",
        "Nissan Leaf 40 kWh",
        "2018-2019",
        21_000,
        26_000,
        214,
        "60,000-110,000 km",
        "Battery health varies widely; ask for a state-of-health reading.",
        "August 2026",
        "2026-08-25",
    ),
    PricePage(
        "ev-en-0107",
        "Nissan Leaf 24 kWh",
        "2013-2015",
        9_500,
        13_000,
        96,
        "100,000-160,000 km",
        "Many cars have less than 75% battery health and a real range below 100 km.",
        "August 2026",
        "2026-08-25",
    ),
    PricePage(
        "ev-en-0108",
        "Tesla Model 3 rear-wheel drive",
        "2020-2021",
        52_000,
        60_000,
        88,
        "50,000-90,000 km",
        "Most cars are repaired US auction imports.",
        "August 2026",
        "2026-08-25",
    ),
    PricePage(
        "ev-en-0109",
        "Tesla Model Y Long Range",
        "2021-2022",
        78_000,
        92_000,
        41,
        "40,000-80,000 km",
        "Supply is limited; prices vary with repair history.",
        "August 2026",
        "2026-08-25",
    ),
    PricePage(
        "ev-en-0110",
        "Hyundai Ioniq 5",
        "2022",
        74_000,
        86_000,
        37,
        "30,000-70,000 km",
        "Popular with families; 800-volt fast charging.",
        "August 2026",
        "2026-08-25",
    ),
    PricePage(
        "ev-en-0111",
        "Chevrolet Bolt",
        "2020-2021",
        36_000,
        43_000,
        73,
        "40,000-90,000 km",
        "Check that the battery recall work was done.",
        "August 2026",
        "2026-08-25",
    ),
    PricePage(
        "ev-en-0112",
        "Volkswagen ID.4 (China market)",
        "2021-2022",
        58_000,
        66_000,
        52,
        "20,000-60,000 km",
        "China-market cars use the GB/T DC charging connector.",
        "August 2026",
        "2026-08-25",
    ),
    PricePage(
        "ev-en-0113",
        "Kia Niro EV",
        "2020",
        42_000,
        49_000,
        34,
        "50,000-100,000 km",
        "Liquid-cooled battery; good winter efficiency with heat pump option.",
        "August 2026",
        "2026-08-25",
    ),
    PricePage(
        "ev-en-0114",
        "Toyota Prius hybrid",
        "2016",
        24_000,
        29_000,
        302,
        "120,000-220,000 km",
        "Included for comparison with electric cars.",
        "August 2026",
        "2026-08-25",
    ),
    PricePage(
        "ev-en-0115",
        "Toyota Camry 2.5 petrol",
        "2018",
        38_000,
        45_000,
        187,
        "90,000-160,000 km",
        "Included for comparison with electric cars.",
        "August 2026",
        "2026-08-25",
    ),
    PricePage(
        "ev-en-0116",
        "Nissan Leaf 40 kWh",
        "2018-2019",
        27_000,
        32_000,
        156,
        "40,000-90,000 km",
        "Prices rose after the 2022 fuel price spike.",
        "September 2023",
        "2023-09-12",
    ),
    PricePage(
        "ev-en-0117",
        "Tesla Model 3 rear-wheel drive",
        "2020-2021",
        68_000,
        78_000,
        49,
        "30,000-70,000 km",
        "Imports pay 18% VAT, which keeps prices high.",
        "September 2023",
        "2023-09-12",
    ),
    PricePage(
        "ev-en-0118",
        "Hyundai Ioniq 5",
        "2022",
        95_000,
        110_000,
        12,
        "10,000-40,000 km",
        "Very few cars on the market.",
        "September 2023",
        "2023-09-12",
    ),
)

PRICES_KA = (
    PricePage(
        "ev-ka-0111",
        "Nissan Leaf 40 kWh",
        "2018-2019",
        21_000,
        26_000,
        214,
        "60 000-110 000 კმ",
        "ბატარეის მდგომარეობა ძალიან განსხვავდება; მოითხოვეთ SOH-ის შემოწმება.",
        "2026 წლის აგვისტო",
        "2026-08-25",
    ),
    PricePage(
        "ev-ka-0112",
        "Tesla Model 3 (უკანა წამყვანი)",
        "2020-2021",
        52_000,
        60_000,
        88,
        "50 000-90 000 კმ",
        "მანქანების უმეტესობა აშშ-ის აუქციონიდან შემოყვანილი და შეკეთებულია.",
        "2026 წლის აგვისტო",
        "2026-08-25",
    ),
    PricePage(
        "ev-ka-0113",
        "Chevrolet Bolt",
        "2020-2021",
        36_000,
        43_000,
        73,
        "40 000-90 000 კმ",
        "გადაამოწმეთ, ჩატარდა თუ არა ბატარეის გამოწვევის (recall) სამუშაოები.",
        "2026 წლის აგვისტო",
        "2026-08-25",
    ),
    PricePage(
        "ev-ka-0114",
        "Hyundai Ioniq 5",
        "2022",
        74_000,
        86_000,
        37,
        "30 000-70 000 კმ",
        "პოპულარულია ოჯახებში; 800-ვოლტიანი სწრაფი დატენვა.",
        "2026 წლის აგვისტო",
        "2026-08-25",
    ),
    PricePage(
        "ev-ka-0115",
        "Nissan Leaf 24 kWh",
        "2013-2015",
        9_500,
        13_000,
        96,
        "100 000-160 000 კმ",
        "ბევრ მანქანას ბატარეის 75%-ზე ნაკლები ჯანმრთელობა და 100 კმ-ზე ნაკლები რეალური სავალი აქვს.",
        "2026 წლის აგვისტო",
        "2026-08-25",
    ),
)


def price_en(p: PricePage) -> GeneratedDoc:
    body = f"""# Price index: {p.model} ({p.years})

Georgia Auto Review price index, {p.month}. Typical asking prices for cars in good condition at the Rustavi car market and on the main listing sites.

| Measure | Value |
|---|---|
| Typical price | {p.low:,}-{p.high:,} GEL |
| Listings analysed | {p.listings} |
| Typical mileage | {p.mileage} |

## Notes

{p.note}"""
    title = f"Price index {p.month}: {p.model} {p.years}"
    return GeneratedDoc(p.doc_id, title, p.published, "news", "Georgia Auto Review", body)


def price_ka(p: PricePage) -> GeneratedDoc:
    low, high = (f"{v:,}".replace(",", " ") for v in (p.low, p.high))
    body = f"""# ფასების ინდექსი: {p.model} ({p.years})

ავტომიმოხილვის ფასების ინდექსი, {p.month}. კარგ მდგომარეობაში მყოფი მანქანების ტიპური ფასები რუსთავის ავტობაზრობაზე და განცხადებების მთავარ საიტებზე.

| მაჩვენებელი | მნიშვნელობა |
|---|---|
| ტიპური ფასი | {low}-{high} ლარი |
| გაანალიზებული განცხადებები | {p.listings} |
| ტიპური გარბენი | {p.mileage} |

## შენიშვნა

{p.note}"""
    title = f"ფასების ინდექსი, {p.month}: {p.model} {p.years}"
    return GeneratedDoc(p.doc_id, title, p.published, "news", "ავტომიმოხილვა", body)


# ----------------------------------------------------------------- import statistics


@dataclass(frozen=True)
class ImportStats:
    period_en: str
    period_ka: str
    total: int
    bev_share: float
    hybrid_share: float
    fleet: int
    fleet_date_en: str
    fleet_date_ka: str
    published: str


IMPORTS = {
    "2021": ImportStats(
        "2021",
        "2021 წელი",
        92_000,
        1.8,
        22,
        5_300,
        "31 December 2021",
        "2021 წლის 31 დეკემბერს",
        "2022-02-15",
    ),
    "2022": ImportStats(
        "2022",
        "2022 წელი",
        101_000,
        3.1,
        25,
        7_900,
        "31 December 2022",
        "2022 წლის 31 დეკემბერს",
        "2023-02-14",
    ),
    "2023": ImportStats(
        "2023",
        "2023 წელი",
        108_000,
        5.0,
        27,
        11_600,
        "31 December 2023",
        "2023 წლის 31 დეკემბერს",
        "2024-02-13",
    ),
    "2024": ImportStats(
        "2024",
        "2024 წელი",
        112_000,
        7.2,
        29,
        17_300,
        "31 December 2024",
        "2024 წლის 31 დეკემბერს",
        "2025-02-12",
    ),
    "2025": ImportStats(
        "2025",
        "2025 წელი",
        118_000,
        9.4,
        31,
        24_100,
        "31 December 2025",
        "2025 წლის 31 დეკემბერს",
        "2026-02-11",
    ),
    "H1-2026": ImportStats(
        "the first half of 2026",
        "2026 წლის პირველი ნახევარი",
        63_500,
        11.8,
        32,
        28_400,
        "30 June 2026",
        "2026 წლის 30 ივნისს",
        "2026-07-28",
    ),
}

IMPORTS_EN_IDS = {
    "2021": "ev-en-0119",
    "2022": "ev-en-0120",
    "2023": "ev-en-0121",
    "2024": "ev-en-0122",
    "2025": "ev-en-0123",
    "H1-2026": "ev-en-0124",
}
IMPORTS_KA_IDS = {"2024": "ev-ka-0116", "2025": "ev-ka-0117", "H1-2026": "ev-ka-0118"}


def _counts(s: ImportStats) -> tuple[int, int, int]:
    bev = round(s.total * s.bev_share / 100)
    hybrid = round(s.total * s.hybrid_share / 100)
    return bev, hybrid, s.total - bev - hybrid


def imports_en(key: str) -> GeneratedDoc:
    s = IMPORTS[key]
    bev, hybrid, other = _counts(s)
    body = f"""# Imports of passenger cars by fuel type: {s.period_en}

## Imports

| Fuel type | Cars | Share |
|---|---|---|
| Battery electric (BEV) | {bev:,} | {s.bev_share:.1f}% |
| Hybrid (incl. plug-in) | {hybrid:,} | {s.hybrid_share:.1f}% |
| Petrol, diesel and gas | {other:,} | {100 - s.bev_share - s.hybrid_share:.1f}% |
| Total | {s.total:,} | 100% |

## Registered fleet

The number of battery electric passenger cars registered in Georgia was {s.fleet:,} on {s.fleet_date_en}.

## Methodology

Import figures count vehicles declared for import, including vehicles later re-exported to neighbouring countries. The registered fleet comes from the vehicle registry."""
    title = f"Imports of passenger cars by fuel type, {s.period_en}"
    return GeneratedDoc(
        IMPORTS_EN_IDS[key],
        title,
        s.published,
        "statistics_office",
        "National Statistics Bureau",
        body,
    )


def imports_ka(key: str) -> GeneratedDoc:
    s = IMPORTS[key]
    bev, hybrid, other = _counts(s)

    def n(value: int) -> str:
        return f"{value:,}".replace(",", " ")

    def pct(value: float) -> str:
        return f"{value:.1f}".replace(".", ",") + "%"

    body = f"""# მსუბუქი ავტომობილების იმპორტი საწვავის ტიპის მიხედვით: {s.period_ka}

## იმპორტი

| საწვავის ტიპი | ავტომობილები | წილი |
|---|---|---|
| ელექტრომობილი (BEV) | {n(bev)} | {pct(s.bev_share)} |
| ჰიბრიდი (plug-in-ის ჩათვლით) | {n(hybrid)} | {pct(s.hybrid_share)} |
| ბენზინი, დიზელი და გაზი | {n(other)} | {pct(100 - s.bev_share - s.hybrid_share)} |
| სულ | {n(s.total)} | 100% |

## რეგისტრირებული ავტოპარკი

საქართველოში რეგისტრირებული ელექტრომობილების რაოდენობა {s.fleet_date_ka} შეადგენდა {n(s.fleet)}-ს.

## მეთოდოლოგია

იმპორტის მონაცემები მოიცავს იმპორტისთვის დეკლარირებულ ყველა ავტომობილს, მათ შორის მეზობელ ქვეყნებში რეექსპორტირებულს. რეგისტრირებული ავტოპარკის მონაცემები სატრანსპორტო საშუალებების რეესტრიდანაა."""
    title = f"მსუბუქი ავტომობილების იმპორტი საწვავის ტიპის მიხედვით, {s.period_ka}"
    return GeneratedDoc(
        IMPORTS_KA_IDS[key],
        title,
        s.published,
        "statistics_office",
        "სტატისტიკის ეროვნული ბიურო",
        body,
    )


KA_PUBLISHER_SLUGS = {
    "მწვანე დატენვა": "mtsvane-charge",
    "ავტომიმოხილვა": "avtomimokhilva",
    "სტატისტიკის ეროვნული ბიურო": "national-statistics-bureau",
}


def generate() -> Iterator[GeneratedDoc]:
    yield from (station_en(s) for s in STATIONS_EN)
    yield from (station_ka(s) for s in STATIONS_KA)
    yield from (price_en(p) for p in PRICES_EN)
    yield from (price_ka(p) for p in PRICES_KA)
    yield from (imports_en(k) for k in IMPORTS_EN_IDS)
    yield from (imports_ka(k) for k in IMPORTS_KA_IDS)


def write_all(out: Path) -> list[Path]:
    paths = []
    for doc in generate():
        topic, language, _ = doc.id.split("-")
        path = out / topic / language / f"{doc.id}.md"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(render(doc, KA_PUBLISHER_SLUGS.get(doc.publisher)), encoding="utf-8")
        paths.append(path)
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("corpus"))
    args = parser.parse_args()
    print(f"wrote {len(write_all(args.out))} documents to {args.out}")


if __name__ == "__main__":
    main()
