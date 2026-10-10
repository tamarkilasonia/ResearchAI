# Synthetic world facts (design reference)

This is the "world bible" used to write the synthetic corpus: the canonical
facts, dates and fictional publishers behind the documents. It is kept for
transparency and for writing new cases.

- **The corpus documents are the source of truth.** Every reference fact in the
  evaluation dataset cites documents, and `python -m researchai.evaluation
  validate` checks that each value actually appears in them. This file is never
  read by the agent or by the scorers.
- **Everything here is synthetic** and describes the world as of the corpus
  snapshot (2026-09-01). The numbers are not authoritative facts about Georgia.
- **Planted traps (corpus 1.1.0) are not listed here**; they are in
  [`traps.toml`](traps.toml).

All publishers fictional. Numbers synthetic but plausible. Time-varying facts listed with dates
(older docs state older values -> natural outdated-info cases).

## Fictional publishers
gov: National Revenue Office (NRO, tax+customs); Ministry of Transport and Mobility (MTM);
     Tbilisi City Mobility Office (TCMO); Public Registry Hall (PRH)
regulator: Energy Tariff Commission (ETC)
statistics_office: National Statistics Bureau (NSB)
utility: Tbilisi Grid Company (TGC); Kartli Power Distribution (KPD)
news: Caucasus Business Daily; Tbilisi Tribune; Georgia Auto Review (also price index);
      Black Sea Economic Times; KA: საქმიანი თბილისი; ავტომიმოხილვა; ეკოამბები
industry_report: Caucasus Mobility Institute (CMI); Auto Importers Union (AIU)
ngo: EV Owners Association of Georgia (EVOA); Clean Air Tbilisi; Small Business Support Center (SBSC)
academic: Caucasus Polytechnic Institute, Energy Systems Lab (CPI)
dealer: Rustavi EV Hub; Tbilisi Auto Gallery; BYD Tbilisi (official dealer, fictional entity)
blog/forum: Leaf Owners Georgia forum; "Volt & Vine" blog; "Tbilisi Commuter" blog; accountant blog "Books & Balances"
charging networks: VoltLine; ChargeGE; eRoute; Mtsvane Charge

## A. Import & taxes (EV)
A1 BEV excise on import: 0 (exempt). Stable.
A2 Hybrids: paid 40% of standard excise until 2025-12-31; from 2026-01-01 pay 70%.
A3 Import VAT 18%: BEVs EXEMPT 2024-01-01..2026-12-31 (temporary). Before 2024: 18% applied.
A4 Customs clearance service fee: 150 GEL.
A5 Registration (tech passport+plates): standard 115 GEL (5 working days); same-day 200 GEL.
A7 Shipping US East Coast->Poti container: 2026: 1,600-2,200 USD; 2021: 1,200-1,600 USD.
A8 ~60% of imported used EVs come from US insurance (salvage) auctions (AIU 2025).
A9 Share of EV imports from China: 2023 15%; 2025 38%.
A10 China-market cars use GB/T DC connector; most GE fast chargers CCS2; GB/T->CCS2 adapters 1,400-2,000 GEL, not all cars accept them.

## B. Charging
B1 Public charging points: Jan-2023 210; Jan-2024 290; Jan-2025 385; Jun-2026 512 (MTM registry).
B2 DC fast (>=50 kW) points: Jan-2023 48; Jan-2025 102; Jun-2026 146.
B3 Tbilisi share of public points: 58% (2026) -> ~297.
B4 Batumi 61; Kutaisi 34 (Jun-2026).
B5 E60 Tbilisi-Batumi (~380 km): max gap between DC fast chargers 85 km (Khashuri-Zestaponi, Rikoti pass) as of 2026. In 2023 the gap was 160 km.
B6 Gudauri first DC fast charger opened Dec 2024 (eRoute, 60 kW); Stepantsminda AC only (22 kW).
B7 Networks Jun-2026: VoltLine 190; ChargeGE 120; eRoute 85; Mtsvane Charge 60; other 57 = 512.
B9 EVOA 2025 survey: 14% of fast-charging attempts failed (out of order / occupied / payment).
B10 Home wallbox 7.4 kW install: 1,200-1,800 GEL incl hardware (2026). 41% of Tbilisi EV owners live in apartments without a private charging point (EVOA 2025).
B11 3-phase upgrade (for 11 kW) TGC fee: 900 GEL.

## C. Electricity & fuel
C1 TGC residential (Tbilisi) from 2025-01-01 (valid 2026): 0-101 kWh 0.239; 101-301 0.292; >301 0.345 GEL/kWh incl VAT.
   2022-01-01..2024-12-31: 0.198 / 0.248 / 0.298.
C2 KPD (regions) from 2025-01-01: 0.231 / 0.284 / 0.336.
C3 Public AC 2026: VoltLine 0.55; ChargeGE 0.50; Mtsvane 0.49 GEL/kWh.
C4 Public DC 2026: VoltLine 0.85 (was 0.75 in 2024); ChargeGE 0.79; eRoute 0.89. VoltLine idle fee 0.30 GEL/min after 10 min grace.
C5 Petrol Premium avg Aug-2026 3.05 GEL/l; diesel 3.15; peak Jun-2022 3.65.
C6 No household time-of-use (night) tariff as of 2026; ETC pilot announced for 2027.

## D. Climate & range
D1 Tbilisi Jan avg ~1 C; July avg ~24 C; peaks 38-40 C.
D2 Winter range loss at ~0 C: 15-25% typical (EVOA 2024 test); heat pump cars 10-15%.
D3 Jvari pass 2,379 m; Gudauri winter -10 C -> 35-40% loss.
D4 Leaf 40 kWh (passive cooling) loses ~2.5-3.5% SOH/year in Tbilisi climate (CPI 2025 study, 420 cars).
D5 AC at 35 C+: 5-10% range reduction.

## E. Prices & market (GEL)
E1 Used (Georgia Auto Review index, Aug-2026):
  Leaf 2018-19 40 kWh 21,000-26,000 (2023: 27,000-32,000)
  Leaf 2013-15 24 kWh 9,500-13,000
  Tesla Model 3 2020-21 SR+/RWD 52,000-60,000 (2023: 68,000-78,000)
  Tesla Model Y 2021-22 LR 78,000-92,000
  Hyundai Ioniq 5 2022 74,000-86,000
  Chevrolet Bolt 2020-21 36,000-43,000
  VW ID.4 2021-22 (China-market) 58,000-66,000
  Kia Niro EV 2020 42,000-49,000
  New 2026 dealer: BYD Dolphin 62,500; BYD Atto 3 86,900; BYD Seal 109,000.
  Petrol comparison: Prius 2016 hybrid 24,000-29,000; Camry 2018 petrol 38,000-45,000.
E3 BEV share of imported passenger cars (NSB, by count): 2021 1.8%; 2022 3.1%; 2023 5.0%; 2024 7.2%; 2025 9.4%; H1-2026 11.8%.
E4 Registered BEVs: end-2022 7,900; end-2023 11,600; end-2024 17,300; end-2025 24,100; Jun-2026 28,400.
E5 Hybrids 31% of imports 2025.
E6 FX Aug-2026: 1 USD = 2.71 GEL; 1 EUR = 2.96 GEL. (2023: 1 USD = 2.62)

## F. Ownership
F1 Real-world consumption kWh/100 km (EVOA logbook 2025): Leaf 40 16.8; Model 3 RWD 14.9; Model Y LR 17.2;
   Ioniq 5 18.6; Bolt 16.1; ID.4 18.9; Niro EV 16.3; Atto 3 17.5; Dolphin 15.2.
F2 Petrol: Prius 2016 4.6 l/100 km; Camry 2.5 8.3 l/100 km (Tbilisi mixed).
F3 Maintenance/yr (CMI 2025): EV 280-350 GEL; petrol sedan 850-1,100 GEL (tyres excluded).
F4 CASCO insurance: 3.5-5% of value/yr; Tesla often 4.5-6%.
F5 Battery: Leaf 40 new pack 15,500-18,000 installed; refurbished 8,000-10,500; module repair 2,500-4,500. Model 3 pack 38,000-45,000.
F6 Depreciation 2025: EVs -18% value; petrol -9% (Georgia Auto Review index).
F7 Individuals' property tax: only if household annual income > 40,000 GEL; up to 1% of value; EVs NOT exempt.
F8 Tbilisi parking: free municipal parking for EVs 2021-01-01..2023-12-31 (pilot, ended). From 2024-01-01: 50% discount on annual permit: 180 GEL -> 90 GEL.

## G. Policy & environment
G1 Transport ~60% of NOx emissions in Tbilisi (Clean Air Tbilisi 2024).
G2 MTM 2024 strategy target: >=20% of newly registered passenger cars zero-emission by 2030.
G3 Electricity generation ~78% hydro (2025); winter imports and gas plants fill gaps.

## H. Service
H1 ~45 EV-capable repair shops (2026), cluster in Rustavi.
H2 Grey imports: no manufacturer warranty. BYD Tbilisi: 8 yr / 160,000 km battery warranty, 6 yr / 150,000 km vehicle.
H3 Battery SOH diagnostic: 60-120 GEL.

## TAX world (Individual Entrepreneur)
T1 IE registration at Public Registry Hall: 50 GEL, next business day; same-day 100 GEL.
T2 Small business status: turnover <= 500,000 GEL/yr -> 1% of turnover (since 2018-01-01).
   2011-2017: threshold 100,000 GEL, rate 5% (3% if documented expenses >= 60% of turnover).
T3 If turnover exceeds 500,000 GEL: the excess taxed at 3%; exceeding in two consecutive years -> status lost from next year.
T4 Micro business: turnover <= 30,000 GEL, no employees -> income tax exempt.
T5 VAT: register when taxable turnover in any continuous 12 months > 100,000 GEL; within 2 working days; rate 18%. Applies to small business holders too.
T6 Excluded from small business status: consulting, auditing, notary, currency exchange, gambling, producing excisable goods, activities requiring a license or permit.
T7 Filing: since 2021-01-01 monthly return + payment by 15th of the following month. Before 2021: one annual return by 1 April of the following year.
T8 Penalties: late filing 5% of tax due per month late, max 30%; late payment interest 0.05%/day.
T9 Employees allowed; employer withholds 20% personal income tax; pension 2%+2%+2%.
T10 Self-employed pension contribution 4% of income; people aged 40+ in 2019 could opt out.
T11 Status applied for via NRO e-services; effective from 1st day of following month.
T12 Cash register/POS mandatory for retail cash sales; first violation fine 500 GEL.
T15 June 2026 draft bill proposes raising threshold to 700,000 GEL from 2027 -> NOT law as of snapshot.

## Additions for generated docs
C3b eRoute AC 0.52 GEL/kWh; Mtsvane DC (few 50 kW sites) 0.75 GEL/kWh. ChargeGE/eRoute/Mtsvane no idle fee.
E7 Imported passenger cars total (NSB, incl. later re-exported): 2021 92,000; 2022 101,000; 2023 108,000; 2024 112,000; 2025 118,000; H1-2026 63,500.
   BEV count = share*total: 2021 1,656; 2022 3,131; 2023 5,400; 2024 8,064; 2025 11,092; H1-2026 7,493.
   Hybrid share: 2021 22%; 2022 25%; 2023 27%; 2024 29%; 2025 31%; H1-2026 32%.
E8 Price index pages (Georgia Auto Review, Aug 2026) also: Model Y LR 78-92k; ID.4 58-66k; Niro EV 42-49k; Leaf 24 9.5-13k; Prius 2016 24-29k; Camry 2018 38-45k.
E4b Registered BEVs end-2021: 5,300.
E9 Ioniq 5 2022 price Sep-2023: 95,000-110,000 (12 listings). Sea View Hotel Batumi AC 0.60.
T16 Small business status holders: 2017 61,000; 2019 128,000; end-2025 192,000.
T17 (distractors) LLC corporate income tax 15% on distributed profits; company property tax up to 1% of avg annual book value.
T18 Turnover from foreign clients counts toward the 500,000 GEL limit.
T19 Small business IE: no property tax on business assets unless rented out.
T20 Status decision usually within 5 working days.
