# Data sources, licences and attribution

Every dataset this project reads appears below with its source, its licence and
the credit its licence requires. The panel agent `ethics-licensing` checked each
licence against the publisher's own page on 2026-09-15. A licence marked READ
comes from that page. A licence marked UNCONFIRMED has no page that states one,
and no result from that source enters a table until it does.

Five of these sources use a Creative Commons Attribution 4.0 licence. That
licence is free to use and free to redistribute, and it requires credit. This
file is that credit.

## 1. The two reported domains

### Australian electricity prices and demand

Source: AEMO Aggregated Price and Demand Data, Victoria.
Files: `Data/AEMO/<year>/PRICE_AND_DEMAND_<YYYYMM>_VIC1.csv`.
Reader: `Data/preprocess_aemo.py`.
Licence: READ. AEMO's copyright permissions page gives general permission to use
AEMO material for any purpose with accurate attribution, and no specific
permission is needed.
Credit: "Contains AEMO data."

A caution for a later reader. Some AEMO operational demand data carries a
"not for commercial use" mark. That is a different dataset — the near real-time
operational forecast. This project does not read it.

### British regional carbon intensity

Source: the Carbon Intensity API, National Grid ESO.
Endpoint: `api.carbonintensity.org.uk/regional/intensity/...`.
Files: `Data/GBCI_SthEng_*.csv`, `Data/GBCI_SthWest_*.csv`.
Readers: `fetch_carbon_regions.py`, `extract_carbon_region.py`.
Licence: READ, CC-BY 4.0, from the project's own terms repository.
Credit: "Carbon Intensity data from the National Grid ESO Carbon Intensity API,
used under CC-BY 4.0."
The API needs no key. The fetch script sends only a descriptive User-Agent.

### The scheduling problem itself

Source: CSPlib problem 059, Energy-Cost Aware Scheduling.
Files: `Data/sampleinstances/*/instance.txt`.
Reader: `behavior_instances.py`, and the whole ICON pipeline.
Licence: READ, CC-BY 4.0, from csplib.org/Problems/prob059.
Credit: "ICON energy-cost aware scheduling instances by Helmut Simonis, Barry
O'Sullivan, Deepak Mehta, Barry Hurley and Milan De Cauwer, CSPlib problem 059,
used under CC-BY 4.0."

## 2. The baseline's own data

Source: Azure LLM Inference Traces, Microsoft.
Licence: READ, CC-BY.
Credit: "Azure LLM Inference Traces, Microsoft, used under CC-BY."
The published 3D-Learning paper evaluates on this data. This project does not.

## 3. British demand, used to build the carbon features

Source: NESO data portal, package `8f2fe0af-871c-488d-8bad-960426f24601`.
Reader: `fetch_neso_demand.py`.
Licence: READ. The NESO open licence is based on the Open Government Licence
v3.0, which the portal states is compatible with CC-BY 4.0.
Credit: "Contains NESO data, used under the NESO Open Licence."
No key needed.

## 4. Screened domains — measured, not reported

These sources fed the domain screen in DOMAIN_SCREEN_2026-08-28.md. None carries
a reported number today.

| source | reader | licence | credit owed |
|---|---|---|---|
| TfNSW road traffic counts | `run_cycles.py` | READ, CC-BY 4.0 | yes |
| GitHub Archive event data | `fetch_gharchive.py` | READ, CC-BY 4.0 | yes |
| Wikipedia pageviews | `fetch_wikipedia.py` | READ, CC0 | no |
| USDA NASS QuickStats | `fetch_crop.py` | READ, CC0 | no |
| BTS on-time performance | `run_cycles.py` | READ, US public domain | no |
| NYC TLC taxi | `get_nyctaxi.py` | INFERRED favourable | no restriction found |
| USDA AMS Market News | `fetch_grapes.py` | UNCONFIRMED | unknown |
| Binance archives | `run_cycles.py` | UNCONFIRMED | unknown |

Credit for the two that need it:
"Transport for NSW road traffic counts, used under CC-BY 4.0."
"GH Archive event data, used under CC-BY 4.0."

THE TWO UNCONFIRMED ROWS BIND A RULE. No screen result from USDA AMS Market News
or from Binance enters any table until someone reads the publisher's licence page
and records it here.

## 5. Keys and credentials

No key, token or password sits anywhere in this repository. The panel swept the
whole tree, the archive, 1,344 scheduler logs and all 29 commits on 2026-09-15
and found none. Two data downloads need a key. Both read it from an environment
variable and neither writes it to disk.

    NASS_KEY   USDA NASS QuickStats
    MMN_KEY    USDA AMS Market News

## 6. The code

This repository is the 3D-Learning authors' own release, extended. `LICENSE`
carries "Copyright (c) 2026 JiaqiWen-25" under the MIT licence. `README.md` links
the paper, arXiv 2602.02943, accepted at INFOCOM 2026. MIT permits use,
modification and redistribution with the copyright notice and the licence text
kept, which this repository does.

Gurobi is a commercial solver. This project calls it under an academic licence
and never bundles or redistributes it.
