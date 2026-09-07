# Horse Race Predictor

Paste a Racing & Sports race page (or type the date, country, venue and race number),
press **Search & predict**, and the app pulls that country's open feeds, rates every
runner, blends the rating with the market and simulates the race 20,000 times.

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Input

The paste is the race identity. The breadcrumb links in an R&S page carry the country,
venue, date and race number (`/thoroughbred/australia/grafton/2026-09-07/R2`), and the
header carries the distance line, class, prize and start time. Both the plain select-all
copy and the Markdown export (`* [Race 2](...)`, pipe tables) are accepted.

When the paste is a **Full Fields** page the vendored `rs_parser` also reads each runner's
last-10 run table, which becomes the richest Australian form source. A header-only paste is
normal: the runners then come from the feeds.

## Feeds (all free, no login, verified 7 Sep 2026)

| Country | Source | What it gives |
|---|---|---|
| Australia (and any meeting it prices: UK, IRE, US, FR) | Ladbrokes AU affiliate JSON | barrier, weight, jockey, trainer, gear, last-20 form string, jockey/trainer last-50 records, speed-map label + settling position, fixed win/place odds, **every price fluctuation with a timestamp**, race preview, tips |
| France | PMU `programme` API | field, musique, career counts, earnings, draw, weight, blinkers/shoes, **every past run with the full finishing order** (beaten lengths rebuilt from the gaps), tote win odds with opening reference |
| UK / Ireland | Sporting Life JSON | draw, weight, official rating, headgear, claim, jockey, trainer, commentary, lifetime stats, previous results (position, field, class, going, BHA rating, SP), Sky Bet price with history |
| Hong Kong | HKJC race card + horse pages + sectional pages | rating and rating change, draw, weight, body weight, gear, days since run, every past run (class, draw, LBW, running positions, time), **last-400 m sectional** of the latest run |
| South Africa | Winning Form legacy racecards | merit rating, mass, draw, jockey/trainer last-30 records, wet/course/distance records, career record, stakes, every recent run with lengths behind, early bookmaker price |
| USA | Horse Racing Nation entries | program number, post, sire, trainer, jockey, scratches, morning line, HRN power rating |
| all | Betfair Exchange read-only feed | last traded / tight back-lay mid price, used where the country has no priced feed (HK, SA) |

Sources that are bot-walled to a plain request and therefore **not** used: Racing & Sports
itself (paste instead), Punters, Racenet, Sky Racing, TAB, Racing Post, Timeform,
Oddschecker, At The Races, Racing TV, Equibase, DRF, TwinSpires, bet365.

## Model

1. **Form** – each usable past run becomes a beaten margin at today's distance, adjusted for
   weight carried, field size and class (prizemoney ratio in the same currency, or class
   number), weighted by recency and distance relevance. Feeds without beaten lengths get a
   margin estimated from the finishing position at reduced evidence weight.
2. **Records** – career strike and place conversion; distance, course, surface, going and
   first-up records shrunk toward the runner's own career rate (prior = 15 starts).
3. **Connections** – jockey and trainer recent strike rates shrunk toward 10%.
4. **Conditions** – barrier scaled by distance, weight vs the field, days since last run,
   published rating vs the field, early speed and the race's pace shape, HK last-400 m sectional.
5. **Market** – de-vigged by the power method; thin-evidence runners are shrunk toward the
   market's implied rating (never toward uniform); log-space blend; the move since opening is
   a small sentiment term and drives the steamer/drifter flags.
6. **Simulation** – 20,000 Plackett-Luce draws for every finishing position, exactas and trifectas.

Weights are hand-set per country (`model.COUNTRY_DEFAULTS`), not fitted. The market has beaten
every form-only model graded in this repo, so treat the model-vs-market gap as a prompt to look
closer rather than as an edge. Edge is only quoted when the prices form a coherent book (102–180%).

## Layout

`app.py` (Streamlit page, stale-deployment guard derived from its own source) ·
`rs_paste.py` (paste identity + Markdown cleaning, wraps `rs_parser.py`) ·
`sources/*.py` (one adapter per feed, all returning `common.RaceCard`) ·
`pipeline.py` (parallel fetch, merge by saddlecloth then name, rate, simulate) ·
`model.py` · `charts.py` (Plotly) · `test_app.py` (offline snapshots in `fixtures/`).
