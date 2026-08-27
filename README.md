# HarnessPredict

Harness branch of **suraj-betting-apps**. Streamlit app that parses a
**Racing & Sports harness Enhanced Form** page (select-all → copy → paste) and
produces a tactical speed map, market-blended win probabilities, fair prices and
value flags.

| App | Branch | Entry point | Status |
|---|---|---|---|
| GreyhoundPredictor | `greyhound` | `app.py` | ✅ live |
| HorsePredictor | `horse` | `app.py` | ✅ live |
| **HarnessPredict** | `harness` | `app.py` | ✅ live |

## Files

- `harness_parser.py` — tolerant parser for the R&S harness Enhanced Form
  clipboard format.
- `harness_model.py` — mile-rate speed, tactical pace map, connections and
  official rating, blended with the market and simulated to a finishing order.
- `place_finder.py` — placegetter shortlist and criteria (see below).
- `app.py` — Streamlit UI.
- `test_parser.py` + `tests_fixture_gloucesterpark_r2.txt` — parser regression suite.
- `test_place_finder.py` — Place Finder regression suite.

## What was wrong with the old parser

The previous harness parser accepted **only markdown pipe-table rows**:

```python
if not re.match(r"^\|\s*\d{1,2}\s*\|", line.strip()): continue
```

A plain select-all/copy from the live R&S page is **tab separated**, so every
row was rejected and the app parsed **zero runners** — it could never work on a
real paste.

## Parser design

The field table is read through its **header row**, never by column position,
and rows keep their empty cells when split. Three rules keep it honest:

1. **Columns are found by header label** (`Tab`, `Runner`, `Driver`, `Trainer`,
   plus bookmaker columns). Locating bookmaker columns by name matters here:
   harness field tables are full of money columns (`Tot $PM`, `Car$/St`,
   `Dri L50`, `Dri PM`, `Tra L50`, `Tra PM`) that a right-most-number scan would
   happily mistake for a price.
2. **A row needs only a tab number and a horse name** to produce a runner.
3. **A gap in the tab sequence raises a warning** instead of silently shrinking
   the field.

Harness-specific handling:

- **IMR / OHR.** Each run carries an individual mile rate after the literal
  `IMR`, optionally rank-prefixed (`40th 2:00.24`), and often an official
  handicap rating as a bare number on the line *before* the literal `OHR`. Both
  are read positionally against their own label, not by counting.
- **Start position** comes from `HCP Fr5` / `HCP Sr2` inside the results line —
  front or second row plus gate — since harness field tables carry no barrier
  column. For today's race the tab number doubles as the gate.
- **Glued panels.** The Filters and Facts panels render as `<value><next label>`
  on one line (`0-2-312m` = `0-2-3` then `12m`). Because some labels start with a
  digit, a regex alone misreads the starts; the parser walks each panel in order
  and peels the known next label off the end. Harness filter panels carry
  surface categories (AW/Turf/Dirt/Sand), not going categories.
- **Label-anchored strike rates.** `Last50` appears twice per runner (driver then
  trainer), so each is read from the segment belonging to its own label.
- **Block boundaries.** A runner's form figures and price sit *above* its own
  name header, so each block ends where the next block *starts*, not at the next
  header — otherwise one runner's block swallows the next runner's form and price.
- **Odd price lines.** Prices route by bookmaker prefix; the token may contain a
  pipe (`USR|GRS$15`) and some runners have no price line at all.
- **Barrier trials are excluded** from the recent-run list — only completed races
  count toward form, speed and reliability.

## Model

Ported unchanged from the `harness-predictor` branch of `horse-race-predictor`.
Components and weights: speed 0.30, tactics 0.18, track/distance 0.12, form 0.11,
connections 0.10, rating 0.08, sectionals 0.05, reliability 0.04, freshness 0.02,
blended with the market at α = 0.56 and simulated with Plackett–Luce.

Betfair prices are parsed and displayed, but the model's market term currently
uses the **TAB price only** — adding an exchange blend would be a deliberate
recalibration, not a free win.

## Place Finder

A tab that shortlists and colour-codes probable placegetters. The criteria were
**measured on thoroughbred racing** (one Wolverhampton card — six races, 16
placegetters) and transplanted here, so they are split by how well they should
travel.

**Kept on, because the reasoning is structural**

| Rule | Why |
|---|---|
| Require the market to agree — model top 5 ∩ market top 3 | The biggest single lift there: precision 33.3% → 47.1% at the same ~3 picks. Betting markets being hard to beat is not a thoroughbred peculiarity |
| Cap at 3 selections | Coverage is not the goal; a shortlist you can bet is. Overflow shows as a reserve, not dropped |
| Use the right place count | A top-3 probability overstates the real chance when only two places pay, so the app switches to `Top2%` itself |
| Shrink toward the base rate | Pulls toward `places / runners`, trimming over-confidence at the top and lifting it at the bottom, with no curve fitted to a small sample |

**Shipped off, because the number is thoroughbred-specific**

| Rule | Why not |
|---|---|
| Exclude F/M ≥ 2.0× | On that card such horses placed 1 time in 20 — but there is **no harness evidence** for that threshold. The slider is there if you want it; the F/M column is shown either way |

F/M is fundamental ÷ market probability. The harness model does not return it
directly, so it is derived as `p_fund / p_mkt`.

Rows are colour-coded — 🟩 selection, 🟦 reserve, 🟨 excluded, ⬜ outside the model
shortlist. **Fair place $ = 1 ÷ adjusted place probability**; enter real place
odds as `tab:price` pairs to get an edge per runner.

> **Scratchings.** `predict()` filters them itself and `result["order"]` indexes
> its own active list, so the table is built from `result["runners"]` rather than
> whatever the caller passed. Getting that wrong would shift every row by one in
> any race with a scratching — the Gloucester Park fixture has two, and the test
> suite pins it.

**Nothing here is validated on harness racing yet.** It is a sensible starting
method borrowed from a neighbouring code, and the app says so on the tab.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Regression test

205 field-level checks against a real Gloucester Park R2 page capture, with the
expected values read off the page rather than from parser output:

```bash
python test_parser.py
```

Expect `PASS 205  FAIL 0`. Run it whenever Racing & Sports change their markup.

The Place Finder has its own suite covering the place-terms rule, the derived
F/M ratio, the shrink behaviour, the consensus gate and cap, and — most
importantly — that a race with scratchings stays index-aligned with the model:

```bash
python test_place_finder.py
```

Expect `PASS 60  FAIL 0`.

## Deploy (Streamlit Community Cloud)

New app → repo `sbachagian1101/suraj-betting-apps` → branch `harness` → main
file `app.py` → Advanced settings → Python **3.13**. Pushing to the branch
auto-redeploys.

---

*Prediction is probabilistic decision support, not a guaranteed outcome. Gamble
responsibly — Gambling Help 1800 858 858, [gamblinghelponline.org.au](https://www.gamblinghelponline.org.au).*
