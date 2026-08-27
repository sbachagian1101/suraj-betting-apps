# HorsePredictor

Thoroughbred branch of **suraj-betting-apps**. Streamlit app that parses a
**Racing & Sports thoroughbred Enhanced Form** page (select-all → copy → paste)
and produces market-anchored win probabilities, fair prices and value flags.

| App | Branch | Entry point | Status |
|---|---|---|---|
| GreyhoundPredictor | `greyhound` | `app.py` | ✅ live |
| **HorsePredictor** | `horse` | `app.py` | ✅ live |
| **HarnessPredict** | `harness` | `app.py` | ✅ live |

## Files

- `horse_parser.py` — tolerant parser for the R&S thoroughbred Enhanced Form
  clipboard format.
- `horse_model.py` — Shin de-vig → conditional-logit fundamentals → Benter
  blend → discounted Plackett–Luce finishing-order simulation.
- `race_quality.py` — grades the race itself before any runner is chosen.
- `place_finder.py` — placegetter shortlist and criteria (see below).
- `results_log.py` — results ledger, performance tracking and threshold tuner.
- `app.py` — Streamlit UI.
- `test_parser.py` + `tests_fixture_tamworth_r4.txt` — parser regression suite.
- `test_place_finder.py` — Place Finder regression suite.
- `test_race_quality.py` — race-grading regression suite.
- `test_results_log.py` — ledger and tuner regression suite.

## Parser design

The field table is read through its **header row**, never by column position.
R&S ship several column layouts for the same page and freely leave cells empty —
a horse with no declared jockey, a meeting with no Bet365 column — and any
positional reader mis-assigns every column to the right of the first blank.

Three rules keep it honest:

1. **Columns are found by header label** (`Tab`, `Horse`, `WT`, `BP`, `Jockey`,
   `JRat`, `Trainer`, `TRat`, plus bookmaker columns), and rows keep their empty
   cells when split so positions survive.
2. **A row needs only a tab number and a horse name** to produce a runner.
   Weight, barrier, jockey, ratings and price are all best-effort, so a missing
   optional column can never discard a runner.
3. **A gap in the tab sequence raises a warning.** Silent field-shrinkage is the
   failure mode that matters most, so it is made loud.

Two subtler traps the parser handles explicitly:

- **Glued panels.** R&S render the Filters and Facts panels as
  `<value><next label>` on one line, e.g. `0-1-1212m` = `0-1-12` then `12m`.
  Because some labels start with a digit, a regex alone reads starts as `1212`;
  the parser walks each panel in order and peels the *known* next label off the
  end of every line.
- **Label-anchored strike rates.** `Last50` appears twice per horse (jockey then
  trainer). A horse with no declared jockey has an *empty* jockey `Last50`, so
  counting occurrences hands the trainer's strike rate to the jockey. Each rate
  is read from the segment belonging to its own label instead.

Prices are routed by their bookmaker prefix: `betfair$x` is an exchange price,
`Tab$x` is a TAB price, and anything else (`Ladbrokes$x`, `TABtouch$x`) is
treated as a fallback bookmaker price rather than being filed as Betfair.

## Model

Ported from `horse-race-predictor` and unchanged apart from two guards for
single-runner fields and one bug fix:

> **Shin solver.** The original de-vig used a fixed-point iteration that
> oscillated between two values and never converged, running all 300 iterations
> and leaving the result at an arbitrary point in that cycle — the returned
> insider fraction *z* did not even correspond to the returned probabilities.
> It is now solved by **bisection**, which is guaranteed to converge because the
> sum of implied probabilities is monotonically decreasing in *z*.

Pipeline: Shin (1993) de-vig → Betfair/TAB market blend → Bolton & Chapman
(1986) conditional logit on z-scored fundamentals → Benter (1994) log-odds
blend at weight α → Lo, Bacon-Shone & Busche (1995) discounted Plackett–Luce
Monte Carlo for place and top-3 probabilities.

### Going column fix

The model reads a horse's record on today's going. It used to collapse every
surface into two buckets:

```
race going HEAVY  ->  read the Soft_win column     # the Heavy column was ignored
race going FIRM   ->  read the Good_win column     # likewise Firm
```

A horse's Heavy, Firm, AW and Turf records are all parsed. The model now reads
the matching column, with a fallback chain (Heavy → Soft → Good) for horses with
fewer than two starts on it, and shrinkage toward the career rate so a 1-from-1
record is not read as a 100% strike rate. Synthetic and all-weather surfaces use
the `AW` column.

### Extended fundamentals (opt-out, unvalidated weights)

An audit found **115 scalar fields parsed per horse plus 5 recent runs × 21
fields**, of which the model scored ten — and of the ~105 recent-run values it
used exactly one, the last-start finishing position. Three of the strongest
unused signals were added:

| Feature | Weight | What it fixes |
|---|---|---|
| Course & distance record | 0.20 | Only the *distance* record was used; course and C&D were ignored |
| Record at this run of the preparation | 0.20 | `freshness` was `−\|days − 21\|`, a V-shape that ignored whether the horse actually performs fresh |
| Jockey/trainer partnership strike rate | 0.15 | The model used `jrat + trat` (R&S *ratings*); the partnership's real record was parsed and unused |

The run-of-preparation feature needed a parser addition: R&S tag each horse
`(FU)`, `(2U)`, `(3U)`… in the *Days Since Last Run* line, and that marker was
being discarded. It is now captured as `runup`.

> **These three weights are informed priors, not fitted values.** There is no
> labelled horse dataset in this repo to fit against — unlike the soccer model,
> where the blend weights were chosen by walk-forward backtest on 491 matches.
> They are deliberately smaller than the established terms they overlap with, and
> the whole block can be switched off with **Extended fundamentals** in the
> sidebar to compare against the ten originally calibrated features.

**One coupling worth knowing.** F/M is the ratio of fundamental to market
probability, and strengthening the fundamentals mechanically raises it for
horses the form model likes. On the Tamworth fixture the top pick's F/M moved
**1.51 → 2.02**, which trips the Place Finder's `F/M ≥ 2.0` exclusion and drops
the selections from three to two. The threshold and the feature set are not
independent — if you run the extended features, consider raising the F/M cut.

Barrier position (`BP`) and the apprentice claim are parsed and displayed but are
still **not** model features.

## Race Quality

The Place Finder answers *which runner*. This answers the question that comes
first: **is this a race where winners and placegetters can be found at all?**

### Why it exists

A full-strength LightGBM (117 features, within-race normalisation, trained on
2,959 races and scored on races it had never seen) was a **worse ranker than the
market price** at every depth:

| finding the winner | top 1 | top 2 | top 3 |
|---|---|---|---|
| Market price | **0.370** | 0.534 | 0.664 |
| Model, win-trained | 0.350 | 0.534 | **0.679** |
| Model, place-trained | 0.324 | 0.522 | 0.667 |
| Model, no market term | 0.287 | 0.492 | 0.632 |

Placegetter precision told the same story — the market's top 3 contained 1.53
placegetters on average, and no model beat it. There is no ranking edge here.

What there *is* — and it is much larger than anything a model added — is the
difference between **races**. Sorted by the favourite's price and the field
size, the favourite's place strike rate runs from 78% down to 50%. That signal
needs no training data at all: the price and the field size are already on the
page.

### The four grades

Measured on the 1,480 most recent races, none used to fit anything:

| Grade | Rule | Races | Fav wins | Fav places | Top 3 holds winner |
|---|---|---|---|---|---|
| 🟢 **PRIME** | favourite under $2.50 | 31% | **54.7%** | **78.3%** | 84.3% |
| 🔵 **STRONG** | favourite under $4, 8–10 runners | 21% | 37.7% | 74.5% | 68.7% |
| 🟡 **FAIR** | favourite under $4, other field sizes | 26% | 31.3% | 59.8% | 66.8% |
| 🔴 **SKIP** | favourite $4 or longer | 21% | 22.6% | 50.0% | 50.6% |
| — | *all races* | 100% | 38.2% | 66.6% | 69.3% |

Checked across four successive time folds: the grade ordering held in 3 or 4 of
4 folds on both win and place, and no fold reversed it.

**Skipping the red grade is the single largest improvement available.** It is a
fifth of all races and the one where the favourite places only half the time.

### Two things this deliberately does not do

**It does not use the model.** Adding "the form model also rates the favourite
top 2" moved the place rate about one point while discarding 82 of 566 races —
a difference the sample cannot separate from noise, and the win rate actually
went the other way. The model's opinion is shown as a note on the tab, never as
a gate.

**It does not promise profit.** These are strike rates. A favourite that places
78% of the time is usually priced to place about 78% of the time, and TAB's
overround on this data was 1.244 — about 19.6 cents in the dollar. The grade
tells you where the *result* is predictable, which is what it was asked for; it
does not claim those races are profitable.

Field size is not a difficulty measure on its own, which is why the bands are
where they are: fields under 8 pay only **two** places, so their lower place
rate is arithmetic rather than a warning.

## Place Finder

A dedicated tab that highlights likely placegetters. The criteria came from
scoring the model against a real Wolverhampton card — six races, 16 placegetters
— and each rule earned its place:

| Rule | Why |
|---|---|
| **Require the market to agree** — model top 5 ∩ market top 3 | The biggest single lift: precision **33.3% → 47.1%** at the same ~3 picks per race |
| **Cap at 3 selections** | A shortlist you can actually bet. Overflow is shown as a reserve, not dropped |
| **Exclude F/M ≥ 2.0×** | Horses the form model rates at twice the market's opinion placed **1 time in 20** |
| **Use the right place count** | `Top3%` is a *top-three* number. Fields of 5–7 pay only two places, so the app switches to `Top2%` automatically |
| **Shrink toward the base rate** | Observed place rates ran *above* the model in its low band and *below* it in its high band. A shrink toward `places / runners` corrects both tails at once |

### Why consensus rather than a shorter model list

Measured on the Wolverhampton card, precision per selection was:

| Rule | Precision | Picks/race |
|---|---|---|
| Model top 3 | 33.3% | 3.0 |
| Model top 5 | 40.0% | 5.0 |
| Market top 3 | 44.4% | 3.0 |
| **Model top 5 ∩ market top 3, F/M < 2** | **47.1%** | **2.8** |

Simply cutting the model's list from 5 to 3 made it *worse*, because the model's
ordering inside its own top 5 is close to noise — its 4th and 5th picks placed as
often as its 2nd and 3rd. The market supplies the discrimination the model lacks,
so the app keeps a wide model pool and lets the market choose from it.

Rows are colour-coded — 🟩 selection, 🟦 reserve, 🟨 excluded (market or F/M),
⬜ outside the model shortlist — and every threshold is adjustable in the sidebar.
Set "Market must rate inside its top" to the field size to switch the consensus
gate off entirely.

**Fair place $ = 1 ÷ adjusted place probability.** Enter actual place odds as
`tab:price` pairs and the table shows the edge on each.

### Why the shrink is a shrink and not a fitted curve

Fitting a calibration curve to 16 observations would be over-fitting dressed up
as rigour. Shrinking toward `places / runners` — what a dart throw scores — is
principled, moves both tails in the direction the data indicated, and degrades
gracefully. Set it to 0 for the raw model.

### Honest limits

These criteria rest on **one meeting**. They are a sensible working method, not
a proven edge, and the app says so on the tab. On that same card the three
shortest Betfair prices found **more** placegetters than the model did (6 of 16
versus 4 of 16). Treat the shortlist as a starting point for pricing against the
actual place market, not as a tip sheet.

## Results & Tuning

Log a prediction, enter the finishing order once the race is run, and the app
tracks how the selections are actually doing. Two things it is careful about:

**What can and cannot be learned from outcomes.** The model's feature weights
were calibrated on roughly 1,700 races. The five Place Finder thresholds *can* be
tuned from your results, because those are the part currently resting on sixteen
placegetters.

Re-fitting the model's own weights needs the full feature vector for hundreds of
races — so the ledger now **stores it**. Every logged runner carries all 13
feature values, the raw parsed fields they were built from (going and surface
records, first-up/second-up splits, jockey/trainer partnership, course and C&D
records, run-of-preparation) and the race context. That is 111 columns per
runner, kept so that features nobody has thought of yet can still be derived from
races logged today. Once a few hundred races have settled, the weights become an
empirical question instead of a judgement call.

**How much data that actually needs.** From a two-proportion power calculation at
80% power:

| Question you want answered | Races needed |
|---|---|
| Has a rule quietly broken (47% → 20%)? | **~17** |
| Does it beat a dart throw? | ~45 |
| Does consensus really beat model top-3? | ~71 |
| Is 47% genuinely better than 40%? | **~273** |

So the ledger earns its keep from race one by **monitoring**, while the tuner
stays locked until 40 settled races. Below that it reports how far off it is
rather than offering a suggestion nobody should take.

When it does unlock, tuning is scored by **leave-one-race-out cross-validation** —
thresholds chosen on the other races, then applied to the held-out one. Choosing
and scoring on the same races would manufacture an edge that is not there, and
the app shows the in-sample and cross-validated figures side by side so the gap
between them (which *is* the over-fitting) is visible. If cross-validation does
not beat the current defaults, it says so and tells you to leave them alone.

Performance reporting includes a **95% confidence interval on the strike rate**,
and warns explicitly when that interval still includes the base rate — that is,
when the selections are not yet distinguishable from picking at random.

> **Storage.** Streamlit Community Cloud wipes its filesystem on restart, so the
> **download button is the real save**. Keep the ledger CSV and re-upload it next
> session.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Regression test

255 field-level checks against a real Tamworth R4 page capture, with the
expected values read off the page rather than from parser output:

```bash
python test_parser.py
```

Expect `PASS 265  FAIL 0`. Run it whenever Racing & Sports change their markup.
If a page ever parses to fewer runners than its field table shows, save the
paste as a new fixture and add it to `FIXTURES` in `test_parser.py`.

The Place Finder has its own suite covering the place-terms rule, the shrink
behaviour, the F/M exclusion and the odds maths:

```bash
python test_place_finder.py
```

Expect `PASS 65  FAIL 0`.

The race grades have their own suite covering the tier boundaries, the price
fallback chain and its 999 sentinel, and the internal consistency of the
measured tables — a tier whose numbers drifted out of order would mislead
quietly rather than fail loudly:

```bash
python test_race_quality.py
```

Expect `PASS 104  FAIL 0`.

The ledger and tuner have their own suite. Its most important check is that
`results_log.selections_for_race` agrees exactly with `place_finder.build`: the
ledger re-derives selections from stored columns so old races can be re-scored
under new thresholds, and if those two ever drift apart every historic number
silently becomes wrong.

```bash
python test_results_log.py
```

Expect `PASS 65  FAIL 0`.

## Deploy (Streamlit Community Cloud)

New app → repo `sbachagian1101/suraj-betting-apps` → branch `horse` → main file
`app.py` → Advanced settings → Python **3.13**. Pushing to the branch
auto-redeploys.

---

*Prediction is probabilistic decision support, not a guaranteed outcome. Gamble
responsibly — Gambling Help 1800 858 858, [gamblinghelponline.org.au](https://www.gamblinghelponline.org.au).*
