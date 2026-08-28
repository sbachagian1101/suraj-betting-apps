# RacingScorePredictor

`score` branch of **suraj-betting-apps**. Paste a Racing & Sports **Enhanced
Form** page; every runner is scored **0–100** across the fifteen weighted
categories of the *Universal Horse-Racing Scoring Framework (Reference V2)*, and
the field is ranked.

| App | Branch | Input |
|---|---|---|
| GreyhoundPredictor | `greyhound` | pasted Enhanced Form page |
| HorsePredictor | `horse` | pasted Enhanced Form page |
| HarnessPredict | `harness` | pasted Enhanced Form page |
| SoccerPredict | `soccer` | season CSVs |
| FormPredict | `form` | uploaded `-T.xlsx` meeting files |
| **RacingScorePredictor** | `score` | **pasted Enhanced Form page** |

## Files

- `rs_parser.py` — Enhanced Form parser (AU / UK / IRE / South Africa).
- `scoring.py` — the framework: categories, weights, multipliers, availability.
- `app.py` — Streamlit UI.
- `test_scoring.py` + `tests_fixture_greyville_r7.txt` — 433-check suite.

## The calculation

For each horse and each of the fifteen categories:

1. **Category score** 0–10, read from the form page.
2. **Adjusted weight** = base × distance multiplier × surface multiplier ×
   race-type multiplier (× output profile), renormalised so the set totals 100.
3. **Available flag** — 1 if the page carried evidence, 0 if not.
4. **Contribution** = (score ÷ 10) × adjusted weight.
5. **Final** = `100 × Σ contributions ÷ Σ available adjusted weights`.
6. **Field index** = 100 × horse score ÷ best in field.
7. **Confidence** — separately, from career starts, coverage and recency.

### Base weights

| Category | Wt | Category | Wt |
|---|---|---|---|
| Recent Form | 15 | Weight / Claim / Allowances | 6 |
| Ability / Ratings / Speed | 14 | Direct H2H / Comparable Races | 6 |
| Pace / Race Shape | 10 | Course / Track Suitability | 4 |
| Class / Opposition Strength | 9 | Barrier / Draw | 3 |
| Distance / Stamina | 9 | Fitness / Preparation | 3 |
| Surface / Going | 8 | Jockey | 3 |
| Sectionals / Efficiency | 6 | Trainer | 2 |
| | | Trip / Gear / Stewards | 2 |

Distance bands (§4) move draw from ×1.35 in sprints to ×0.60 beyond 2000m and
stamina the other way; surface (§5) lifts going and sectionals on turf and
halves going terminology on synthetics; race type (§17) cuts weight to ~2.5% for
weight-for-age and lifts ability and class for Group races. Every combination
renormalises to exactly 100 — the test suite checks all 144 of them.

## The four rules that shape the implementation

**Missing means unknown, not poor.** A category with no evidence leaves the
denominator entirely rather than scoring zero. A horse never raced on today's
going is not punished for it — the framework is explicit that untested is
*neutral*. The app shows a **Weight covered** column so you can see how much of
the 100 points each horse was actually judged on.

**Field-relative thinking.** Official ratings, jockey and trainer ratings,
effective weight and draw are all scored against today's opponents.

**No double counting.** Class takes the single strongest applicable piece of
evidence; a higher-class win does not also collect same-class win and place
points.

**The market stays out.** Odds are parsed and displayed but never scored. A test
rewrites every price in the field to $1.01 and asserts every score is unchanged
to within 1e-9.

## What the page cannot give

Three sub-parameters are marked unavailable rather than guessed: **true
sectional times** (L600/L400/L200), **trials and workouts**, and **track
configuration**. On the sample race **85 of the 100 base points** were scoreable
for every runner; Class and Head-to-Head are the two that occasionally go
missing for an individual horse.

## Parser notes

Built on the `horse` branch parser, with three additions this page needed:

- **Per-run official ratings.** The OHR sits as a bare number on the line
  directly above the literal `OHR`. Without it the Ability category — the
  second-heaviest at 14 points — reads as missing on every page that shows
  ratings only per past run.
- **`Place at 800m`.** Australian pages write "Place on settling"; South African
  ones write "Place at 800m". Both are the early position that running style is
  derived from, so either is accepted.
- **Head-to-head.** The `Head to Head` block is parsed into meetings kept whole
  — a comparison only means something as a pair. It feeds the 6-point H2H
  category, and the sample race yields 37 meetings across the field.

Long-form dates (`2m 7d ago`) are also converted to days; leaving them unparsed
silently dropped the recency of every run older than about a month.

## What this is not

**Nothing here has been validated against results.** The framework itself says
its weights are *"a rational baseline for testing, not universal scientific
constants"* and asks you to freeze the rules, then track winner rank, top-3 and
top-4 recall and calibration across many completed races and several countries
before trusting them.

This app is a faithful implementation of that stated method — the thing you
would validate, not evidence that the method works. The **Model %** column is an
uncalibrated transform of the score (adjustable in the sidebar), a ranking
expressed as percentages rather than a probability forecast.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Regression test

```bash
python test_scoring.py
```

Expect `PASS 433  FAIL 0`. The suite pins the document's own numbers (base
weights, distance multipliers, the pace-pressure formula, the confidence bands),
checks that adjusted weights renormalise to 100 across every band × surface ×
race type × profile, and asserts the missing-data rule, the no-stacking rule and
the market-independence rule directly.

> It has already earned its keep: it caught a bug where every `BM` class
> collapsed to the same rank, which would have flattened the 9-point Class
> category on every Australian and South African benchmark race.

## Deploy (Streamlit Community Cloud)

New app → repo `sbachagian1101/suraj-betting-apps` → branch `score` → main file
`app.py` → Advanced settings → Python **3.13**.

---

*Prediction is probabilistic decision support, not a guaranteed outcome. Gamble
responsibly — Gambling Help 1800 858 858,
[gamblinghelponline.org.au](https://www.gamblinghelponline.org.au).*
