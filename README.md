# FormPredict

`form` branch of **suraj-betting-apps**. Streamlit app that reads Racing &
Sports **meeting exports** (`<date>-<TRACK>-T.xlsx`) and gives every horse a
form score and a win probability.

| App | Branch | Entry point | Input |
|---|---|---|---|
| GreyhoundPredictor | `greyhound` | `app.py` | pasted Enhanced Form page |
| HorsePredictor | `horse` | `app.py` | pasted Enhanced Form page |
| HarnessPredict | `harness` | `app.py` | pasted Enhanced Form page |
| SoccerPredict | `soccer` | `app.py` | season CSVs |
| **FormPredict** | `form` | `app.py` | **uploaded `-T.xlsx` meeting files** |

Upload one or more meetings and every race on the card is scored. Nothing is
pasted and nothing is typed.

## Files

- `meeting_parser.py` — reads the stacked-race-block layout of the meeting export.
- `form_model.py` — form score, win probability, place probability.
- `app.py` — Streamlit UI: Upload, Whole Card, Method.
- `test_form_model.py` — regression suite for both modules.

## The one fact that shapes everything

**The meeting export contains no market prices.** Not a blank column — the
format has no odds column at all. Which means:

- This is necessarily a **form-only** model.
- The finding from the other branches — that a trained model could not out-rank
  the market — **does not apply here**, because there is no market to lose to.
- Nothing in this app is a claim about value or profit.

If you want anything price-aware, the `<date>-<track>-rNN.xlsx` single-race
export has 128 columns including `Best Fixed Odds`. Its `Finish Result` column
updates after the race — capture it, and that data can train a much stronger
model.

## What it scored

65 races from 27 August 2026 — nine meetings across Australia, Britain and
Ireland, 712 runners, results from the published board.

| | top pick wins | its top 3 place | winner in top 3 |
|---|---|---|---|
| dart throw | 9.2% | 25.6% | 33.8% |
| **this model** | **21.5%** | **42.6%** | **55.4%** |

Win probabilities are calibrated: across six probability bands the mean gap
between predicted and actual was **1.5 points**, and the most confident band
predicted 31.8% against 33.3% actual.

## Read the top three as a group, not an order

| form rank | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|
| win% | 21.5 | 16.9 | 16.9 | 12.3 | 3.1 |
| place% | 44.6 | 43.1 | 40.0 | 24.6 | 20.0 |

Ranks 1–3 are indistinguishable; rank 4 falls away. The score resolves a *group
of three*, not a ranking within it — and the app says so above every table.

## Why the weights are fixed rather than fitted

Fitting was tried first and lost. A 28-feature conditional logit scored **16.9%**
on top pick — *worse than half a dozen single columns* — and gave negative
weights to "better record at this distance" and "better record on this surface".
One cause for both: 65 races cannot identify 28 free parameters, so the fit
chases noise and lands on signs that are physically backwards.

The fixed weights cost less than they look. Against **200 random weightings** of
the same 14 columns they sit *inside* the spread on every measure (top-1 median
0.200, range 0.169–0.246; these weights 0.215). Removing the single
highest-weighted column entirely moves top-1 from 0.215 to 0.200. **The choice of
columns does the work, not the numbers on them.**

## Two things deliberately not done

**No place-probability correction.** Plackett–Luce overstates the place chance at
the top, and does here — 0.62 predicted against 0.57 actual. A shrink toward the
base rate was measured at six strengths: runner-weighted calibration error came
out 0.038, 0.033, 0.035, 0.036, 0.030, 0.032 for shrink 0.00→0.30.
Non-monotonic, spanning half a point — noise, not signal. Picking the minimum
would be tuning on the same 65 races it is scored against, so the figures ship
uncorrected and the optimism is stated instead. The slider is in the sidebar.

**No claim it beats its best single column.** Last-start beaten margin used alone
scores 24.6% against the model's 21.5%. Three races separate them at n=65. The
model wins clearly on placegetters (42.6% vs 35.4%) and top-three coverage,
which is why it ships — but the honest statement is that it beats a dart throw
and nothing finer is resolvable yet.

## Parser design

The export is stacked race blocks on one sheet. Two rules keep it honest:

1. **Blocks are found by their header row**, never by counting lines — the
   number of runners and of blank rows between blocks both vary.
2. **A race ends at the first blank tab cell**, not at a fixed offset.

The percentage triplets (`12-28-40`) are **win% – place% – starts** and are split
into three real columns. Read as text they are useless; read as only their first
number they throw away both the place rate and the sample size, and `0-0-0` from
a first-starter would be indistinguishable from a genuine 0% off forty runs.

Form strings (`x604x`) count `0` as **10**, not as a win — the single most
dangerous misread in the format.

A race whose tab numbers are not `1..n` raises a warning rather than silently
shrinking the field.

## Sample size

65 races. Every rate here carries a 95% interval roughly ±10 points wide. Treat
them as "clearly better than random", not as precise figures.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Regression test

```bash
python test_form_model.py
```

Expect `PASS 105  FAIL 0` when the nine 2026-08-27 meeting files are in
`Downloads`, `PASS 92  FAIL 0` without them — the suite builds its own synthetic
meeting so it still means something on a machine that has no spreadsheets.

## Deploy (Streamlit Community Cloud)

New app → repo `sbachagian1101/suraj-betting-apps` → branch `form` → main file
`app.py` → Advanced settings → Python **3.13**. Pushing to the branch
auto-redeploys.

> `openpyxl` is in `requirements.txt` and must stay there — `pandas.read_excel`
> needs it for `.xlsx` and does not vendor it. Without it every upload fails on
> Cloud while working fine locally.

---

*Prediction is probabilistic decision support, not a guaranteed outcome. Gamble
responsibly — Gambling Help 1800 858 858,
[gamblinghelponline.org.au](https://www.gamblinghelponline.org.au).*
