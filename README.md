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
- `app.py` — Streamlit UI.
- `test_parser.py` + `tests_fixture_tamworth_r4.txt` — regression suite.

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

Barrier position (`BP`) is now parsed and displayed but is **not** a model
feature — the feature weights were validated without it, so adding it is a
deliberate recalibration rather than a free win.

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

Expect `PASS 255  FAIL 0`. Run it whenever Racing & Sports change their markup.
If a page ever parses to fewer runners than its field table shows, save the
paste as a new fixture and add it to `FIXTURES` in `test_parser.py`.

## Deploy (Streamlit Community Cloud)

New app → repo `sbachagian1101/suraj-betting-apps` → branch `horse` → main file
`app.py` → Advanced settings → Python **3.13**. Pushing to the branch
auto-redeploys.

---

*Prediction is probabilistic decision support, not a guaranteed outcome. Gamble
responsibly — Gambling Help 1800 858 858, [gamblinghelponline.org.au](https://www.gamblinghelponline.org.au).*
