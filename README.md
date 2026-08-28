# SoccerPredict

Soccer branch of **suraj-betting-apps**. Upload season match data, pick a fixture
from the dropdowns, and get **1X2**, **BTTS Yes/No** and **Over/Under 2.5**
probabilities from a Dixon–Coles team-strength model.

| App | Branch | Entry point | Status |
|---|---|---|---|
| GreyhoundPredictor | `greyhound` | `app.py` | ✅ live |
| HorsePredictor | `horse` | `app.py` | ✅ live |
| HarnessPredict | `harness` | `app.py` | ✅ live |
| **SoccerPredict** | `soccer` | `app.py` | ✅ live |

## Files

- `soccer_data.py` — CSV loading, cleaning and validation.
- `soccer_model.py` — Dixon–Coles fit, score matrix, markets, bet signal, walk-forward backtest.
- `app.py` — Streamlit UI.
- `test_model.py` — 86-check regression suite.
- `sample_data/` — Latvian Virsliga 2024, 2025, 2026 as bundled demo data.

## The bet signal

A banner above the 1X2 numbers that fires when the **1X2 view and the two most
likely scorelines agree**. It adds no new information — it re-reads figures the
model already produced — but it turns out to be a useful filter.

| | condition |
|---|---|
| ✅ **green** | the side's win probability is **> 45%** *and* **both** of the top two scorelines are wins for it |
| ⚠️ **yellow** | win probability **> 45%**, the **top** scoreline is a win for it, but the **second** is a draw |
| *(nothing)* | anything else — including a **draw as the most likely score**, which is the distribution contradicting the 1X2 number rather than confirming it |

The threshold is strict (`> 45%`, not `≥`) and adjustable in the sidebar. Both
sides are tested, though two sides can only both clear 45% if the draw is under
10%, which essentially never happens.

### How it scored

Replayed on **429 walk-forward predictions** from the three bundled Latvian
Virsliga seasons — every match predicted by a model fitted only on earlier
matches, so nothing saw its own result:

| | bets | named team won | 95% CI | drawn |
|---|---|---|---|---|
| ✅ **green** | 187 | **80.2%** | 73.9–85.3% | 10.7% |
| ⚠️ **yellow** | 34 | **47.1%** | 31.5–63.3% | **29.4%** |
| either | 221 | 75.1% | | 13.6% |
| *every side the model rates > 45%* | 302 | *70.9%* | | *14.6%* |

**The bottom row is the comparison that matters.** Green beats simply backing
every side the model already likes, by 9.3 points, while skipping 27% of them
(Fisher exact **p = 0.025**; the two groups overlap, so this understates rather
than overstates the gap).

### Yellow is a caution, not a weaker green

This is the finding worth acting on. Yellow selections won **47.1%** — *below*
the **54.0%** the model itself assigned them — and **29.4%** of those matches
were drawn against **14.6%** for over-45% sides generally. **The draw sitting in
second place roughly doubles the draw risk.** Green versus yellow separates at
Fisher exact **p = 0.00014**, so the colour distinction is real and not a
presentational flourish.

The app states this on the yellow banner rather than letting the colour imply
"nearly a green". At n = 34 yellow is still a small sample; treat it as a flag to
look closer, not a bet.

## What data it uses

From every **completed** match: the final score, **xG for each side**, **shots on
target**, the date, and the two team names. Bookmaker odds are read *only* to
benchmark the model in the backtest — they never feed the fit.

Two data traps are handled explicitly, because both would quietly corrupt ratings:

- **`-1` sentinels.** Shot and foul columns use `-1` for "not recorded". Averaged
  in as a real value it drags a team's attacking rating down. Treated as missing.
- **Both-zero xG.** Three matches in the sample carry `0.00` xG for *both* sides —
  one of them a 6–1. That is missing data, not a real goalless-chance game, so the
  xG is dropped rather than believed.

Unplayed fixtures, abandonments and suspensions are excluded via `status`.

## Method

A **Dixon–Coles** team-strength model. Each team gets an attack and a defence
parameter, and a fixture's expected goals are

```
λ_home = exp(μ + attack_home + defence_away + γ)
λ_away = exp(μ + attack_away + defence_home)
```

with `γ` the home advantage **fitted from the data** (0.20 goals on the Latvian
sample), not assumed. Because attack is always measured against the specific
opponent's defence, a team gets no credit for feasting on the league's worst side.

Three refinements, each settled by backtest rather than taste:

1. **The response is a blend** — 50% goals, 25% xG, 25% shots on target converted at
   the league's own goals-per-SoT rate. Goals are what happened; xG and SoT are
   less noisy measures of how a team actually played. The blend beat pure goals on
   *all three* markets.
2. **Light time decay** — `exp(-ξ × days ago)` with ξ = 0.0005. Tuning strongly
   preferred a small ξ: in a ten-team league, aggressive recency-chasing throws
   away more signal than it gains. (ξ = 0.005 was clearly worse: 0.9226 vs 0.8869.)
3. **Low-score correction** — independent Poissons misprice 0–0, 1–0, 0–1 and 1–1,
   which is exactly where football lives. The Dixon–Coles `τ` term is fitted on the
   real scorelines.

Newly promoted teams are pulled toward the league average by an L2 shrinkage term,
so a two-game hot streak doesn't make a team a title favourite.

### From expected goals to markets

The two λ values build a joint **score matrix**, and every market is a sum over its
cells: 1X2 from the win/draw/loss regions, BTTS from cells with both scores ≥ 1,
Over/Under 2.5 from cells totalling ≥ 3 or ≤ 2. Because all three come from one
distribution they are **mutually consistent by construction**.

## Measured accuracy

`test_model.py` and the app's Backtest tab both run a **walk-forward** validation:
the model is re-fitted before every matchday and predicts only that day's fixtures,
so no match ever contributes to its own prediction.

On the bundled Latvian data, **429 out-of-sample matches**:

| | 1X2 log-loss | 1X2 RPS | Accuracy |
|---|---|---|---|
| **This model** | **0.8667** | **0.1709** | **61.5%** |
| Bookmaker closing odds (de-vigged) | 0.8436 | 0.1638 | 62.5% |
| League base rates | 1.0601 | 0.2368 | — |

BTTS log-loss **0.6864**, Over/Under 2.5 log-loss **0.6764** (a coin flip is 0.6931).

### Honest limitations

- **The market is sharper than this model on 1X2.** Blending the two did not beat
  the market alone in testing, so the odds input is there for *comparison*, not as
  a free upgrade. Treat a large disagreement as a prompt to look closer.
- **BTTS carries the least signal** — 0.6864 against a 0.6931 coin flip is a thin
  edge. Over/Under 2.5 is the stronger of the two goals markets.
- **A calibration layer was tried and rejected.** Platt scaling improved in-sample
  but failed out-of-sample (Over 2.5 went from 0.6488 to 0.6780, and the fitted
  slope swung from 0.44 to 1.05 between halves), so it is not shipped.
- **Season is derived from the match date's year.** That is right for calendar-year
  leagues like Latvia's; a league running August–May will show two "seasons" per
  campaign. It affects display only, never the model, which works purely off dates.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Regression test

```bash
python test_model.py
```

Expect `PASS 86  FAIL 0`. The suite checks data integrity against hand-counted
values, mathematical invariants that must hold for any input (probabilities summing
to one, home advantage having the right sign, a stronger defence reducing expected
goals), golden values pinning the current fit, and that the backtest still beats
league base rates and a coin flip.

## Deploy (Streamlit Community Cloud)

New app → repo `sbachagian1101/suraj-betting-apps` → branch `soccer` → main file
`app.py` → Advanced settings → Python **3.13**. Pushing to the branch auto-redeploys.

---

*Predictions are probabilistic decision support, not a guaranteed outcome. Gamble
responsibly — Gambling Help 1800 858 858, [gamblinghelponline.org.au](https://www.gamblinghelponline.org.au).*
