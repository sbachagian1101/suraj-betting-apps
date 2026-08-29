# SoccerPredict

`soccer` branch of **suraj-betting-apps**. A Dixon–Coles team-strength model,
**tuned on your league**, **calibrated**, and **explicit that it sits behind the
bookmaker**.

Upload season CSVs, pick a fixture, get 1X2 / BTTS / Over-Under 2.5 — shown next
to the market, with a flag on the disagreement.

## The finding this app is built around

The intuitive idea is that where a model departs from the market it has found
something. **Measured, the opposite is true.**

Splitting 723 Dutch Eerste Divisie matches by how far the model and the market
are apart:

| disagreement | matches | model | market | model − market |
|---|---|---|---|---|
| smallest half | 362 | 1.0160 | 1.0072 | **+0.0088** |
| 50th–75th pct | 181 | 1.0030 | 0.9939 | +0.0091 |
| 75th–90th pct | 108 | 1.0537 | 1.0245 | +0.0292 |
| top 10% | 73 | 1.0192 | 0.9526 | **+0.0666** |

Positive means the model is **worse**. Its disadvantage *grows* with the
disagreement. And where the two name different favourites (13.7% of matches) the
market is right **45.5%** against the model's **32.3%**.

So a large gap is evidence that **the model** is wrong — usually because the
market has absorbed team news the model cannot see. The flag says *look closer*,
never *back this*. Tab 3 recomputes this from whatever data you load rather than
asking you to take it on faith.

## The recommendation

The app names a bet, states a confidence, and explains itself. All three are
grounded in measurement rather than invention.

**The selection** is the model's most likely outcome.

**The confidence** is a band whose label means something, because the model's
stated probability and the realised rate line up:

| model says | matches | actually won |
|---|---|---|
| 55%+ → **HIGH** | 156 | **63.5%** |
| 45–55% → **MEDIUM** | 285 | **48.8%** |
| under 45% → **LOW** | 281 | **41.6%** |

Confidence is downgraded one step whenever the market names a different
favourite, because on this league's history that is when the model is least
reliable.

**The reasons** are the actual drivers: the expected goals, where each side
ranks in attack and defence, the fitted home advantage, how concentrated the
scoreline distribution is, and whether the bookmaker agrees.

### And it tells you the recommendation is not a bet

Flat-staking each bucket over the same 723 matches:

| bucket | matches | strike | ROI | 95% CI |
|---|---|---|---|---|
| agree, HIGH confidence | 273 | 57.5% | **−10.8%** | −20% to −1% |
| agree, MEDIUM | 305 | 46.9% | −7.8% | −19% to +3% |
| agree, LOW | 46 | 52.2% | +11.6% | −20% to +43% |
| they disagree | 99 | 32.3% | −8.6% | −35% to +19% |
| *back the favourite every week* | 723 | 51.0% | *−6.3%* | |

Only the first row's interval clears zero, so it is the only bucket that
**definitely** lost money; the rest establish nothing either way, and the app
says exactly that rather than letting the one favourable-looking point estimate
stand. The book's margin here is **7.7%** — larger than any edge the model has.

So the recommendation names the most likely outcome and shows the expected
return at your price. A 57.5% strike rate at short odds still loses. That is
the honest shape of it.

## Where it stands

Walk-forward on three Dutch seasons — refitted before every matchday, so no
match contributes to its own prediction:

| | 1X2 log-loss |
|---|---|
| league base rates | 1.0557 |
| **this model** | **≈1.021** |
| the bookmaker | 1.0008 |

The model captures roughly two-thirds of the available signal. The gap to the
bookmaker is about **+0.02 with a 95% interval clear of zero** — real, not noise.
It is well calibrated (mean absolute error ≈ **0.010** across probability bands):
honest about its own uncertainty even while the market out-predicts it.

## Tuned per league, because the defaults were not yours

The shipped constants were tuned on a **ten-team** Latvian league. On a
**twenty-team** Dutch one they rank **23rd of 27** settings tried. Every league
now gets its own grid search on an inner time split, so the settings are chosen
on matches that come *after* the ones they were fitted on.

The gain is about **0.0025 log-loss** — real, and nowhere near enough to close
the gap to the market. The top few settings sit inside the noise, so the app
calls the winner *a reasonable setting*, not the optimum.

## Measured and rejected

Everything below was tested on this data and did not work:

| | result |
|---|---|
| a third season of the same league | **+0.0002** log-loss — nothing |
| pooling other leagues | home advantage runs 0.164–0.415 across nine leagues and the low-score correction **flips sign**; pooling would hurt |
| blending with the market | choosing the weight on a validation half put **zero** weight on the model |
| the feed's own pre-match PPG/xG columns | made it **worse** — 1.0455 against 1.0348 for the model alone |
| the goals markets | over/under 2.5 scored 0.6677 against a **0.6622 base rate** — worse than always predicting the league average |

> ⚠️ The previous version of this README benchmarked over/under against a **coin
> flip (0.6931)**. That is the wrong benchmark and it flattered the result. The
> base rate is the honest comparison, and against it the model loses.

## What would actually help

Not more rows. In order:

1. **Closing odds** rather than pre-match prices — far sharper, and the movement
   between the two is itself signal: it is the market absorbing team news.
2. **Lineups and injuries** — absent from this feed entirely, and probably the
   largest missing variable in football modelling.
3. For a promoted or relegated side, **data from the division it came from** —
   otherwise the model just shrinks it to league average.

## The model

Each team gets an attack and a defence parameter:

```
λ_home = exp(μ + attack_home + defence_away + γ)
λ_away = exp(μ + attack_away + defence_home)
```

`γ` (home advantage) is **fitted from your data**, not assumed. Attack is always
measured against the specific opponent's defence, so a team gets no credit for
feasting on the league's worst side. The Dixon–Coles `τ` term repairs the 0–0 /
1–0 / 0–1 / 1–1 cells that independent Poissons misprice — which is where
football lives. The response blends goals, xG and shots on target: goals are what
happened, the other two are less noisy measures of how a team played.

All markets come from one score matrix, so 1X2, BTTS and Over/Under are
**mutually consistent by construction**.

Two data traps are handled explicitly: `-1` sentinels in the shot columns are
treated as missing rather than averaged in as real values, and matches recording
`0.00` xG for *both* sides have their xG dropped — that is missing data, not a
goalless-chance game.

## Files

- `soccer_data.py` — CSV loading, cleaning, validation.
- `soccer_model.py` — Dixon–Coles fit, score matrix, markets, walk-forward.
- `assess.py` — per-league tuning, calibration, the disagreement flag.
- `app.py` — Streamlit UI.
- `test_model.py` — 151-check suite.
- `sample_data/` — Dutch Eerste Divisie 2024-25, 2025-26, 2026-27.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Regression test

```bash
python test_model.py
```

Expect `PASS 151  FAIL 0`. The suite is deliberately **data-agnostic** — an
earlier version hardcoded Latvian team names and golden log-loss values, so
swapping the bundled league broke it for reasons unrelated to the code. It now
asserts properties that hold for any league, including the claim the whole app
rests on: that the model's disadvantage against the market grows with
disagreement.

---

*Prediction is probabilistic decision support, not a guaranteed outcome. Gamble
responsibly — Gambling Help 1800 858 858,
[gamblinghelponline.org.au](https://www.gamblinghelponline.org.au).*
