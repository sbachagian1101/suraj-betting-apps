# HorsePredictorPro

`predictor` branch of **suraj-betting-apps**. An ensemble race model that
**beats the market on log-loss**, validated walk-forward on held-out periods.

Upload a Racing & Sports single-race sheet; get win and place probabilities from
a blend of a conditional logit, a gradient-boosted classifier, a LambdaRank
model and the de-vigged market price.

## Results

Split by **date**, never at random: train → validate (blend weights only) →
test. The test block is touched once.

### Accuracy — 465 races nothing had seen

| | log-loss | vs market | RPS | top-1 |
|---|---|---|---|---|
| market, Shin de-vigged | 1.8275 | — | 0.1857 | 38.1% |
| conditional logit (no market) | 1.9368 | +0.1093 | 0.2163 | 31.8% |
| gradient boosting (no market) | 2.0270 | +0.1994 | 0.2175 | 28.0% |
| LambdaRank (no market) | 2.0294 | +0.2018 | 0.2280 | 25.6% |
| blend, fundamentals only | 1.9361 | +0.1086 | 0.2157 | 31.4% |
| **blend + market** | **1.8111** | **−0.0164** | **0.1891** | 35.5% |

Edge **−0.0164** log-loss, 95% CI −0.0023 to −0.0307 (paired bootstrap over
races), P(edge > 0) = **0.989**, positive in **5 of 5** walk-forward folds.

**Read the top-1 column honestly.** The blend picks the winner *less* often than
the favourite (35.5% vs 38.1%). Its gain is in pricing the whole field, not in
naming the winner — which is why the betting strategy backs value across the
card rather than backing its own top pick.

### Money — backing every positive-edge runner, refitting each fold

| | |
|---|---|
| bets | 3,841 |
| strike rate | 11.4% (about one in nine) |
| average price | $21.60 (median $13) |
| **ROI** | **+22.6%**, 95% CI +5.8% to +40.5% |
| folds positive | **5/5** |

## Jockey-only mode

A second, deliberately blinkered model that sees **33 columns about the rider
and nothing about the horse**: earnings, starts, wins, places, strike rate and
ROI over the last 100 rides, 12 months, this season and last season, plus the
apprentice claim.

| | top-1 | top-3 | log-loss |
|---|---|---|---|
| a dart throw | 11.3% | — | — |
| **jockey only** | **22.4%** | 54.6% | 2.0888 |
| the market | 38.1% | 66.5% | 1.8275 |
| everything + market | 35.5% | 65.8% | 1.8111 |

The jockey columns **roughly double a random pick**, so they carry real signal.
They are also well behind the market and the full model — and when the jockey
model was blended with the market, the fitted weight on the market was
**100%**: the rider's record adds nothing on top of the price.

The strongest columns by a distance are the **ROI figures** — this season, 12
months, last season, last 100 — ahead of strike rates and average earnings. ROI
captures whether a jockey beats the prices they ride at, which is closer to
skill than a raw win rate.

Selectable in the sidebar. The app carries a banner in this mode stating all of
the above, because a top pick shown without that context reads far stronger
than it is.

## A CSV trap worth knowing about

The Racing & Sports **CSV** export ends every data row with a trailing comma, so
rows carry **129 fields against a 128-name header**. Pandas resolves that by
promoting the first column to the index, which shifts every column one to the
left:

| column | naive `read_csv` | correct |
|---|---|---|
| `Horse Name` | `[5, 4, 4]` (ages) | `['Manoora', 'Curie', 'Dunquin']` |
| `Best Fixed Odds` | `[60.0, 59.5, 59.0]` (weights) | `[2.9, 3.4, 8.0]` |

Nothing raises. The frame looks fine and every prediction from it is nonsense.
`data.read_race_file()` reads CSVs with `index_col=False` and then checks that
horse names are not numbers, refusing the file if they are. The `.xlsx` export
is unaffected. The regression test builds both a clean and a trailing-comma file
and asserts the naive read really does corrupt while the safe one recovers.

## The three things that would kill it

**Price slippage.** Same bets, settled worse:

| price taken | ROI | folds positive |
|---|---|---|
| exactly as quoted | +22.6% | 5/5 |
| 5% worse | +16.5% | 5/5 |
| 10% worse | +10.4% | 3/5 |
| 20% worse | −1.9% | 2/5 |

These bets average $22. If you cannot reliably take the quoted best price, you
do not have this edge.

**Concentration.** Removing each fold's five best-priced winners — 25 bets of
3,841 — turns three of the five folds negative.

**Variance.** One winner in nine. In validation the longest losing run was **48
bets** and the worst drawdown **190 units** at flat stakes. That is the normal
shape of the strategy, not a malfunction.

## The models

Four different mathematical objects, combined because they optimise genuinely
different criteria. An ensemble of near-identical models is just a slower single
model.

**Conditional logit** (Bolton & Chapman, 1986) — the likelihood that matches the
question: exactly one horse wins, so a runner's probability is a softmax over its
own field. Fitted by maximising the grouped log-likelihood directly (L-BFGS, L2
penalty). Linear, so it cannot overfit the way a tree can, and calibrated by
construction. It takes the largest fundamental weight, 0.22 of the final blend.

**Gradient-boosted classifier** — binary win/lose, renormalised within the race.
Non-linear; catches interactions the logit cannot. Weight 0.11.

**LambdaRank** — optimises the *order* inside a race rather than a per-runner
probability, so its errors are shaped differently. It takes weight 0.00 in the
current fit: it ranks well but is not calibrated, and the blend is chosen on
log-loss. It is kept because the weight is re-chosen on every refit.

**Market**, Shin (1993) de-vigged, weight 0.67. Solved by **bisection** — the
usual fixed-point iteration oscillates instead of converging on books with a big
favourite.

**Plackett–Luce** — not fitted. Given win probabilities it samples whole
finishing orders (Gumbel top-k) for place probabilities. Sampling an order
respects the fact that exactly one horse can be first; independent per-horse
draws do not, and will happily give two horses a 60% chance of winning.

Blended as a **weighted geometric mean in log space** — the form Benter (1994)
uses for combining a fundamental model with the market. Weights are chosen by
grid search on a validation period the models were never fitted on.

## Features

3,094 races, 29,926 runners, **217 fundamental features** from 110 populated
columns.

Every column becomes a **within-race z-score** *and* a **within-race rank**. A
handicap rating of 82 says nothing on its own and a great deal relative to
today's opponents. Ranks sit alongside z-scores because prize money and earnings
are heavily skewed and a rank is immune to that. Columns constant within a race
(distance, prize) are detected automatically and passed through raw rather than
normalised to a column of zeros.

The market is kept in a separate feature set that can be switched off entirely,
so the fundamental rating can be compared against a price it has never seen.

## Files

- `data.py` — loading, race ids, Shin de-vig, within-race features.
- `models.py` — conditional logit, boosted models, blending, Plackett–Luce, metrics.
- `train.py` — fits and saves `model_bundle.joblib`.
- `predict.py` — scores one race from the bundle (`feature_set='all'` or `'jockey'`).
- `app.py` — Streamlit UI.
- `test_model.py` — 172-check suite.
- `model_bundle.joblib` — the trained model (2.2 MB).

## Retraining

```bash
python train.py          # refits on the workbook and rewrites the bundle
```

The method is validated; the shipped model is then fitted on every race, because
holding data back from the final fit buys nothing once the method is settled.
The blend weights are the exception — they are always chosen on the most recent
15% of races, before the final refit, so they are never tuned on races the models
also learned from.

## Regression test

```bash
python test_model.py
```

Expect `PASS 172  FAIL 0`. It pins mathematical properties rather than numbers:
the de-vig removes the overround and sums to one, probabilities sum to one inside
each race, the conditional logit recovers weights from data generated with known
ones, place probabilities sum to the number of places paid, and a race with no
prices falls back to fundamentals rather than failing.

## What was not done

No result was used to choose the features, the model family or the blend method.
The blend weights are the only tuned quantity, and they are chosen out of sample.
**None of this has been tested on live racing** — every number above is a
backtest, and a backtest is the most optimistic estimate you will ever see of a
strategy.

## Deploy (Streamlit Community Cloud)

New app → repo `sbachagian1101/suraj-betting-apps` → branch `predictor` → main
file `app.py` → Advanced settings → Python **3.13**.

---

*Prediction is probabilistic decision support, not a guaranteed outcome. Gamble
responsibly — Gambling Help 1800 858 858,
[gamblinghelponline.org.au](https://www.gamblinghelponline.org.au).*
