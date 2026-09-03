# Racing EV Lab

A Streamlit research application for parsing **Racing & Sports Enhanced Form** text for:

- thoroughbred horse racing;
- harness racing; and
- greyhound racing.

The app turns copied race pages into structured race, runner and historical-start data; estimates each active runner's win probability; converts those probabilities to fair odds; removes the bookmaker margin from the entered market; and calculates expected value (EV).

> **Status:** working research prototype. The bundled model is a transparent, shrinkage-controlled form baseline. Its probabilities are not claimed to be calibrated or profitable until a sufficiently large labelled history is trained and evaluated out of time.

## Streamlit deployment

Use these settings in Streamlit Community Cloud:

| Setting | Value |
|---|---|
| Repository | `sbachagian1101/suraj-betting-apps` |
| Branch | `racing-ev-lab` |
| Main file path | `app.py` |

For local use:

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
streamlit run app.py
```

## Input workflow

1. Open a complete Racing & Sports **Enhanced Form** race page.
2. Use **Select All → Copy**, then paste the complete text into the app, or save it as `.md`/`.txt` and upload it.
3. Let the app auto-detect `thoroughbred`, `harness` or `greyhound`.
4. Inspect the Parsed Data tab. Correct any missing or misread current odds in the Prediction & EV tab.
5. Keep every entered price from the **same bookmaker/exchange and the same timestamp**.

The application does not scrape websites. It parses text supplied by the user.

## What is parsed

### Race context

- discipline, country, meeting date/time and track;
- race number, name, type/class and age/sex conditions;
- distance, surface, going and track direction where supplied;
- prize and active field size.

### Current runner snapshot

Common fields:

- runner number/name and scratching/reserve status;
- trainer and jockey/driver, with available strike rates;
- career, last-12-month, course, distance, course-and-distance, going and preparation-stage records;
- days since last run, current weight and market price.

Discipline-specific fields:

- **Thoroughbred:** barrier, carried weight, official rating, jockey/trainer ratings, gear and trials;
- **Harness:** front/second-row draw, standing-start handicap metres, driver/trainer, mile-rate data, gait/disqualification indicators and gear;
- **Greyhound:** box, body weight, box-by-box history, track/distance best time, first split, runner time and sectional data.

### Historical starts

Where present, the parser records:

- date, track/country, distance, surface/going and class;
- finishing position, field size and beaten margin;
- race time, sectional time, runner time, best-of-meeting time and adjusted time measures;
- barrier/box/handicap, carried/body weight, jockey/driver and trainer;
- starting price, gear, stewards comments, in-running position, tempo and sectional strings;
- disqualifications and trials separately from official starts.

## Canonical datasets

The **Data builder** tab creates three downloadable tables:

| Table | Grain | Purpose |
|---|---|---|
| `races.csv` | one row per current race | race context and source tracking |
| `runner_features.csv` | one row per current runner | pre-race feature store and future labels |
| `historical_starts.csv` | one row per prior start | longitudinal form database |

A production archive should also maintain:

```text
odds_snapshots.csv
race_id, runner_id, bookmaker, market_type, captured_at_utc, decimal_odds, exchange_commission

results.csv
race_id, runner_id, official_finish, won, placed, scratched, result_status
```

Do not overwrite odds as the market moves. Append timestamped snapshots so backtests can use the exact price that was actually available at the intended betting time.

## Probability model

### Built-in baseline

For each discipline, the app derives form components from information available before the current race:

- recent finishing strength;
- recent win/place signals;
- beaten-margin strength;
- speed/time/sectional strength;
- official or supplied ratings;
- course, distance and going suitability;
- trainer plus jockey/driver strength;
- fitness/rest pattern;
- barrier, box or handicap setup;
- reliability/disqualification history; and
- trials when available.

Each component is robustly standardised **within the current field**, combined using discipline-specific weights and converted to a mutually exclusive race probability with softmax:

```text
p_i = exp(score_i / T) / sum_j exp(score_j / T)
```

Sparse or debut profiles are shrunk toward the equal-chance field prior so that missing data does not create false precision. Current market odds are excluded from the independent form score.

### Trained model

The Model lab can fit a discipline-specific ensemble from labelled pre-race feature rows:

- regularised logistic regression;
- histogram gradient boosting;
- race-level normalisation; and
- temperature calibration on a chronological validation period.

Required columns:

```text
race_id, race_date, discipline, won
```

plus the engineered feature columns exported by the app. Exactly one runner per completed race should normally have `won=1`. Train **one discipline at a time**. The app requires at least 80 labelled races to run; 300+ is only a practical starting point, while several thousand diverse races are preferable for stable subgroup evaluation.

Offline training is also available:

```bash
python scripts/train_model.py labelled_runner_features.csv racing_ev_model.joblib
```

## Leakage controls

A feature is legal only when it was knowable at the chosen prediction timestamp. In particular:

- do not train the independent model on current odds, market rank or Racing & Sports neural price/rank;
- do not use official results, post-race ratings, future winners from the same form cycle, or comments added after the target race;
- keep every runner from a race in the same train/validation/test fold;
- split chronologically, never randomly across runner rows;
- calculate trainer, jockey, driver, track and box statistics using records available **before** the target race;
- fit calibration only on races not used to fit the base estimators; and
- reserve a still-later untouched test period for the final claim.

## Market probabilities and EV

For decimal odds `O_i`, raw implied probability is:

```text
q_i = 1 / O_i
```

Because the raw probabilities usually sum to more than 100%, the app removes the margin using either a power-method or proportional de-vig. It then reports:

```text
Model fair odds     = 1 / p_i
Probability edge    = p_i - p_market_i
EV per unit staked  = p_i * O_net_i - 1
EV percent          = 100 * EV
```

For an exchange commission `c` applied to winnings:

```text
O_net = 1 + (O - 1) * (1 - c)
```

Example: a calibrated model probability of 25% at net odds 5.00 gives:

```text
EV = 0.25 * 5.00 - 1 = +0.25, or +25%
```

A positive estimate is not proof of value. It can be caused by model bias, missing data, a stale/mixed price snapshot or chance.

## Evaluation standard

Do not select a model by strike rate alone. On an untouched chronological test set, report at least:

- multiclass race log loss;
- Brier score and calibration/reliability plots;
- top-pick win rate and winner coverage in top 2/top 3;
- results by discipline, country, track, distance, class, field size and odds band;
- return on turnover at the exact recorded price after commission/deductions;
- number of bets, average edge, maximum drawdown and confidence intervals; and
- comparison against equal chance, market favourite and no-vig market probability benchmarks.

A credible EV policy should be frozen before the final test, for example minimum data quality, minimum probability edge, minimum EV and maximum price. Do not tune the same test period repeatedly.

## Project structure

```text
app.py                       Streamlit interface
racing_ev/parser.py          Three-discipline tolerant text parser
racing_ev/features.py        Market-independent feature engineering
racing_ev/model.py           Baseline and trained-model probability blend
racing_ev/odds.py            De-vig, commission, fair price and EV
racing_ev/training.py        Chronological training and calibration
scripts/train_model.py       Offline training command
tests/test_core.py           Parser/probability/EV regression tests
```

## Tests

```bash
pip install -r requirements-dev.txt
pytest -q
```

The parser should be extended through regression fixtures whenever Racing & Sports changes a layout. Never “fix” a page by hard-coding one runner count or one positional table layout.

## Responsible use

Racing outcomes remain uncertain. Estimated probabilities and EV can be materially wrong, and positive-EV selections can lose repeatedly. This project is for analysis and model validation, not a promise of returns or a staking instruction.
