# SoccerPredictorPro

Upload two seasons of FootyStats data, pick a fixture, and get **1X2**,
**half-time**, **HT/FT** and **Asian handicap** probabilities — each with a
confidence band — read off one shared score distribution.

## What it takes

Either FootyStats export shape; the app detects which you gave it.

| Shape | File name | What it adds |
|---|---|---|
| **Team** | `...-teams-...csv` | One row per team per season, already aggregated. Simplest route — last season's file is a completed table, this season's is a to-date snapshot. |
| **Match** | `...-matches-...csv` | One row per fixture. Adds a Dixon–Coles ρ *fitted* on your own scorelines, and time-decay weighting of older matches. |

Upload one shape at a time. Mixing them would count a season twice, so the app
uses whichever shape supplies more files and says which it ignored.

## How much data

**Two seasons: last season plus the current one.** Measured walk-forward on
1,147 matches:

| History | Log-loss | Accuracy |
|---|---|---|
| Last season only | 1.0468 | 43.7% |
| This season to date only | 1.0481 | 43.7% |
| **This season + last season** | **1.0307** | **46.7%** |
| This season + last 2 | 1.0284 | 45.7% |
| All seven seasons | 1.0334 | 43.9% |
| *Market odds* | *1.0044* | *48.4%* |

A third season is a dead heat; seven is worse than two. Two seasons also lifts
coverage from 75.6% of fixtures to 99.0%, because a promoted side becomes
priceable once it has played a few matches.

Team-file blending weights each season by matches played **and** decays older
seasons by 0.55 per season back. Without the decay a completed season three
years ago (30 matches) outranks the current one (4 matches so far) and ties last
season exactly — no recency preference at all.

## The markets

Everything comes from one full-time score matrix plus a half-time/second-half
pair, so the tabs can never contradict each other. The HT/FT joint is raked onto
the full-time and half-time marginals, and a test asserts they match exactly.

Halves are produced by **splitting** the full-time expectation by the league's
measured first-half share, not by rating each half independently. Rating
half-time directly from a ratio of two team rates compounds two extremes: on the
bundled data an Esteghlal–Persepolis fixture came out with half-time λ of
1.27/0.80 against a league half-time average of 0.41/0.38, implying a second
half quieter than the first. Second halves actually carry ~37% more goals, and
for some pairings the implied second half went negative.

### Why there is no Over/Under or BTTS

Measured walk-forward on 1,162 matches, **every goals market scored worse than
simply quoting the league base rate**:

| Market | Model | Base rate |
|---|---|---|
| Over 1.5 | 0.6998 | 0.6822 |
| Over 2.5 | 0.6346 | 0.6189 |
| Over 3.5 | 0.4405 | 0.4297 |
| BTTS | 0.6905 | 0.6679 |

The bookmaker cannot beat the base rate on them either — it ties on Over 2.5 and
is worse than the base rate on Over 3.5. Results depend on the *difference*
between two teams' strengths, which ratings capture; total goals depends on the
*sum*, which nothing in this data predicts. A test fails the build if a goals
market reappears.

## Honest accuracy

On the bundled league the full-time 1X2 model was right **46.5%** of the time
against a **35.3%** base rate — and the bookmaker's own odds were right
**48.4%**. This is a real edge over guessing and it is still **behind the closing
line**. Confidence bands describe the model's own certainty (how clear the
leading outcome is, and how many matches back the two ratings), not a promised
hit rate.

## Running it

```bash
streamlit run app.py
```

```bash
python -m pytest test_spp.py -q
```

## Files

| File | Role |
|---|---|
| `app.py` | UI — two tabs, Altair visuals |
| `spp_data.py` | Loads and detects both export shapes |
| `spp_model.py` | Ratings, score matrix, the four markets, confidence |
| `soccer_model.py` | Dixon–Coles MLE fit (vendored from the `soccer` branch) |
| `soccer_data.py` | Match-file cleaner (vendored from the `soccer` branch) |
| `sample_data/` | Three seasons of Iranian Persian Gulf Pro League, both shapes |

## Deploying

Streamlit Community Cloud, branch `soccerpro`, main file `app.py`. In **Advanced
settings** set the Python version to **3.13** — the numpy/pandas pins are
verified against it, and the form defaults to 3.14.
