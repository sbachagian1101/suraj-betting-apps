# RaceSim

Paste a Racing & Sports **Enhanced Form** page (and, optionally, the **Speed Map**
page) and RaceSim rates the field from the form, then animates how the model
expects the race to be run: the jump, the settle, mid-race, the turn and the
finish. The animation is at most 30 seconds long and downloads as a GIF.

Live app: deployed from the `racesim` branch of this repo on Streamlit Community Cloud.

## What you get

- **Race animation** - a top-down running of the race with saddlecloth colours,
  a live running-order panel with lengths, distance-to-go and a predicted race
  clock. 10-30 seconds, 8-15 fps, downloadable GIF. Four "key moment" stills.
- **Ratings & probabilities** - rating in lengths, win and place %, model price
  vs market price, and a stacked bar showing where every horse's rating comes from.
- **Race call & positions** - running order at each stage and a lengths-behind-
  the-leader chart through the race.
- **Scenario modes** - the model's expected running, or random race-day draws
  that add noise so you can see how differently the same field can run.

## How the model works

Everything is measured in **lengths at the finish of today's race**, so the
components add up transparently and the sum drives both the win probabilities
and the finishing margins that are animated.

| Component | Source on the page | Rule |
|---|---|---|
| Class | Official rating (OHR) beside recent runs | +0.3 L per point above the field mean |
| Weight | WT minus apprentice claim | -0.4 L per kg above the field mean (scaled by distance) |
| Form | Last 5 runs: class, prizemoney, margin, weight carried, recency | class points - 3 x margin, recency-weighted |
| Speed | Race times vs a distance par, adjusted for Heavy/Soft/Good | median of best 3, damped |
| Fitness | Days since last run, run of the prep, first-up record, age | peak 7-35 days, 2nd/3rd-up bonus, spell penalty |
| Barrier | Gate (Speed Map gate if supplied), distance, early speed | free to gate 4, then -0.12 L/gate, steeper past 10 |
| Jockey | JRat and last-50 strike rate | 0.25 L per rating point |
| Trainer | TRat and last-50 strike rate | 0.15 L per rating point |
| Combos | J/T and J/H records | capped +-0.5 L |
| Going | Record on today's going, Soft/Heavy transfer, turf vs synthetic | shrunk success rate vs career |
| Distance | Raced range, distance record, winning distances, C&D | -0.5 L per 100 m beyond the longest trip |
| Late speed | Speed Map AFS | 0.35 L per standard deviation |

The fundamental rating is centred and its spread is capped so the model cannot
be over-confident. **The market is blended in afterwards at a weight you set
(default 5 %, hard cap 10 %)** - odds never dominate the rating. Win and place
probabilities come from 20,000 simulated finishes with 3 lengths of race-day noise.

Early positions in the animation use the Speed Map's average early speed (AES)
plus where each horse usually settles; the finish uses the ratings; the turn
blends the two, with high-AFS closers held back longer. Gaps are exaggerated
3.5x on the track drawing for legibility; the side panel shows true lengths.

## Files

| File | Purpose |
|---|---|
| `app.py` | Streamlit UI |
| `horse_parser.py` | Enhanced Form parser (header-driven field table, glued panels, per-run OHR/times) and Speed Map parser |
| `race_model.py` | Lengths-based rating engine and Monte Carlo probabilities |
| `race_sim.py` | Checkpoint gaps and lanes -> smooth trajectories |
| `race_anim.py` | Pillow GIF / PNG renderer |
| `fixture_canberra_r5*.txt` | Sample race (Canberra R5, 4 Sep 2026) used by the tests and the "Load sample" button |
| `test_racesim.py` | Parser, model, simulation, renderer and app smoke tests |

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

```bash
python -m pytest -q
```

## Deploy

Streamlit Community Cloud: repository `sbachagian1101/suraj-betting-apps`,
branch `racesim`, main file `app.py`, Python 3.13.

Ratings and animations are model output for entertainment and analysis. Check
fields, scratchings and track conditions with an official source.
