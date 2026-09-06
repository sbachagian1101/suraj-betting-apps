# Mauritius Gallops

Scores a Champ de Mars race from the published training sheet
(`GALLOPS for Race N [...]`). Paste the text, get a ranking with win %,
and a breakdown of why each horse rates where it does.

## Running it

```
pip install -r requirements.txt
streamlit run app.py
```

`run.bat` does the same on Windows; `run.bat test` runs the tests.

## What the sheet actually contains

The columns are `200P 2400P 2200P 600M 400M 200M`. Only two of them are
real clocks: the **600 m** time and the **last 200 m** time. The 400 m
column is always the midpoint of those two, and the first two split
columns are always `600M - 400M`, so they carry no extra information.
Everything below is built from the 600 m clock, the 200 m clock, the
equipment column and the comment.

Equipment: `Lib` = free (no urging), `Leg Acc` = light acceleration,
`Acc` = accelerated, `Sol` = asked for effort, `(BT)` = barrier trial.
Comments: `encie X` = worked in company with X, `bat X` = beat X,
`bab X` = beaten by X.

## Features per runner

| Group | Feature | Meaning |
|---|---|---|
| Workload | `workload` | gallops per week across the prep |
| | `gallops_14d` | gallops in the last 14 days |
| Freshness | `days_since` | days from last gallop to race day |
| | `fresh_penalty` | 0 if the last gallop was 3-9 days out; grows outside that band |
| Conditions | `best_adj600` | fastest 600 m after adding an effort penalty (Lib 0, Leg Acc +0.35, Acc +0.5, Sol +0.8) |
| | `lib_share` | share of gallops done freely |
| | `finish_ratio` | last 200 m over the average 200 m of the first 400 m; below 1 = quickened |
| Consistency | `std600` | spread of the raw 600 m clocks |
| | `trend600` | seconds per week the 600 m clock is changing; negative = improving |
| Taper | `taper` | last timed 600 m minus the best one; positive = sharp work mid-prep, easy work in race week |
| Company | `beat_n`, `beaten_n` | times it beat / was beaten by a companion |
| Data | `missing_600`, `barrier_trial` | incomplete lines, trial present |

Barrier trials are kept for freshness and workload but excluded from the clocks.

## Scoring

Each feature is standardised within the race (z-score) and multiplied by
its weight; the sum is the score, and win % is a softmax of the score.
The `Why` expander on the Predict page shows each feature's contribution.

The default weights are **judgment weights, not fitted**. They were set
after looking at two races (6 September 2026, races 1 and 2), where the
winner in both had the heaviest workload of the field and an easier final
gallop than its sharpest one, while the fastest-clock horses that did
their best work inside three days of the race (Brand New World, Zacatoo)
did not win. That is a training-theory story, not evidence: with two
races the model gets both winners in-sample by construction and that
proves nothing.

## Making it a real model

Paste each meeting's sheets with an `Actual results: A - B - C` line under
each race and press **Save races with results**. The History page reports
winner log-loss against a no-skill baseline and, once at least 15 races
are stored, **Fit weights** runs a coordinate search over the weights and
saves them. Treat the report as honest only for races saved *after* the
weights were last fitted.

## Files

* `gallops.py` -- parser, features, scoring, evaluation, fitting, storage.
* `app.py` -- the Streamlit interface.
* `test_gallops.py` -- tests.
* `data/sample_06sep2026.txt` -- the two sample races.
