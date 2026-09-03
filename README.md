# FootyStats xG

A goals model for any league you have FootyStats match exports for. Time-weighted
Poisson with attack/defence strengths per team, a home advantage, and the
Dixon-Coles low-score correction. Fits to match xG by default (goals is a toggle;
matches without recorded xG use goals either way). One model per league, chosen in
the sidebar.

## Run

```bash
pip install -r requirements.txt
streamlit run app.py
```

Tests: `python -m pytest -q`.

## Data

`data/` holds FootyStats exports. Bundled:

- Bulgaria Second League, 2025-26 and 2026-27
- Kazakhstan First Division, 2025 and 2026

To add a league or update a season, export the *matches* CSV from FootyStats and
drop it in the sidebar uploader (or copy it into `data/`), keeping FootyStats'
file name `<league>-matches-<y1>-to-<y2>-stats.csv`; league and season are read
from the name. A newer file for the same season replaces overlapping rows. Teams
and players files are accepted and ignored.

The player CSVs in the original Bulgarian zip were empty.

## What the backtest says

Walk-forward, refitting before every match day, xG fit, 60-day half-life:

| league (matches with odds) | bookmaker (fair) | model | 25% blend | base rates |
|---|---:|---:|---:|---:|
| Bulgaria (239) — log loss | 0.995 | 1.019 | 0.996 | 1.068 |
| Kazakhstan (210) — log loss | 0.824 | 0.912 | 0.832 | 0.997 |

The bookmaker wins in both leagues, by a wide margin in Kazakhstan, and blending
the model into the market does not beat the market alone. The app is a calibrated
second opinion and a goal-line pricer, not an edge. The Backtest tab reproduces
these numbers live for whichever league and settings you pick.

## Deploy (Streamlit Cloud)

Main file `bulgaria-xg/app.py`, Python 3.13 (`runtime.txt`). No secrets needed.
Uploads on Streamlit Cloud do not persist across restarts; commit new CSVs to
`data/` instead.
