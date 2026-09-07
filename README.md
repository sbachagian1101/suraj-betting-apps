# Soccerway Predictor

Paste two Soccerway team links, get a prediction for their next match built from
each side's last N games (form, xG, xGA, player ratings) and the published line-up.

```bash
pip install -r requirements.txt
streamlit run app.py
```

## What it fetches

Soccerway runs on the Flashscore engine. The app uses the same public endpoints the
site's own pages call. No login, no browser automation, no API key.

| Data | Where it comes from |
|---|---|
| Results and fixtures (ids, dates, scores, home/away, competition) | embedded feed in the team page HTML |
| xG, xGA, shots, possession per match | `global.flashscore.ninja/2035/x/feed/df_st_1_<match id>` |
| Formation, starting XI, per-player ratings, team average rating | `2035.ds.lsapp.eu/pq_graphql?_hash=dlie2&eventId=<match id>` |
| bet365 pre-match 1X2 odds (current and opening) | `global.ds.lsapp.eu/odds/pq_graphql?_hash=ope2` |

The team id is the 8-character code in the link (`.../team/groningen/MBUGcjb9/`).

Soccerway and Flashscore share team ids, so a Flashscore team link
(`https://www.flashscore.com/team/groningen/MBUGcjb9/`) works too.

## Two pages

* **Two teams** (`views/two_teams.py`): paste two team links, get the prediction for their
  next meeting with the full breakdown.
* **League day** (`views/league_day.py`), two tabs:
  * **Pick a match**: press *Fetch Today Matches*, then choose country -> league -> match
    (Home v Away) from dropdowns filled from the day's feed, and press *Predict this match*.
    The *Keep updating* toggle re-reads the score, odds and line-ups every minute, so the
    panel stays current up to and through kick-off.
  * **Whole day**: paste league names one per line in Flashscore's wording
    (`ENGLAND: Premier League`) and the app predicts every match of that day in those
    leagues: a summary table (CSV download) plus one expander per match with the same
    breakdown as the two-team page. Budget about 10 seconds per match.

  * **Live board**: pick leagues (every scheduled match of the day in them goes on the
    board) and a background worker inside the server keeps the predictions fresh on a
    tiered schedule: every 60 min beyond 3 h from kick-off, 30 min within 3 h, 15 min
    within 1 h, 5 min within 30 min. Started matches are frozen (score and status keep
    updating every 5 min). The table has start time, starts-in, country, league, match,
    Home/Draw/Away %, the three model-minus-book gaps and the selection; it redraws every
    30 s. Click a row for charts (model v book bars, 1X2 pie, scoreline heatmap, recent xG
    history, XI ratings) and the insights. The worker lives in `board.py`, charts in
    `charts.py` (plotly), the tab itself in `board_ui.py`. The **Only matches with odds**
    toggle (on by default) hides unpriced matches; the filter is applied when the table
    redraws, so a match appears by itself once the book prices it on a later refresh.
    **Kick-off alerts** (`alerts.py`): when a listed match comes within 5 minutes of
    kick-off the bell beeps once (a short WAV via `st.audio(autoplay=True)`), a bell banner
    names the match, and its row blinks pale purple (the table redraws once a second and
    alternates the row colour) until it starts; started and finished rows are rose. The
    odds cells keep their own colours on top of the row colour: green under 2.0, yellow
    under 3.0, pale blue at 3.0 and above.

  Only matches played *before* each fixture count as its history, so finished matches are
  predicted pre-match and shown against the actual score. Both teams are fetched in
  parallel and team data is cached for ten minutes, so re-runs are fast.

## Bet signal

Under the prediction row a black-on-lime banner says **BET HOME TEAM**, **BET DRAW** or
**BET AWAY TEAM** when, for the side with the biggest model-over-book gap, both hold:
the model puts that side **above 50%**, and the gap between the model and the book's implied
probability (margin removed) is **between 5.0 and 20.0 points**. Gaps above 20 points are
treated as the model being wrong rather than the book. Anything else shows **NO BET** with the
reasons. The whole-day table and the live board carry the same signal in their selection column.

## Desktop app (Windows)

The same code ships as a desktop app in `D:\01_PREDICTION MODELS\SoccerwayPredictor`:
`run.bat` builds a private `.venv` on first run and starts the app; `SoccerwayPredictor.vbs`
(and the desktop shortcut) open it in its own window with no console; `launcher.py` picks a
free port and reuses a running copy. It needs the internet while open, because every
prediction reads the Flashscore feeds live.

The daily match list comes from `global.flashscore.ninja/2035/x/feed/f_1_<day offset>_<utc offset>_en-us_1`,
which carries every match of the day with league headers and team ids.

Shared code: `pipeline.py` (fetching, caching, the analysis object) and `ui.py` (rendering).
`app.py` is the entry point with `st.navigation`.

## How the prediction works

1. Attack rate = 70% xG-for + 30% goals-for per game over the last N matches (default 12,
   up to 15), **recency-weighted**: match k back counts 0.85^k, so the newest match has
   weight 1 and the twelfth about 0.17. Defence rate the same from xGA and goals-against.
   Matches without xG use goals only.
2. Both rates are shrunk toward a league average (1.45 goals) with a prior weight of 2 games,
   against the effective sample size (sum of the weights).
3. Player ratings and the line-up comparison use a **shorter window** (default 5 matches,
   up to 10), because who is in form changes faster than team strength.
4. Expected goals: `home = att_home x def_away / avg x 1.12`, `away = att_away x def_home / avg x 0.90`.
5. Line-up adjustment: each of today's starters gets their mean Soccerway rating over the
   last N matches. The gap between the XI's mean and the team's mean rating scales the attack
   by `exp(0.35 x gap)` and the opponent's attack by `exp(-0.20 x gap)`. Starters with no
   recent rating are neutral. Regulars (started at least half the matches) who are not in
   today's XI are listed.
6. Independent Poisson grid gives 1X2, top scorelines, over 2.5 and both-teams-to-score.
7. If the fixture has odds, the model's probabilities are shown against the book's implied
   probabilities with the margin removed.

Line-ups appear on Soccerway roughly an hour before kick-off. Before that the app uses the
last match's XI and labels it "Probable XI from last match".

## Caveat

Three matches is a very small sample and one freak game moves the numbers a lot. In every
soccer model graded in this repo the closing market beat short-form models, so treat a large
model-vs-book gap as a prompt to look for what the book knows, not as an edge.

## Tests

```bash
python -m pytest -q test_app.py
```

Unit tests run offline on captured feed snippets. Two live tests hit Soccerway and are
skipped when there is no network.
