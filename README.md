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

## How the prediction works

1. Attack rate = 70% xG-for + 30% goals-for per game over the last N matches.
   Defence rate the same from xGA and goals-against. Matches without xG use goals only.
2. Both rates are shrunk toward a league average (1.45 goals) with a prior weight of 2 games,
   because N=3 is a tiny sample.
3. Expected goals: `home = att_home x def_away / avg x 1.12`, `away = att_away x def_home / avg x 0.90`.
4. Line-up adjustment: each of today's starters gets their mean Soccerway rating over the
   last N matches. The gap between the XI's mean and the team's mean rating scales the attack
   by `exp(0.35 x gap)` and the opponent's attack by `exp(-0.20 x gap)`. Starters with no
   recent rating are neutral. Regulars (started at least half the matches) who are not in
   today's XI are listed.
5. Independent Poisson grid gives 1X2, top scorelines, over 2.5 and both-teams-to-score.
6. If the fixture has odds, the model's probabilities are shown against the book's implied
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
