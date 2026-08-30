# FT Score Predictor

Paste the two FootyStats **team-comparison panels** for a fixture and get a
full-time score matrix — every scoreline with its probability — plus 1X2, BTTS,
over/under and a total-goals distribution.

The panels sit side by side on FootyStats, so one copy usually brings back
both: paste that into the left box and leave the right one empty. The first
panel is taken as the home side, and a **Swap** toggle fixes it if they come
back the other way round.

## Why panels rather than match pages

A panel carries each team's season aggregates **split by venue** — Scored,
Conceded, xG, xGA, BTTS, clean sheets, all across Overall / Home / Away. That
is half a season behind each venue figure, where five pasted match pages give
two or three. For a scoreline model, which lives or dies on the two scoring
rates, that is a much better trade.

## The rates

    λ_home = mean(home scored at home,  away conceded away)
    λ_away = mean(away scored away,     home conceded at home)

and the same pairing again on xG/xGA, the two blended.

**No home-advantage multiplier is applied.** The Home and Away columns already
contain it — a home scoring rate *is* a home scoring rate — so a further home
factor would count the same effect twice. That is the one specification error
this shape of input invites.

Venue figures are shrunk toward each team's Overall column: eight or nine
matches is enough to be worth using and not enough to take at face value. The
grid is Poisson with the **Dixon–Coles** low-score correction, optionally
negative-binomial.

## The parsing trap

The panel's numbers arrive **concatenated, with no separator at all**:

    AVG2.942.713.10   ->   2.94, 2.71, 3.10

As a string that is genuinely ambiguous. `1.542.231.2` could split
1.54/2.23/1.2 or 1.5/42.2/31.2, and FootyStats drops trailing zeros so the
field widths are not fixed. The disambiguator is arithmetic, not a guess:
**Overall is a weighted average of Home and Away, so it must lie between
them.** Every tokenisation is enumerated and that one constraint keeps the
right one — it resolved all 90 values across the five test fixtures, including
`xGA1.21.131.28` → 1.20 / 1.13 / 1.28.

A second trap: `xGA1.57…` also starts with `xG`. Matching the shorter label
first leaves `A1.57…`, which cannot be split, so **xGA silently vanished from
every panel** until labels were matched longest-first.

## Tested against five matches with known results

| match | expected | most likely | actual | rank of actual |
|---|---|---|---|---|
| AIK v Hammarby | 1.27–1.76 | 1–1 | **3–2** | outside top 10 |
| Hønefoss v Follo | 2.14–1.29 | 2–1 | **2–1** | **1st** |
| Bodø/Glimt v Rosenborg | 2.09–0.91 | 2–0 | **4–2** | outside top 10 |
| Stabæk II v Staal | 2.14–1.81 | 2–1 | **2–1** | **1st** |
| Dinamo Minsk II v Smorgon | 1.38–1.34 | 1–1 | **3–2** | outside top 10 |

- exact scoreline **2/5**
- 1X2 correct **4/5**
- BTTS correct **5/5**
- over/under 2.5 correct **5/5**
- scoreline log-loss **3.257** against **3.425** for a league-average Poisson

**Two exact scorelines from five looks extraordinary and mostly is not.** The
most likely scoreline in a football match carries roughly an 11% chance, so
hitting two or more from five happens about **10%** of the time by luck alone.

The model expected 3.23 goals a game against 4.40 observed. The standard error
of a five-match mean at that rate is about 0.80 goals, so that gap is roughly
**1.5 standard errors** — the sort of thing five matches produce routinely, not
evidence of bias.

And the sample itself is one-sided: the home side won all five, both teams
scored in all five, and all five went over 2.5. That is five matches someone
chose, not five drawn at random.

## What it cannot do

A score matrix is a distribution, not a prediction. Even a well-specified model
puts only about 11% on its most likely scoreline, so that single cell is wrong
roughly nine times in ten. Read the grid as a shape; the markets carry more.

There is no market price in this input, and the measured record is five
matches — enough to catch a badly broken model, nowhere near enough to
establish a good one. Nothing here is tuned on those five.

## Files

| | |
|---|---|
| `panel.py` | parses the team panels, including the concatenated numbers |
| `model.py` | rates, Dixon–Coles grid, markets |
| `backtest.py` | the five known results |
| `app.py` | the Streamlit interface |

## Running it

```bash
pip install -r requirements.txt
streamlit run app.py
```

```bash
python test_model.py && python test_app.py
```

453 checks — 419 on the parser and model, 34 driving the app. The app checks
paste a fixture, read the rendered grid, swap the sides, and run the whole
scoring path again with **matplotlib blocked**, because that is the difference
between a development machine and Streamlit Cloud.

---

*Probabilistic decision support, not a guaranteed outcome. Gamble responsibly —
Gambling Help 1800 858 858, [gamblinghelponline.org.au](https://www.gamblinghelponline.org.au).*
