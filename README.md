# OddsPredictor

Paste a betting screen. The app strips the bookmaker margin out of **both**
price pools, averages them, and prices every runner against what is actually
on offer — then tells you whether to back it to win, to place, or not at all,
at how many points and with what confidence.

Everything is on one page: paste box, **Parse**, **Predict**, and the results
underneath.

## What it actually does

Both price sources carry a margin. Across the two graded races the fixed-odds
book ran **122–127%** and the tote book **118%**. Summing `1/price` and
dividing through removes that margin and leaves probabilities that sum to 1.

The model is then deliberately almost nothing — **average the two de-vigged
pools.** That is not modesty for its own sake. It is what the grading showed:

| Method | Winner rank | Mean |
|---|---|---:|
| Fixed odds alone | 2, 7 | 4.5 |
| Tote alone | 1, 7 | 4.0 |
| **Blend 50/50** | **1, 6** | **3.5** |
| Random guess | | 6.8 |

## Signals that were tried and dropped

- **Steam money.** At Ipswich the biggest mover (Cantarito, 4.20 → 3.00) won.
  At Murray Bridge the biggest mover by a mile (Magic Island, 14.00 → 7.50,
  **+87%**) ran nowhere, and the winner had *drifted* 15%. One from two, and
  adding a steam bonus never changed a single ranking. It is charted as
  context and not scored.
- **Drift as an elimination.** Refuted twice. Sweet Pretender drifted 35% and
  ran second; Play Bouzouki and Win to Retire both drifted 15% and ran first
  and second.
- **Tote priority.** Won race one decisively, lost race two. Equal-weighted.

## Where an edge could actually exist

Not in out-predicting the market — **the probabilities are the market.** It is
in the disagreement. When the two pools differ, the fair estimate sits near
the average while one pool is offering a longer price than that average.
Backing the longer of the two is then +EV against your own estimate. Every
positive expected value this app reports comes from that gap and nowhere else.

## Honest record

**Two races graded.** The winner was ranked **1st** and **6th**. Both
recommended bets **lost**. That is the entire track record, and it is printed
in the sidebar rather than buried here.

The confidence score describes how *well-defined* the market is — whether the
pools agree, how much margin is in the book, the field size, and how clearly
one runner leads. It is **not** a hit rate.

## Two bugs worth knowing about

**A bare runner number matches the price pattern.** `12` on its own line looks
exactly like a price, so the parser was consuming the next runner's number as
a fifth price and silently dropping **every second runner** — 12 became 6. The
next-runner test now runs before the price test, and a check pins the full set
of saddlecloths.

**The header line is not a result.** It reads
`Stewards Comments Overcast Soft5 1400m 8,9,12,2`, and that trailing list of
valid saddlecloths looks exactly like a first four. Ipswich showed `1,11,6,4`
and the race was won by 6 from 11, 12, 2. It is a stewards-comment runner
list. The parser records it as a note and never as finishing order.

## Staking

One point = 1% of your bank. Stakes are **fractional Kelly** (quarter by
default) and capped, because probabilities read off the market itself mean a
large computed edge is far more likely to be a stale price than a real one.

Runners under **6%** (win) or **15%** (place) are never recommended. An early
build touted a 0.7% chance at $37.50 to place, because it estimated the place
probability from the fixed book and then compared it to the tote price. Both
books are now blended, and thin runners are excluded outright.

## Running it

```bash
streamlit run app.py
```

```bash
python test_app.py
```

## Files

| File | Role |
|---|---|
| `app.py` | Single-page UI — paste, parse, predict, charts, recommendation |
| `odds_parser.py` | Betting-screen text to runners and prices |
| `odds_model.py` | De-vig, blend, EV, Kelly, confidence, insights |
| `samples/` | The two graded races |

## Deploying

Streamlit Community Cloud, branch `oddspredictor`, main file `app.py`. In
**Advanced settings** set the Python version to **3.13** — the form defaults
to 3.14. No matplotlib: charts are Altair, which ships with Streamlit.
