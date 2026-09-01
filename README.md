# RaceForm

Paste a **Racing & Sports _Full Fields_ page** — **greyhound, thoroughbred or
harness** — and get a rated field. The code is detected from the page itself.

This is a separate app from the `greyhound`, `horse` and `harness` branches,
which read the *Enhanced Form* page. Full Fields is the page that carries each
runner's **last-10 run table** (`FP / Marg / Date / Trk / Race / $R.PM / Dist /
SOT / … / SP / Winner`), and that table is what this rating is built from.

| | |
|---|---|
| Branch | `dogform` |
| Entry point | `app.py` |
| Python | 3.13 (override the Streamlit Cloud default) |

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Files

| File | What it is |
|---|---|
| `rs_parser.py` | Code-aware parser for the Full Fields page |
| `rating.py` | The rating: weighted margins → softmax → market blend |
| `app.py` | Streamlit UI |
| `backtest.py` | Scores the rating against races whose results we know |
| `test_app.py` | 73 regression tests over five real page captures |
| `fixtures/` | Bulli, Q Lakeside, Horsham (greyhound), Deauville (thoroughbred), Cabourg (harness) |

## The model

Each runner's last-10 runs become a **weighted average beaten margin** in lengths
at today's distance, weighted by recency (`exp(−days/τ)`), distance relevance
(`exp(−((d−target)/σ)²)`), **track shape** (straight tracks have no first turn,
so that form does not transfer to or from a circle track) and **surface** — a
separate axis, so straight turf form is discounted on both counts.

That margin plus `class`, `conversion`, `distance`, `course`, `surface`, `going`
and `layoff` gives a rating in lengths. A softmax turns ratings into
race-conditional win probabilities, blended in log space with the market after a
power de-vig. Every record term is shrunk toward the runner's **own career
strike rate** with a prior worth 15 starts — an earlier version used a prior
worth 7, which let a *two-start* distance record move a dog 1.9 lengths, and the
two runners it penalised hardest in a live race finished first and second.

### What differs per code

| | Greyhound | Thoroughbred | Harness |
|---|---|---|---|
| Margins | lengths | lengths | **metres**, ÷2.5 |
| Extra terms | early speed, box | **weight**, barrier | **reliability (DQG)** |
| Past-run adjustment | — | weight carried vs today's mark | distance handicap credited |
| Default spread | 2.4L | 4.0L | 9.0L |
| Field-size credit | 0.45 L/runner off 4.5 | 0.15 off 12 | 0.15 off 13 |

**Thoroughbred.** A handicapper's own margins are not comparable across its runs
until they are on the same weight. A run under 56kg when today's mark is 61.5kg
flatters it by ~3.8 lengths at the default scale.

**Harness.** A `DQG` — broke gait, disqualified — is **not a result**, and R&S
print a sentinel margin of `99m` against it. Those runs are dropped from the
margin average and the *rate* is scored separately, because on a French trot page
it is the most predictive thing there. At Cabourg, 48 of 134 past runs in the
field were non-finishes. Runs off a `+25m` distance handicap get credit for the
extra ground.

**The field-size credit is code-scaled and that matters.** The greyhound
coefficient (0.45 lengths per runner) applied to a 20-runner thoroughbred field
subtracts seven lengths and turns a 4th-of-20 into an apparent *winner*. There
is a test pinning it.

## What it gets wrong

**There is no opposition-strength adjustment.** Margins are scaled for distance
and field size but not for the quality of the field beaten, so a runner beaten 1L
in a weak race outrates one beaten 5L in a strong one. Over the greyhound races
this was built on, the runner with the best weighted margin finished 3rd and 4th.
`form` is the weakest column and everything else sits on it. The real fix is
fitting abilities jointly across many races — not another hand-tuned term.

**The scored record is three greyhound races.** `python backtest.py`:

```
RACE                      PICK               WON                   blend   model  market    unif
bulli r6                  LIZZIE LONG LEGS   LIZZIE LONG LEGS     1.3363  1.3886  1.4053  2.0794
qlakeside r7              DAWN SURE CAN      WHO'S IDEA           1.5980  1.5059  1.7120  1.6094
horsham r8                PAW PALMER         PAW PALMER           1.1133  1.3303  1.0198  1.7918

mean log loss                                                     1.3492  1.4083  1.3790  1.8269
picks: 2 from 3
```

Two from three, blend 0.03 nats/race ahead of the market. **Neither number means
anything at n=3** — at Horsham the market beat the model outright. **There is no
scored result at all for thoroughbred or harness yet**, so read those ratings as
a structured summary of the form, not as a proven edge. The backtest exists so a
change to the parser or the rating shows up as a number instead of a vibe.

## Parser notes

R&S ship several column sets for the same page and freely leave cells empty, so:

- every table is read through **its own header row** — the Cabourg page ships
  two different run-table column sets *on the same page*, some runners carrying
  a `Draw` column and some not;
- **empty cells are kept** — a blank `Sec.Time`, `Draw`, or a blank finishing
  position on a non-finish must not shift the columns. (Stripping block lines
  broke exactly this; there is a test for it now.)
- panel labels are **glued to their values** (`Career17: 1 1 0`,
  `W% - P%6% - 12%`), so known labels are peeled in order;
- the runner's **name precedes** the tag block and the **trainer comes last** —
  a reader that takes the last capitalised line reports the trainer (greyhound),
  the **weight** (thoroughbred) or the **driver** (harness) as the runner;
- R&S emit impossible finishing lines such as `6 of 4`; those field sizes are
  discarded and imputed, and Diagnostics lists every one;
- the meeting's track code is inferred by preferring a code whose letters appear
  in order in the track's name (`DEA`/DEAUVILLE, `CBRG`/CABOURG) — frequency
  alone picks Chantilly at Deauville, because thoroughbreds travel and
  greyhounds do not.

## Responsible gambling

For free and confidential support call **1800 858 858** or visit
[gamblinghelponline.org.au](https://www.gamblinghelponline.org.au).
