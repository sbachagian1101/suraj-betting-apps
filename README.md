# DogForm

Paste a **Racing & Sports greyhound _Full Fields_ page**, get a rated field.

This is a separate app from the `greyhound` branch (GreyhoundPredictor), which
reads the *Enhanced Form* page. Full Fields is the page that carries each
runner's **last-10 run table** — `FP / Marg / Date / Trk / Race / $R.PM / Dist /
SOT / Box / SP / Sec.Time / Winner` — and that table is what this rating is
built from. The two apps consume different pages and use different models.

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
| `rs_parser.py` | Tolerant parser for the Full Fields page |
| `rating.py` | The rating: weighted margins → softmax → market blend |
| `app.py` | Streamlit UI |
| `backtest.py` | Scores the rating against the races whose results we know |
| `test_app.py` | 37 regression tests over three real page captures |
| `fixtures/` | Three real pastes: Bulli, Q Lakeside, Horsham (1 Sep 2026) |

## The model

Each runner's last-10 runs become a **weighted average beaten margin** in
lengths at today's distance. Each past run is weighted by recency
(`exp(−days/τ)`), distance relevance (`exp(−((d−target)/σ)²)`), **track shape**
(straight tracks have no first turn, so that form does not transfer to or from a
circle track) and **surface** (turf form is discounted in an all-weather race).
Track shape and surface are separate axes — straight turf form is discounted on
both counts.

That margin plus `class`, `conversion`, `distance`, `course`, `surface`,
`early speed`, `layoff` and `box` terms gives a rating in lengths. A softmax
turns ratings into race-conditional win probabilities, which are blended in log
space with the market after a power de-vig.

Every record term is shrunk toward the runner's **own career strike rate** with a
prior worth 15 starts. That number matters: an earlier version used a prior
worth 7 starts, which let a *two-start* distance record move a dog 1.9 lengths,
and the two runners it penalised hardest in a live race finished first and
second.

## What it gets wrong

**There is no opposition-strength adjustment.** Margins are scaled for distance
and field size but not for the quality of the field beaten, so a dog beaten 1L in
a weak race outrates one beaten 5L in a strong one. Over the three races this was
built on, the runner with the best weighted margin finished 3rd and 4th. `form`
is the weakest column in the model and everything else sits on top of it. The
real fix is fitting abilities jointly across many races — not another term.

**The record is three races.** `python backtest.py`:

```
RACE                      PICK               WON                   blend   model  market    unif
bulli r6                  LIZZIE LONG LEGS   LIZZIE LONG LEGS     1.3330  1.3841  1.4053  2.0794
qlakeside r7              DAWN SURE CAN      WHO'S IDEA           1.5980  1.5059  1.7120  1.6094
horsham r8                PAW PALMER         PAW PALMER           1.1133  1.3303  1.0198  1.7918

mean log loss                                                     1.3481  1.4068  1.3790  1.8269
picks: 2 from 3
```

Two from three, and the blend is 0.03 nats/race ahead of the market. **Neither
number means anything at n=3.** At Horsham the market beat the model outright.
The backtest exists so a change to the parser or the rating shows up as a number
instead of a vibe — not as evidence of an edge.

## Parser notes

R&S ship several column sets for the same page and freely leave cells empty, so:

- the run table is read through its **header row**, never by column position;
- **empty cells are kept** when splitting — a blank `Sec.Time` must not shift
  the columns;
- panel labels are **glued to their values** (`Career17: 1 1 0`,
  `W% - P%6% - 12%`), so known labels are peeled in order;
- the dog's **name precedes** the tag block and the **trainer follows the box** —
  a reader that takes the last capitalised line reports the trainer as the
  runner (box 1 at Bulli is EXPLORE, *trained by* Frank Micallef);
- R&S emit impossible finishing lines such as `6 of 4`; the field size is
  discarded and imputed, and the Diagnostics tab lists every one;
- a gap in the box sequence raises a warning rather than silently shrinking the
  field.

## Responsible gambling

For free and confidential support call **1800 858 858** or visit
[gamblinghelponline.org.au](https://www.gamblinghelponline.org.au).
