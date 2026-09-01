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
| `test_app.py` | 89 regression tests over six real page captures |
| `fixtures/` | Bulli, Q Lakeside, Horsham (greyhound), Deauville (thoroughbred), Cabourg early + late market (harness) |

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
bulli r6                  BLAZING ACE        LIZZIE LONG LEGS     1.4182  1.5256  1.4053  2.0794
qlakeside r7              DAWN SURE CAN      WHO'S IDEA           1.6303  1.5963  1.7120  1.6094
horsham r8                PAW PALMER         PAW PALMER           1.0876  1.2542  1.0198  1.7918

mean log loss                                                     1.3787  1.4587  1.3790  1.8269
picks: 1 from 3
```

**This was 2 from 3 before 2026-09-01, and the regression was deliberate.** Four
specification defects were fixed that day (see below); together they cost the
Bulli pick and took the blend's edge over the market from +0.030 to +0.000
nats/race. The old Bulli pick depended on confidence being driven by recency,
which charged a spelled runner for staleness three times over. Keeping a known
double-count because it won two races out of three is exactly the mistake this
file exists to prevent — three races cannot adjudicate a specification. The
numbers are here so the trade is visible rather than quietly forgotten.

**There is still no scored result at all for thoroughbred or harness**, so read
those ratings as a structured summary of the form, not as a proven edge.

## The four fixes of 2026-09-01

Prompted by a live Cabourg page showing **"+88% EV" on a 101/1 shot**:

1. **The going was whitelisted.** The later page read `2750m SAND STANDARD`;
   `STANDARD` was not in the hardcoded list, so the whole distance line failed,
   `dist_m` was `None`, the target distance fell back to 400m, every past run
   fell outside the distance kernel, and **every runner ended with zero usable
   evidence** — without anything visibly breaking. The going is now whatever the
   last word is.
2. **An uninformative model was manufacturing value.** Shrinking thin evidence
   toward the *field mean* drives the model to uniform, and a uniform model is
   not a neutral input: the log-space blend then reduces to `p ∝ p_market^(1−w)`,
   which flattens the market and inflates every longshot (0.7% → 1.9%, hence the
   "+88%"). Thin evidence is now anchored on the **market**, so a runner the
   model knows nothing about simply gets the market's price and shows no edge.
3. **Confidence was driven by recency**, which charged staleness three times —
   in the recency weighting of the average, in the layoff term, and again as low
   confidence. Blazing Ace had ten usable runs and a confidence of 0.36 purely
   because they were old. Confidence now measures how much *relevant* form
   exists; recency stays where it belongs.
4. **Opposition strength**, the model's oldest named weakness, is now partly
   addressed via prizemoney as a class proxy — **but only in France**. Across
   the fixtures a single Australian greyhound grade (`GR 5`) spans \$1.4k–\$9.0k,
   a 6.4× spread, so there prizemoney tracks the venue rather than the class;
   switching it on took the backtest from 2/3 to 1/3 and put the blend behind
   the market. In France prizemoney *is* the ladder (CL1 ≈€88k > CL2 ≈€51k >
   CL3 ≈€29k, each band tight to 1.3×; trot D > E > F > G > H), so it is kept
   for thoroughbred and harness and set to zero for greyhound.

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
