# AustraliaPdfHorseRacing

Ranks the runners in an Australian race from a **Racing & Sports meeting form
guide PDF** — the whole meeting, one file, no spreadsheet export and no odds.

## Why the PDF is worth parsing

The field table is the obvious part. The valuable part is the per-runner
history buried in the detail pages: about **400 past runs per meeting**, and
every one of them carries a finishing position out of a known field size, the
date, track, distance, going, margin, race time, sectional, jockey, weight,
barrier, gear change, the first three finishers, the running positions — and
**the starting price**.

That is a labelled training set. The upcoming races in these files have no
prices and no results; their history does.

## What it is

A ridge regression on **finishing percentile** — where a horse finished as a
fraction of its field, so a 3rd of 8 and a 4th of 11 are comparable — fitted on
6,168 past runs from 1,481 horses between September 2023 and August 2026.

The obvious choice, a rank-ordered conditional logit, was tried and lost. A
past race is visible here only through the runners entered again this weekend,
so a within-race ranking likelihood can use just the 60% of rows that land in a
group of two or more, while a percentile target uses all of them.

| on identical held-out groups | log-loss | top-1 |
|---|---|---|
| **ridge on finishing percentile** | **0.9998** | **53.3%** |
| rank-ordered conditional logit | 1.0164 | 50.5% |
| gradient boosting | 1.1016 | 47.4% |

## What it is worth

Walk-forward, out of sample, on 3,147 held-out races in fields averaging 10.3
runners:

| | top pick wins | top pick places |
|---|---|---|
| **this model** | **14.9%** | **39.0%** |
| the starting price | 16.0% | 40.6% |
| a random pick from the same horses | 12.5% | 34.9% |

It recovers about **70% of the market's edge over a random pick without ever
seeing a price**. It does not beat the market and is not built to — these files
carry no odds, so the job is to be useful when there is no price to consult.

Confidence comes from how far clear the top pick is, because that is the one
thing whose payoff can be measured here:

| gap to second | races | top pick won | 95% CI |
|---|---|---|---|
| under 0.62 | 616 | 15.4% | 12.8 – 18.5 |
| 0.62 – 1.11 | 309 | 18.4% | 14.5 – 23.1 |
| over 1.11 | 306 | **23.2%** | 18.8 – 28.2 |

## Traps in the data

**`Odds 0.3F` is decimal odds minus one**, with `F` marking favourite. Read
literally it turns a $1.30 winner into a 0.3 shot. Calibration caught it: with
the correction the actual win rate tracks the implied probability across every
price band, and without it the shortest bucket claimed a 97% implied
probability against a 43% actual strike rate.

**The tab number is sometimes on its own line and sometimes glued** to the form
figures. Matching only the glued form found 4 of 9 runners at Strathalbyn.

**The form figures are glued to the horse name.** The split is made on case;
matching the figures case-insensitively produced `x7036D` + `ance Dance Dance`.

**`date | track | distance` merges divided races.** 676 of 3,904 such groups
held two different field sizes and 90 held two winners. Before this was fixed,
field size — a race constant a conditional logit cannot even identify — carried
the largest coefficient in the model, because it was telling the two divisions
apart.

**The field tables are pre-acceptance.** They can list more runners than start,
and barriers repeat where a runner has been scratched.

## What was rejected

- **Gear changes.** Blinkers going on looked worth +0.069 of a finishing
  percentile at p=0.001. But horses about to get blinkers had a *worse*
  previous run (0.557 against 0.465), and controlling for it leaves +0.017 at
  p=0.48 — regression to the mean. Gear is displayed, not modelled.
- **Per-fold temperature calibration.** Worse (1.0245 against 1.0077); the
  fitted scale swung between 0.19 and 1.31 on slices that were too small.
- **Within-race centring of features.** No effect (1.1322 against 1.1313).

## Files

| | |
|---|---|
| `pdf_parser.py` | field tables — tab, horse, jockey, weight, barrier, gear |
| `past_form.py` | past-run histories, the training labels and the prices |
| `features.py` | per-runner features, built strictly from earlier runs |
| `model.py` | rank-ordered conditional logit, temperature search |
| `train.py` | walk-forward validation, confidence bands, the bundle |
| `predict.py` | scoring one race, and the reasons |
| `app.py` | the Streamlit app |

## Running it

```bash
pip install -r requirements.txt
streamlit run app.py
```

Three sample meetings ship in `sample_data/`, so it can be tried without a file
to hand. To refit, put meeting PDFs somewhere and run `python train.py`.

```bash
python test_model.py && python test_app.py
```

111 checks — 85 on the model and parser, 26 driving the app itself. The app
checks exercise the prediction tab, not just the landing page: a Streamlit app
with a fatal error deeper in the script still serves HTTP 200 and still renders
its first tab.

---

*Probabilistic decision support, not a guaranteed outcome. Gamble responsibly —
Gambling Help 1800 858 858, [gamblinghelponline.org.au](https://www.gamblinghelponline.org.au).*
