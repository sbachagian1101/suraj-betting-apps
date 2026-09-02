# WorksheetPredictor

Paste or upload a Racing & Sports **Worksheets** export. The app keeps R&S's
running order exactly as published and rebuilds the percentages attached to
it, per jurisdiction.

## Why

Measured on **61 races and 551 runners** from 1 September 2026 — Wodonga,
Scone, Wolverhampton, Ripon, Lingfield, Brighton, Deauville, Gowran Park:

| | rate | random |
|---|---:|---:|
| R&S top-rated **wins** | **31.1%** | 12.4% |
| R&S top three hold the winner | **60.7%** | 37.3% |

**The order is good.** The percentages are not. Published `PER` scores a
log-loss of **2.6500**, which is *worse than calling every runner equally
likely* (2.1445), because the top pick claims 43.0% and wins 31.1% — it is
**1.38× overconfident**.

## The correction is not the same everywhere

| Region | Meetings | Races | R&S claims | Actually wins | Ratio | |
|---|---:|---:|---:|---:|---:|---|
| **AUS** | 2 | 16 | 33.3% | **50.0%** | 0.67× | *under*confident |
| FR | 1 | 8 | 33.4% | 37.5% | 0.89× | provisional |
| IRE | 1 | 8 | 46.3% | 25.0% | 1.85× | provisional |
| **UK** | 4 | 29 | 50.0% | **20.7%** | 2.42× | badly over |

A single global constant would push Australia and the UK the wrong way from
each other, so the temperature is per region. Australia is **sharpened**
(t = 0.60); the UK is flattened hard (t = 6.55).

## How the constants are chosen

Two per region, both by **moment matching** — the top pick is made to claim
the strike rate it actually achieves — and graded **leave-one-meeting-out**,
never on the meeting they score.

| | |
|---|---|
| win temperature | so the top pick's percentage matches its real strike rate |
| place temperature | extra flattening before Harville, which otherwise reads far too high off sharpened probabilities |

Fitting on **log-loss** instead was tried first and rejected: it left the
favourite **12.8 points understated** out of sample (14.9 in Australia), and
scored *worse on log-loss too* — 2.0352 against 2.0293.

Out-of-sample log-loss **2.0293**, against 2.6500 published and 2.1445
uniform. In-sample calibration after the fix:

| Region | win claim / actual | top-3 claim / actual |
|---|---|---|
| AUS | 47.5% / 50.0% | 81.2% / 81.2% |
| FR | 29.0% / 37.5% | 52.3% / 50.0% |
| IRE | 27.0% / 25.0% | 41.0% / 37.5% |
| UK | 20.7% / 20.7% | 55.0% / 55.2% |
| **ALL** | **29.6% / 31.1%** | **59.7% / 59.0%** |

## Two things it deliberately does not do

**It never reorders the field.** A test asserts the order is identical to
R&S's for every region. All that changes is how confident the numbers are —
the part their own figures get wrong.

**It never invents an expected value.** With no market price there is no EV,
so the headline is the **break-even price**. Enter what you can actually get
and it becomes a fractional-Kelly stake in points, capped.

## Honest limits

This is **one day of racing**. It is a first measurement, not a track record.
FR and IRE rest on a single meeting each — when that meeting is held out no
region constant exists for it and the fold silently grades the *global* one,
so those constants were never actually validated. They are shrunk halfway to
global and the app labels them **provisional**.

A second day of worksheets would settle whether the UK figure is a real bias
or one bad Tuesday.

## Running it

```bash
streamlit run app.py
```

```bash
python test_app.py
```

```bash
python fit_calibration.py
```

## Files

| File | Role |
|---|---|
| `app.py` | Single-page UI — load, parse, predict, charts, recommendation |
| `ws_parser.py` | Worksheets CSV/paste to races and runners, region detection |
| `ws_model.py` | Calibration, Harville places, staking, confidence, insights |
| `fit_calibration.py` | Refits `calibration.json` and proves it out of sample |
| `samples/` | The eight graded meetings plus `results.json` |

74 checks, including re-deriving every accuracy claim the app prints from the
bundled data — a stale `calibration.json` fails the build.

## Deploying

Streamlit Community Cloud, branch `worksheet`, main file `app.py`. In
**Advanced settings** set the Python version to **3.13** — the form defaults
to 3.14. No matplotlib: charts are Altair, which ships with Streamlit.
