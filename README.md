# Top Race Predictor

The **HorseRacingTextPredictor** engine, on the web. Paste a race table, or
upload a Racing & Sports `-T.xlsx` meeting export and pick a race from it.

`engine.py` is the original v1.0.1 engine, unchanged apart from one added
parameter (below). Its own test suite still passes untouched. The adaptive
scorecard, the context weighting, the Monte Carlo, the factor breakdown, the
exacta and trifecta structures and the exports all behave exactly as they do in
the desktop app.

## What is new

**Meeting files.** The `-T.xlsx` export is laid out exactly like the text people
paste — race header, `Type :` line, distance, the `Tab Horse Form L5 BP`
header, one row per runner — just spread across spreadsheet cells. So
`meeting.py` joins the cells back into text and hands it to the untouched
parser. A whole card is one upload.

**Calibrated confidence** (on by default, one click to turn off).

## Tested against actual results

CARNARVON (WA), 30 August 2026, races 1–5, scored against the finishing order:

| race | field | its pick | stated win % | winner | pick finished |
|---|---|---|---|---|---|
| 1 | 9 | 1 Any Questions | 78.0% | tab 4 | unplaced |
| 2 | 7 | 5 Majestuoso Phoenix | 62.4% | tab 7 | unplaced |
| 3 | 11 | 2 Awesome Lily | 37.7% | tab 6 | unplaced |
| 4 | 12 | 5 Solar System | 52.5% | tab 7 | **placed** |
| 5 | 10 | 2 Hard Questions | 55.5% | tab 6 | unplaced |

**Top pick won 0 of 5, placed once.**

Five races cannot condemn the selections. A model with no skill at all goes
0-for-5 in these field sizes **57%** of the time. But they do condemn the
percentages: if those probabilities were right the expected count was **2.9
winners**, and 0-for-5 has probability about **1%**.

## The confidence problem, measured without results

Across **119 races from 16 meetings**, the original engine's top pick averages a
stated win probability of **60.3%** in fields averaging 11.3 runners. It claims
over 80% in **18%** of races, and as high as 99.9%.

From **7,787 past runs** parsed out of Racing & Sports Australian form guides,
the **market favourite wins 37.3%** in fields averaging 10.3. The favourite is
the best-informed selection available anywhere, and nothing suggests this
scorecard beats it.

The cause is a single uncalibrated constant. The simulation draws each runner as
`score + gauss(0, σ)` with σ ≈ 6.5 points, and the spread of scores across a
field is about that size, so the top-scorer wins nearly every simulated running.

## The calibration

`analyse_race_text` gained `uncertainty_scale`, which multiplies σ. At `1.0` the
engine is bit-for-bit the original.

| | mean top-pick win % | claims > 80% |
|---|---|---|
| original (`1.0`) | 60.3% | 18% of races |
| **calibrated (`2.0`)** | **38.5%** | **1% of races** |
| favourites in reality | 37.3% | — |

It **does not change the selections**. Across the same 119 races the top pick is
identical in **98%** and the top three are the same set in **92%**. Only the
confidence moves — which is why it is the default, with the original one click
away.

## What is still unknown

Whether the picks are any good. Five races cannot answer that. The finishing
order for the other 15 meetings already on file — about **114 more races** —
would settle it at a useful precision.

## Traps found while building this

**`TABTOUCH` starts with "tab".** Skipping race names beginning `tab` to avoid
the `Tab Horse Form L5` header row silently swallowed the TABTOUCH Carnarvon
Cup, merging it into the previous race, which then reported 19 runners. TABtouch
sponsors races across Australia. The header row is already excluded because it
does not carry a number in column 0.

**A runner row looks like a race header.** Both have a number in column 0 and
text in column 2 — the runner's form figures. The race header is the one with
column 1 *empty*; without that test every runner starts a new race.

**A `BytesIO` has no `.name`.** Sniffing the file extension off the stream sent
an uploaded `.xlsx` to the CSV reader, which died on `UnicodeDecodeError` at the
first binary byte. The filename is passed explicitly.

## Files

| | |
|---|---|
| `engine.py` | the original v1.0.1 engine, plus `uncertainty_scale` |
| `meeting.py` | splits a `-T.xlsx` meeting into per-race text |
| `app.py` | the Streamlit interface |
| `backtest.py` | scores picks against recorded finishing order |
| `test_engine.py` | the original engine's own suite, unchanged |

## Running it

```bash
pip install -r requirements.txt
streamlit run app.py
```

```bash
python -m unittest test_engine.py && python test_top.py && python test_app.py
```

78 checks — 4 original engine, 53 splitter and calibration, 21 driving the app.
The app checks exercise the prediction tab, not just the landing page: a
Streamlit app with a fatal error deeper in the script still serves HTTP 200 and
still renders its first tab.

---

*A transparent heuristic scorecard, not a trained or calibrated betting model —
the original README says so too. Probabilistic decision support, not a
guaranteed outcome. Gamble responsibly — Gambling Help 1800 858 858,
[gamblinghelponline.org.au](https://www.gamblinghelponline.org.au).*
