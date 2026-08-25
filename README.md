# suraj-betting-apps

Personal suite of quantitative racing prediction apps.

| App | Branch | Entry point | Status |
|---|---|---|---|
| **GreyhoundPredictor** | `greyhound` | `app.py` | ✅ live |
| HorsePredictor (thoroughbred) | `horse` | — | planned |
| HarnessPredictor | `harness` | — | planned |

## GreyhoundPredictor

Streamlit app that parses a **Racing & Sports greyhound Enhanced Form** page
(select-all → copy → paste) and produces market-anchored win probabilities,
a speed map, and value flags.

- `greyhound_parser.py` — tolerant parser for both the live-page clipboard
  format (labels glued to values, `betfair$x` / `Tab$x` prices) and the older
  line-per-label format.
- `greyhound_model.py` — ensemble model: 60% market (Betfair-weighted), 40%
  fundamentals (speed, early pace, box, track/dist, form, trainer, freshness),
  Plackett–Luce simulation for place probabilities.
- `app.py` — Streamlit UI.

### Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

### Regression test

134 field-level checks against a real Geelong R9 page capture:

```bash
python test_parser.py   # expect: PASS 134  FAIL 0
```

Run this whenever Racing & Sports change their page markup.

### Deploy (Streamlit Community Cloud)

New app → repo `sbachagian1101/suraj-betting-apps` → branch `greyhound` →
main file `app.py`. Pushing to the branch auto-redeploys.
