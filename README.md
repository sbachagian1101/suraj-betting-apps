# suraj-betting-apps

Personal suite of quantitative racing prediction apps.

| App | Branch | Entry point | Status |
|---|---|---|---|
| **GreyhoundPredictor** | `greyhound` | `app.py` | ✅ live |
| **HorsePredictor** (thoroughbred) | `horse` | `app.py` | ✅ live |
| **HarnessPredict** | `harness` | `app.py` | ✅ live |

## GreyhoundPredictor

Streamlit app that parses a **Racing & Sports greyhound Enhanced Form** page
(select-all → copy → paste) and produces market-anchored win probabilities,
a speed map, and value flags.

- `greyhound_parser.py` — tolerant parser for both the live-page clipboard
  format (labels glued to values, `betfair$x` / `Tab$x` prices) and the older
  line-per-label format. The field table is read via its **header row**, not by
  column position, because R&S ship several column sets for the same Enhanced
  Form page (some carry a `WT` column, some carry prizemoney/bookmaker columns
  instead). A row needs only a tab number and a name to produce a runner, and a
  gap in the tab sequence raises a warning rather than silently shrinking the
  field.
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

270 field-level checks against two real page captures, one per column layout:

| Fixture | Layout |
|---|---|
| `tests_fixture_geelong_r9.txt` | has a `WT` column, price in a trailing unnamed column |
| `tests_fixture_qlakeside_r9.txt` | no `WT` column; prizemoney + `Bet365`/`Tab` columns |

```bash
python test_parser.py   # expect: PASS 270  FAIL 0
```

Run this whenever Racing & Sports change their page markup. If you hit a page
that parses to fewer runners than the field table shows, save the paste as a
third fixture and add it to `FIXTURES` in `test_parser.py`.

### Deploy (Streamlit Community Cloud)

New app → repo `sbachagian1101/suraj-betting-apps` → branch `greyhound` →
main file `app.py`. Pushing to the branch auto-redeploys.
