# Bet365 Predictor

The **Bet365 Multi-Meeting Horse Race Predictor Pro v1.0** desktop engine, on
the web. Paste a Bet365 race-card text export covering any number of meetings,
predict the full finishing order of every active runner, and teach the model
from actual results after racing.

`bet365_parser.py`, `bet365_model.py`, `results_parser.py` and `storage.py` are
the original engine, unchanged. Only the Tkinter front end was replaced.

## The one thing that is different on the web

**Retraining lives in the browser session, not on disk.** The desktop version
writes `data/model_state.json` after every retrain and picks it up next launch.
Streamlit Cloud gives an app a fresh, empty filesystem on every restart, so that
cannot work here — a write would look successful and quietly vanish.

Instead:

- the trained state shipped in `data/` is the starting point every time;
- retraining updates the state held in the session, and every figure on the
  page comes from it;
- **Download trained state** in the sidebar saves it as JSON, and **Restore**
  loads it back at the start of the next session.

The app says this on the Input tab rather than letting a retrain appear to
persist.

## The five tabs

1. **Input** — one shared box for all meetings, `.txt` upload, two bundled
   cards, adjustable finishing-position simulations, *Predict all races*.
2. **Parsed meetings** — per meeting: race table with distance, going, rail,
   discipline, active vs declared runners and scratchings; per race: the full
   runner table and the Bet365 overview.
3. **Race predictions** — per race, four views: **Final trained**, **Model
   agreement**, **Specialist ranks**, **Horse explanations**. Every active
   runner gets a predicted position, win and top-three probability, expected
   rank, confidence 0–9, model fair odds, classification and its rank under
   each specialist.
4. **Results & training** — paste results in the raw Bet365 style
   (`1-5-4-3Fixed OddsQuaddie`) or the simplified style (`R1: 1-5-4-3`).
   Partial results and dead heats (`R7: 6-3-4,5`) are valid. The mapping onto
   parsed races is shown before anything is saved, then *Save and retrain*.
5. **Model & diagnostics** — training race and pair counts, learned-layer
   influence, training metrics, learned coefficients, training history.

## Four specialist models plus a learned layer

| | model | what it uses |
|---|---|---|
| 1 | Bet365 analyst source | suggested play, order of named chances, analyst wording |
| 2 | Independent recent form & class | last-five form, field-size-adjusted finishes, margins, recency, class |
| 3 | Distance / going / fitness | distance and going similarity, days since run, prep stage, age, weight |
| 4 | Pace / draw / weight | expected race map, likely leaders, on-pace and closing evidence, barrier |

Results become **pairwise finishing relationships** — from `1-5-4-3` the model
learns that 1 beat everyone, 5 beat everyone but 1, and so on, without
inventing an order for unreported runners. A regularised pairwise logistic
ranking layer learns from all accumulated races; its influence grows with the
sample and stays capped so the four transparent models keep mattering.

## The odds firewall

Current and historical prices are excluded from every predictive feature, and
the app calculates no EV or stake sizing. Bet365's written analyst view is kept
as one separately visible source model, not as a price. **Odds influence reads
0% because it is structurally zero**, not because it happens to be small.

## What the bundled numbers mean

The shipped state was trained on the developer's original five meetings — 35
labelled races, 1,358 pairwise relationships. Leave-one-meeting-out:

| metric | result |
|---|---:|
| winner ranked first | 25.71% |
| winner ranked top three | 48.57% |
| mean actual-winner rank | 4.37 |
| pairwise finishing accuracy | 62.00% |
| known top-four overlap | 47.86% |

All five meetings were on the **same date**, so that is a cross-meeting
diagnostic, not a forward-looking one. The honest test is a meeting predicted
before its results are known — which is what the Results tab is for.

## Bugs found porting it

**Signals are `(label, weight)` pairs, not strings.** Joining them raised
`sequence item 0: expected str instance, tuple found` and took out the whole
Horse explanations view.

**`training_metrics` mixes int, float and str.** Formatting only the floats
left one object column of mixed types; Arrow cannot type that, and Streamlit
ships dataframes as Arrow, so the diagnostics table failed in the browser while
every Python test passed. Same family as the matplotlib and NaN-in-`attrs`
traps.

**The sidebar renders before the retrain block runs.** On the pass where
retraining happened the counts were still the pre-training ones. Fixed with a
rerun after retraining, carrying the confirmation across.

**Runner fields differ from the obvious guesses**: `historical_runs` not
`history`, and `status` (`ACTIVE`/`SCRATCHED`) rather than a `scratched` flag.

## Running it

```bash
pip install -r requirements.txt
streamlit run app.py
```

```bash
python -m unittest tests.test_predictor && python test_app.py
```

57 checks — the engine's own 8, unchanged, plus 49 driving the app. One engine
test asserted the Tkinter GUI had exactly one `tk.Text`; its *intent* — a
single shared input box rather than one per meeting — was ported to the
Streamlit widget that replaced it rather than deleted.

The app checks parse a card, predict every race, walk all four specialist
views, map a real result, retrain, and run the whole path again with
**matplotlib blocked**, because that is the difference between a development
machine and Streamlit Cloud.

---

*Probabilistic decision support, not a guaranteed outcome. Gamble responsibly —
Gambling Help 1800 858 858, [gamblinghelponline.org.au](https://www.gamblinghelponline.org.au).*
