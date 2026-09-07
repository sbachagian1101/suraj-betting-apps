"""Horse Race Predictor - paste a Racing & Sports page (or type the race), the
app pulls the country's open feeds, rates the field and simulates the race."""
from __future__ import annotations

import importlib
import io
import os
import re
from datetime import date

import pandas as pd
import streamlit as st

st.set_page_config(page_title="Horse Race Predictor", page_icon="🏇", layout="wide")

HERE = os.path.dirname(os.path.abspath(__file__))

# --- stale-deployment guard (runs BEFORE any helper import) --------------------
# Streamlit Cloud has served a new app.py against a cached old helper module
# more than once.  The required-name list is DERIVED from this file so it cannot
# drift: every `module.name` reference below is checked against the live module.
BUILD = "2026-09-07c"      # every helper module carries the same string; a mismatch = stale copy
_src = io.open(__file__, encoding="utf-8").read()
_REQUIRED = {m: sorted(set(re.findall(rf"\b{m}\.([A-Za-z_]\w*)", _src)))
             for m in ("pipeline", "model", "charts", "rs_paste", "common")}
_missing: list[str] = []
for _mod, _names in _REQUIRED.items():
    try:
        _m = importlib.import_module(_mod)
    except Exception as _e:  # noqa: BLE001
        _missing.append(f"{_mod} (import failed: {type(_e).__name__}: {_e})")
        continue
    _missing += [f"{_mod}.{n}" for n in _names if not hasattr(_m, n)]
    # a name can exist with an old signature (analyse() without speed_raw did exactly
    # that on 7 Sep 2026), so every helper also carries a build stamp
    if getattr(_m, "BUILD", None) != BUILD:
        _missing.append(f"{_mod} (build {getattr(_m, 'BUILD', 'none')} ≠ {BUILD})")
if _missing:
    st.error("**This deployment is running stale code.** Missing: `" + "`, `".join(_missing)
             + "`. Streamlit Cloud pulled the new files but kept an old module in memory. "
             "Fix it with **Manage app → ⋮ → Reboot app**.", icon=":material/error:")
    st.stop()

import charts      # noqa: E402
import common      # noqa: E402
import model       # noqa: E402
import pipeline    # noqa: E402
import rs_paste    # noqa: E402

COUNTRIES = list(common.COUNTRIES)          # display names
SAMPLE = os.path.join(HERE, "fixtures", "rs_grafton_r2_2026-09-07_header.txt")


def pct(x) -> str:
    return "—" if x is None else f"{x:.0%}"


def price(x) -> str:
    return "—" if not x else f"{x:.2f}"


# --- sidebar: inputs --------------------------------------------------------------

with st.sidebar:
    st.title("🏇 Horse Race Predictor")
    st.caption("Australia · France · UK · Ireland · Hong Kong · South Africa · USA")
    race_day = st.date_input("Date", value=date.today())
    country_name = st.selectbox("Race country", COUNTRIES, index=0)
    venue = st.text_input("Race venue", placeholder="Grafton")
    race_no = st.number_input("Race no", min_value=1, max_value=20, value=1, step=1)
    st.markdown("**Paste the Racing & Sports page** (Full Fields or Enhanced Form tab, "
                "select-all → copy). The breadcrumb inside it identifies the race, so the fields "
                "above are only needed when you paste nothing. The Speed Map page can be pasted "
                "underneath it or in the second box.")
    if st.button("Load sample paste (Grafton R2, 7 Sep 2026)"):
        st.session_state["raw"] = io.open(SAMPLE, encoding="utf-8").read()
    raw = st.text_area("Racing & Sports paste", key="raw", height=220,
                       placeholder="* [Race 2](https://www.racingandsports.com.au/form-guide/thoroughbred/australia/grafton/2026-09-07/R2)\n...")
    speed_raw = st.text_area("Speed Map paste (optional)", key="speed_raw", height=120,
                             placeholder="Copy the R&S Speed Map tab (Pace Values table: Tab, Horse, WT, Jockey, JR, BP, AES, AFS)",
                             help="Adds each runner's average early speed (AES), finishing speed (AFS), "
                                  "jockey rating (JR) and the barrier after scratchings. You can also "
                                  "append the Speed Map page under the main paste.")
    go = st.button("🔍 Search & predict", type="primary", width="stretch")
    st.divider()
    st.header("Model settings")
    country_code = common.COUNTRIES[country_name]
    idn_preview = rs_paste.identity(raw or "")
    d = model.defaults_for(idn_preview.country or country_code)
    form_w = st.slider("Weight on the form model", 0.0, 1.0, float(d.market_weight), 0.05,
                       help="0 = follow the market exactly, 1 = ignore the market. "
                            "Blended as a weighted geometric mean in log space.")
    spread = st.slider("Performance spread (lengths)", 1.5, 6.0, float(d.spread), 0.1,
                       help="Standard deviation of a runner's performance. Larger = flatter probabilities.")
    k_move = st.slider("Market-move term", 0.0, 2.0, float(d.k_move), 0.1,
                       help="Lengths credited per log-unit of firming since the opening price.")
    show_sens = st.checkbox("Run robustness check (slower)", value=False)

params = model.replace(d, market_weight=form_w, spread=spread, k_move=k_move)

# --- run ------------------------------------------------------------------------

if go:
    log: list[str] = []
    with st.status("Searching the feeds…", expanded=True) as box:
        def progress(msg: str):
            log.append(msg)
            box.write(msg)
        try:
            A = pipeline.analyse(raw or "", country=country_code, venue=venue.strip(),
                                 race_no=int(race_no), race_day=race_day, params=params,
                                 progress=progress, speed_raw=speed_raw or "")
            st.session_state["analysis"] = A
            box.update(label=f"Done in {A.seconds:.0f}s", state="complete", expanded=False)
        except ValueError as exc:
            st.session_state.pop("analysis", None)
            box.update(label="Could not identify the race", state="error")
            st.error(str(exc))
        except Exception as exc:  # noqa: BLE001
            st.session_state.pop("analysis", None)
            box.update(label="Failed", state="error")
            st.exception(exc)

A: pipeline.Analysis | None = st.session_state.get("analysis")
if A is None:
    st.title("🏇 Horse Race Predictor")
    st.markdown("""
Paste a **Racing & Sports** race page in the sidebar (or type the date, country, venue and
race number) and press **Search & predict**. The app then:

1. reads the race identity from the paste (country, venue, date, race number, distance, going, class, prize),
2. pulls the country's open feeds in parallel - form, career records, jockey and trainer
   strike rates, barrier, weight, gear, run style / speed map, sectionals (Hong Kong),
   fixed odds with every fluctuation, exchange prices,
3. rates every runner in lengths, de-vigs the market, blends the two, and
4. simulates the race 20,000 times for win, place, exacta and trifecta probabilities.

| Country | Feeds used |
|---|---|
| Australia | Ladbrokes AU (form, speed map, flucs), Betfair exchange, your R&S paste |
| France | PMU (participants, every past run, tote odds), Ladbrokes AU, Betfair |
| UK / Ireland | Sporting Life (rides, OR, previous results, Sky Bet price history), Ladbrokes AU (flucs, speed map), Betfair |
| Hong Kong | HKJC race card + horse records + sectional times, Betfair |
| South Africa | Winning Form racecards (merit ratings, records, past runs, early prices), Betfair |
| USA | Horse Racing Nation entries (morning line), Ladbrokes AU (fixed odds, flucs, form string), Betfair |

**Caveat.** Across every racing model graded in this repo the market has been the strongest
single predictor. Read the model-vs-market gap as a prompt to look closer, not as an edge.
""")
    st.stop()

# re-rate cheaply when the sliders move
if (A.params.market_weight, A.params.spread, A.params.k_move) != (params.market_weight, params.spread, params.k_move):
    A = pipeline.rerate(A, params)
    st.session_state["analysis"] = A

card, rows, sim = A.card, A.rows, A.sim
idn = A.identity

# --- header ---------------------------------------------------------------------

st.title(f"{card.venue} · Race {card.race_no} · {common.CODE_TO_COUNTRY.get(card.country, card.country)}")
bits = [x for x in (card.name, f"{card.dist_m}m" if card.dist_m else "", card.surface.title() if card.surface else "",
                    card.going.title() if card.going else "", card.race_class,
                    f"{card.currency} {card.prize:,.0f}" if card.prize else "",
                    card.race_date.strftime("%a %d %b %Y") if card.race_date else "",
                    f"{card.start_time} local" if card.start_time else "", card.weather.title() if card.weather else "",
                    f"Rail: {card.rail}" if card.rail else "") if x]
st.markdown(" · ".join(bits))
ok = [s for s in A.status if s.ok]
bad = [s for s in A.status if not s.ok]
st.caption("Sources: " + ", ".join(f"✅ {s.name} ({s.runners})" for s in ok)
           + (" · " if ok and bad else "") + ", ".join(f"⚠️ {s.name}" for s in bad)
           + (" · R&S paste" if "Racing & Sports paste" in card.sources else ""))
for w in card.warnings[:4]:
    st.warning(w)

if not rows:
    st.error("The race could not be rated.")
    for n in A.notes:
        st.write(n)
    st.stop()

top = rows[0]
fav = min((r for r in rows if r.odds), key=lambda r: r.odds, default=None)
steam = min((r for r in rows if r.move_pct is not None), key=lambda r: r.move_pct, default=None)
drift = max((r for r in rows if r.move_pct is not None), key=lambda r: r.move_pct, default=None)
odds_all = [r.odds for r in rows if r.odds]


def card_box(col, label: str, value: str, sub: str, colour: str = "#2563eb"):
    col.markdown(
        f"<div style='border:1px solid rgba(128,128,128,.25);border-left:5px solid {colour};"
        f"border-radius:8px;padding:10px 12px;min-height:92px'>"
        f"<div style='font-size:.8em;opacity:.7'>{label}</div>"
        f"<div style='font-size:1.25em;font-weight:600;line-height:1.2'>{value}</div>"
        f"<div style='font-size:.85em;opacity:.8'>{sub}</div></div>", unsafe_allow_html=True)


c1, c2, c3, c4, c5 = st.columns(5)
card_box(c1, "Top pick", f"{top.number} {top.name}", f"{top.p_final:.0%} win · {top.p_top3:.0%} top-3")
card_box(c2, "Favourite", f"{fav.number} {fav.name}" if fav else "—",
         f"{fav.odds:.2f} ({fav.price_source})" if fav else "no prices", "#64748b")
card_box(c3, "Biggest steamer", f"{steam.number} {steam.name}" if steam and steam.move_pct < 0 else "—",
         f"{steam.entry.odds_open:g} → {steam.entry.odds:g} ({steam.move_pct:+.0%})"
         if steam and steam.move_pct < 0 else "no movement data", "#16a34a")
card_box(c4, "Biggest drifter", f"{drift.number} {drift.name}" if drift and drift.move_pct > 0 else "—",
         f"{drift.entry.odds_open:g} → {drift.entry.odds:g} ({drift.move_pct:+.0%})"
         if drift and drift.move_pct > 0 else "no movement data", "#dc2626")
card_box(c5, "Book %", f"{model.book_percentage(odds_all):.0f}%" if len(odds_all) == len(rows) else "—",
         f"{len(odds_all)}/{len(rows)} priced", "#f59e0b")
st.write("")

st.subheader("Insights")
for line in A.insights:
    st.markdown(f"- {line}")

# --- tabs -----------------------------------------------------------------------

t_pred, t_speed, t_form, t_market, t_conn, t_sim, t_data, t_method = st.tabs(
    ["Prediction", "Speed map & pace", "Form", "Market", "Jockeys & trainers", "Simulation",
     "Runner data", "Method"])

with t_pred:
    coherent = len(odds_all) == len(rows) and 102 <= model.book_percentage(odds_all) <= 180
    table = pd.DataFrame([{
        "No": r.number, "Horse": r.name, "Bar": r.entry.barrier, "Wt": r.entry.weight,
        "Jockey": r.entry.jockey, "Trainer": r.entry.trainer,
        "Win %": r.p_final * 100, "Top-3 %": r.p_top3 * 100, "Form model %": r.p_model * 100,
        "Market %": (r.p_market or 0) * 100 if r.p_market else None,
        "Price": r.odds, "Fair": r.fair, "Edge %": (r.ev * 100 if r.ev is not None and coherent else None),
        "Move": (f"{r.move_pct:+.0%}" if r.move_pct is not None else ""),
        "Rating (L)": r.rating, "Runs used": r.used_runs,
    } for r in rows])
    st.dataframe(table, hide_index=True, width="stretch", height=min(40 + 36 * len(rows), 700),
                 column_config={
                     "Win %": st.column_config.ProgressColumn(format="%.0f%%", min_value=0, max_value=100),
                     "Top-3 %": st.column_config.NumberColumn(format="%.0f%%"),
                     "Form model %": st.column_config.NumberColumn(format="%.0f%%"),
                     "Market %": st.column_config.NumberColumn(format="%.0f%%"),
                     "Price": st.column_config.NumberColumn(format="%.2f"),
                     "Fair": st.column_config.NumberColumn(format="%.2f"),
                     "Edge %": st.column_config.NumberColumn(format="%+.0f%%"),
                     "Wt": st.column_config.NumberColumn(format="%.1f"),
                     "Rating (L)": st.column_config.NumberColumn(format="%.2f"),
                 })
    if not coherent and odds_all:
        st.caption("Edge is not quoted: the prices do not form a coherent book (102-180%).")
    st.plotly_chart(charts.probability_chart(rows), width="stretch")
    st.plotly_chart(charts.terms_chart(rows), width="stretch")
    if show_sens:
        with st.spinner("Jittering the model constants…"):
            tally = model.sensitivity(card, params)
        st.plotly_chart(charts.sensitivity_chart(tally), width="stretch")

with t_speed:
    st.plotly_chart(charts.speed_map(rows, card.dist_m), width="stretch")
    sm = pd.DataFrame([{"No": r.number, "Horse": r.name, "Barrier": r.entry.barrier,
                        "Run style": r.entry.speed_label or "—",
                        "AES": r.entry.extras.get("aes"), "AFS": r.entry.extras.get("afs"),
                        "JR": r.entry.extras.get("jr"),
                        "Settling (L off lead)": r.entry.settling,
                        "Early speed 0-10": (r.speed_score * 10) if r.speed_score is not None else None,
                        "Last-400m (s)": r.entry.sectional_600,
                        "Speed term (L)": r.terms.get("speed", 0.0),
                        "Late speed term (L)": r.terms.get("late speed", 0.0),
                        "Sectional term (L)": r.terms.get("sectional", 0.0)} for r in rows])
    st.dataframe(sm, hide_index=True, width="stretch",
                 column_config={"AES": st.column_config.NumberColumn(format="%.1f"),
                                "AFS": st.column_config.NumberColumn(format="%.1f"),
                                "JR": st.column_config.NumberColumn(format="%.1f"),
                                "Early speed 0-10": st.column_config.NumberColumn(format="%.1f")})
    if any(r.entry.extras.get("aes") is not None for r in rows):
        st.caption("AES / AFS are Racing & Sports pace values from your Speed Map paste: average early "
                   "speed and average finishing speed (higher = faster). Early speed here is AES ranked "
                   "within the field; the late-speed term is AFS versus the field in standard deviations.")
    elif not any(r.speed_score is not None for r in rows):
        st.info("No run-style data for this race from the feeds. Paste the R&S Speed Map page in the "
                "sidebar's second box to add AES / AFS pace values.")

with t_form:
    st.plotly_chart(charts.form_heatmap(rows), width="stretch")
    pick = st.selectbox("Runner", [f"{r.number} {r.name}" for r in rows])
    r = next(x for x in rows if f"{x.number} {x.name}" == pick)
    e = r.entry
    cc1, cc2, cc3, cc4 = st.columns(4)
    cc1.metric("Career", f"{e.career[0]}: {e.career[1]}-{e.career[2]}-{e.career[3]}")
    cc2.metric("Days since last run", e.last_run_days if e.last_run_days is not None else "—")
    cc3.metric("Rating", f"{e.official_rating:g}" if e.official_rating is not None else "—")
    cc4.metric("Weighted margin", f"{r.avg_margin:+.2f} L" if r.avg_margin is not None else "—",
               f"{r.used_runs} runs used", delta_color="off")
    if e.records:
        st.caption("Records (starts: wins-2nds-3rds): " + " · ".join(
            f"{k} {v[0]}: {v[1]}-{v[2]}-{v[3]}" for k, v in e.records.items()))
    if e.comment:
        st.markdown(f"*{e.comment}*")
    if e.runs:
        runs = pd.DataFrame([{
            "Date": run.run_date, "Days": run.days_ago, "Track": run.track, "Race": run.race_class or run.race_name,
            "Dist": run.dist_m, "Surface": run.surface, "Going": run.going,
            "Fin": (run.comment[:4] if run.non_finish else (f"{run.pos}/{run.field_size or '?'}" if run.pos else "")),
            "Margin (L)": run.margin_l, "Wt": run.weight, "Bar": run.barrier, "Jockey": run.jockey,
            "SP": run.sp, "Rating": run.rating, "Time": run.time_s, "Run positions": run.running_pos,
            "Prize": run.prize, "Comment": run.comment if not run.non_finish else "",
        } for run in e.runs])
        st.dataframe(runs, hide_index=True, width="stretch")
    else:
        st.info("No past-run table for this runner from the feeds.")

with t_market:
    st.plotly_chart(charts.odds_movement(rows), width="stretch")
    mk = pd.DataFrame([{
        "No": r.number, "Horse": r.name, "Open": r.entry.odds_open, "Now": r.odds,
        "Source": r.price_source, "Move": r.move_label, "Move %": (r.move_pct * 100) if r.move_pct is not None else None,
        "Place": r.entry.place_odds, "Exchange": r.entry.exchange_odds,
        "Market %": (r.p_market or 0) * 100 if r.p_market else None,
        "Market-move term (L)": r.terms.get("market move", 0.0),
        "Flucs": " → ".join(f"{f.price:g}" for f in r.entry.flucs) if r.entry.flucs else "",
    } for r in rows])
    st.dataframe(mk, hide_index=True, width="stretch",
                 column_config={"Move %": st.column_config.NumberColumn(format="%+.0f%%"),
                                "Market %": st.column_config.NumberColumn(format="%.0f%%")})
    if card.extras.get("forecast"):
        st.caption("Betting forecast: " + card.extras["forecast"])

with t_conn:
    st.plotly_chart(charts.connections_chart(rows), width="stretch")

    def fmt(s):
        return f"{s[1]}/{s[0]} ({s[1] / s[0]:.0%} win, {s[2] / s[0]:.0%} place)" if s and s[0] else "—"
    st.dataframe(pd.DataFrame([{
        "No": r.number, "Horse": r.name, "Jockey": r.entry.jockey, "Jockey record": fmt(r.entry.jockey_stats),
        "Trainer": r.entry.trainer, "Trainer record": fmt(r.entry.trainer_stats),
        "Combination": fmt(r.entry.combo_stats),
        "Jockey term (L)": r.terms.get("jockey", 0.0), "Trainer term (L)": r.terms.get("trainer", 0.0),
    } for r in rows]), hide_index=True, width="stretch")

with t_sim:
    if sim:
        st.plotly_chart(charts.finish_heatmap(sim), width="stretch")
        s1, s2 = st.columns(2)
        with s1:
            st.markdown("**Most likely exactas**")
            st.dataframe(pd.DataFrame(sim.exactas, columns=["1st → 2nd", "Prob"]), hide_index=True,
                         width="stretch", column_config={"Prob": st.column_config.NumberColumn(format="%.1%")})
        with s2:
            st.markdown("**Most likely trifectas**")
            st.dataframe(pd.DataFrame(sim.trifectas, columns=["1st → 2nd → 3rd", "Prob"]), hide_index=True,
                         width="stretch", column_config={"Prob": st.column_config.NumberColumn(format="%.1%")})
        st.dataframe(pd.DataFrame({"No": sim.numbers, "Horse": sim.names,
                                   "Expected finish": sim.expected_pos,
                                   "Win %": sim.position_matrix[:, 0] * 100,
                                   "Top-3 %": sim.position_matrix[:, :3].sum(axis=1) * 100}),
                     hide_index=True, width="stretch",
                     column_config={"Expected finish": st.column_config.NumberColumn(format="%.1f"),
                                    "Win %": st.column_config.NumberColumn(format="%.0f%%"),
                                    "Top-3 %": st.column_config.NumberColumn(format="%.0f%%")})

with t_data:
    st.markdown("**Source status**")
    st.dataframe(pd.DataFrame([{"Source": s.name, "OK": s.ok, "Detail": s.detail, "Seconds": round(s.seconds, 1)}
                               for s in A.status]), hide_index=True, width="stretch")
    if card.extras.get("preview"):
        st.markdown("**Race preview**")
        st.write(card.extras["preview"])
    if card.extras.get("conditions"):
        st.caption(card.extras["conditions"][:600])
    st.markdown("**Merged runner records**")
    st.dataframe(pd.DataFrame([{
        "No": e.number, "Horse": e.name, "Scr": e.scratched, "Bar": e.barrier, "Wt": e.weight, "Age": e.age,
        "Sex": e.sex, "Jockey": e.jockey, "Trainer": e.trainer, "Form": e.form_string,
        "Career": f"{e.career[0]}: {e.career[1]}-{e.career[2]}-{e.career[3]}" if e.career[0] else "",
        "Prize money": e.prize_money, "Rating": e.official_rating, "Days": e.last_run_days,
        "Gear": e.gear, "Runs": len(e.runs), "Odds": e.odds, "Open": e.odds_open, "Exch": e.exchange_odds,
        "Sources": ", ".join(e.sources),
    } for e in card.entries]), hide_index=True, width="stretch")
    st.caption("Identity read from the paste: " + ", ".join(
        f"{k}={v}" for k, v in (("country", idn.country), ("venue", idn.venue), ("race", idn.race_no),
                                ("date", idn.race_date), ("time", idn.start_time), ("dist", idn.dist_m),
                                ("surface", idn.surface), ("going", idn.going), ("type", idn.race_type),
                                ("prize", idn.prize)) if v))

with t_method:
    p = A.params
    st.markdown(f"""
**Rating (in lengths)** = form + class + consistency + distance/course/surface/going records +
fitness + rating + jockey + trainer + combination + barrier + weight + speed/pace + sectional +
market move + tips.

* **Form**: each usable past run becomes a beaten margin at today's distance, adjusted
  {p.lengths_per_kg} L per kg of weight difference, {p.k_field} L per runner of field size, and
  up to ±{1.5 * p.k_class_adj:.1f} L for the prizemoney ratio (same currency only). Runs are
  weighted by recency (e^(-days/{p.tau_days:.0f})) and distance relevance (σ = {p.sigma_dist:.0f} m).
  Feeds without beaten lengths (Sporting Life) get a margin estimated from the finishing
  position at {p.w_estimated_margin:.0%} evidence weight.
* **Records** are shrunk toward the runner's own career strike rate with a prior of
  {p.prior_starts:.0f} starts, so a two-start distance record cannot move a horse two lengths.
* **Jockey / trainer** strike rates are shrunk toward 10% with a 40-start prior; each log-unit
  above that is worth {p.k_jockey} / {p.k_trainer} L.
* **Barrier** costs up to {p.k_barrier} L for the widest gate at 1200 m, scaled by 1200/distance.
  **Speed**: run style vs field (R&S AES ranked within the field when a Speed Map is pasted, else the
  Ladbrokes settling position or label), with a hot pace (3+ leaders) penalising leaders and a slow pace
  rewarding a lone leader. **Late speed**: {p.k_late} L per SD of R&S AFS vs the field. **JR**: when no
  jockey record exists, {p.k_jr} L per point of R&S jockey rating vs the field mean.
  **Sectional** (HK): {p.k_sectional} L per second of last-400 m vs the field median.
* **Market**: prices are de-vigged by the power method (longshots shrunk more). Where a runner
  has little evidence its rating is shrunk toward the market's implied rating, never toward the
  field mean. Final = exp({p.market_weight:.2f}·log(form) + {1 - p.market_weight:.2f}·log(market)).
  A move since opening enters as −{p.k_move} × log(now/open) lengths.
* **Simulation**: 20,000 Plackett-Luce draws on the final probabilities give every finishing
  position, exactas and trifectas. Top-3 in the table is the exact enumeration.

**Honest limits.** Weights are hand-set, not fitted; there is no joint ability fit across races
(opposition strength is only proxied by prizemoney and class); Sporting Life and US feeds carry no
beaten lengths; sectionals are free only in Hong Kong. The market has beaten every form-only model
graded in this repo, so the value of the app is the ranking and the *disagreement* it surfaces.
""")
