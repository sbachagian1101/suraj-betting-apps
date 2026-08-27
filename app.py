"""HorsePredictor - Streamlit UI for Racing & Sports thoroughbred Enhanced Form."""
from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st

import horse_model as model
import horse_parser as parser
import place_finder as pf
import race_quality as rq
import results_log as rl

st.set_page_config(page_title="HorsePredictor", page_icon=":horse_racing:",
                   layout="wide", initial_sidebar_state="expanded")
st.markdown("""
<style>
.block-container {padding-top: 1.15rem; padding-bottom: 3rem;}
div[data-testid="stMetric"] {border:1px solid rgba(128,128,128,.22);border-radius:.7rem;padding:.55rem .8rem;}
.hero {padding:1rem 1.1rem;border-radius:.85rem;background:linear-gradient(135deg,rgba(21,128,61,.18),rgba(0,0,0,0));border:1px solid rgba(128,128,128,.22);margin-bottom:1rem;}
.pick {padding:1rem 1.2rem;border:1px solid rgba(46,204,113,.35);border-radius:.8rem;background:rgba(46,204,113,.08);margin:.4rem 0 1rem 0;}
.muted {opacity:.72;font-size:.9rem;}
</style>
""", unsafe_allow_html=True)


def reset_app() -> None:
    st.session_state["paste_input"] = ""
    for k in ("header", "runners", "warnings", "result"):
        st.session_state.pop(k, None)


def race_title(h: dict[str, Any]) -> str:
    bits = [str(h.get("track", "")).strip(),
            f"R{h.get('race_no')}" if h.get("race_no") else "",
            str(h.get("race_name", "")).strip()]
    return " · ".join(x for x in bits if x) or "Parsed thoroughbred race"


def race_subtitle(h: dict[str, Any]) -> str:
    going = str(h.get("going", ""))
    if h.get("going_rating"):
        going = f"{going} {h['going_rating']}"
    bits = [f"{h.get('distance_m')}m" if h.get("distance_m") else "",
            h.get("surface", ""), going, h.get("race_type", ""),
            h.get("prize", ""), h.get("date", "")]
    return " | ".join(str(x) for x in bits if x)


def wps(r: dict[str, Any], key: str) -> str:
    return f"{r.get(key+'_wins',0)}-{r.get(key+'_places',0)}-{r.get(key+'_starts',0)}"


def parsed_df(runners: list[dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for r in runners:
        runs = r.get("recent_runs", [])
        last = runs[0] if runs else {}
        rows.append({
            "Tab": r.get("tab"), "Horse": r.get("horse"),
            "Scr": "Y" if r.get("scratched") else "",
            "WT": r.get("wt"), "BP": r.get("bp") or None,
            "Jockey": r.get("jockey", ""), "JRat": r.get("jrat"),
            "Trainer": r.get("trainer", ""), "TRat": r.get("trat"),
            "Clm": r.get("claim") or None,
            "TAB$": r.get("tab_odds"), "BF$": r.get("bf_odds"),
            "Form": r.get("form", ""),
            "Jky L50%": round(100 * r.get("jky_win", 0), 1),
            "Trn L50%": round(100 * r.get("trn_win", 0), 1),
            "Career": wps(r, "Car"), "12m": wps(r, "M12"), "Crs": wps(r, "Crs"),
            "Dist": wps(r, "Dist"), "C&D": wps(r, "CrsDist"),
            "Good": wps(r, "Good"), "Soft": wps(r, "Soft"), "Heavy": wps(r, "Heavy"),
            "1st-up": wps(r, "FU"), "DLS": r.get("dslr"),
            "Last Fin": last.get("finish"), "Last Mgn": last.get("margin"),
            "Last SP": last.get("sp"), "Last Trk": last.get("track", ""),
            "Last Dist": last.get("distance"), "Runs": len(runs),
        })
    return pd.DataFrame(rows)


def prediction_df(active: list[dict[str, Any]], result: dict[str, Any]) -> pd.DataFrame:
    rows = []
    for rank, i in enumerate(result["order"], start=1):
        r = active[i]
        rows.append({
            "Pred": rank, "Tab": r.get("tab"), "Horse": r.get("horse"),
            "BP": r.get("bp") or None, "WT": r.get("wt"),
            "Jockey": r.get("jockey", ""),
            "Market%": 100 * result["p_mkt"][i], "Fund%": 100 * result["p_fund"][i],
            "Win%": 100 * result["p_win"][i], "Top3%": 100 * result["top3"][i],
            "E[pos]": result["exp_pos"][i],
            "Fair$": 1.0 / max(result["p_win"][i], 1e-9),
            "TAB$": r.get("tab_odds"), "BF$": r.get("bf_odds"),
            "EV": result["ev_win"][i], "F/M": result["fund_mkt_ratio"][i],
            "Conf": int(result["conf"][i]), "Recommendation": result["recs"][i],
        })
    return pd.DataFrame(rows)


for k, v in (("header", {}), ("runners", []), ("warnings", []), ("result", None)):
    st.session_state.setdefault(k, v)

st.markdown(
    '<div class="hero"><h1 style="margin:0">🐎 HorsePredictor</h1>'
    '<div class="muted">Racing &amp; Sports thoroughbred Enhanced Form parser · '
    'Shin de-vig + Benter market/form blend + Plackett–Luce simulation</div></div>',
    unsafe_allow_html=True)

with st.sidebar:
    st.header("Model settings")
    alpha = st.slider(
        "Market weight α", .50, .98, float(model.MARKET_ALPHA), .01,
        help="Benter blend: how much weight the betting market gets against the "
             "form model. 0.85 is the validated optimum.")
    bf_w = st.slider(
        "Betfair weight within market", .0, 1.0, float(model.BF_WEIGHT), .05,
        help="Exchange price vs Shin-corrected TAB price inside the market term.")
    sims = st.slider("Finishing-order simulations", 5_000, 60_000,
                     int(model.SIMS), 5_000)
    extended = st.checkbox(
        "Extended fundamentals", value=True,
        help="Adds course & distance record, record at this run of the "
             "preparation, and the jockey/trainer partnership strike rate. "
             "Their weights are informed priors, not fitted values — there is no "
             "labelled horse dataset here to fit against. Untick to fall back to "
             "the ten originally calibrated features and compare.")
    if extended:
        st.caption("13 fundamentals active — 10 calibrated + 3 added "
                   "(**unvalidated weights**).")
    else:
        st.caption("10 calibrated fundamentals active.")
    seed = st.number_input("Random seed", value=42, step=1)
    st.divider()
    st.subheader("Place Finder criteria")
    place_max = st.slider(
        "Maximum selections", 1, 5, pf.DEFAULT_MAX_PICKS, 1,
        help="Hard cap on how many horses are flagged. Anything that passes every "
             "filter beyond the cap is shown as a reserve.")
    place_market_top = st.slider(
        "Market must rate inside its top", 1, 6, pf.DEFAULT_MARKET_TOP, 1,
        help="The biggest single lift. Requiring the market to agree raised "
             "precision from 33.3% (model top 3 alone) to 47.1% at the same ~3 "
             "picks per race. Set to the field size to switch this off.")
    place_top_n = st.slider(
        "Model shortlist to draw from", 3, 8, pf.DEFAULT_TOP_N, 1,
        help="The pool the consensus is taken from. The model's ordering inside "
             "its own top 5 proved close to noise, so widening this and letting "
             "the market choose beat narrowing it.")
    place_fm_max = st.slider(
        "Exclude F/M at or above", 1.0, 5.0, pf.DEFAULT_FM_MAX, 0.1,
        help="Horses the form model rates far above the market. On the "
             "Wolverhampton card, F/M ≥ 2.0 placed 1 time in 20. NOTE: this "
             "threshold is coupled to the feature set — turning on the extended "
             "fundamentals raises F/M for horses the form model likes, so 2.0 "
             "bites harder than it did on the ten calibrated features.")
    place_shrink = st.slider(
        "Shrink toward base rate", 0.0, 0.5, pf.DEFAULT_SHRINK, 0.05,
        help="Pulls place probabilities toward places/runners. Corrects the "
             "observed over-confidence at the top and under-confidence at the "
             "bottom without fitting a curve to a tiny sample. 0 = raw model.")
    st.divider()
    st.caption(
        "Fundamentals: official rating → weight carried → career strike rate → "
        "distance record → going record → jockey/trainer form → last-start "
        "finish → freshness → R&S ratings.")
    st.caption("Prediction is probabilistic decision support, not a guaranteed outcome.")

paste_tab, parsed_tab, pred_tab, place_tab, results_tab, explain_tab, method_tab = st.tabs(
    ["1 · Paste Data", "2 · Parsed Data", "3 · Prediction", "4 · Place Finder",
     "5 · Results & Tuning", "6 · Explanations", "Method"])

st.session_state.setdefault("ledger", rl.empty_ledger())

with paste_tab:
    st.subheader("Paste the full Racing & Sports thoroughbred Enhanced Form page")
    st.caption("Select all on the Enhanced Form page, copy, then paste below. "
               "Scratchings are detected from the field table.")
    st.text_area("Paste", key="paste_input", height=340,
                 label_visibility="collapsed",
                 placeholder="Paste thoroughbred Enhanced Form text here…")
    c1, c2, _ = st.columns([1, 1, 4])
    with c1:
        parse_clicked = st.button("Parse race ▶", type="primary", width="stretch")
    with c2:
        st.button("Clear", on_click=reset_app, width="stretch")

    if parse_clicked:
        raw = st.session_state.get("paste_input", "")
        if not raw.strip():
            st.warning("Nothing pasted yet.")
        else:
            try:
                h, rs, ws = parser.parse(raw)
            except Exception as exc:                      # noqa: BLE001
                st.error(f"Parse failed: {exc}")
            else:
                st.session_state.update(header=h, runners=rs, warnings=ws, result=None)
                active = [r for r in rs if not r.get("scratched")]
                if not rs:
                    st.error("No runners found. Make sure the field table was included.")
                else:
                    st.success(f"Parsed {len(rs)} runners "
                               f"({len(active)} active, {len(rs)-len(active)} scratched).")
                for w in ws:
                    st.warning(w)

with parsed_tab:
    runners = st.session_state["runners"]
    if not runners:
        st.info("Parse a race in **1 · Paste Data** first.")
    else:
        h = st.session_state["header"]
        st.subheader(race_title(h))
        st.caption(race_subtitle(h))
        for w in st.session_state["warnings"]:
            st.warning(w)
        st.dataframe(parsed_df(runners), width="stretch", hide_index=True)

with pred_tab:
    runners = st.session_state["runners"]
    if not runners:
        st.info("Parse a race in **1 · Paste Data** first.")
    else:
        h = st.session_state["header"]
        active = [r for r in runners if not r.get("scratched")]
        if len(active) < 2:
            st.error(f"Only {len(active)} active runner(s); nothing to model.")
        else:
            if st.button("Predict race ▶", type="primary", key="predict"):
                with st.spinner(f"Running {int(sims):,} simulations…"):
                    result = model.predict(active, h, alpha=float(alpha),
                                           bf_weight=float(bf_w),
                                           sims=int(sims), seed=int(seed),
                                           extended=bool(extended))
                    st.session_state["result"] = result
            result = st.session_state["result"]
            if result:
                df = prediction_df(active, result)
                top = df.iloc[0]
                st.markdown(
                    f'<div class="pick"><div class="muted">MODEL TOP PICK</div>'
                    f'<h2 style="margin:.15rem 0">#{top["Tab"]} · {top["Horse"]}</h2>'
                    f'<b>Win {top["Win%"]:.1f}%</b> · Top 3 {top["Top3%"]:.1f}% · '
                    f'Fair ${top["Fair$"]:.2f} · EV {top["EV"]:.2f} · '
                    f'Confidence {top["Conf"]}/9</div>', unsafe_allow_html=True)
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Overall confidence", f"{result['overall_conf']}/9")
                m2.metric("Market weight α", f"{alpha:.2f}")
                m3.metric("TAB overround", f"{result['overround_tab']:.3f}")
                m4.metric("Active field", str(len(active)))
                st.dataframe(
                    df, width="stretch", hide_index=True,
                    column_config={
                        "Win%": st.column_config.ProgressColumn(
                            "Win%", format="%.1f%%", min_value=0.0,
                            max_value=float(max(df["Win%"].max(), 1))),
                        "Market%": st.column_config.NumberColumn(format="%.1f%%"),
                        "Fund%": st.column_config.NumberColumn(format="%.1f%%"),
                        "Top3%": st.column_config.NumberColumn(format="%.1f%%"),
                        "Fair$": st.column_config.NumberColumn(format="$%.2f"),
                        "EV": st.column_config.NumberColumn(format="%.2f"),
                        "F/M": st.column_config.NumberColumn(format="%.2fx"),
                        "E[pos]": st.column_config.NumberColumn(format="%.2f"),
                    })
                st.caption(
                    f"Shin insider fraction z = {result['shin_z']:.4f} · "
                    f"Betfair book = {result['book_bf']:.3f} · "
                    f"{result['sims']:,} simulations. "
                    "EV is win probability × Betfair price; 1.00 is break-even.")
                st.download_button(
                    "Download prediction as CSV",
                    df.to_csv(index=False).encode("utf-8"),
                    file_name=f"{h.get('track','race')}_R{h.get('race_no','')}_prediction.csv",
                    mime="text/csv")

with place_tab:
    runners = st.session_state["runners"]
    result = st.session_state["result"]
    if not runners:
        st.info("Parse a race in **1 · Paste Data** first.")
    elif not result:
        st.info("Run the prediction in **3 · Prediction** first — the place table is "
                "built from the same finishing-order simulation.")
    else:
        active = [r for r in runners if not r.get("scratched")]
        st.subheader("Place Finder")

        # Which race, before which runner. A full-strength model could not
        # out-rank the market, but the gap between an easy race and a hard one
        # is large and needs no model at all - so it is shown first.
        quality = rq.assess(active, result)
        box = {rq.PRIME: st.success, rq.STRONG: st.info,
               rq.FAIR: st.warning, rq.SKIP: st.error}[quality["tier"]]
        box(rq.headline(quality))
        with st.expander(f"Why this race is graded **{quality['label']}** — "
                         "and what that grade has been worth"):
            st.markdown(rq.detail(quality))
            st.markdown("**What each market pick has historically returned in "
                        "races of this grade:**")
            st.dataframe(
                rq.summary_table(quality).style.format({
                    "Historic win%": "{:.1f}%", "Historic place%": "{:.1f}%",
                    "Typical $": "${:.2f}"}),
                width="stretch", hide_index=True)
            st.caption(
                "These are **strike rates, not profit**. A favourite that places "
                "78% of the time is usually priced to place about 78% of the "
                "time; the grade tells you where the result is predictable, not "
                "where it is profitable. Skipping the red grade is the single "
                "biggest improvement available — it is a fifth of all races and "
                "the one where the favourite places only half the time.")

        auto = pf.places_paid(len(active))
        c1, c2 = st.columns([1, 3])
        places = c1.selectbox(
            "Places paid", [0, 1, 2, 3, 4],
            index=[0, 1, 2, 3, 4].index(auto),
            help="Defaults to standard terms: 3 places for 8+ runners, 2 for 5-7, "
                 "none under 5. Override if your bookmaker differs.")
        with st.expander("Optional: enter place odds to price the bets"):
            st.caption("Enter the place price for any runner you are considering. "
                       "Edge = adjusted place probability × your price − 1.")
            po_text = st.text_input(
                "Place odds as `tab:price` pairs", value="",
                placeholder="e.g. 4:2.10, 7:3.40, 11:5.00")
        place_odds = {}
        for chunk in po_text.replace(";", ",").split(","):
            if ":" in chunk:
                t, _, o = chunk.partition(":")
                try:
                    place_odds[int(t.strip())] = float(o.strip())
                except ValueError:
                    pass
        if po_text and not place_odds:
            st.warning("Could not read any `tab:price` pairs from that text.")

        table, meta = pf.build(active, result, top_n=int(place_top_n),
                               market_top=int(place_market_top),
                               max_picks=int(place_max),
                               fm_max=float(place_fm_max), shrink=float(place_shrink),
                               places=int(places), place_odds=place_odds or None)
        if meta["no_place_market"]:
            st.warning(pf.summary_line(meta))
        else:
            st.caption(pf.summary_line(meta))

        picks = table[table["Status"] == pf.STATUS_QUALIFY]
        if len(picks):
            names = " · ".join(
                f'#{int(row["Tab"])} {row["Horse"]} ({row["Place% (adj)"]:.1f}%)'
                for _, row in picks.iterrows())
            st.markdown(
                f'<div class="pick"><div class="muted">PLACE SELECTIONS</div>'
                f'<div style="font-size:1.05rem;margin-top:.3rem">{names}</div></div>',
                unsafe_allow_html=True)
        else:
            st.info("No runner met all the criteria in this race. That is a real "
                    "answer, not a failure - a race where the model and the market "
                    "disagree is one to leave alone.")

        st.dataframe(pf.style(table), width="stretch", hide_index=True)
        st.caption(
            "🟩 **SELECTION** — in the model's shortlist, rated inside the market's "
            "top group, past the F/M filter, within the cap.  "
            "🟦 **reserve** — passed everything but fell outside the cap.  "
            "🟨 **excluded** — either the market does not rate it, or the form model "
            "rates it far above the market (F/M), which was a trap for places.  "
            "⬜ outside the model shortlist.  "
            "**Fair place $** is 1 ÷ adjusted place probability — only bet when the "
            "actual place market pays more than that.")
        st.warning(
            "These criteria were derived from **one meeting — six races, 16 "
            "placegetters**. Measured on that card the consensus rule hit "
            "**47.1%** of its selections, against 33.3% for the model's own top 3 "
            "and 44.4% for the market's top 3 alone. That is a sensible working "
            "method, not a proven edge — the sample is far too small to be sure.")
        c1, c2 = st.columns(2)
        with c1:
            st.download_button(
                "Download place table as CSV",
                table.to_csv(index=False).encode("utf-8"),
                file_name=f"{st.session_state['header'].get('track','race')}"
                          f"_R{st.session_state['header'].get('race_no','')}_places.csv",
                mime="text/csv", width="stretch")
        with c2:
            if st.button("📋 Log this race for results tracking", width="stretch"):
                snap = rl.snapshot(st.session_state["header"], active, result, table, meta)
                st.session_state["ledger"] = rl.merge(st.session_state["ledger"], snap)
                st.success(f"Logged **{rl.race_id(st.session_state['header'])}**. "
                           "Enter the finishing order in **5 · Results & Tuning** "
                           "once the race is run.")

with results_tab:
    st.subheader("Results & Tuning")
    ledger = st.session_state["ledger"]

    with st.expander("Load or save your ledger", expanded=ledger.empty):
        st.caption(
            "Streamlit Cloud wipes its filesystem on every restart, so **the "
            "download is the real save**. Keep the CSV and re-upload it next session.")
        up = st.file_uploader("Upload a saved ledger CSV", type=["csv"], key="ledger_up")
        if up is not None:
            try:
                loaded = pd.read_csv(up)
                core_missing = [c for c in ("race_id", "tab", "model_rank")
                                if c not in loaded.columns]
                if core_missing:
                    st.error("That does not look like a HorsePredictor ledger — "
                             f"missing {', '.join(core_missing)}.")
                else:
                    added = [c for c in rl.LEDGER_COLUMNS if c not in loaded.columns]
                    st.session_state["ledger"] = rl.conform(loaded)
                    ledger = st.session_state["ledger"]
                    st.success(f"Loaded {ledger['race_id'].nunique()} race(s), "
                               f"{len(ledger)} runner rows.")
                    if added:
                        st.info(f"That ledger predates {len(added)} column(s) added "
                                "since — they are blank for those races, which only "
                                "limits what can be re-derived from them.")
            except Exception as exc:                       # noqa: BLE001
                st.error(f"Could not read that CSV: {exc}")
        if not ledger.empty:
            st.download_button(
                "💾 Download ledger CSV", ledger.to_csv(index=False).encode("utf-8"),
                file_name="horsepredictor_ledger.csv", mime="text/csv",
                width="stretch")

    if ledger.empty:
        st.info("Nothing logged yet. Run a prediction, open **4 · Place Finder**, "
                "and press **Log this race for results tracking**.")
    else:
        pending = sorted(ledger[ledger["placed"].isna()]["race_id"].unique())
        st.markdown("### Enter a finishing order")
        if not pending:
            st.success("Every logged race has a result recorded.")
        else:
            rid = st.selectbox("Race awaiting a result", pending)
            fld = int(ledger[ledger["race_id"] == rid]["field_size"].iloc[0])
            pl = int(ledger[ledger["race_id"] == rid]["places_paid"].iloc[0])
            st.caption(f"{fld} runners, {pl} places paid. Enter the finishing order as "
                       f"**tab numbers, winner first** — the first {pl} are enough.")
            fin_txt = st.text_input("Finishing order", key=f"fin_{rid}",
                                    placeholder="e.g. 6, 5, 1")
            if st.button("Save result", type="primary"):
                try:
                    fins = [int(x) for x in fin_txt.replace(";", ",").split(",") if x.strip()]
                except ValueError:
                    fins = []
                valid = set(ledger[ledger["race_id"] == rid]["tab"].astype(int))
                unknown = [t for t in fins if t not in valid]
                if not fins:
                    st.error("Could not read any tab numbers from that.")
                elif unknown:
                    st.error(f"Tab number(s) {unknown} are not in that race's field.")
                elif len(fins) < pl:
                    st.error(f"That race pays {pl} places — enter at least {pl} finishers.")
                elif len(set(fins)) != len(fins):
                    st.error("The same tab number appears twice.")
                else:
                    st.session_state["ledger"] = rl.record_result(ledger, rid, fins)
                    ledger = st.session_state["ledger"]
                    st.success(f"Recorded {rid}: {' - '.join(str(t) for t in fins)}")

        perf = rl.performance(ledger)
        state, msg = rl.readiness(perf.get("races", 0))
        st.divider()
        st.markdown("### Measured performance")
        {"empty": st.info, "thin": st.info, "monitor": st.warning,
         "tune": st.success, "confident": st.success}[state](msg)

        if perf.get("races", 0):
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Settled races", perf["races"])
            m2.metric("Selections", perf["selections"])
            if "precision" in perf:
                m3.metric("Selections that placed", f"{perf['precision']:.1%}",
                          f"base rate {perf['base_rate']:.1%}")
                m4.metric("95% interval",
                          f"{perf['ci_low']:.0%}–{perf['ci_high']:.0%}")
            st.caption(
                f"Place-probability log-loss **{perf['logloss']:.4f}**, Brier "
                f"**{perf['brier']:.4f}**. Predicted place rate "
                f"{perf['mean_predicted']:.1%} vs actual {perf['actual_rate']:.1%}.")
            if "precision" in perf and perf["ci_low"] <= perf["base_rate"]:
                st.caption("⚠️ The interval still includes the base rate — on this much "
                           "data the selections are not yet distinguishable from "
                           "picking at random.")
            if perf["calibration"]:
                st.markdown("**Calibration** — does a 40% call place 40% of the time?")
                cal = pd.DataFrame(perf["calibration"])
                cal["predicted"] = (100 * cal["predicted"]).round(1)
                cal["actual"] = (100 * cal["actual"]).round(1)
                st.dataframe(cal.rename(columns={
                    "band": "Predicted band", "n": "n",
                    "predicted": "Mean predicted %", "actual": "Actual placed %"}),
                    width="stretch", hide_index=True)

        st.divider()
        st.markdown("### Threshold tuning")
        s_df = rl.settled(ledger)
        n_races = s_df["race_id"].nunique() if not s_df.empty else 0
        if n_races < rl.MIN_RACES_TO_TUNE:
            st.info(
                f"**Locked — {n_races} of {rl.MIN_RACES_TO_TUNE} races.** Tuning "
                "thresholds on fewer races fits noise and would make the app worse, "
                "not better. Keep logging; the ledger is already earning its keep "
                "by monitoring for a broken rule above.")
            with st.expander("How much data does tuning actually need?"):
                st.markdown(
                    "| Question | Races |\n|---|---|\n"
                    "| Has a rule quietly broken (47% → 20%)? | ~17 |\n"
                    "| Does it beat a dart throw? | ~45 |\n"
                    "| Does consensus really beat model top-3? | ~71 |\n"
                    "| Is 47% genuinely better than 40%? | ~273 |\n\n"
                    "Derived from a two-proportion power calculation at 80% power. "
                    "The core model's feature weights are **not** tuned here — those "
                    "were calibrated on roughly 1,700 races and cannot be moved by "
                    "logging finishing positions.")
        else:
            if st.button("Run tuner", type="primary"):
                with st.spinner("Cross-validating threshold combinations…"):
                    st.session_state["tuned"] = rl.tune(s_df)
            tuned = st.session_state.get("tuned")
            if tuned and tuned.get("ok"):
                p = tuned["params"]
                c1, c2, c3 = st.columns(3)
                c1.metric("Current default", f"{tuned['current']['precision']:.1%}")
                c2.metric("Suggested (in-sample)", f"{tuned['in_sample']['precision']:.1%}")
                c3.metric("Suggested (cross-validated)", f"{tuned['cv_precision']:.1%}",
                          help="The only number that means anything: thresholds chosen "
                               "on the other races, then applied to the held-out one.")
                st.markdown(
                    f"**Suggested thresholds** — model shortlist **{p['top_n']}**, "
                    f"market top **{p['market_top']}**, F/M below **{p['fm_max']}**, "
                    f"cap **{p['max_picks']}**.")
                gap = tuned["in_sample"]["precision"] - tuned["cv_precision"]
                if gap > 0.05:
                    st.warning(
                        f"In-sample beats cross-validated by {gap:.1%}. That gap **is** "
                        "the over-fitting — trust the cross-validated figure, not the "
                        "headline.")
                if tuned["cv_precision"] <= tuned["current"]["precision"]:
                    st.info("Cross-validated tuning does **not** beat the current "
                            "defaults. Leave them alone.")
                else:
                    st.success("Cross-validated tuning beats the defaults. Set these "
                               "values in the sidebar if you want to adopt them.")
                if not tuned["trustworthy"]:
                    st.caption("Sample is still below the confident threshold — treat "
                               "this as provisional.")

        with st.expander("The full ledger"):
            st.dataframe(ledger, width="stretch", hide_index=True)

with explain_tab:
    result = st.session_state["result"]
    runners = st.session_state["runners"]
    if not result:
        st.info("Run the prediction first.")
    else:
        active = [r for r in runners if not r.get("scratched")]
        st.subheader(f"Runner explanations · overall confidence {result['overall_conf']}/9")
        for rank, i in enumerate(result["order"], start=1):
            r = active[i]
            with st.expander(
                    f"{rank}. #{r.get('tab')} {r.get('horse')} — "
                    f"Win {result['p_win'][i]*100:.1f}% · {result['recs'][i]}",
                    expanded=rank <= 3):
                st.text(result["why"][i])

with method_tab:
    st.markdown("""
### How HorsePredictor works

**1 · Field parsing.** The field table is read through its **header row**, not by
column position. R&S ship several column layouts and leave cells blank (a horse
with no declared jockey, a meeting with no Bet365 column), so a positional reader
mis-assigns columns the moment a blank appears. A row needs only a tab number and
a horse name to produce a runner; weight, barrier, jockey, ratings and price are
all best-effort. **A gap in the tab sequence raises a warning** rather than
silently shrinking the field.

**2 · Market.** TAB prices are de-vigged with **Shin (1993)**, which strips the
bookmaker margin *and* the favourite–longshot bias, solved by bisection on the
insider fraction *z*. Betfair prices are normalised directly, then blended with
the Shin-corrected TAB prices.

**3 · Fundamentals.** A **Bolton & Chapman (1986)** conditional logit over
z-scored features: official rating, weight carried, career strike rate, distance
placing record, record on today's going, jockey and trainer L50 strike rates,
last-start finish, freshness, and the R&S jockey/trainer ratings.

**4 · Blend.** Market and fundamental probabilities are combined in log-odds
space per **Benter (1994)**, at weight α.

**5 · Finishing order.** A discounted **Plackett–Luce** Monte Carlo
(**Lo, Bacon-Shone & Busche, 1995** discounts) gives place and top-3
probabilities plus expected finishing position.

**Confidence** per runner combines position-distribution sharpness (45%),
market/form agreement (35%) and data depth (20%).

---
*Prediction is probabilistic decision support, not a guaranteed outcome.
Gamble responsibly — Gambling Help 1800 858 858.*
""")
