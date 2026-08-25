"""HarnessPredict - Streamlit UI for Racing & Sports harness Enhanced Form."""
from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st

import harness_model as model
import harness_parser as parser

st.set_page_config(page_title="HarnessPredict", page_icon="🏇",
                   layout="wide", initial_sidebar_state="expanded")
st.markdown("""
<style>
.block-container {padding-top: 1.15rem; padding-bottom: 3rem;}
div[data-testid="stMetric"] {border:1px solid rgba(128,128,128,.22);border-radius:.7rem;padding:.55rem .8rem;}
.hero {padding:1rem 1.1rem;border-radius:.85rem;background:linear-gradient(135deg,rgba(37,99,235,.18),rgba(0,0,0,0));border:1px solid rgba(128,128,128,.22);margin-bottom:1rem;}
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
    return " · ".join(x for x in bits if x) or "Parsed harness race"


def race_subtitle(h: dict[str, Any]) -> str:
    bits = [f"{h.get('distance_m')}m" if h.get("distance_m") else "",
            h.get("surface", ""), h.get("going", ""),
            f"Age {h['race_type']}" if h.get("race_type") else "",
            h.get("prize", ""), h.get("date", "")]
    return " | ".join(str(x) for x in bits if x)


def wps(r: dict[str, Any], key: str) -> str:
    return f"{r.get(key+'_wins',0)}-{r.get(key+'_places',0)}-{r.get(key+'_starts',0)}"


def mmss(secs: Any) -> str:
    try:
        s = float(secs)
    except (TypeError, ValueError):
        return ""
    if s <= 0:
        return ""
    return f"{int(s // 60)}:{s % 60:05.2f}"


def parsed_df(runners: list[dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for r in runners:
        runs = r.get("recent_runs", [])
        last = runs[0] if runs else {}
        rows.append({
            "Tab": r.get("tab"), "Horse": r.get("horse"),
            "Scr": "Y" if r.get("scratched") else "", "Gate": r.get("gate"),
            "Driver": r.get("driver", ""), "Trainer": r.get("trainer", ""),
            "TAB$": r.get("tab_odds"), "BF$": r.get("bf_odds"),
            "Form": r.get("form", ""),
            "Dri L50%": round(100 * r.get("driver_win", 0), 1),
            "Trn L50%": round(100 * r.get("trainer_win", 0), 1),
            "D/T%": round(100 * r.get("driver_trainer_win", 0), 1),
            "Career": wps(r, "career"), "Crs": wps(r, "course"),
            "Dist": wps(r, "distance"), "C&D": wps(r, "course_distance"),
            "1st-up": wps(r, "first_up"), "DLS": r.get("dls"),
            "Best IMR": mmss(r.get("best_imr")), "OHR": r.get("latest_ohr"),
            "Last Fin": last.get("finish"), "Last Mgn": last.get("margin"),
            "Last SP": last.get("sp"), "Last HCP": last.get("hcp", ""),
            "Last Trk": last.get("track", ""), "Runs": len(runs),
        })
    return pd.DataFrame(rows)


def speed_map_df(result: dict[str, Any]) -> pd.DataFrame:
    active = result["runners"]
    rows = []
    for idx in result["early_order"]:
        r = active[idx]
        rows.append({
            "Early Rank": int(result["early_rank"][idx]), "Tab": r.get("tab"),
            "Horse": r.get("horse"), "Gate": r.get("gate"),
            "Driver": r.get("driver", ""),
            "Pace Score": float(result["components"]["pace"][idx]),
            "Win%": 100 * result["p_win"][idx],
        })
    return pd.DataFrame(rows)


def prediction_df(result: dict[str, Any]) -> pd.DataFrame:
    active = result["runners"]
    rows = []
    for rank, i in enumerate(result["order"], start=1):
        r = active[i]
        rows.append({
            "Pred": rank, "Tab": r.get("tab"), "Horse": r.get("horse"),
            "Gate": r.get("gate"), "Driver": r.get("driver", ""),
            "Early": int(result["early_rank"][i]),
            "Market%": 100 * result["p_mkt"][i], "Fund%": 100 * result["p_fund"][i],
            "Win%": 100 * result["p_win"][i], "Top2%": 100 * result["top2"][i],
            "Top3%": 100 * result["top3"][i], "E[pos]": result["exp_pos"][i],
            "Fair$": result["fair"][i], "TAB$": r.get("tab_odds"),
            "BF$": r.get("bf_odds"), "EV": result["ev_win"][i],
            "Conf": int(result["conf"][i]), "Recommendation": result["recs"][i],
        })
    return pd.DataFrame(rows)


for k, v in (("header", {}), ("runners", []), ("warnings", []), ("result", None)):
    st.session_state.setdefault(k, v)

st.markdown(
    '<div class="hero"><h1 style="margin:0">🏇 HarnessPredict</h1>'
    '<div class="muted">Racing &amp; Sports harness Enhanced Form parser · '
    'mile-rate speed, tactical pace map and market-blended probabilities</div></div>',
    unsafe_allow_html=True)

with st.sidebar:
    st.header("Model settings")
    alpha = st.slider(
        "Market weight α", .20, .95, float(model.MARKET_ALPHA), .01,
        help="How much weight the betting market gets against the form model.")
    sims = st.slider("Finishing-order simulations", 5_000, 60_000, 20_000, 5_000)
    seed = st.number_input("Random seed", value=42, step=1)
    st.divider()
    st.caption(
        "Fundamentals: mile-rate speed → tactical pace/gate → track & distance "
        "→ recent form → driver/trainer connections → official rating → "
        "sectionals → steward reliability → freshness.")
    st.caption("Prediction is probabilistic decision support, not a guaranteed outcome.")

paste_tab, parsed_tab, speed_tab, pred_tab, explain_tab, method_tab = st.tabs(
    ["1 · Paste Data", "2 · Parsed Data", "3 · Speed Map", "4 · Prediction",
     "5 · Explanations", "Method"])

with paste_tab:
    st.subheader("Paste the full Racing & Sports harness Enhanced Form page")
    st.caption("Select all on the Enhanced Form page, copy, then paste below. "
               "Scratchings are detected from the field table.")
    st.text_area("Paste", key="paste_input", height=340,
                 label_visibility="collapsed",
                 placeholder="Paste harness Enhanced Form text here…")
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


def _run_prediction(runners, h):
    return model.predict(runners, h, alpha=float(alpha), sims=int(sims), seed=int(seed))


with speed_tab:
    runners = st.session_state["runners"]
    if not runners:
        st.info("Parse a race in **1 · Paste Data** first.")
    else:
        h = st.session_state["header"]
        if st.button("Build speed map ▶", type="primary", key="speedmap"):
            try:
                st.session_state["result"] = _run_prediction(runners, h)
            except ValueError as exc:
                st.error(str(exc))
        result = st.session_state["result"]
        if result:
            sdf = speed_map_df(result)
            leader = sdf.iloc[0]
            st.markdown(
                f'<div class="pick"><div class="muted">PROJECTED LEADER</div>'
                f'<h3 style="margin:.15rem 0">#{leader["Tab"]} · {leader["Horse"]} '
                f'(gate {leader["Gate"]})</h3>'
                f'<span class="muted">Driver {leader["Driver"]}</span></div>',
                unsafe_allow_html=True)
            st.dataframe(sdf, width="stretch", hide_index=True,
                         column_config={
                             "Win%": st.column_config.NumberColumn(format="%.1f%%"),
                             "Pace Score": st.column_config.NumberColumn(format="%.2f"),
                         })
            st.caption("Early rank blends recent in-running positions with the gate "
                       "draw. Harness races are won from the front more often than "
                       "not, so a clear leader is a real edge.")

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
                    st.session_state["result"] = _run_prediction(runners, h)
            result = st.session_state["result"]
            if result:
                df = prediction_df(result)
                top = df.iloc[0]
                st.markdown(
                    f'<div class="pick"><div class="muted">MODEL TOP PICK</div>'
                    f'<h2 style="margin:.15rem 0">#{top["Tab"]} · {top["Horse"]}</h2>'
                    f'<b>Win {top["Win%"]:.1f}%</b> · Top 3 {top["Top3%"]:.1f}% · '
                    f'Fair ${top["Fair$"]:.2f} · EV {top["EV"]:+.2f} · '
                    f'Confidence {top["Conf"]}/9</div>', unsafe_allow_html=True)
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Overall confidence", f"{result['overall_conf']}/9")
                m2.metric("Market weight α", f"{alpha:.2f}")
                m3.metric("Projected leader",
                          str(result["runners"][result["early_order"][0]]["horse"]))
                m4.metric("Active field", str(len(active)))
                st.dataframe(
                    df, width="stretch", hide_index=True,
                    column_config={
                        "Win%": st.column_config.ProgressColumn(
                            "Win%", format="%.1f%%", min_value=0.0,
                            max_value=float(max(df["Win%"].max(), 1))),
                        "Market%": st.column_config.NumberColumn(format="%.1f%%"),
                        "Fund%": st.column_config.NumberColumn(format="%.1f%%"),
                        "Top2%": st.column_config.NumberColumn(format="%.1f%%"),
                        "Top3%": st.column_config.NumberColumn(format="%.1f%%"),
                        "Fair$": st.column_config.NumberColumn(format="$%.2f"),
                        "EV": st.column_config.NumberColumn(format="%+.2f"),
                        "E[pos]": st.column_config.NumberColumn(format="%.2f"),
                    })
                st.caption(
                    "EV is win probability × listed TAB price, minus 1; 0.00 is "
                    "break-even. Betfair prices are parsed and shown, but the "
                    "market term currently uses the TAB price only.")
                st.download_button(
                    "Download prediction as CSV",
                    df.to_csv(index=False).encode("utf-8"),
                    file_name=f"{h.get('track','race')}_R{h.get('race_no','')}_harness.csv",
                    mime="text/csv")

with explain_tab:
    result = st.session_state["result"]
    if not result:
        st.info("Run the prediction first.")
    else:
        active = result["runners"]
        st.subheader(f"Runner explanations · overall confidence {result['overall_conf']}/9")
        for rank, i in enumerate(result["order"], start=1):
            r = active[i]
            with st.expander(
                    f"{rank}. #{r.get('tab')} {r.get('horse')} — "
                    f"Win {result['p_win'][i]*100:.1f}% · {result['recs'][i]}",
                    expanded=rank <= 3):
                st.write(result["why"][i])

with method_tab:
    st.markdown("""
### How HarnessPredict works

**1 · Field parsing.** The field table is read through its **header row**, not by
column position, and rows keep their empty cells. The previous harness parser
accepted only markdown pipe-table rows, so a plain select-all/copy from the live
page — which is tab separated — produced **zero runners**. Bookmaker columns are
located by header name so the money columns (`Tot $PM`, `Dri L50`, `Tra PM`)
can never be mistaken for a price. A gap in the tab sequence raises a warning
rather than silently shrinking the field.

**2 · Speed.** Each past run's **race mile rate adjustment** and **individual
mile rate (IMR)** are recency-weighted, with extra weight for runs at a similar
distance and at today's track.

**3 · Tactics.** Recent in-running and bell-lap positions build a pace profile,
adjusted for the gate draw. A runner projected to lead clearly gets a bonus; a
contested lead penalises both protagonists.

**4 · The rest.** Track/distance record, recent form, driver and trainer strike
rates plus the driver/horse and driver/trainer combination records, official
handicap rating (**OHR**), sectionals, steward-comment reliability, and a
freshness curve on days since last run.

**5 · Blend and simulate.** Market and fundamental probabilities are combined in
log-odds space at weight α, then a Plackett–Luce Monte Carlo produces place and
top-3 probabilities and expected finishing position.

**Barrier trials are excluded** from the form figures — only completed races count.

---
*Prediction is probabilistic decision support, not a guaranteed outcome.
Gamble responsibly — Gambling Help 1800 858 858.*
""")
