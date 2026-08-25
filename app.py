"""Streamlit app for Racing & Sports greyhound form parsing and prediction."""
from __future__ import annotations

import io
from typing import Any

import pandas as pd
import streamlit as st

import greyhound_model as model
import greyhound_parser as parser

st.set_page_config(page_title="GreyhoundPredictor", page_icon="🐕", layout="wide", initial_sidebar_state="expanded")
st.markdown("""
<style>
.block-container {padding-top: 1.15rem; padding-bottom: 3rem;}
div[data-testid="stMetric"] {border:1px solid rgba(128,128,128,.22);border-radius:.7rem;padding:.55rem .8rem;}
.hero {padding:1rem 1.1rem;border-radius:.85rem;background:linear-gradient(135deg,rgba(124,77,255,.18),rgba(0,0,0,0));border:1px solid rgba(128,128,128,.22);margin-bottom:1rem;}
.pick {padding:1rem 1.2rem;border:1px solid rgba(46,204,113,.35);border-radius:.8rem;background:rgba(46,204,113,.08);margin:.4rem 0 1rem 0;}
.muted {opacity:.72;font-size:.9rem;}
</style>
""", unsafe_allow_html=True)


def reset_app() -> None:
    st.session_state["paste_input"] = ""
    for k in ("header", "runners", "warnings", "result"):
        st.session_state.pop(k, None)


def race_title(h: dict[str, Any]) -> str:
    bits = [str(h.get("track", "")).strip(), f"R{h.get('race_no')}" if h.get("race_no") else "", str(h.get("race_name", "")).strip()]
    return " · ".join(x for x in bits if x) or "Parsed greyhound race"


def race_subtitle(h: dict[str, Any]) -> str:
    bits = [f"{h.get('distance_m')}m" if h.get("distance_m") else "", h.get("surface", ""), h.get("going", ""), h.get("race_type", ""), h.get("prize", ""), h.get("date", "")]
    return " | ".join(str(x) for x in bits if x)


def rec_str(r: dict[str, Any], prefix: str) -> str:
    return str(r.get(f"{prefix}_rec", "0-0-0"))


def parsed_df(runners: list[dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for r in runners:
        box = int(r.get("box", 0) or 0)
        bs = r.get("box_stats", {}).get(box, {}) if box else {}
        recent = r.get("recent_runs", []); last = recent[0] if recent else {}
        rows.append({
            "Tab": r.get("tab"), "Box": box or "", "Greyhound": r.get("horse"), "Scr": "Y" if r.get("scratched") else "",
            "Wt": r.get("weight"), "Trainer": r.get("trainer"), "TAB$": r.get("tab_odds"), "BF$": r.get("bf_odds"),
            "Form": r.get("form", ""), "Tra W%": 100*r.get("trainer_win", 0), "Tra P%": 100*r.get("trainer_place", 0),
            "Tra/Dist Best": r.get("tra_dist_best") or "", "Career": rec_str(r,"career"), "Course": rec_str(r,"course"),
            "Dist": rec_str(r,"distance"), "C&D": rec_str(r,"course_distance"), "DLS": r.get("dls"),
            "Box W-S": f"{bs.get('wins',0)}-{bs.get('starts',0)}", "Last Fin": last.get("finish", ""),
            "Last Mgn": last.get("margin", ""), "Last MRKΔ": last.get("mrk_delta", ""), "Last Split": last.get("first_split", ""),
            "Last Settle": last.get("settle_pos", ""), "Last Note": last.get("stewards", ""),
        })
    return pd.DataFrame(rows)


def prediction_df(result: dict[str, Any]) -> pd.DataFrame:
    rows = []; active = result["runners"]; c = result["components"]
    for rank, i in enumerate(result["order"], start=1):
        r = active[i]
        rows.append({
            "Pred": rank, "Tab": r.get("tab"), "Box": r.get("box"), "Greyhound": r.get("horse"),
            "Early": int(result["early_rank"][i]), "Market%": 100*result["p_mkt"][i], "Fund%": 100*result["p_fund"][i],
            "Win%": 100*result["p_win"][i], "Top2%": 100*result["top2"][i], "Top3%": 100*result["top3"][i],
            "E[pos]": result["exp_pos"][i], "Fair$": result["fair"][i], "BF$": r.get("bf_odds"), "EV": result["ev_win"][i],
            "Conf": result["conf"][i], "Speed Z": c["speed"][i], "Early Z": c["early"][i], "Box Z": c["box"][i],
            "C&D Z": c["trackdist"][i], "Form Z": c["form"][i], "Recommendation": result["recs"][i],
        })
    return pd.DataFrame(rows)


def speed_map_df(result: dict[str, Any]) -> pd.DataFrame:
    active = result["runners"]; rows = []
    for idx in result["early_order"]:
        r = active[idx]
        rows.append({"Early Rank": int(result["early_rank"][idx]), "Box": r.get("box"), "Greyhound": r.get("horse"),
                     "Early Score": result["components"]["early_raw"][idx], "Win%": 100*result["p_win"][idx]})
    return pd.DataFrame(rows)


def csv_bytes(df: pd.DataFrame) -> bytes:
    s = io.StringIO(); df.to_csv(s, index=False); return s.getvalue().encode("utf-8")

for k, v in (("header", {}), ("runners", []), ("warnings", []), ("result", None)):
    st.session_state.setdefault(k, v)

st.markdown('<div class="hero"><h1 style="margin:0">🐕 GreyhoundPredictor</h1><div class="muted">Racing & Sports Enhanced Form parser · greyhound speed-map + probability model</div></div>', unsafe_allow_html=True)

with st.sidebar:
    st.header("Model settings")
    alpha = st.slider("Market weight α", .30, .90, float(model.MARKET_ALPHA), .01,
                      help="Higher = final probabilities follow market odds more closely; lower = fundamentals matter more.")
    sims = st.select_slider("Finishing-order simulations", options=[5_000,10_000,20_000,30_000,50_000], value=20_000)
    seed = st.number_input("Random seed", 0, 999999, 42, 1)
    st.divider()
    st.caption("Greyhound fundamentals: adjusted speed/MRK → early pace → box history → track & distance → recent form → trainer → freshness.")
    st.caption("Prediction is probabilistic decision support, not a guaranteed outcome.")

paste_tab, parsed_tab, speed_tab, pred_tab, explain_tab, method_tab = st.tabs([
    "1 · Paste Data", "2 · Parsed Data", "3 · Speed Map", "4 · Prediction", "5 · Explanations", "Method"
])

with paste_tab:
    st.subheader("Paste the full Racing & Sports greyhound Enhanced Form page")
    st.caption("Select all on the Enhanced Form page, copy, then paste below. Scratches and reserves are handled automatically where the field table identifies them.")
    pasted = st.text_area("Race data", key="paste_input", height=430, placeholder="Paste greyhound Enhanced Form text here…", label_visibility="collapsed")
    c1,c2,c3 = st.columns([1,1,5])
    with c1: parse_clicked = st.button("Parse race ▶", type="primary", use_container_width=True)
    with c2: st.button("Clear", on_click=reset_app, use_container_width=True)
    if parse_clicked:
        if len(pasted.strip()) < 300:
            st.error("The pasted text looks too short. Paste the full Enhanced Form page.")
        else:
            try:
                h, rs, ws = parser.parse(pasted)
                st.session_state.update(header=h, runners=rs, warnings=ws, result=None)
                active = [r for r in rs if not r.get("scratched")]
                if active: st.success(f"Parsed {len(rs)} listed runners: {len(active)} active and {len(rs)-len(active)} scratched.")
                else: st.error("No active runners were found.")
            except Exception as exc: st.exception(exc)
    if st.session_state["runners"]: st.info(f"Current race: **{race_title(st.session_state['header'])}**")
    if st.session_state["warnings"]:
        with st.expander(f"Parser warnings ({len(st.session_state['warnings'])})"):
            for w in st.session_state["warnings"]: st.warning(w)

with parsed_tab:
    rs, h = st.session_state["runners"], st.session_state["header"]
    if not rs: st.info("Parse a race in **1 · Paste Data** first.")
    else:
        st.subheader(race_title(h)); st.caption(race_subtitle(h)); df = parsed_df(rs)
        st.dataframe(df, use_container_width=True, hide_index=True, height=520,
                     column_config={"Tra W%":st.column_config.NumberColumn(format="%.1f%%"),"Tra P%":st.column_config.NumberColumn(format="%.1f%%")})
        st.download_button("Download parsed data (CSV)", data=csv_bytes(df), file_name="greyhound_parsed.csv", mime="text/csv")

with speed_tab:
    rs, h = st.session_state["runners"], st.session_state["header"]
    if not rs: st.info("Parse a race first.")
    else:
        if st.button("Build speed map ▶", type="primary"):
            try: st.session_state["result"] = model.predict(rs, h, alpha=float(alpha), sims=int(sims), seed=int(seed))
            except Exception as exc: st.exception(exc)
        result = st.session_state["result"]
        if result:
            sdf = speed_map_df(result); leader = sdf.iloc[0]
            st.markdown(f"### Projected leader: **Box {leader['Box']} · {leader['Greyhound']}**")
            st.dataframe(sdf, use_container_width=True, hide_index=True,
                         column_config={"Win%": st.column_config.NumberColumn(format="%.1f%%"), "Early Score": st.column_config.NumberColumn(format="%.3f")})
            st.caption("The speed map uses recent settling position/start notes plus the current box profile. Raw first-split seconds are not compared blindly across different tracks/distances.")

with pred_tab:
    rs, h = st.session_state["runners"], st.session_state["header"]
    if not rs: st.info("Parse a race first.")
    else:
        st.subheader(race_title(h)); st.caption(race_subtitle(h))
        if st.button("Predict race ▶", type="primary", key="predict"):
            try:
                with st.spinner("Running greyhound ensemble and finishing-order simulation…"):
                    st.session_state["result"] = model.predict(rs, h, alpha=float(alpha), sims=int(sims), seed=int(seed))
            except Exception as exc: st.exception(exc)
        result = st.session_state["result"]
        if result:
            df = prediction_df(result); winner = df.iloc[0]
            st.markdown(f'<div class="pick"><div class="muted">MODEL TOP PICK</div><h2 style="margin:.1rem 0">Box {winner["Box"]} · {winner["Greyhound"]}</h2><b>Win {winner["Win%"]:.1f}%</b> · Top 3 {winner["Top3%"]:.1f}% · Fair ${winner["Fair$"]:.2f} · EV {winner["EV"]:+.2f} · Confidence {int(winner["Conf"])}/9</div>', unsafe_allow_html=True)
            m1,m2,m3,m4 = st.columns(4)
            m1.metric("Overall confidence", f"{result['overall_conf']}/9"); m2.metric("Market weight", f"{alpha:.2f}")
            m3.metric("Projected leader", str(result["runners"][result["early_order"][0]]["horse"])); m4.metric("Active field", str(len(result["runners"])))
            st.dataframe(df, use_container_width=True, hide_index=True, height=520, column_config={
                "Market%":st.column_config.NumberColumn(format="%.1f%%"),"Fund%":st.column_config.NumberColumn(format="%.1f%%"),
                "Win%":st.column_config.ProgressColumn(format="%.1f%%",min_value=0,max_value=100),"Top2%":st.column_config.NumberColumn(format="%.1f%%"),
                "Top3%":st.column_config.NumberColumn(format="%.1f%%"),"E[pos]":st.column_config.NumberColumn(format="%.2f"),
                "Fair$":st.column_config.NumberColumn(format="$%.2f"),"BF$":st.column_config.NumberColumn(format="$%.2f"),"EV":st.column_config.NumberColumn(format="%+.2f"),
                "Speed Z":st.column_config.NumberColumn(format="%+.2f"),"Early Z":st.column_config.NumberColumn(format="%+.2f"),"Box Z":st.column_config.NumberColumn(format="%+.2f"),
                "C&D Z":st.column_config.NumberColumn(format="%+.2f"),"Form Z":st.column_config.NumberColumn(format="%+.2f"),})
            st.download_button("Download prediction (CSV)", data=csv_bytes(df), file_name="greyhound_prediction.csv", mime="text/csv")

with explain_tab:
    result = st.session_state["result"]
    if not result: st.info("Run the prediction first.")
    else:
        st.subheader(f"Runner explanations · confidence {result['overall_conf']}/9")
        for rank, i in enumerate(result["order"], start=1):
            r = result["runners"][i]
            with st.expander(f"{rank}. Box {r.get('box')} · {r.get('horse')} — Win {result['p_win'][i]*100:.1f}% · {result['recs'][i]}", expanded=rank<=3):
                st.markdown(f"**{result['recs'][i]}**"); st.write(result["why"][i]); c = result["components"]
                st.dataframe(pd.DataFrame([{"Speed Z":c["speed"][i],"Early Z":c["early"][i],"Box Z":c["box"][i],"Track/Dist Z":c["trackdist"][i],
                                            "Form Z":c["form"][i],"Trainer Z":c["trainer"][i],"Freshness Z":c["freshness"][i]}]), hide_index=True, use_container_width=True)

with method_tab:
    st.subheader("How the greyhound model works")
    st.markdown("""
1. **Field parsing** — extracts the race, active runners, scratches/reserves, weights, trainers and prices.
2. **Adjusted speed** — recent R&S **BOM Time Adj** and **MRK delta** are recency-weighted, with extra weight for the same/similar distance and track.
3. **Early pace / speed map** — settling positions and steward start notes estimate early position. The current box profile is added and a mild pace-clash adjustment is applied.
4. **Box model** — uses each dog's own historical win/place/start record from the **current assigned box**, shrunk toward a broad baseline when the sample is small.
5. **Track & distance** — Course, Distance and Course & Distance records plus trainer/distance best time where available.
6. **Form, trainer and freshness** — recent finishing position/margins, trainer L50 strike rates and days since last start.
7. **Fundamental probability** — component scores are standardized across the field and combined into a greyhound-specific ensemble.
8. **Market blend** — TAB/Betfair probabilities are de-vigged and blended with fundamentals. The sidebar α controls market influence.
9. **Finishing-order simulation** — Plackett-Luce simulations estimate Top-2, Top-3 and expected finishing position.
10. **Value screen** — model fair odds are compared with the listed exchange price to estimate EV.

**Important:** track-specific box bias and fully normalized sectional databases would improve the model further. This version deliberately avoids comparing raw split seconds across unrelated tracks/distances when that would be misleading.
""")
