"""HorsePredictor - Streamlit UI for Racing & Sports thoroughbred Enhanced Form."""
from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st

import horse_model as model
import horse_parser as parser

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
    seed = st.number_input("Random seed", value=42, step=1)
    st.divider()
    st.caption(
        "Fundamentals: official rating → weight carried → career strike rate → "
        "distance record → going record → jockey/trainer form → last-start "
        "finish → freshness → R&S ratings.")
    st.caption("Prediction is probabilistic decision support, not a guaranteed outcome.")

paste_tab, parsed_tab, pred_tab, explain_tab, method_tab = st.tabs(
    ["1 · Paste Data", "2 · Parsed Data", "3 · Prediction", "4 · Explanations", "Method"])

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
                                           sims=int(sims), seed=int(seed))
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
