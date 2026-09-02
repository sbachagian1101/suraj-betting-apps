"""WorksheetPredictor - recalibrate R&S worksheet percentages."""
from __future__ import annotations

import json
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

st.set_page_config(page_title="WorksheetPredictor",
                   page_icon=":material/insights:", layout="wide")

# Streamlit Cloud pulls new files but keeps imported modules in sys.modules,
# so a deploy can run new app code against stale helpers. Fail loudly.
try:
    import ws_model as MD
    import ws_parser as PS
    for _m, _a in [(PS, "parse"), (PS, "region_for"),
                   (PS, "meeting_name_from_filename"),
                   (MD, "analyse_race"), (MD, "recommend"), (MD, "insights"),
                   (MD, "load_calibration"), (MD, "band")]:
        if not hasattr(_m, _a):
            raise ImportError(f"{_m.__name__} is missing {_a}")
except ImportError as exc:
    st.error(f"**Stale modules** ({exc}).\n\nOn Streamlit Cloud: "
             "**Manage app -> menu -> Reboot app**. A rerun is not enough.",
             icon=":material/error:")
    st.stop()

SAMPLES = Path("samples")
CAL = MD.load_calibration()
REC = CAL.get("record", {})

CSS = """
<style>
.wp-hero{border-radius:18px;padding:22px 26px;margin:2px 0 16px 0;color:#fff;
  background:linear-gradient(135deg,#123524 0%,#1f6b46 55%,#3fa06a 100%);
  box-shadow:0 8px 26px rgba(0,0,0,.18)}
.wp-hero h1{margin:0;font-size:1.62rem;letter-spacing:-.02em}
.wp-hero p{margin:6px 0 0 0;opacity:.93;font-size:.90rem;max-width:74ch}
.wp-row{display:flex;gap:12px;flex-wrap:wrap;margin:6px 0 16px 0}
.wp-card{flex:1 1 160px;border-radius:14px;padding:13px 16px;color:#fff;
  box-shadow:0 5px 16px rgba(0,0,0,.15)}
.wp-card .lab{font-size:.67rem;letter-spacing:.11em;text-transform:uppercase;
  opacity:.93}
.wp-card .val{font-size:1.6rem;font-weight:700;line-height:1.16;margin-top:3px}
.wp-card .sub{font-size:.73rem;opacity:.93;margin-top:2px}
.wp-a{background:linear-gradient(135deg,#14532d,#2f9e44)}
.wp-b{background:linear-gradient(135deg,#1e3a5f,#3b82c4)}
.wp-c{background:linear-gradient(135deg,#5b3a1e,#c2822f)}
.wp-d{background:linear-gradient(135deg,#3f2a56,#7c5bab)}
.wp-e{background:linear-gradient(135deg,#5b1e1e,#c23f3f)}
.wp-bet{border-radius:16px;padding:20px 24px;margin:4px 0 14px 0;
  border-left:9px solid var(--ac);background:rgba(127,127,127,.09)}
.wp-bet .hd{font-size:.69rem;letter-spacing:.13em;text-transform:uppercase;
  opacity:.75}
.wp-bet .big{font-size:1.48rem;font-weight:750;margin:4px 0 2px 0;
  color:var(--ac)}
.wp-bet .why{font-size:.87rem;opacity:.88;margin-top:8px;max-width:90ch}
.wp-chip{display:inline-block;padding:3px 11px;border-radius:999px;
  background:rgba(127,127,127,.17);font-size:.76rem;margin:0 6px 6px 0}
.wp-ins{border-left:4px solid #2f9e44;padding:8px 14px;margin:7px 0;
  background:rgba(127,127,127,.07);border-radius:0 9px 9px 0}
.wp-ins b{font-size:.80rem;letter-spacing:.03em}
.wp-ins span{display:block;font-size:.86rem;opacity:.9;margin-top:2px}
</style>
"""


def cards(items):
    html = "<div class='wp-row'>"
    for lab, val, sub, cls in items:
        html += (f"<div class='wp-card {cls}'><div class='lab'>{lab}</div>"
                 f"<div class='val'>{val}</div>"
                 f"<div class='sub'>{sub}</div></div>")
    st.markdown(html + "</div>", unsafe_allow_html=True)


@st.cache_data(show_spinner=False)
def do_parse(text: str, meeting: str):
    return PS.parse(text, meeting)


st.markdown(CSS, unsafe_allow_html=True)
st.markdown(
    "<div class='wp-hero'><h1>WorksheetPredictor</h1><p>Paste or upload a "
    "Racing &amp; Sports <b>Worksheets</b> export. Their running order is "
    "good &mdash; the top pick wins 31% against 12% for random. Their "
    "percentages are not: the top pick claims 43% and wins 31%. This "
    "rebuilds the percentages per jurisdiction and leaves the order "
    "untouched.</p></div>", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("### Settings")
    region_over = st.selectbox(
        "Region", ["Detect from meeting name", "AUS", "NZ", "UK", "IRE",
                   "FR", "OTHER"],
        help="Calibration is per jurisdiction. Australia is if anything "
             "underconfident; the UK is out by 2.4x.")
    price = st.number_input("Price available on the selection", 0.0, 1000.0,
                            0.0, 0.5,
                            help="Leave at 0 to see the break-even price "
                                 "instead. No price means no expected value.")
    kelly = st.select_slider("Kelly fraction", [0.10, 0.25, 0.50, 1.0], 0.25)
    cap = st.slider("Maximum stake (points)", 1.0, 10.0, 5.0, 0.5)
    st.divider()
    st.markdown("### Measured record")
    if REC:
        st.markdown(
            f"**{CAL['fitted_on']['races']} races, "
            f"{CAL['fitted_on']['runners']} runners** "
            f"({CAL['fitted_on']['meetings']} meetings, "
            f"{CAL['fitted_on']['date']}).\n\n"
            f"- R&S top pick wins **{REC['top_pick_win']*100:.1f}%** "
            f"(random {REC['random_win']*100:.1f}%)\n"
            f"- Top three hold the winner **{REC['top_three']*100:.1f}%** "
            f"(random {REC['random_top_three']*100:.1f}%)\n"
            f"- Log-loss **{REC['logloss_calibrated_oos']:.4f}** calibrated "
            f"vs **{REC['logloss_published']:.4f}** published and "
            f"**{REC['logloss_uniform']:.4f}** uniform")
        st.caption("Log-loss is out of sample, leave-one-meeting-out. "
                   "One day of racing: read it as a first measurement, not "
                   "a track record.")

st.markdown("#### Load a worksheet")
c1, c2, c3 = st.columns(3)
files = sorted(SAMPLES.glob("*.csv"))
pick = c1.selectbox("Bundled meeting", ["-"] + [f.stem for f in files],
                    label_visibility="collapsed")
if c2.button("Load bundled meeting", width="stretch") and pick != "-":
    st.session_state["wp_raw"] = (SAMPLES / f"{pick}.csv").read_text(
        encoding="utf-8-sig")
    st.session_state["wp_name"] = pick.replace("-", " ").title()
    st.session_state.pop("wp_mt", None)
    st.session_state.pop("wp_done", None)
    st.session_state.pop("wp_race", None)
if c3.button("Clear", width="stretch"):
    st.session_state["wp_raw"] = ""
    for k in ("wp_mt", "wp_done", "wp_race", "wp_upload"):
        st.session_state.pop(k, None)

up = st.file_uploader("...or upload a Worksheets CSV", type=["csv", "txt"])
if up is not None:
    # The uploader hands back the same file on EVERY rerun, so this has to be
    # keyed on the file itself. Ingesting unconditionally wiped the parsed
    # meeting each time any other widget moved, which is what sent the user
    # back to Parse/Predict the moment they changed race.
    fresh, fp, text, mname = PS.ingest_upload(
        st.session_state.get("wp_upload"), up.name, up.getvalue())
    if fresh:
        st.session_state["wp_upload"] = fp
        st.session_state["wp_raw"] = text
        st.session_state["wp_name"] = mname
        for k in ("wp_mt", "wp_done", "wp_race"):
            st.session_state.pop(k, None)

if "wp_name" not in st.session_state:
    st.session_state["wp_name"] = ""
name = st.text_input("Meeting name", key="wp_name", placeholder="e.g. Ripon")
st.text_area("Worksheet text", key="wp_raw", height=190,
             label_visibility="collapsed",
             placeholder="Tab,Horse,RFS,DLS,12m,BRR,FORM,COND,CONS,BP,JOCK,"
                         "JC,FR,EM,PER,DIV ... one block per race, blank "
                         "line between races.")

b1, b2 = st.columns(2)
if b1.button("Parse", width="stretch"):
    if not st.session_state.get("wp_raw", "").strip():
        st.warning("Nothing to parse yet.")
    else:
        st.session_state["wp_mt"] = do_parse(st.session_state["wp_raw"], name)
        st.session_state.pop("wp_done", None)
if b2.button("Predict", type="primary", width="stretch"):
    if st.session_state.get("wp_raw", "").strip() and "wp_mt" not in \
            st.session_state:
        st.session_state["wp_mt"] = do_parse(st.session_state["wp_raw"], name)
    if "wp_mt" not in st.session_state:
        st.warning("Parse a worksheet first.")
    else:
        st.session_state["wp_done"] = True

mt = st.session_state.get("wp_mt")

if mt is not None:
    if not mt.races:
        st.error("No races were found. Each race needs a `Tab,Horse,...` "
                 "header or a blank line between blocks.",
                 icon=":material/error:")
        st.stop()
    region = mt.region if region_over.startswith("Detect") else region_over
    live_tot = sum(len(r.live) for r in mt.races)
    st.divider()
    st.markdown(
        f"<span class='wp-chip'>{name or mt.name}</span>"
        f"<span class='wp-chip'>region {region}</span>"
        f"<span class='wp-chip'>{len(mt.races)} races</span>"
        f"<span class='wp-chip'>{live_tot} live runners</span>",
        unsafe_allow_html=True)
    st.success(f"Parsed **{len(mt.races)} races**, {live_tot} live runners, "
               f"{sum(len(r.runners) - len(r.live) for r in mt.races)} "
               f"scratched.", icon=":material/check_circle:")
    for w in mt.warnings[:4]:
        st.warning(w, icon=":material/warning:")

if mt is not None and st.session_state.get("wp_done"):
    region = mt.region if region_over.startswith("Detect") else region_over
    det = CAL.get("region_detail", {}).get(region, {})
    t, validated = MD.temperature(CAL, region)
    if not validated:
        st.info(
            f"The **{region}** constant rests on "
            f"{det.get('meetings', 0)} meeting(s) and is **provisional**. "
            f"Regions with two or more meetings behind them (AUS, UK) are "
            f"validated leave-one-meeting-out; these are not.",
            icon=":material/info:")

    labels = [f"Race {r.index}  ({len(r.live)} runners)"
              for r in mt.races if len(r.live) >= 2]
    idx_map = [r for r in mt.races if len(r.live) >= 2]
    if not idx_map:
        st.error("No race has two or more unscratched runners.")
        st.stop()
    sel = st.selectbox("Race", labels, key="wp_race")
    if sel not in labels:                     # field changed under the key
        sel = labels[0]
    race = idx_map[labels.index(sel)]

    rows, meta = MD.analyse_race(race.live, region, CAL)
    rec = MD.recommend(rows, meta, price=(price if price > 1.0 else None),
                       kelly_fraction=kelly, max_points=cap)
    conf = meta["confidence"]
    lab, colour = MD.band(conf)
    top = rows[0]

    st.markdown("### Prediction")
    if rec["action"] == "No bet":
        st.markdown(
            f"<div class='wp-bet' style='--ac:#9b2c2c'>"
            f"<div class='hd'>Betting recommendation</div>"
            f"<div class='big'>No bet</div>"
            f"<div>Top rated is <b>#{top.tab} {top.horse}</b> at "
            f"{top.prob*100:.1f}% to win, {top.place*100:.1f}% for a place."
            f"</div><div class='why'>{rec['why']}</div></div>",
            unsafe_allow_html=True)
    else:
        stake = (f"<b>{rec['points']} points</b> at <b>${rec['price']:.2f}</b>"
                 f" &nbsp;&middot;&nbsp; expected value "
                 f"<b>{rec['ev']:+.1%}</b>" if rec.get("points")
                 else f"needs better than <b>${rec['fair']:.2f}</b> "
                      f"&nbsp;&middot;&nbsp; enter a price in the sidebar "
                      f"for a stake")
        st.markdown(
            f"<div class='wp-bet' style='--ac:{colour}'>"
            f"<div class='hd'>Betting recommendation</div>"
            f"<div class='big'>{rec['market'].upper()} &mdash; "
            f"#{rec['line'].tab} {rec['line'].horse}</div>"
            f"<div>{stake} &nbsp;&middot;&nbsp; calibrated chance "
            f"<b>{rec['prob']*100:.1f}%</b> &nbsp;&middot;&nbsp; confidence "
            f"<b>{conf:.0f}/100 ({lab})</b></div>"
            f"<div class='why'>{rec['why']}</div></div>",
            unsafe_allow_html=True)

    cards([
        ("Top rated", f"#{top.tab}", top.horse[:20], "wp-b"),
        ("Win chance", f"{top.prob*100:.1f}%",
         f"R&amp;S said {top.published:.1f}% &middot; fair "
         f"${top.fair_win:.2f}", "wp-a"),
        ("Place chance", f"{top.place*100:.1f}%",
         f"top {meta['places']} &middot; fair ${top.fair_place:.2f}", "wp-c"),
        ("Confidence", f"{conf:.0f}", f"{lab} &mdash; evidence, not hit rate",
         "wp-d"),
        (f"{region} correction", f"{det.get('ratio', 1.0):.2f}x",
         f"t = {t:.2f} on {det.get('races', 0)} graded races", "wp-e"),
    ])

    df = pd.DataFrame([{
        "Horse": f"#{r.tab} {r.horse}", "R&S %": r.published,
        "Calibrated %": r.prob * 100, "Shift": r.shift,
        "Place %": r.place * 100, "Fair win": r.fair_win,
        "Fair place": r.fair_place, "R&S price": r.div,
        "FR": r.fr, "EM": r.em,
    } for r in rows])
    order = list(df["Horse"])

    left, right = st.columns([3, 2])
    with left:
        st.markdown("**Published against recalibrated**")
        melt = df.melt(id_vars=["Horse"],
                       value_vars=["R&S %", "Calibrated %"],
                       var_name="Source", value_name="pct")
        st.altair_chart(
            alt.Chart(melt).mark_bar().encode(
                y=alt.Y("Horse:N", sort=order, title=None),
                x=alt.X("pct:Q", title="win probability (%)"),
                yOffset="Source:N",
                color=alt.Color("Source:N", scale=alt.Scale(
                    range=["#9aa5b1", "#2f9e44"])),
                tooltip=["Horse", "Source", alt.Tooltip("pct", format=".1f")]
            ).properties(height=36 * len(df)), width="stretch")
    with right:
        st.markdown("**Who the correction moves**")
        st.altair_chart(
            alt.Chart(df).mark_bar(cornerRadius=3).encode(
                y=alt.Y("Horse:N", sort=order, title=None),
                x=alt.X("Shift:Q", title="change in percentage points"),
                color=alt.condition(alt.datum.Shift > 0,
                                    alt.value("#2f9e44"),
                                    alt.value("#c23f3f")),
                tooltip=["Horse", alt.Tooltip("Shift", format="+.1f")]
            ).properties(height=36 * len(df)), width="stretch")
        st.caption("Flattening an overconfident book lifts the outsiders and "
                   "trims the favourite. In Australia it runs the other way.")

    st.markdown("**Win and place, side by side**")
    st.altair_chart(
        alt.Chart(df).mark_circle(size=150, opacity=.85).encode(
            x=alt.X("Calibrated %:Q", title="win %"),
            y=alt.Y("Place %:Q", title="top-3 %"),
            color=alt.Color("Calibrated %:Q",
                            scale=alt.Scale(scheme="greens"), legend=None),
            tooltip=["Horse", alt.Tooltip("Calibrated %", format=".1f"),
                     alt.Tooltip("Place %", format=".1f"),
                     alt.Tooltip("Fair win", format="$.2f")]
        ).properties(height=260), width="stretch")

    st.markdown("**Every runner**")
    st.dataframe(df, hide_index=True, width="stretch", column_config={
        "R&S %": st.column_config.NumberColumn(format="%.1f%%"),
        "Calibrated %": st.column_config.ProgressColumn(
            "Calibrated %", format="%.1f%%", min_value=0.0,
            max_value=float(max(df["Calibrated %"].max(), 1))),
        "Shift": st.column_config.NumberColumn(format="%+.1f"),
        "Place %": st.column_config.NumberColumn(format="%.1f%%"),
        "Fair win": st.column_config.NumberColumn(format="$%.2f"),
        "Fair place": st.column_config.NumberColumn(format="$%.2f"),
        "R&S price": st.column_config.NumberColumn(format="$%.2f"),
    })

    ic, cc = st.columns([3, 2])
    with ic:
        st.markdown("**Insights**")
        for title, body in MD.insights(rows, meta, CAL):
            st.markdown(f"<div class='wp-ins'><b>{title}</b>"
                        f"<span>{body}</span></div>", unsafe_allow_html=True)
    with cc:
        st.markdown("**What the confidence score is made of**")
        cp = pd.DataFrame([{"Factor": k, "Score": v * 100}
                           for k, v in meta["confidence_parts"].items()])
        st.altair_chart(
            alt.Chart(cp).mark_bar(cornerRadiusEnd=5).encode(
                y=alt.Y("Factor:N", sort=list(cp["Factor"]), title=None),
                x=alt.X("Score:Q", title=None,
                        scale=alt.Scale(domain=[0, 100])),
                color=alt.Color("Score:Q", scale=alt.Scale(
                    scheme="redyellowgreen"), legend=None),
                tooltip=["Factor", alt.Tooltip("Score", format=".0f")]
            ).properties(height=132), width="stretch")
        st.info(f"**{conf:.0f}/100 — {lab}.** This reflects how much graded "
                f"evidence stands behind the {region} constant, how clearly "
                f"one runner leads and the field size. It is **not** a hit "
                f"rate.", icon=":material/info:")

    with st.expander("How the calibration was measured"):
        rd = CAL.get("region_detail", {})
        st.dataframe(pd.DataFrame([{
            "Region": k, "Meetings": v["meetings"], "Races": v["races"],
            "R&S claims": v["claims"] * 100, "Actually wins": v["actual"] * 100,
            "Ratio": v["ratio"],
            "Win t": CAL["regions"].get(k),
            "Place t": CAL.get("place_regions", {}).get(k),
            "Validated": "yes" if v["validated"] else "provisional",
        } for k, v in sorted(rd.items())]), hide_index=True, width="stretch",
            column_config={
                "R&S claims": st.column_config.NumberColumn(format="%.1f%%"),
                "Actually wins": st.column_config.NumberColumn(
                    format="%.1f%%"),
                "Ratio": st.column_config.NumberColumn(format="%.2fx")})
        st.markdown(
            "Both constants are chosen by **moment matching** — the top pick "
            "is made to claim the strike rate it actually achieves — and "
            "graded **leave-one-meeting-out**, never on the meeting they "
            "score. Fitting on log-loss instead left the favourite 12.8 "
            "points understated, and scored worse on log-loss too.\n\n"
            "**The order is never changed.** Only the confidence attached to "
            "it moves. One day of racing is a first measurement, not a track "
            "record — a second day would settle whether the UK figure is a "
            "real bias or one bad Tuesday.")
