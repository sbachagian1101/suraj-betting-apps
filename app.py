"""OddsPredictor - paste a betting screen, get de-vigged probabilities."""
from __future__ import annotations

from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

st.set_page_config(page_title="OddsPredictor", page_icon=":material/paid:",
                   layout="wide")

# Streamlit Cloud pulls new files but keeps imported modules in sys.modules,
# so a deploy can run new app code against stale helpers. Fail loudly.
try:
    import odds_model as MD
    import odds_parser as PS
    for _m, _a in [(PS, "parse"), (MD, "analyse"), (MD, "insights"),
                   (MD, "band"), (MD, "paid_places")]:
        if not hasattr(_m, _a):
            raise ImportError(f"{_m.__name__} is missing {_a}")
except ImportError as exc:
    st.error(f"**Stale modules** ({exc}).\n\nOn Streamlit Cloud: "
             "**Manage app -> menu -> Reboot app**. A rerun is not enough.",
             icon=":material/error:")
    st.stop()

SAMPLES = Path("samples")

CSS = """
<style>
.op-hero{border-radius:18px;padding:22px 26px;margin:2px 0 18px 0;color:#fff;
  background:linear-gradient(135deg,#0f2c4d 0%,#1d5e8c 55%,#2a8fa8 100%);
  box-shadow:0 8px 26px rgba(0,0,0,.18)}
.op-hero h1{margin:0;font-size:1.65rem;letter-spacing:-.02em}
.op-hero p{margin:6px 0 0 0;opacity:.92;font-size:.90rem;max-width:70ch}
.op-row{display:flex;gap:12px;flex-wrap:wrap;margin:6px 0 16px 0}
.op-card{flex:1 1 165px;border-radius:14px;padding:13px 16px;color:#fff;
  box-shadow:0 5px 16px rgba(0,0,0,.15)}
.op-card .lab{font-size:.68rem;letter-spacing:.11em;text-transform:uppercase;
  opacity:.93}
.op-card .val{font-size:1.65rem;font-weight:700;line-height:1.16;margin-top:3px}
.op-card .sub{font-size:.74rem;opacity:.93;margin-top:2px}
.op-a{background:linear-gradient(135deg,#14532d,#2f9e44)}
.op-b{background:linear-gradient(135deg,#1e3a5f,#3b82c4)}
.op-c{background:linear-gradient(135deg,#5b3a1e,#c2822f)}
.op-d{background:linear-gradient(135deg,#3f2a56,#7c5bab)}
.op-e{background:linear-gradient(135deg,#5b1e1e,#c23f3f)}
.op-bet{border-radius:16px;padding:20px 24px;margin:4px 0 14px 0;
  border-left:9px solid var(--ac);background:rgba(127,127,127,.09)}
.op-bet .hd{font-size:.70rem;letter-spacing:.13em;text-transform:uppercase;
  opacity:.75}
.op-bet .big{font-size:1.5rem;font-weight:750;margin:4px 0 2px 0;
  color:var(--ac)}
.op-bet .why{font-size:.87rem;opacity:.88;margin-top:8px;max-width:88ch}
.op-chip{display:inline-block;padding:3px 11px;border-radius:999px;
  background:rgba(127,127,127,.17);font-size:.76rem;margin:0 6px 6px 0}
.op-ins{border-left:4px solid #3b82c4;padding:8px 14px;margin:7px 0;
  background:rgba(127,127,127,.07);border-radius:0 9px 9px 0}
.op-ins b{font-size:.80rem;letter-spacing:.03em}
.op-ins span{display:block;font-size:.86rem;opacity:.9;margin-top:2px}
</style>
"""


def cards(items):
    html = "<div class='op-row'>"
    for lab, val, sub, cls in items:
        html += (f"<div class='op-card {cls}'><div class='lab'>{lab}</div>"
                 f"<div class='val'>{val}</div>"
                 f"<div class='sub'>{sub}</div></div>")
    st.markdown(html + "</div>", unsafe_allow_html=True)


@st.cache_data(show_spinner=False)
def do_parse(text: str):
    return PS.parse(text)


st.markdown(CSS, unsafe_allow_html=True)
st.markdown(
    "<div class='op-hero'><h1>OddsPredictor</h1><p>Paste a betting screen. "
    "The bookmaker margin comes out of both price pools, the two are "
    "averaged, and every runner is priced against what is actually on "
    "offer. Any value it finds comes from the two pools disagreeing &mdash; "
    "not from a view about the horses.</p></div>",
    unsafe_allow_html=True)

# ----------------------------------------------------------------- controls
with st.sidebar:
    st.markdown("### Settings")
    tote_w = st.slider("Weight on the tote pool", 0.0, 1.0, 0.5, 0.05,
                       help="0 = bookmakers only, 1 = tote only. 0.5 scored "
                            "best over the two graded races.")
    kelly = st.select_slider("Kelly fraction", [0.10, 0.25, 0.50, 1.0],
                             value=0.25,
                             help="Full Kelly is far too aggressive for "
                                  "probabilities read off the market itself.")
    cap = st.slider("Maximum stake (points)", 1.0, 10.0, 5.0, 0.5)
    st.divider()
    st.markdown("### Measured record")
    st.markdown(
        "**2 races graded.** Winner ranked **1st** and **6th** "
        "(random would average 6.8). Both recommended bets **lost**.\n\n"
        "That is the entire track record. Treat every number in this app "
        "as a description of the market, not a prediction that beats it.")

st.markdown("#### Paste the betting screen")
c1, c2, c3 = st.columns([1, 1, 1])
if c1.button("Load Ipswich sample", width="stretch"):
    st.session_state["op_raw"] = (SAMPLES / "ipswich_cl1.txt").read_text(
        encoding="utf-8")
    st.session_state.pop("op_race", None)
if c2.button("Load Murray Bridge sample", width="stretch"):
    st.session_state["op_raw"] = (SAMPLES / "murray_bridge_cl1.txt").read_text(
        encoding="utf-8")
    st.session_state.pop("op_race", None)
if c3.button("Clear", width="stretch"):
    st.session_state["op_raw"] = ""
    st.session_state.pop("op_race", None)
    st.session_state.pop("op_done", None)

st.text_area("Betting screen text", key="op_raw", height=210,
             label_visibility="collapsed",
             placeholder="Paste the whole race page here - runner numbers, "
                         "names, opening/top prices and both sets of "
                         "win/place prices.")

b1, b2 = st.columns(2)
if b1.button("Parse", type="secondary", width="stretch"):
    txt = st.session_state.get("op_raw", "")
    if not txt.strip():
        st.warning("Nothing to parse yet.")
    else:
        st.session_state["op_race"] = do_parse(txt)
        st.session_state.pop("op_done", None)
if b2.button("Predict", type="primary", width="stretch"):
    txt = st.session_state.get("op_raw", "")
    if txt.strip() and "op_race" not in st.session_state:
        st.session_state["op_race"] = do_parse(txt)
    if "op_race" not in st.session_state:
        st.warning("Parse a race first.")
    else:
        st.session_state["op_done"] = True

race = st.session_state.get("op_race")

# ------------------------------------------------------------------- parsed
if race is not None:
    if not race.active:
        st.error("No priced runners were found. The screen needs a runner "
                 "number, a name, then the win and place prices.",
                 icon=":material/error:")
        st.stop()

    st.divider()
    bits = [race.meeting, race.title, race.race_class,
            f"{race.distance}m" if race.distance else "",
            race.going, race.weather]
    st.markdown(
        " ".join(f"<span class='op-chip'>{b}</span>" for b in bits if b),
        unsafe_allow_html=True)
    st.success(
        f"Parsed **{len(race.active)} priced runners**"
        + (f", {len(race.scratched)} scratched" if race.scratched else ""),
        icon=":material/check_circle:")
    if race.scratched:
        st.caption("Scratched: " + ", ".join(
            f"#{r.number} {r.name}" for r in race.scratched))

    with st.expander("Parsed prices", expanded=not st.session_state.get(
            "op_done")):
        st.dataframe(pd.DataFrame([{
            "#": r.number, "Horse": r.name, "Open": r.opening,
            "Top": r.top, "Fixed win": r.fixed_win,
            "Fixed place": r.fixed_place, "Tote win": r.tote_win,
            "Tote place": r.tote_place,
        } for r in race.active]), hide_index=True, width="stretch")

# --------------------------------------------------------------- prediction
if race is not None and st.session_state.get("op_done"):
    lines, meta = MD.analyse(race, tote_weight=tote_w, kelly_fraction=kelly,
                             max_points=cap)
    if meta.get("error"):
        st.error(meta["error"])
        st.stop()

    rec = meta["recommendation"]
    conf = meta["confidence"]
    lab, colour = MD.band(conf)
    top = lines[0]

    st.divider()
    st.markdown("### Prediction")

    if rec["action"] == "No bet":
        st.markdown(
            f"<div class='op-bet' style='--ac:#9b2c2c'>"
            f"<div class='hd'>Betting recommendation</div>"
            f"<div class='big'>No bet</div>"
            f"<div>Top rated is <b>#{top.number} {top.name}</b> at "
            f"{top.p_win*100:.1f}% &mdash; fair price "
            f"<b>${top.fair_win:.2f}</b>, best available "
            f"<b>${top.best_win:.2f}</b>.</div>"
            f"<div class='why'>{rec['why']}</div></div>",
            unsafe_allow_html=True)
    else:
        L = rec["line"]
        p = L.p_win if rec["market"] == "Win" else L.p_place
        st.markdown(
            f"<div class='op-bet' style='--ac:#14804a'>"
            f"<div class='hd'>Betting recommendation</div>"
            f"<div class='big'>{rec['market'].upper()} &mdash; "
            f"#{L.number} {L.name}</div>"
            f"<div><b>{rec['points']} points</b> at "
            f"<b>${rec['price']:.2f}</b> ({rec['source']}) &nbsp;&middot;&nbsp; "
            f"expected value <b>{rec['ev']:+.1%}</b> &nbsp;&middot;&nbsp; "
            f"model probability <b>{p*100:.1f}%</b> &nbsp;&middot;&nbsp; "
            f"confidence <b>{conf:.0f}/100 ({lab})</b></div>"
            f"<div class='why'>{rec['why']} One point = 1% of your bank; "
            f"the stake is {kelly:g} Kelly, capped at {cap:g}.</div></div>",
            unsafe_allow_html=True)

    cards([
        ("Top rated", f"#{top.number}", top.name[:19], "op-b"),
        ("Win chance", f"{top.p_win*100:.1f}%",
         f"fair ${top.fair_win:.2f} vs ${top.best_win:.2f}", "op-a"),
        ("Confidence", f"{conf:.0f}", f"{lab} &mdash; market clarity, "
         f"not hit rate", "op-c"),
        ("Field", f"{meta['runners']}",
         f"{meta['places_paid']} places paid", "op-d"),
        ("Margin removed", f"{(meta['book_fixed']-1)*100:.1f}%",
         "from the fixed book", "op-e"),
    ])

    df = pd.DataFrame([{
        "no": x.number, "Horse": f"#{x.number} {x.name}",
        "Win %": x.p_win * 100, "Place %": x.p_place * 100,
        "Bookmakers": x.p_fixed * 100, "Tote": x.p_tote * 100,
        "Fair": x.fair_win, "Best": x.best_win, "Src": x.best_win_src,
        "EV %": x.ev_win * 100, "Move %": x.move * 100,
        "Points": x.kelly_win * 100,
    } for x in lines])
    order = list(df["Horse"])

    left, right = st.columns([3, 2])
    with left:
        st.markdown("**Win probability, margin removed**")
        base = alt.Chart(df).encode(
            y=alt.Y("Horse:N", sort=order, title=None))
        bar = base.mark_bar(cornerRadiusEnd=5).encode(
            x=alt.X("Win %:Q", title="win probability (%)"),
            color=alt.Color("Win %:Q", scale=alt.Scale(scheme="tealblues"),
                            legend=None),
            tooltip=["Horse", alt.Tooltip("Win %", format=".1f"),
                     alt.Tooltip("Fair", format="$.2f"),
                     alt.Tooltip("Best", format="$.2f"),
                     alt.Tooltip("EV %", format="+.1f")])
        txt = base.mark_text(align="left", dx=4, fontSize=11).encode(
            x="Win %:Q", text=alt.Text("Win %:Q", format=".1f"))
        st.altair_chart((bar + txt).properties(height=28 * len(df)),
                        width="stretch")

    with right:
        st.markdown("**Where the two pools disagree**")
        melt = df.melt(id_vars=["Horse"], value_vars=["Bookmakers", "Tote"],
                       var_name="Pool", value_name="pct")
        st.altair_chart(
            alt.Chart(melt).mark_circle(size=125, opacity=.85).encode(
                y=alt.Y("Horse:N", sort=order, title=None),
                x=alt.X("pct:Q", title="implied probability (%)"),
                color=alt.Color("Pool:N", scale=alt.Scale(
                    range=["#2f6fb0", "#d98b2b"])),
                tooltip=["Horse", "Pool", alt.Tooltip("pct", format=".1f")]
            ).properties(height=28 * len(df)), width="stretch")
        st.caption("A wide gap is the only place value can come from: the "
                   "longer of the two prices beats the blended estimate.")

    st.markdown("**Money since the opening price**  "
                "&mdash; shortened is green, drifted is red")
    st.altair_chart(
        alt.Chart(df).mark_bar(cornerRadius=3).encode(
            y=alt.Y("Horse:N", sort=order, title=None),
            x=alt.X("Move %:Q", title="price movement (%)"),
            color=alt.condition(alt.datum["Move %"] > 0,
                                alt.value("#2f9e44"), alt.value("#c23f3f")),
            tooltip=["Horse", alt.Tooltip("Move %", format="+.0f")]
        ).properties(height=24 * len(df)), width="stretch")
    st.caption("Context only. Steam money won one graded race and failed the "
               "other, and drifting runners hit the board in both, so this "
               "chart is deliberately not scored.")

    st.markdown("**Every runner**")
    st.dataframe(
        df.drop(columns=["no"]), hide_index=True, width="stretch",
        column_config={
            "Win %": st.column_config.ProgressColumn(
                "Win %", format="%.1f%%", min_value=0.0,
                max_value=float(max(df["Win %"].max(), 1))),
            "Place %": st.column_config.NumberColumn(format="%.1f%%"),
            "Bookmakers": st.column_config.NumberColumn(format="%.1f%%"),
            "Tote": st.column_config.NumberColumn(format="%.1f%%"),
            "Fair": st.column_config.NumberColumn("Fair price",
                                                  format="$%.2f"),
            "Best": st.column_config.NumberColumn("Best available",
                                                  format="$%.2f"),
            "Src": st.column_config.TextColumn("From"),
            "EV %": st.column_config.NumberColumn(format="%+.1f%%"),
            "Move %": st.column_config.NumberColumn(format="%+.0f%%"),
            "Points": st.column_config.NumberColumn("Kelly pts",
                                                    format="%.2f"),
        })

    ic, cc = st.columns([3, 2])
    with ic:
        st.markdown("**Insights**")
        for title, body in MD.insights(lines, meta, race):
            st.markdown(f"<div class='op-ins'><b>{title}</b>"
                        f"<span>{body}</span></div>",
                        unsafe_allow_html=True)
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
        st.info(
            f"**{conf:.0f}/100 &mdash; {lab}.** This measures how "
            f"well-defined the market is: whether the two pools agree, how "
            f"much margin is in the book, the field size, and how clearly "
            f"one runner leads. It is **not** a hit rate, and nothing here "
            f"has been shown to beat the market.",
            icon=":material/info:")
