"""RaceSim - paste a Racing & Sports Enhanced Form page (plus, optionally, the
Speed Map) and watch how the model expects the race to be run: the jump, the
settle, the turn and the finish, as a downloadable GIF of at most 30 seconds.

The rating engine weighs class, form, speed figures, fitness, weight, barrier,
jockey, trainer, going, distance and late speed in lengths. The market is a
separate, capped blend (default 5 %, never more than 10 %).
"""
from __future__ import annotations

import importlib
import os

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="RaceSim", page_icon=":material/sprint:", layout="wide",
                   initial_sidebar_state="expanded")

# Streamlit Cloud keeps already-imported helper modules in memory across a
# redeploy, so a commit that adds a name to a helper can leave the new app.py
# running against the old module. Check up front and say so in one sentence.
_REQUIRED = {
    "horse_parser": ("parse", "parse_speed_map", "track_direction"),
    "race_model": ("rate_field", "fmt_time", "MARKET_WEIGHT_CAP", "SIGMA_LENGTHS"),
    "race_sim": ("simulate", "running_calls"),
    "race_anim": ("render_gif", "snapshots", "to_png_bytes", "pretty_name", "MAX_DURATION_S"),
}
_missing: list[str] = []
_mods: dict = {}
for _name, _syms in _REQUIRED.items():
    try:
        _mods[_name] = importlib.import_module(_name)
    except Exception as exc:                      # noqa: BLE001
        _missing.append(f"{_name} ({exc.__class__.__name__})")
        continue
    _missing += [f"{_name}.{s}" for s in _syms if not hasattr(_mods[_name], s)]
if _missing:
    st.error("**This deployment is running stale code.** Missing: `" + "`, `".join(_missing)
             + "`. Streamlit Cloud pulled the new files but kept old modules in memory. "
             "Fix it with **Manage app → ⋮ → Reboot app**.", icon=":material/error:")
    st.stop()
hp, rm, rs, ra = _mods["horse_parser"], _mods["race_model"], _mods["race_sim"], _mods["race_anim"]

_HERE = os.path.dirname(os.path.abspath(__file__))

st.markdown("""
<style>
.block-container {padding-top: 1.1rem; padding-bottom: 3rem;}
div[data-testid="stMetric"] {border:1px solid rgba(128,128,128,.22); border-radius:.7rem; padding:.5rem .8rem;}
.pick {border:1px solid rgba(255,202,40,.55); background:rgba(255,202,40,.08); border-radius:.8rem;
       padding:.7rem 1rem; margin:.3rem 0 .8rem;}
.muted {opacity:.65; font-size:.8rem;}
</style>
""", unsafe_allow_html=True)


# --------------------------------------------------------------------------
# cached pipeline
# --------------------------------------------------------------------------
@st.cache_data(show_spinner="Reading the form page…")
def parse_form(raw: str):
    return hp.parse(raw)


@st.cache_data(show_spinner=False)
def parse_speed(raw: str):
    return hp.parse_speed_map(raw) if raw.strip() else {}


@st.cache_data(show_spinner="Rating the field…")
def rate(header, runners, speed_map, market_weight, sigma):
    return rm.rate_field(header, runners, speed_map, market_weight=market_weight, sigma=sigma)


@st.cache_data(show_spinner="Running the race…", max_entries=12)
def build(rated, header, mode, seed, clockwise, duration, fps):
    sim = rs.simulate(rated, mode=mode, seed=seed)
    gif = ra.render_gif(sim, header, clockwise=clockwise, duration_s=duration, fps=fps)
    snaps = [(label, ra.to_png_bytes(img)) for label, img in ra.snapshots(sim, header, clockwise)]
    calls = rs.running_calls(sim)
    progress = {
        "tabs": sim.tabs, "names": sim.names, "s_grid": sim.s_grid[::20].tolist(),
        "gap": sim.gap[:, ::20].tolist(),
    }
    return gif, snaps, calls, progress, sim.finish_order, sim.finish_margin.tolist()


def load_fixture(name: str) -> str:
    path = os.path.join(_HERE, name)
    return open(path, encoding="utf-8").read() if os.path.exists(path) else ""


# --------------------------------------------------------------------------
# sidebar
# --------------------------------------------------------------------------
with st.sidebar:
    st.header("Race data")
    if st.button("Load sample race (Canberra R5, 1400m Soft 6)", use_container_width=True):
        st.session_state["form_text"] = load_fixture("fixture_canberra_r5.txt")
        st.session_state["speed_text"] = load_fixture("fixture_canberra_r5_speedmap.txt")
    form_text = st.text_area("Enhanced Form page (select-all, copy, paste)", key="form_text",
                             height=180, placeholder="Paste the whole Racing & Sports Enhanced Form page here…")
    speed_text = st.text_area("Speed Map page (optional, improves the early positions)",
                              key="speed_text", height=110,
                              placeholder="Paste the Speed Map 'Pace Values' table here…")

    st.header("Animation")
    duration = st.slider("Length (seconds)", 10, ra.MAX_DURATION_S, 24, 1)
    fps = st.select_slider("Smoothness (frames per second)", [8, 10, 12, 15], value=12)
    mode_label = st.radio("Scenario", ["Model expectation", "Random race-day draw"], index=0,
                          help="Expectation shows the margins the model predicts. A random draw adds "
                               "race-day luck so you can see how differently the same field can run.")
    mode = "expected" if mode_label.startswith("Model") else "random"
    seed = st.number_input("Draw number (seed)", 0, 9999, 1, 1, disabled=(mode == "expected"))

    st.header("Model")
    market_pct = st.slider("Market weight in the rating (%)", 0, int(rm.MARKET_WEIGHT_CAP * 100), 5, 1,
                           help="Hard-capped at 10 %. The fundamentals - class, form, speed, fitness, weight, "
                                "barrier, jockey, trainer, going, distance - always carry at least 90 %.")
    sigma = st.slider("Race-day noise (lengths)", 1.5, 4.5, float(rm.SIGMA_LENGTHS), 0.1,
                      help="Standard deviation of a runner's finishing margin around its rating. "
                           "Bigger = flatter win probabilities.")
    dir_choice = st.radio("Track direction", ["Auto-detect", "Clockwise", "Anti-clockwise"], index=0)


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------
st.title("RaceSim")
st.caption("How the race is likely to be run - from the jump to the post - built from the form, "
           "not from the odds.")

if not (form_text or "").strip():
    st.info("Paste a Racing & Sports **Enhanced Form** page in the sidebar (or load the sample race) "
            "to rate the field and animate the run.", icon=":material/content_paste:")
    with st.expander("How the model works"):
        st.markdown(open(os.path.join(_HERE, "README.md"), encoding="utf-8").read()
                    .split("## How the model works", 1)[-1].split("## ", 1)[0]
                    if os.path.exists(os.path.join(_HERE, "README.md")) else "See README.md")
    st.stop()

header, runners, warnings = parse_form(form_text)
active = [r for r in runners if not r.get("scratched")]
if not active:
    st.error("No runners could be read from that text. Make sure the whole Enhanced Form page "
             "(including the field table) was copied.", icon=":material/error:")
    st.stop()
speed_map = parse_speed(speed_text or "")
for w in warnings:
    st.warning(w, icon=":material/warning:")
if speed_text.strip() and not speed_map:
    st.warning("The Speed Map text was pasted but no pace values could be read from it; "
               "early positions fall back to each horse's settling history.", icon=":material/warning:")

rated = rate(header, runners, speed_map, market_pct / 100.0, sigma)
rr = rated["runners"]
meta = rated["meta"]

if dir_choice == "Auto-detect":
    clockwise = hp.track_direction(runners, header.get("track", "")) == "clockwise"
else:
    clockwise = dir_choice == "Clockwise"

# ---- race card -------------------------------------------------------------
going = f"{str(header.get('going') or '').title()} {header.get('going_rating') or ''}".strip()
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Race", f"{header.get('track', '?')} R{header.get('race_no', '?')}")
c2.metric("Distance / going", f"{header.get('distance_m', '?')}m {going}")
c3.metric("Runners", f"{len(rr)} ({len(runners) - len(rr)} scr)")
c4.metric("Predicted winning time", rm.fmt_time(meta["pred_time_s"]))
c5.metric("Market weight used", f"{int(meta['market_weight'] * 100)}%")
if header.get("race_name"):
    st.caption(f"{header['race_name']} · {header.get('race_type', '')} · {header.get('date', '')} "
               f"{header.get('time', '')} · {'clockwise' if clockwise else 'anti-clockwise'}"
               + (" · speed map loaded" if speed_map else " · no speed map (early speed from settling history)"))

top = rr[0]
sec = rr[1]
st.markdown(
    f'<div class="pick"><b>Model pick:</b> #{top["tab"]} {ra.pretty_name(top["horse"])} '
    f'({top["win_prob"]:.0%} win, ${top["model_odds"]} model vs '
    f'{"$" + format(top["odds"], ".1f") if top["odds"] else "no price"} market) &nbsp;·&nbsp; '
    f'<b>Danger:</b> #{sec["tab"]} {ra.pretty_name(sec["horse"])} ({sec["win_prob"]:.0%}) &nbsp;·&nbsp; '
    f'<span class="muted">expected winning margin {sec["exp_margin"]:.1f}L</span></div>',
    unsafe_allow_html=True)

tab_anim, tab_ratings, tab_calls, tab_why = st.tabs(
    ["Race animation", "Ratings & probabilities", "Race call & positions", "How it's calculated"])

gif, snaps, calls, progress, finish_order, finish_margin = build(
    rated, header, mode, int(seed), clockwise, float(duration), int(fps))

with tab_anim:
    left, right = st.columns([3, 1])
    with left:
        st.image(gif, caption=f"{duration}s animation · {mode_label.lower()} · {fps} fps", use_container_width=True)
    with right:
        fname = f"racesim_{header.get('track', 'race')}_R{header.get('race_no', '')}_{mode}{seed if mode == 'random' else ''}.gif"
        st.download_button("Download GIF", gif, file_name=fname.replace(" ", "_"), mime="image/gif",
                           use_container_width=True, type="primary")
        st.metric("GIF size", f"{len(gif) / 1e6:.1f} MB")
        st.markdown("**Predicted finish**")
        for k, i in enumerate(finish_order[:5], 1):
            m = finish_margin[i]
            st.markdown(f"{k}. #{progress['tabs'][i]} {ra.pretty_name(progress['names'][i])}"
                        + (f" &nbsp;<span class='muted'>{m:.1f}L</span>" if k > 1 else ""),
                        unsafe_allow_html=True)
        if mode == "random":
            st.caption("A random draw is one plausible running, not the most likely one. "
                       "Change the draw number for another.")
    st.markdown("#### Key moments")
    cols = st.columns(4)
    for col, (label, png) in zip(cols, snaps):
        col.image(png, caption=label, use_container_width=True)

with tab_ratings:
    df = pd.DataFrame([{
        "Rank": r["rank"], "Tab": r["tab"], "Horse": ra.pretty_name(r["horse"]), "BP": r["bp"],
        "Carried kg": round(r["carried"], 1), "Jockey": r["jockey"].title(), "OHR": r["ohr"],
        "Rating (L)": round(r["rating"], 2), "Win %": round(100 * r["win_prob"], 1),
        "Place %": round(100 * r["place_prob"], 1), "Model $": r["model_odds"],
        "Market $": r["odds"], "Market %": round(100 * r["market_prob"], 1),
        "Edge (pp)": round(100 * (r["win_prob"] - r["market_prob"]), 1),
        "Settles": "leader" if r["early_speed"] > 1.0 else "on pace" if r["early_speed"] > 0.3
                   else "midfield" if r["early_speed"] > -0.5 else "back",
        "Exp. margin (L)": round(r["exp_margin"], 1),
    } for r in rr])
    st.dataframe(df, hide_index=True, use_container_width=True,
                 column_config={"Win %": st.column_config.ProgressColumn(format="%.1f", min_value=0, max_value=100),
                                "Place %": st.column_config.ProgressColumn(format="%.1f", min_value=0, max_value=100)})
    st.caption("Rating is in lengths relative to the field average. Edge = model win % minus the market's "
               "implied win % (overround removed). Market % is shown for comparison only; it enters the rating "
               f"at {int(meta['market_weight'] * 100)}%.")

    st.markdown("#### Where each rating comes from (lengths)")
    comp_rows = []
    for r in rr:
        for k, v in r["components"].items():
            comp_rows.append({"Horse": f"#{r['tab']} {ra.pretty_name(r['horse'])}", "Component": k,
                              "Lengths": round(v, 2), "rank": r["rank"]})
    cdf = pd.DataFrame(comp_rows)
    order = [f"#{r['tab']} {ra.pretty_name(r['horse'])}" for r in rr]
    chart = (alt.Chart(cdf).mark_bar().encode(
        y=alt.Y("Horse:N", sort=order, title=None),
        x=alt.X("Lengths:Q", title="lengths (positive = better)"),
        color=alt.Color("Component:N", legend=alt.Legend(orient="bottom", columns=6)),
        tooltip=["Horse", "Component", "Lengths"],
    ).properties(height=26 * len(rr) + 40))
    st.altair_chart(chart, use_container_width=True)

with tab_calls:
    st.markdown("#### Running order at each stage")
    stage_cols = st.columns(len(calls))
    for col, c in zip(stage_cols, calls):
        col.markdown(f"**{c['checkpoint']}**  \n<span class='muted'>{int(c['to_go'])}m to go</span>",
                     unsafe_allow_html=True)
        lines = []
        for k, (tab, name, gap) in enumerate(c["order"][:8], 1):
            g = "" if k == 1 else f" ({gap:.1f}L)"
            lines.append(f"{k}. #{tab} {ra.pretty_name(name)[:14]}{g}")
        col.markdown("  \n".join(lines))
    st.markdown("#### Lengths behind the leader through the race")
    rows = []
    for i, tab in enumerate(progress["tabs"]):
        for s, g in zip(progress["s_grid"], progress["gap"][i]):
            rows.append({"Horse": f"#{tab} {ra.pretty_name(progress['names'][i])}", "Distance run (m)": s,
                         "Lengths behind": round(g, 2)})
    pdf = pd.DataFrame(rows)
    sel = alt.selection_point(fields=["Horse"], bind="legend")
    line = (alt.Chart(pdf).mark_line(interpolate="monotone").encode(
        x=alt.X("Distance run (m):Q"), y=alt.Y("Lengths behind:Q", scale=alt.Scale(reverse=True)),
        color=alt.Color("Horse:N", legend=alt.Legend(orient="right")),
        opacity=alt.condition(sel, alt.value(1), alt.value(0.15)),
        tooltip=["Horse", "Distance run (m)", "Lengths behind"],
    ).add_params(sel).properties(height=360))
    st.altair_chart(line, use_container_width=True)
    st.caption("Click a horse in the legend to highlight it.")

with tab_why:
    st.markdown(f"""
**Everything is measured in lengths at the finish of today's race**, so the parts add up and the sum
drives both the win probabilities and the margins you see animated.

| Component | What it uses | Rule |
|---|---|---|
| Class | Official handicap rating (OHR) of recent runs | +0.3 L per point above the field mean |
| Weight | Allotted weight minus apprentice claim | −0.4 L per kg above the field mean (scaled by distance) |
| Form | Last 5 runs: race class + prizemoney, beaten margin, weight carried, recency | class points − 3 × margin, recency-weighted, ÷3 → lengths (×0.75) |
| Speed | Race times of past runs vs a par speed for that distance, adjusted for Heavy/Soft/Good | best 3 of 4, median, damped ×0.5 |
| Fitness | Days since last run, run of the preparation, first-up record, age | peak 7–35 days; 2nd/3rd-up +0.25 L; long spell −0.7 to −1.0 L unless proven fresh |
| Barrier | Gate (Speed Map gate if supplied) vs distance and early speed | free to gate 4, −0.12 L/gate to 10, steeper beyond; leaders penalised less |
| Jockey | R&S jockey rating and last-50 strike rate | 0.25 L per rating point, 3 L per 100 % strike-rate difference |
| Trainer | R&S trainer rating and last-50 strike rate | 0.15 L per point, 2 L per 100 % strike-rate difference |
| Combos | Jockey/trainer and jockey/horse records | small, capped ±0.5 L |
| Going | Record on today's going (Soft ↔ Heavy form transfers), turf vs synthetic | shrunk success rate vs career, ±1.2 L cap |
| Distance | Raced-distance range, distance record, winning distances, course & distance | −0.5 L per 100 m beyond the longest trip tried |
| Late speed | Speed Map AFS | 0.35 L per standard deviation |

The fundamental rating is centred and its spread is capped at {rm.MAX_RATING_SPREAD} L (standard deviation)
so no single component can make the model over-confident. **The market is then blended in at
{int(meta['market_weight'] * 100)} %** (you set it; the hard cap is {int(rm.MARKET_WEIGHT_CAP * 100)} %).
Win and place probabilities come from 20,000 simulated finishes with {sigma:.1f} L of race-day noise.

**Animation.** Early positions use the Speed Map's early-speed values (AES) plus where each horse usually
settles; the finish uses the ratings; the turn blends the two, with high-AFS closers held back longer.
Gaps between horses are exaggerated 3.5× on the track drawing so the field is readable; the running-order
panel shows the true lengths.
""")
    st.caption("Ratings and animations are model output for entertainment and analysis. Check fields, "
               "scratchings and track conditions with an official source.")
