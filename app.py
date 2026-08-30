"""Top Race Predictor — the HorseRacingTextPredictor engine, on the web."""
from __future__ import annotations

import io

import pandas as pd
import streamlit as st

st.set_page_config(page_title="Top Race Predictor",
                   page_icon=":material/trophy:", layout="wide")

# Streamlit Cloud pulls new files but keeps imported modules in sys.modules, so
# a deploy can run new app code against stale helpers. Fail loudly with the fix.
try:
    import engine as E
    import meeting as M
    for _m, _a in [(E, "analyse_race_text"), (M, "split_races")]:
        if not hasattr(_m, _a):
            raise ImportError(f"{_m.__name__} is missing {_a}")
    if "uncertainty_scale" not in E.analyse_race_text.__code__.co_varnames:
        raise ImportError("engine.analyse_race_text has no uncertainty_scale")
except ImportError as exc:
    st.error(f"**Stale modules** ({exc}).\n\nOn Streamlit Cloud: "
             "**Manage app → ⋮ → Reboot app**. A rerun is not enough.",
             icon=":material/error:")
    st.stop()

CALIBRATED = 2.0
FAV_STRIKE = 37.3          # measured, see the Accuracy tab
RAW_MEAN = 60.3
CAL_MEAN = 38.5

SAMPLE_MEETING = "sample_data/2026-08-30-CARNARVON-T.xlsx"
SAMPLE_TEXT = "sample_data/sample_race.txt"


@st.cache_data(show_spinner=False)
def split_meeting(data: bytes, name: str):
    df = M.read_meeting(io.BytesIO(data), filename=name)
    return M.split_races(df), M.meeting_label(df)


@st.cache_data(show_spinner=False)
def analyse(text: str, sims: int, scale: float, overrides: tuple,
            higher_better: bool):
    dist, surf, going, hcap = overrides
    ov = {}
    if dist:
        ov["distance_m"] = dist
    if surf:
        ov["surface"] = surf
    if going:
        ov["going"] = going
    if hcap is not None:
        ov["is_handicap"] = hcap
    return E.analyse_race_text(text, simulations=sims, race_overrides=ov or None,
                               higher_rating_is_better=higher_better,
                               uncertainty_scale=scale)


# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.title("Top Race Predictor")
    st.caption("Adaptive scorecard and Monte Carlo, from a pasted race table "
               "or a Racing & Sports meeting file.")

    src = st.radio("Input", ["Meeting file", "Paste a race"], horizontal=True)
    up = None
    if src == "Meeting file":
        up = st.file_uploader("`-T.xlsx` meeting export", type=["xlsx", "csv"])
        use_sample = st.toggle("Use the bundled Carnarvon meeting", value=False)
    else:
        use_sample = False

    st.divider()
    calibrate = st.toggle(
        "Calibrated confidence", value=True,
        help="Scales the simulation noise so the top pick's stated win "
             "probability matches the strike rate favourites actually "
             "achieve. Changes the numbers, not the selections.")
    sims = st.select_slider("Monte Carlo runs",
                            [2000, 5000, 10000, 25000, 50000], value=25000)

    with st.expander("Race context overrides"):
        o_dist = st.number_input("Distance (m), 0 = detected", 0, 5000, 0, 50)
        o_surf = st.selectbox("Surface", ["detected", "Turf", "Dirt",
                                          "All-weather", "Synthetic"])
        o_going = st.selectbox("Going", ["detected", "Firm", "Good", "Soft",
                                         "Heavy", "Standard"])
        o_hcap = st.selectbox("Handicap", ["detected", "yes", "no"])
        higher_better = not st.checkbox(
            "JRat / TRat: lower is better", value=False,
            help="Some sources rank 1 as the strongest.")

    scale = CALIBRATED if calibrate else 1.0
    overrides = (int(o_dist) or None,
                 None if o_surf == "detected" else o_surf,
                 None if o_going == "detected" else o_going,
                 None if o_hcap == "detected" else (o_hcap == "yes"))

    if calibrate:
        st.success(f"Top pick averages **{CAL_MEAN:.0f}%**, against the "
                   f"**{FAV_STRIKE:.0f}%** that favourites really win.",
                   icon=":material/verified:")
    else:
        st.warning(f"Original engine. Top pick averages **{RAW_MEAN:.0f}%** "
                   f"in fields of 11 — see the Accuracy tab.",
                   icon=":material/warning:")

in_tab, pred_tab, break_tab, weight_tab, acc_tab = st.tabs(
    ["Race input", "Predictions", "Factor breakdown", "Weights & method",
     "Accuracy"])

# --------------------------------------------------------------- race input
race_text = ""
with in_tab:
    if src == "Paste a race":
        default = ""
        if st.button("Load the included sample race"):
            st.session_state["pasted"] = open(SAMPLE_TEXT, encoding="utf-8").read()
        race_text = st.text_area(
            "Paste the race table", height=320,
            value=st.session_state.get("pasted", default),
            placeholder="Race header, then the row beginning `Tab  Horse  "
                        "Form L5  BP` and one tab-separated row per runner.")
    else:
        blob = name = None
        if up is not None:
            blob, name = up.getvalue(), up.name
        elif use_sample:
            blob, name = open(SAMPLE_MEETING, "rb").read(), SAMPLE_MEETING
        if blob is None:
            st.info("Upload a `-T.xlsx` meeting export, or switch on the "
                    "bundled Carnarvon meeting in the sidebar.",
                    icon=":material/upload_file:")
        else:
            try:
                blocks, label = split_meeting(blob, name)
            except Exception as exc:                          # noqa: BLE001
                st.error(f"Could not read that file — "
                         f"{type(exc).__name__}: {exc}")
                blocks, label = [], ""
            if not blocks:
                st.error("No races were found. The file should hold the "
                         "`Tab Horse Form L5 BP` header for each race.")
            else:
                st.success(f"**{label}** — {len(blocks)} races",
                           icon=":material/table_view:")
                lut = {f"R{b.number} · {b.runners} runners · {b.name}": b
                       for b in blocks}
                choice = st.selectbox("Race", list(lut))
                race_text = lut[choice].text
                with st.expander("What the engine sees"):
                    st.code(race_text, language="text")

# The Accuracy tab is rendered at the end whatever happens, so a race that
# fails to parse still leaves the honest numbers on screen.
a = None
if race_text.strip():
    try:
        a = analyse(race_text, int(sims), scale, overrides, higher_better)
    except Exception as exc:                                  # noqa: BLE001
        with pred_tab:
            st.error(f"The engine could not score that race — "
                     f"{type(exc).__name__}: {exc}")
else:
    with pred_tab:
        st.info("Load a race first.", icon=":material/info:")

# --------------------------------------------------------------- predictions
if a is not None:
    rows = pd.DataFrame([{
        "Rank": p.rank, "Tab": p.tab, "Horse": p.horse,
        "Score": p.model_score, "Win %": p.win_pct, "Top-3 %": p.top3_pct,
        "Fair odds": p.fair_odds, "Data %": p.data_completeness,
    } for p in a.predictions])

    with pred_tab:
        r = a.race
        c = st.columns(5)
        c[0].metric("Distance", f"{r.distance_m or '—'} m")
        c[1].metric("Surface", r.surface or "—")
        c[2].metric("Going", r.going or "—")
        c[3].metric("Field", r.field_size)
        c[4].metric("Confidence", a.confidence_label)
        st.caption(a.parser_summary)

        top = a.predictions[0]
        st.success(f"**{top.horse}** (tab {top.tab}) — {top.win_pct:.1f}% win, "
                   f"{top.top3_pct:.1f}% top three"
                   + (f", fair odds ${top.fair_odds:.2f}" if top.fair_odds else ""),
                   icon=":material/trophy:")
        st.markdown(a.verdict)

        for w in a.warnings:
            st.warning(w, icon=":material/report:")

        st.dataframe(
            rows, width="stretch", hide_index=True,
            column_config={
                "Win %": st.column_config.ProgressColumn(
                    "Win %", format="%.1f%%", min_value=0.0,
                    max_value=float(max(rows["Win %"].max(), 1))),
                "Top-3 %": st.column_config.NumberColumn(format="%.1f%%"),
                "Score": st.column_config.NumberColumn(format="%.1f"),
                "Fair odds": st.column_config.NumberColumn(format="$%.2f"),
                "Data %": st.column_config.NumberColumn(format="%.0f%%"),
            })

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Exacta structures**")
            st.write(", ".join(a.exacta) or "—")
        with c2:
            st.markdown("**Trifecta structures**")
            st.write(" | ".join(a.trifecta) or "—")

        d1, d2 = st.columns(2)
        d1.download_button("Download ranking (CSV)", rows.to_csv(index=False),
                           "top_race_predictor_ranking.csv", "text/csv")
        d2.download_button("Download full report (TXT)",
                           E.analysis_report_text(a),
                           "top_race_predictor_report.txt", "text/plain")

    # ----------------------------------------------------- factor breakdown
    with break_tab:
        pick = st.selectbox("Runner", [f"{p.tab}. {p.horse}"
                                       for p in a.predictions])
        p = a.predictions[[f"{q.tab}. {q.horse}"
                           for q in a.predictions].index(pick)]
        c = st.columns(4)
        c[0].metric("Rank", p.rank)
        c[1].metric("Score", f"{p.model_score:.1f}")
        c[2].metric("Win %", f"{p.win_pct:.1f}%")
        c[3].metric("Data completeness", f"{p.data_completeness:.0f}%")
        s1, s2 = st.columns(2)
        s1.markdown("**Strengths**\n" + "\n".join(f"- {x}" for x in p.strengths))
        s2.markdown("**Risks**\n" + "\n".join(f"- {x}" for x in p.risks))
        st.dataframe(
            pd.DataFrame([{
                "Factor": d.label, "Raw": d.raw_value, "Score": d.score,
                "Weight": d.weight, "Contribution": d.contribution,
                "Available": d.available, "Note": d.note,
            } for d in p.factor_details]),
            width="stretch", hide_index=True,
            column_config={
                "Score": st.column_config.NumberColumn(format="%.0f"),
                "Weight": st.column_config.NumberColumn(format="%.1f"),
                "Contribution": st.column_config.NumberColumn(format="%.2f"),
            })

    # ------------------------------------------------------ weights & method
    with weight_tab:
        st.subheader("Context-adjusted weights")
        st.caption("The scorecard starts from a fixed allocation and shifts it "
                   "with the race conditions — sprints raise barrier and "
                   "sharpness, staying races raise distance record and "
                   "fitness, soft going raises the soft/heavy record.")
        st.dataframe(
            pd.DataFrame([{
                "Factor": E.FACTOR_LABELS.get(k, k),
                "Base": E.BASE_WEIGHTS.get(k, 0.0),
                "This race": v,
                "Change": v - E.BASE_WEIGHTS.get(k, 0.0),
                "Why": a.weight_notes.get(k, ""),
            } for k, v in sorted(a.weights.items(),
                                 key=lambda kv: -kv[1])]),
            width="stretch", hide_index=True,
            column_config={
                "Base": st.column_config.NumberColumn(format="%.1f"),
                "This race": st.column_config.NumberColumn(format="%.1f"),
                "Change": st.column_config.NumberColumn(format="%+.1f"),
            })

# ------------------------------------------------------------------ accuracy
with acc_tab:
    st.subheader("Tested against actual results")
    st.markdown(f"""
The engine was run over **CARNARVON (WA), 30 August 2026**, races 1–5, and
scored against the finishing order.

| race | field | its pick | stated win % | actual winner | where its pick finished |
|---|---|---|---|---|---|
| 1 | 9 | 1 Any Questions | 78.0% | tab 4 | unplaced |
| 2 | 7 | 5 Majestuoso Phoenix | 62.4% | tab 7 | unplaced |
| 3 | 11 | 2 Awesome Lily | 37.7% | tab 6 | unplaced |
| 4 | 12 | 5 Solar System | 52.5% | tab 7 | **placed** |
| 5 | 10 | 2 Hard Questions | 55.5% | tab 6 | unplaced |

**Top pick won 0 of 5. It placed once.** The actual winner made the model's top
three once in five races.

### What five races can and cannot tell you

They **cannot** tell you the selections are bad. A model with no skill at all
goes 0-for-5 in these field sizes **57%** of the time. That result is
unremarkable and proves nothing either way.

They **can** reject the confidence. If the stated probabilities above were
right, the expected number of winners was **2.9 of 5**, and going 0-for-5 has a
probability of about **1%**. The selections are unproven; the percentages are
wrong.

### The confidence problem, measured without any results

Across **119 races from 16 meetings**, the original engine's top pick averages
a stated win probability of **{RAW_MEAN:.1f}%** in fields averaging 11.3
runners. It claims over 80% in **18%** of races, and as high as 99.9%.

For comparison, from **7,787 past runs** parsed out of Racing & Sports
Australian form guides, the **market favourite wins {FAV_STRIKE:.1f}%** of the
time in fields averaging 10.3. The favourite is the single best-informed
selection available anywhere. Nothing suggests this scorecard beats it, yet it
routinely claims to be far more certain.

The cause is one uncalibrated number. The simulation draws each runner's
performance as `score + gauss(0, σ)` with σ ≈ 6.5 points. The spread of scores
across a field is about that size, so the top-scorer wins nearly every
simulated running. σ was never fitted to anything.

### What the calibration does

Multiplying σ by **{CALIBRATED:.1f}** brings the average top-pick probability to
**{CAL_MEAN:.1f}%**, in line with the {FAV_STRIKE:.1f}% favourites really
achieve, and drops the share of >80% claims from 18% to 1%.

It **does not change the selections**: across those 119 races the top pick is
the same in **98%** and the top three are the same set in **92%**. Only the
confidence moves. That is why it is on by default, with the original one click
away in the sidebar.

### What is still not known

Five races is not a measurement of whether the picks are any good. To answer
that properly, the finishing order for the other 15 meetings on file — about
**114 more races** — would settle it at a useful precision. Send the results
strips and it can be scored.

---

*This is a transparent heuristic scorecard, not a trained or calibrated betting
model — the original README says so too. Probabilistic decision support, not a
guaranteed outcome. Gamble responsibly — Gambling Help 1800 858 858,
[gamblinghelponline.org.au](https://www.gamblinghelponline.org.au).*
""")
