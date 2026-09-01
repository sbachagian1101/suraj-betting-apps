"""SoccerPredictorPro - fixture prediction from FootyStats season exports.

Upload two (or three) seasons, pick a home and an away team, and get 1X2,
half-time, HT/FT and Asian-handicap probabilities read off one shared score
distribution, each with a confidence band.

Over/Under and BTTS are absent on purpose - see the note in the Prediction tab.
"""
from __future__ import annotations

import glob
import importlib
import io
import os

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="SoccerPredictorPro",
                   page_icon=":material/sports_soccer:",
                   layout="wide", initial_sidebar_state="expanded")

# --- stale-module guard -------------------------------------------------------
# Streamlit Cloud pulls new files but keeps already-imported modules in
# sys.modules, so a commit that ADDS a name to a helper can leave the new
# app.py running against the old module and fail with an opaque AttributeError
# deep inside whichever tab uses it. This must run BEFORE any `import spp_*`,
# or the cached module raises on the import line and the guard never runs.
_REQUIRED = {
    "spp_data": ("load", "teams", "KIND_TEAM", "KIND_MATCH", "sniff"),
    "spp_model": ("build", "predict", "asian_handicap", "htft", "half_matrices",
                  "confidence", "result_probs", "HTFT_LABELS", "CONF_COLOUR",
                  "fair_line", "expected_goals", "DEFAULT_RHO"),
}
_problems: list[str] = []
_mods: dict[str, object] = {}
for _name, _need in _REQUIRED.items():
    try:
        _m = importlib.import_module(_name)
    except Exception as exc:                                        # noqa: BLE001
        _problems.append(f"`{_name}` failed to import ({type(exc).__name__})")
        continue
    _mods[_name] = _m
    _missing = [n for n in _need if not hasattr(_m, n)]
    if _missing:
        _problems.append(f"`{_name}` is missing `" + "`, `".join(_missing) + "`")
if _problems:
    st.error("**This deployment is running stale code.** " + "; ".join(_problems) +
             ". Streamlit Cloud pulled the new files but kept the old modules in "
             "memory. Fix it with **Manage app → ⋮ → Reboot app**.",
             icon=":material/error:")
    st.stop()

spd = _mods["spp_data"]
spm = _mods["spp_model"]

_HERE = os.path.dirname(os.path.abspath(__file__))

HOME_C, DRAW_C, AWAY_C = "#2E86DE", "#F2B705", "#E5476D"
RESULT_COLOURS = [HOME_C, DRAW_C, AWAY_C]

st.markdown("""
<style>
.block-container {padding-top: 1.4rem; padding-bottom: 3rem;}
div[data-testid="stMetric"] {border:1px solid rgba(128,128,128,.22);
  border-radius:.8rem; padding:.6rem .9rem; background:rgba(128,128,128,.05);}
.pick-card {border-radius:1rem; padding:1.1rem 1.4rem; margin:.3rem 0 1rem;
  background:linear-gradient(100deg, rgba(46,134,222,.16), rgba(229,71,109,.13));
  border:1px solid rgba(128,128,128,.25);}
.pick-head {font-size:.72rem; letter-spacing:.14em; text-transform:uppercase; opacity:.7;}
.pick-body {font-size:1.55rem; font-weight:700; line-height:1.25; margin-top:.15rem;}
.pick-why {font-size:.88rem; opacity:.75; margin-top:.3rem;}
.muted {opacity:.62; font-size:.8rem;}
</style>
""", unsafe_allow_html=True)


# --- data plumbing ------------------------------------------------------------

@st.cache_data(show_spinner="Reading your files…")
def load_uploads(payloads):
    return spd.load([io.BytesIO(b) for _, b in payloads], [n for n, _ in payloads])


@st.cache_data(show_spinner="Reading the bundled sample league…")
def load_sample(pattern):
    return spd.load(sorted(glob.glob(os.path.join(_HERE, "sample_data", pattern))))


@st.cache_data(show_spinner="Building team ratings…")
def build_ratings(kind, frame):
    return spm.build(kind, frame)


def pct(x: float) -> str:
    return f"{100 * float(x):.1f}%"


def conf_badge(conf: dict) -> str:
    return f":{conf['colour']}-badge[{conf['band']} confidence]"


# --- sidebar ------------------------------------------------------------------

with st.sidebar:
    st.markdown("### :material/database: Data")
    ups = st.file_uploader(
        "Season CSVs", type=["csv"], accept_multiple_files=True,
        help="FootyStats exports. Either the **team** files "
             "(`...-teams-...csv`) or the **match** files (`...-matches-...csv`) — "
             "the app detects which you gave it. Two seasons is the measured "
             "sweet spot: last season plus the current one.")
    use_sample = st.toggle("Use the bundled sample league", value=not ups,
                           help="Iranian Persian Gulf Pro League, so you can try "
                                "the app before uploading anything.")
    sample_kind = st.radio("Sample shape", ["Team files", "Match files"],
                           horizontal=True, disabled=not use_sample,
                           label_visibility="collapsed" if not use_sample else "visible")

    st.markdown("### :material/tune: Model")
    st.caption("Two seasons beat one, and beat three. Adding a third measured a "
               "dead heat (1.0284 vs 1.0307 log-loss); all seven seasons was "
               "worse than two.")

if ups:
    kind, frame, notes = load_uploads(tuple((f.name, f.getvalue()) for f in ups))
elif use_sample:
    pat = "*-teams-*.csv" if sample_kind == "Team files" else "*-matches-*.csv"
    kind, frame, notes = load_sample(pat)
else:
    kind, frame, notes = None, pd.DataFrame(), []

st.title("SoccerPredictorPro")
st.caption("Team-strength modelling for 1X2, half-time, HT/FT and Asian handicap — "
           "every market read off one score distribution.")

tab_data, tab_pred = st.tabs([":material/upload_file: Data & fixture",
                              ":material/insights: Prediction"])

# --- tab 1 --------------------------------------------------------------------

with tab_data:
    if frame.empty:
        st.info("Upload your season CSVs in the sidebar, or switch on the bundled "
                "sample league to try the app.", icon=":material/info:")
        team_list, ratings = [], None
    else:
        label = "team-season tables" if kind == spd.KIND_TEAM else "individual matches"
        st.success(f"Loaded **{len(frame):,}** {label} — "
                   f"detected as **{kind} files**.", icon=":material/check_circle:")
        for n in notes:
            st.caption(n)

        ratings = build_ratings(kind, frame)
        team_list = spd.teams(kind, frame)

        # A prediction left over from a different upload would name teams that
        # are no longer on the dropdowns, so drop it when the data changes.
        sig = (kind, tuple(team_list))
        if st.session_state.get("data_sig") != sig:
            st.session_state["data_sig"] = sig
            st.session_state.pop("prediction", None)

        st.markdown("### :material/stadium: Pick the fixture")
        c1, c2, c3 = st.columns([3, 3, 2], vertical_alignment="bottom")
        with c1:
            home = st.selectbox("Home team", team_list, index=0,
                                key="home_team", help="The side playing at home.")
        with c2:
            away_opts = [t for t in team_list if t != home]
            away = st.selectbox("Away team", away_opts,
                                index=min(1, len(away_opts) - 1) if away_opts else 0,
                                key="away_team")
        with c3:
            go = st.button("Predict", type="primary", width="stretch",
                           icon=":material/play_arrow:")

        if go:
            try:
                st.session_state["prediction"] = spm.predict(ratings, home, away)
                st.session_state["pred_meta"] = {"kind": kind, "source": ratings.source}
            except KeyError as exc:
                st.error(str(exc), icon=":material/error:")
            else:
                st.success("Prediction ready — open the **Prediction** tab above.",
                           icon=":material/insights:")

        with st.expander("Team ratings behind the model", icon=":material/table_chart:"):
            if kind == spd.KIND_TEAM:
                rows = [{"Team": t,
                         "Matches": ratings.sample(t),
                         "Scored/home": ratings.atk_home[t], "Conceded/home": ratings.def_home[t],
                         "Scored/away": ratings.atk_away[t], "Conceded/away": ratings.def_away[t]}
                        for t in ratings.teams]
                rt = pd.DataFrame(rows).sort_values("Scored/home", ascending=False)
            else:
                import soccer_data as sd
                rt = sd.team_table(frame, ratings.fitted)
            st.dataframe(rt.round(3), width="stretch", hide_index=True)
            st.caption(f"League baseline — home {ratings.lg_home:.2f} goals, away "
                       f"{ratings.lg_away:.2f}. First-half share of goals "
                       f"{100 * ratings.ht_share:.0f}%. Dixon–Coles ρ {ratings.rho:+.3f} "
                       + ("(fitted on your scorelines)." if kind == spd.KIND_MATCH else
                          f"(team files carry no individual scorelines, so the measured "
                          f"default {spm.DEFAULT_RHO} is used)."))

# --- tab 2 --------------------------------------------------------------------

with tab_pred:
    p = st.session_state.get("prediction")
    if not p:
        st.info("Pick a fixture on the **Data & fixture** tab and press Predict.",
                icon=":material/insights:")
        st.stop()

    home, away = p["home"], p["away"]
    ft, ht = p["ft"], p["ht"]
    names = {"home": home, "draw": "Draw", "away": away}

    top_key = max(ft, key=ft.get)
    conf = p["conf_ft"]
    st.markdown(
        f"<div class='pick-card'><div class='pick-head'>Most likely result · "
        f"{home} vs {away}</div>"
        f"<div class='pick-body'>{names[top_key]} — {pct(ft[top_key])}</div>"
        f"<div class='pick-why'>{conf['band']} confidence · {conf['why']} · "
        f"expected goals {p['lambda_home']:.2f} – {p['lambda_away']:.2f}</div></div>",
        unsafe_allow_html=True)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric(f"{home} win", pct(ft["home"]))
    m2.metric("Draw", pct(ft["draw"]))
    m3.metric(f"{away} win", pct(ft["away"]))
    m4.metric("Fair handicap", f"{p['fair_line']:+.2f}",
              help="The Asian-handicap line at which this fixture is closest to a coin flip.")

    # ---- 1X2: donut + bar ----
    st.markdown("### :material/pie_chart: Full-time result " + conf_badge(p["conf_ft"]))
    left, right = st.columns([1, 1])
    res_df = pd.DataFrame({"Outcome": [home, "Draw", away],
                           "Probability": [ft["home"], ft["draw"], ft["away"]]})
    res_df["Label"] = res_df["Probability"].map(pct)
    scale = alt.Scale(domain=[home, "Draw", away], range=RESULT_COLOURS)

    with left:
        donut = (alt.Chart(res_df)
                 .mark_arc(innerRadius=68, outerRadius=110, stroke="white", strokeWidth=2)
                 .encode(theta=alt.Theta("Probability:Q", stack=True),
                         color=alt.Color("Outcome:N", scale=scale,
                                         legend=alt.Legend(orient="bottom", title=None)),
                         tooltip=["Outcome", alt.Tooltip("Probability:Q", format=".1%")])
                 .properties(height=290))
        st.altair_chart(donut, width="stretch")
    with right:
        bars = (alt.Chart(res_df).mark_bar(cornerRadiusEnd=7, height=42)
                .encode(x=alt.X("Probability:Q", axis=alt.Axis(format="%"),
                                scale=alt.Scale(domain=[0, 1]), title=None),
                        y=alt.Y("Outcome:N", sort=None, title=None),
                        color=alt.Color("Outcome:N", scale=scale, legend=None),
                        tooltip=["Outcome", alt.Tooltip("Probability:Q", format=".1%")])
                .properties(height=290))
        text = bars.mark_text(align="left", dx=6, fontWeight="bold").encode(
            text="Label:N", color=alt.value("#888"))
        st.altair_chart(bars + text, width="stretch")

    # ---- half time ----
    st.markdown("### :material/timer: Half-time result " + conf_badge(p["conf_ht"]))
    cmp_df = pd.DataFrame({
        "Outcome": [home, "Draw", away] * 2,
        "Stage": ["Half time"] * 3 + ["Full time"] * 3,
        "Probability": [ht["home"], ht["draw"], ht["away"], ft["home"], ft["draw"], ft["away"]],
    })
    grouped = (alt.Chart(cmp_df).mark_bar(cornerRadiusEnd=5)
               .encode(x=alt.X("Stage:N", title=None, axis=alt.Axis(labelAngle=0)),
                       y=alt.Y("Probability:Q", axis=alt.Axis(format="%"),
                               scale=alt.Scale(domain=[0, 1]), title=None),
                       color=alt.Color("Outcome:N", scale=scale,
                                       legend=alt.Legend(orient="bottom", title=None)),
                       xOffset=alt.XOffset("Outcome:N", sort=[home, "Draw", away]),
                       tooltip=["Stage", "Outcome", alt.Tooltip("Probability:Q", format=".1%")])
               .properties(height=300))
    hc1, hc2 = st.columns([2, 1])
    hc1.altair_chart(grouped, width="stretch")
    hc2.metric("Half-time draw", pct(ht["draw"]),
               help="Matches are level at half time far more often than at full time, "
                    "which is why this is usually the largest single half-time number.")
    hc2.caption(f"Modelled by splitting the full-time expectation, with "
                f"**{100 * p['ht_share']:.0f}%** of goals arriving before the break — "
                "measured from your own upload.")
    if max(ht, key=ht.get) == "draw" and ht["draw"] < 0.60:
        hc2.caption(":orange-badge[Note] Level at the break is football's default "
                    "state, so a high-confidence half-time draw is largely the "
                    "league base rate rather than a read on these two teams.")

    # ---- HT/FT ----
    st.markdown("### :material/grid_view: Half-time / full-time " + conf_badge(p["conf_htft"]))
    j = p["htft"]
    lab = {"H": home, "D": "Draw", "A": away}
    jr = [{"Half time": lab[h], "Full time": lab[f], "Probability": float(j[i // 3, i % 3])}
          for i, (h, f) in enumerate(spm.HTFT_LABELS)]
    jdf = pd.DataFrame(jr)
    order = [home, "Draw", away]
    heat = (alt.Chart(jdf).mark_rect(cornerRadius=5, stroke="white", strokeWidth=3)
            .encode(x=alt.X("Full time:N", sort=order, axis=alt.Axis(labelAngle=0)),
                    y=alt.Y("Half time:N", sort=order),
                    color=alt.Color("Probability:Q",
                                    scale=alt.Scale(scheme="viridis"),
                                    legend=alt.Legend(format="%", title="Probability")),
                    tooltip=["Half time", "Full time",
                             alt.Tooltip("Probability:Q", format=".1%")])
            .properties(height=300))
    # Viridis runs dark-purple (low) to bright-yellow (high), so the LIGHT cells
    # are the high-probability ones and need dark text, not white.
    lbl = heat.mark_text(fontSize=15, fontWeight="bold").encode(
        text=alt.Text("Probability:Q", format=".1%"),
        color=alt.condition(alt.datum.Probability > 0.14,
                            alt.value("#18202b"), alt.value("white")))
    jc1, jc2 = st.columns([2, 1])
    jc1.altair_chart(heat + lbl, width="stretch")
    best = jdf.loc[jdf["Probability"].idxmax()]
    jc2.metric(f"Most likely path · {best['Half time']} → {best['Full time']}",
               pct(best["Probability"]))
    turn = jdf[(jdf["Half time"] != jdf["Full time"])]["Probability"].sum()
    jc2.metric("Result changes after the break", pct(turn),
               help="Chance the half-time state (leading, level or trailing) does not "
                    "hold to full time.")

    # ---- Asian handicap ----
    st.markdown("### :material/balance: Asian handicap " + conf_badge(p["conf_ah"]))
    ah = p["asian"].copy()
    ah["Line"] = ah["line"].map(lambda v: f"{v:+.2f}".replace("+0.00", "0.00"))
    ah["Home covers"] = ah["home_no_push"]
    band = ah.dropna(subset=["Home covers"])
    ahc = (alt.Chart(band).mark_area(
                line={"color": HOME_C, "strokeWidth": 3}, opacity=0.30,
                color=alt.Gradient(gradient="linear",
                                   stops=[alt.GradientStop(color="white", offset=0),
                                          alt.GradientStop(color=HOME_C, offset=1)],
                                   x1=1, x2=1, y1=1, y2=0))
           .encode(x=alt.X("line:Q", title=f"Handicap applied to {home}",
                           scale=alt.Scale(nice=False)),
                   y=alt.Y("Home covers:Q", axis=alt.Axis(format="%"),
                           scale=alt.Scale(domain=[0, 1]),
                           title=f"{home} covers (stake back on a push excluded)"),
                   tooltip=[alt.Tooltip("line:Q", title="Line", format="+.2f"),
                            alt.Tooltip("Home covers:Q", format=".1%")])
           .properties(height=310))
    rule = alt.Chart(pd.DataFrame({"y": [0.5]})).mark_rule(
        strokeDash=[6, 4], color=DRAW_C, strokeWidth=2).encode(y="y:Q")
    fair = alt.Chart(pd.DataFrame({"x": [p["fair_line"]]})).mark_rule(
        color=AWAY_C, strokeWidth=2).encode(x="x:Q")
    ac1, ac2 = st.columns([2, 1])
    ac1.altair_chart(ahc + rule + fair, width="stretch")
    with ac2:
        st.metric("Fair line", f"{p['fair_line']:+.2f}",
                  help="Where the amber 50% line crosses — the handicap that makes "
                       "this fixture an even-money proposition.")
        st.metric("Level handicap (draw no bet)", pct(p["p_level"]),
                  help=f"Chance {home} wins with the stake returned on a draw. "
                       "This is what the confidence band above is measured on.")
        show = ah[ah["line"].isin([-1.5, -1.0, -0.5, -0.25, 0.0, 0.25, 0.5, 1.0, 1.5])]
        st.dataframe(
            show[["Line", "home_win", "push", "away_win"]]
            .rename(columns={"home_win": f"{home} ✓", "push": "Push",
                             "away_win": f"{away} ✓"}),
            hide_index=True, width="stretch",
            # "percent" scales the fraction; a printf like "%.1f%%" formats the
            # raw 0.206 and renders a 20.6% push as "0.2%".
            column_config={
                "Push": st.column_config.NumberColumn(format="percent"),
                f"{home} ✓": st.column_config.ProgressColumn(
                    format="percent", min_value=0.0, max_value=1.0),
                f"{away} ✓": st.column_config.ProgressColumn(
                    format="percent", min_value=0.0, max_value=1.0)})

    # ---- scorelines ----
    st.markdown("### :material/sports_score: Most likely scorelines")
    sc1, sc2 = st.columns([1, 1])
    tops = pd.DataFrame([{"Score": f"{a}–{b}", "Probability": c}
                         for a, b, c in p["top_scorelines"]])
    tops["Label"] = tops["Probability"].map(pct)
    tb = (alt.Chart(tops).mark_bar(cornerRadiusEnd=6, color=HOME_C)
          .encode(x=alt.X("Probability:Q", axis=alt.Axis(format="%"), title=None),
                  y=alt.Y("Score:N", sort="-x", title=None),
                  tooltip=["Score", alt.Tooltip("Probability:Q", format=".1%")])
          .properties(height=270))
    sc1.altair_chart(tb + tb.mark_text(align="left", dx=5).encode(
        text="Label:N", color=alt.value("#888")), width="stretch")

    mm = p["ft_matrix"][:6, :6]
    grid = pd.DataFrame([{"Home": i, "Away": jj, "Probability": float(mm[i, jj])}
                         for i in range(6) for jj in range(6)])
    # Most cells of a 6x6 score grid sit near zero, so a linear scale renders the
    # whole thing as one dark block. A square-root scale spreads the low end and
    # makes the shape of the distribution visible.
    gh = (alt.Chart(grid).mark_rect(stroke="white", strokeWidth=1.5)
          .encode(x=alt.X("Away:O", title=f"{away} goals",
                          axis=alt.Axis(labelAngle=0)),
                  y=alt.Y("Home:O", title=f"{home} goals", sort="descending"),
                  color=alt.Color("Probability:Q",
                                  scale=alt.Scale(scheme="tealblues", type="sqrt"),
                                  legend=alt.Legend(format="%", title=None)),
                  tooltip=[alt.Tooltip("Home:O"), alt.Tooltip("Away:O"),
                           alt.Tooltip("Probability:Q", format=".2%")])
          .properties(height=270))
    sc2.altair_chart(gh, width="stretch")

    # ---- honesty footer ----
    with st.expander("What this model does and does not do",
                     icon=":material/help_outline:"):
        st.markdown(f"""
**Where the numbers come from.** Ratings were built from your {p['source']},
giving {home} an expected **{p['lambda_home']:.2f}** goals and {away}
**{p['lambda_away']:.2f}**. Every market on this page is read off the same score
distribution, so they cannot contradict one another — the half-time and HT/FT
figures are raked onto the full-time ones exactly.

**Why there is no Over/Under or BTTS.** Measured walk-forward on 1,162 matches,
every goals market scored **worse than simply quoting the league base rate** —
Over 2.5 at 0.6346 against a 0.6189 base, BTTS at 0.6905 against 0.6679. The
bookmaker cannot beat the base rate on them either. Results depend on the
*difference* between two teams' strengths, which ratings capture; total goals
depends on the *sum*, which nothing in this data predicts.

**How good is it.** On the bundled league the full-time 1X2 model was right
**46.5%** of the time against a **35.3%** base rate — but the bookmaker's own
odds were right **48.4%**. This model is a genuine edge over guessing and it is
still **behind the closing line**. Treat the confidence bands as the model's own
certainty, not as a promise about hit rate.

**Confidence bands** combine how far the leading outcome is clear of the next
one with how many matches sit behind the two teams' ratings
({p['n_matches']:.0f} here). The weaker of the two decides the band.
""")
