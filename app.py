"""DogForm - paste a Racing & Sports greyhound Full Fields page, get a rating."""

from __future__ import annotations

import glob
import importlib
import io
import os

import altair as alt
import pandas as pd
import streamlit as st

import rating
import rs_parser

st.set_page_config(page_title="DogForm", page_icon="🐕", layout="wide")

HERE = os.path.dirname(os.path.abspath(__file__))

# --- stale-deployment guard ---------------------------------------------------
# Streamlit Cloud has repeatedly served a NEW app.py against a CACHED helper
# module after a push that added a module-level name, producing an opaque
# AttributeError halfway down the page.  Turn that into one actionable line.
_REQUIRED = {
    "rs_parser": ["parse", "Race", "Runner", "STRAIGHT_TRACKS"],
    "rating": ["rate", "Params", "quinellas", "sensitivity", "Rated"],
}
_missing = []
for _mod, _names in _REQUIRED.items():
    _m = importlib.import_module(_mod)
    _missing += [f"{_mod}.{n}" for n in _names if not hasattr(_m, n)]
if _missing:
    st.error(
        "This deployment is running stale code: " + ", ".join(_missing)
        + " is missing. Open **Manage app → ⋮ → Reboot app** to force a clean "
          "rebuild. The pushed source is fine.")
    st.stop()


# --- helpers ------------------------------------------------------------------

def samples() -> dict[str, str]:
    out = {}
    for path in sorted(glob.glob(os.path.join(HERE, "fixtures", "*.txt"))):
        out[os.path.basename(path).replace(".txt", "")] = path
    return out


def money(v: float | None, dp: int = 2) -> str:
    """Format a price for a *markdown* context.

    Streamlit renders markdown with LaTeX enabled, so two bare `$` on one line
    are swallowed as inline math ("$10 against a fair $8.60" loses both signs).
    Escape them.
    """
    if not v:
        return "—"
    return "\\$" + (f"{v:g}" if dp is None else f"{v:.{dp}f}")


def fmt_odds(v: float | None) -> str:
    """Plain (unescaped) price, for non-markdown contexts such as st.metric."""
    return f"${v:g}" if v else "—"


def prediction_frame(rated: list[rating.Rated]) -> pd.DataFrame:
    return pd.DataFrame([{
        "Box": r.box,
        "Runner": r.name,
        "Odds": r.odds,
        "Win %": r.p_final * 100,
        "Top 2 %": r.p_top2 * 100,
        "Place %": r.p_top3 * 100,
        "Fair $": r.fair,
        "EV %": (r.ev * 100) if r.ev is not None else None,
        "Model %": r.p_model * 100,
        "Market %": (r.p_market * 100) if r.p_market is not None else None,
    } for r in rated])


# --- sidebar ------------------------------------------------------------------

with st.sidebar:
    st.header("Model settings")
    st.caption("Defaults are the ones the write-up was built on. "
               "Change them to see how fragile a selection is.")

    p = rating.Params()
    p.market_weight = st.slider(
        "Weight on the form model", 0.0, 1.0, 0.38, 0.02,
        help="0 = follow the market exactly. 1 = ignore the market. "
             "The blend is a weighted geometric mean in log space.")
    p.spread = st.slider(
        "Performance spread (lengths)", 1.5, 4.0, 2.40, 0.05,
        help="Assumed SD of a runner's performance. Higher = flatter, "
             "less confident probabilities.")
    p.prior_starts = st.slider(
        "Shrinkage prior (starts)", 5.0, 30.0, 15.0, 1.0,
        help="How many starts of prior belief the distance / course / surface "
             "records are shrunk toward the career strike rate. Lower values "
             "let tiny samples swing the rating hard — that has cost a race.")

    with st.expander("Advanced"):
        p.tau_days = st.slider("Recency decay (days)", 60.0, 400.0, 200.0, 10.0)
        p.sigma_dist = st.slider("Distance relevance width (m)", 40.0, 160.0, 90.0, 5.0)
        p.w_straight = st.slider("Straight-track form weight", 0.0, 1.0, 0.35, 0.05)
        p.w_offsurface = st.slider("Off-surface form weight", 0.0, 1.0, 0.70, 0.05)
        p.w_foreign = st.slider("Overseas form weight", 0.0, 1.0, 0.75, 0.05)
        p.devig_power = st.slider("De-vig power", 1.0, 1.20, 1.06, 0.01)
        p.k_gap = st.slider(
            "Vacant-box term (lengths per empty box inside)", 0.0, 1.0, 0.0, 0.05,
            help="UNVALIDATED. In one race the finish order was exactly monotone "
                 "in vacant boxes inside each runner. That is a single "
                 "observation, so this ships at zero. Turn it on to explore.")

    run_sens = st.checkbox("Run parameter sensitivity sweep", value=True,
                           help="Re-rates the race a few hundred times with "
                                "jittered constants to see if the selection holds.")

# --- input --------------------------------------------------------------------

st.title("🐕 DogForm")
st.caption("Paste a Racing & Sports greyhound **Full Fields** page. "
           "The app reads each runner's last-10 run table, rates the field in "
           "lengths, and blends with the market.")

sample_map = samples()
c1, c2 = st.columns([3, 1])
with c2:
    pick = st.selectbox("Load a sample race", ["—"] + list(sample_map), index=0)
if pick != "—" and st.session_state.get("_loaded") != pick:
    st.session_state["raw"] = io.open(sample_map[pick], encoding="utf-8").read()
    st.session_state["_loaded"] = pick

raw = st.text_area(
    "Paste the page here (select all → copy on the Full Fields tab)",
    key="raw", height=220,
    placeholder="HomeForm GuideGreyhoundAustralia… / 340m ALL WEATHER GOOD / …")

go = st.button("Analyse race", type="primary")

if not raw or not raw.strip():
    st.info("Paste a Full Fields page, or load one of the sample races above.")
    st.stop()

race = rs_parser.parse(raw)
if not race.field_:
    st.error("No runners could be read from that paste.")
    for w in race.warnings:
        st.warning(w)
    st.stop()

rated, notes = rating.rate(race, p)
if not rated:
    for n in notes:
        st.error(n)
    st.stop()

top = rated[0]
fav = min((r for r in rated if r.odds), key=lambda r: r.odds, default=None)

# --- header -------------------------------------------------------------------

hdr = f"{race.track} R{race.race_no} — {race.dist_m}m {race.surface} {race.going}"
if race.grade:
    hdr += f" · {race.grade}"
st.subheader(hdr)
st.caption(f"{race.race_date:%A, %d %B %Y} · {len(race.field_)} runners"
           + (f" · {sum(1 for r in race.runners if r.scratched)} scratched"
              if any(r.scratched for r in race.runners) else "")
           if race.race_date else f"{len(race.field_)} runners")

with st.container(horizontal=True):
    st.metric("Selection", f"{top.box}. {top.name}", border=True)
    st.metric("Win probability", f"{top.p_final*100:.1f}%", border=True)
    st.metric("Fair price", f"${top.fair:.2f}",
              delta=(f"{top.ev*100:+.0f}% EV at {fmt_odds(top.odds)}"
                     if top.ev is not None else None),
              border=True)
    if fav:
        st.metric("Market favourite", f"{fav.box}. {fav.name}",
                  delta="agrees" if fav is top else "model disagrees",
                  delta_color="normal" if fav is top else "inverse", border=True)

for n in notes:
    st.info(n)
for w in race.warnings:
    st.warning(w, icon=":material/warning:")

tab_pred, tab_run, tab_diag, tab_method = st.tabs(
    ["Prediction", "Runners", "Diagnostics", "Method"])

# --- prediction ---------------------------------------------------------------

with tab_pred:
    df = prediction_frame(rated)
    st.dataframe(
        df, hide_index=True,
        column_config={
            "Odds": st.column_config.NumberColumn("Odds", format="$%.2f"),
            "Win %": st.column_config.ProgressColumn(
                "Win %", format="%.1f%%", min_value=0.0,
                max_value=float(max(df["Win %"].max(), 1.0))),
            "Top 2 %": st.column_config.NumberColumn(format="%.1f%%"),
            "Place %": st.column_config.NumberColumn(format="%.1f%%"),
            "Fair $": st.column_config.NumberColumn(format="$%.2f"),
            "EV %": st.column_config.NumberColumn(format="%+.1f%%"),
            "Model %": st.column_config.NumberColumn(format="%.1f%%"),
            "Market %": st.column_config.NumberColumn(format="%.1f%%"),
        })

    left, right = st.columns(2)
    with left:
        with st.container(border=True):
            st.markdown("**Model vs market**")
            long = []
            for r in rated:
                long.append({"Runner": f"{r.box}. {r.name}",
                             "Source": "Form model", "Probability": r.p_model * 100})
                if r.p_market is not None:
                    long.append({"Runner": f"{r.box}. {r.name}",
                                 "Source": "Market", "Probability": r.p_market * 100})
            chart = alt.Chart(pd.DataFrame(long)).mark_bar().encode(
                x=alt.X("Probability:Q", title="Win probability (%)"),
                y=alt.Y("Runner:N", sort=[f"{r.box}. {r.name}" for r in rated],
                        title=None),
                yOffset="Source:N",
                color=alt.Color("Source:N", title=None),
                tooltip=["Runner", "Source", alt.Tooltip("Probability:Q", format=".1f")],
            ).properties(height=28 * len(rated) + 40)
            st.altair_chart(chart)
            st.caption("Where the two disagree is where the model is taking a "
                       "position. That is also where it can be wrong.")

    with right:
        with st.container(border=True):
            st.markdown("**Value**")
            overlays = [r for r in rated if r.ev is not None and r.ev > 0]
            if overlays:
                for r in overlays:
                    st.write(f"**{r.box}. {r.name}** — {money(r.odds)} "
                             f"against a fair {money(r.fair)} · "
                             f"**{r.ev*100:+.1f}% EV**")
                st.caption("An overlay is the model disagreeing with the market. "
                           "On this model's short record, its value picks have not "
                           "won — treat them as a hypothesis, not a tip.")
            else:
                st.write("No runner is priced above its modelled chance. "
                         "On this race the model has no bet.")

        with st.container(border=True):
            st.markdown("**Most likely first two (either order)**")
            for label, prob in rating.quinellas(rated):
                st.write(f"{label} — {prob*100:.1f}%  ·  fair {money(1/prob, 1)}"
                         if prob > 0 else label)

# --- runners ------------------------------------------------------------------

with tab_run:
    st.caption("Every term is in **lengths**. Positive helps, negative hurts. "
               "`form` is the weighted average beaten margin, negated.")
    for r in rated:
        head = (f"{r.box}. {r.name} — {r.p_final*100:.1f}%  ·  {fmt_odds(r.odds)}"
                f"  ·  rating {r.rating:+.2f}L")   # expander label: not markdown
        with st.expander(head):
            a, b = st.columns([1, 2])
            with a:
                st.markdown("**Rating terms**")
                tf = pd.DataFrame(
                    [{"Term": k, "Lengths": v} for k, v in r.terms.items()])
                st.dataframe(tf, hide_index=True, column_config={
                    "Lengths": st.column_config.NumberColumn(format="%+.2f")})
                st.caption(
                    f"box {r.box} (effective {r.eff_box} from the rail, "
                    f"{r.gap_inside} vacant inside) · evidence "
                    f"{r.evidence:.2f} effective runs")
            with b:
                st.markdown("**Parsed run history**")
                runs = pd.DataFrame([{
                    "Fin": (f"{run.pos} of {run.field_size}"
                            if run.field_size else f"{run.pos} of ?"),
                    "Margin": run.margin,
                    "Date": run.run_date,
                    "Days": run.days_ago,
                    "Trk": run.track,
                    "Dist": run.dist_m,
                    "Surf": run.surface,
                    "Box": run.box,
                    "SP": run.sp,
                    "Sec": run.sectional,
                    "Kind": run.track_kind,
                } for run in r.runner.runs])
                st.dataframe(runs, hide_index=True, column_config={
                    "Margin": st.column_config.NumberColumn(
                        "Margin", format="%.1fL",
                        help="Negative = won by that margin"),
                    "SP": st.column_config.NumberColumn(format="$%.2f")})
                rec = {k: f"{v[0]}: {v[1]}-{v[2]}-{v[3]}"
                       for k, v in r.runner.records.items()}
                st.caption(" · ".join(f"**{k}** {v}" for k, v in rec.items()))

# --- diagnostics --------------------------------------------------------------

with tab_diag:
    if run_sens:
        with st.spinner("Re-rating the race with jittered parameters…"):
            tally = rating.sensitivity(race, p, draws=400)
        with st.container(border=True):
            st.markdown("**How often each runner rates top, over 400 random "
                        "parameter draws**")
            sdf = pd.DataFrame([{"Runner": k, "Top-rated %": v * 100}
                                for k, v in tally.items()])
            st.dataframe(sdf, hide_index=True, column_config={
                "Top-rated %": st.column_config.ProgressColumn(
                    "Top-rated %", format="%.1f%%", min_value=0.0, max_value=100.0)})
            best = max(tally.values()) if tally else 0
            st.caption(
                "A selection near 100% is a property of the form. One near 50% "
                "means the constants are choosing the winner, not the data."
                if best > 0.9 else
                "This selection is **not** robust to the parameter choice — "
                "the race is closer than the headline number suggests.")

    with st.container(border=True):
        st.markdown("**Rating terms across the field (lengths)**")
        keys = list(rated[0].terms)
        tdf = pd.DataFrame([{"Runner": f"{r.box}. {r.name}",
                             **{k: r.terms.get(k, 0.0) for k in keys},
                             "TOTAL": r.rating} for r in rated])
        st.dataframe(tdf, hide_index=True, column_config={
            k: st.column_config.NumberColumn(format="%+.2f")
            for k in keys + ["TOTAL"]})

    with st.container(border=True):
        st.markdown("**Parse report**")
        st.write(f"Track code inferred from run tables: "
                 f"`{race.track_code() or 'unknown'}`")
        susp = [(r.name, run) for r in race.field_ for run in r.runs
                if run.field_size_suspect]
        if susp:
            st.write(f"{len(susp)} run(s) with an impossible `X of Y` field size, "
                     f"imputed as {p.impute_field_size}:")
            st.dataframe(pd.DataFrame(
                [{"Runner": n, "Date": run.run_date, "Trk": run.track,
                  "Shown": f"{run.pos} of ?"} for n, run in susp]),
                hide_index=True)
        if race.warnings:
            for w in race.warnings:
                st.write("• " + w)
        else:
            st.write("No parser warnings.")

# --- method -------------------------------------------------------------------

with tab_method:
    st.markdown("""
### What it does

Each runner's last-10 runs are turned into a **weighted average beaten margin**,
expressed in lengths at today's distance. Each past run is weighted by:

- **recency** — `exp(−days / τ)`
- **distance relevance** — `exp(−((run distance − today) / σ)²)`, so a 520m
  failure barely counts in a 400m race
- **track shape** — straight-track form is discounted for a circle race and vice
  versa. A straight track has no first turn, so railing ability and box speed do
  not transfer in either direction.
- **surface** — turf form is discounted in an all-weather race, and vice versa.
  This is a *separate* axis from track shape: straight turf form gets both.

That margin, plus these terms, gives a rating in lengths:

| Term | What it is |
|---|---|
| `form` | weighted average beaten margin, negated |
| `class` | career strike rate against a 12.5% field baseline |
| `conversion` | how often this dog turns a top-3 finish into a win |
| `distance` | strike rate at today's trip vs its own career rate |
| `course` | strike rate at today's track vs its own career rate |
| `surface` | strike rate on today's surface vs its own career rate |
| `early speed` | own sectionals at this track **and** distance |
| `layoff` | penalty for a spell over 60 days |
| `box` | effective position from the rail, counting vacant boxes |

Ratings go through a softmax to give race-conditional win probabilities, which
are blended in log space with the market after removing the overround with a
power de-vig.

### Things it gets wrong

**There is no opposition-strength adjustment.** Margins are scaled for distance
and field size but not for the quality of the dogs beaten, so a runner beaten 1L
in a weak race outrates one beaten 5L in a strong one. Across the races this was
built on, the runner with the best weighted margin finished 3rd and 4th. `form`
is the weakest column in the model and everything else is built on it. The real
fix is fitting abilities jointly across many races, not another hand-tuned term.

**Small samples used to swing it hard.** The distance term once ran on a prior
worth 7 starts, which let a two-start record move a dog 1.9 lengths; the two
runners it penalised hardest in a live race finished first and second. The prior
now defaults to 15 starts and is on a slider — turn it down and watch the
ratings get louder and worse.

**The record is three races.** Use `backtest.py` to score any change against
them. Three races cannot tell you whether the model beats the market; they only
tell you whether you have broken something that used to work.

### Sanity notes

- The dog's **name comes before** the tag block on this page and the **trainer
  after the box**. A reader that grabs the last capitalised line reports the
  trainer as the runner.
- R&S emit impossible finishing lines such as `6 of 4`. The field size is
  discarded and imputed for those runs; the Diagnostics tab lists them.
- Sectionals are measured to a different marker at each track and distance, so
  only runs at **today's track and distance** are compared.
""")
    st.caption("For free and confidential support call 1800 858 858 or visit "
               "gamblinghelponline.org.au")
