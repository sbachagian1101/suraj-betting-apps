"""RaceForm - paste an R&S Full Fields page (greyhound, thoroughbred or
harness) and get a rated field."""

from __future__ import annotations

import glob
import importlib
import io
import os

import altair as alt
import pandas as pd
import streamlit as st

st.set_page_config(page_title="RaceForm", page_icon="🏇", layout="wide")

HERE = os.path.dirname(os.path.abspath(__file__))

# --- stale-deployment guard ---------------------------------------------------
# Streamlit Cloud has repeatedly served a NEW app.py against a CACHED helper
# module after a push that added module-level names.  This guard MUST run before
# any `from rs_parser import ...` / `import rating`, because a stale module
# raises ImportError on the import line itself and the redacted Cloud traceback
# that follows says nothing useful.  Importing defensively here is the only way
# the guard can turn that into one actionable sentence.
_REQUIRED = {
    "rs_parser": ["parse", "Race", "Runner", "Run", "GREYHOUND", "THOROUGHBRED",
                  "HARNESS", "STRAIGHT_TRACKS", "METRES_PER_LENGTH",
                  "SENTINEL_METRES"],
    "rating": ["rate", "Params", "quinellas", "sensitivity", "Rated",
               "defaults_for", "CODE_DEFAULTS", "replace"],
}
_missing: list[str] = []
for _mod, _names in _REQUIRED.items():
    try:
        _m = importlib.import_module(_mod)
    except Exception as _e:                       # stale module -> ImportError
        _missing.append(f"{_mod} (import failed: {type(_e).__name__}: {_e})")
        continue
    _missing += [f"{_mod}.{n}" for n in _names if not hasattr(_m, n)]
if _missing:
    st.error(
        "**This deployment is running stale code.** " + " · ".join(_missing)
        + "\n\nStreamlit Cloud has served the new `app.py` against a cached "
          "helper module. Open **Manage app → ⋮ → Reboot app** to force a clean "
          "rebuild. The pushed source is fine.")
    st.stop()

import rating          # noqa: E402  (must follow the guard)
import rs_parser       # noqa: E402

GREYHOUND = rs_parser.GREYHOUND
THOROUGHBRED = rs_parser.THOROUGHBRED
HARNESS = rs_parser.HARNESS

CODE_LABEL = {GREYHOUND: "🐕 Greyhound", THOROUGHBRED: "🏇 Thoroughbred",
              HARNESS: "🛞 Harness"}


# --- helpers ------------------------------------------------------------------

def samples() -> dict[str, str]:
    return {os.path.basename(p).replace(".txt", ""): p
            for p in sorted(glob.glob(os.path.join(HERE, "fixtures", "*.txt")))}


def money(v: float | None, dp: int = 2) -> str:
    """Price for a *markdown* context.  Streamlit renders markdown with LaTeX
    on, so two bare `$` on one line are swallowed as inline math."""
    return "—" if not v else "\\$" + f"{v:.{dp}f}"


def fmt_odds(v: float | None) -> str:
    """Plain price, for non-markdown contexts such as st.metric."""
    return f"${v:g}" if v else "—"


def runs_frame(r: rating.Rated, code: str) -> pd.DataFrame:
    rows = []
    for run in r.runner.runs:
        base = {
            "Fin": (run.dq_code if run.disqualified else
                    (f"{run.pos} of {run.field_size}" if run.field_size
                     else f"{run.pos} of ?")),
            "Margin": run.margin,
            "Date": run.run_date,
            "Days": run.days_ago,
            "Trk": run.track,
            "Dist": run.dist_m,
            "Surf": run.surface,
            "SP": run.sp,
        }
        if code == GREYHOUND:
            base |= {"Box": run.box, "Sec": run.sectional, "Kind": run.track_kind}
        elif code == THOROUGHBRED:
            base |= {"Going": run.going, "Wt": run.weight, "BP": run.box,
                     "Jockey": run.jockey}
        else:
            base |= {"Hcp m": run.handicap_m, "Driver": run.jockey,
                     "Raw": (f"{run.margin_raw:g}{run.margin_unit}"
                             if run.margin_raw is not None else None)}
        rows.append(base)
    return pd.DataFrame(rows)


# --- input --------------------------------------------------------------------

st.title("🏇 RaceForm")
st.caption("Paste a Racing & Sports **Full Fields** page — greyhound, "
           "thoroughbred or harness. The app reads each runner's last-10 run "
           "table, rates the field in lengths, and blends with the market.")

sample_map = samples()
_, right = st.columns([3, 1])
with right:
    pick = st.selectbox("Load a sample race", ["—"] + list(sample_map), index=0)
if pick != "—" and st.session_state.get("_loaded") != pick:
    st.session_state["raw"] = io.open(sample_map[pick], encoding="utf-8").read()
    st.session_state["_loaded"] = pick

raw = st.text_area(
    "Paste the page here (select all → copy on the Full Fields tab)",
    key="raw", height=200,
    placeholder="HomeForm GuideThoroughbredFrance… / 1600m TURF GOOD / …")
st.button("Analyse race", type="primary")

if not raw or not raw.strip():
    st.info("Paste a Full Fields page, or load one of the sample races above.")
    st.stop()

race = rs_parser.parse(raw)
if not race.field_:
    st.error("No runners could be read from that paste.")
    for w in race.warnings:
        st.warning(w)
    st.stop()

# --- sidebar (rendered after parsing so the defaults match the code) ----------

d = rating.defaults_for(race.code)
with st.sidebar:
    st.header("Model settings")
    st.caption(f"Detected **{CODE_LABEL[race.code]}**. Defaults are tuned per "
               "code — a greyhound sprint is decided inside 3 lengths, a French "
               "trot can be 70.")
    p = rating.defaults_for(race.code)
    p.market_weight = st.slider(
        "Weight on the form model", 0.0, 1.0, d.market_weight, 0.02,
        help="0 = follow the market exactly. 1 = ignore it. The blend is a "
             "weighted geometric mean in log space.")
    p.spread = st.slider(
        "Performance spread (lengths)", 1.0, max(20.0, d.spread * 2), d.spread,
        0.05, help="Assumed SD of a runner's performance. Higher = flatter, "
                   "less confident probabilities.")
    p.prior_starts = st.slider(
        "Shrinkage prior (starts)", 5.0, 30.0, d.prior_starts, 1.0,
        help="How many starts of prior belief the distance / course / surface "
             "records are shrunk toward the career strike rate. Lower lets tiny "
             "samples swing the rating hard — that has cost a race.")

    if race.code == THOROUGHBRED:
        st.subheader("Thoroughbred")
        p.lengths_per_kg = st.slider(
            "Lengths per kg", 0.0, 1.5, d.lengths_per_kg, 0.05,
            help="Weight scale. A past run under a lighter weight flatters the "
                 "horse relative to today's mark.")
        p.k_barrier = st.slider("Barrier penalty (widest draw)", 0.0, 2.5,
                                d.k_barrier, 0.05)
    elif race.code == HARNESS:
        st.subheader("Harness")
        p.k_dq = st.slider(
            "Non-finish penalty (lengths)", 0.0, 8.0, d.k_dq, 0.1,
            help="A DQG (broke gait) is not a result, so those runs are dropped "
                 "from the margin average. This scores the RATE of it instead.")
        p.handicap_credit = st.slider(
            "Distance-handicap credit", 0.0, 1.0, d.handicap_credit, 0.05,
            help="Share of a +25m handicap credited back to a past margin.")
    else:
        st.subheader("Greyhound")
        p.k_sectional = st.slider("Early-speed weight", 0.0, 1.5,
                                  d.k_sectional, 0.05)
        p.k_gap = st.slider(
            "Vacant-box term (lengths per empty box inside)", 0.0, 1.0, 0.0, 0.05,
            help="UNVALIDATED. In one race the finish order was exactly monotone "
                 "in vacant boxes inside each runner. One observation, so it "
                 "ships at zero.")

    with st.expander("Advanced"):
        p.tau_days = st.slider("Recency decay (days)", 60.0, 400.0, d.tau_days, 10.0)
        p.sigma_dist = st.slider("Distance relevance width (m)", 40.0,
                                 max(400.0, d.sigma_dist * 2), d.sigma_dist, 5.0)
        p.w_offsurface = st.slider("Off-surface form weight", 0.0, 1.0,
                                   d.w_offsurface, 0.05)
        p.w_straight = st.slider("Straight-track form weight", 0.0, 1.0,
                                 d.w_straight, 0.05)
        p.w_foreign = st.slider("Overseas form weight", 0.0, 1.0, d.w_foreign, 0.05)
        p.k_field = st.slider("Field-size credit", 0.0, 1.0, d.k_field, 0.01)
        p.devig_power = st.slider("De-vig power", 1.0, 1.20, d.devig_power, 0.01)

    run_sens = st.checkbox("Run parameter sensitivity sweep", value=True)

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
bits = [CODE_LABEL[race.code], f"{len(race.field_)} runners"]
if race.race_date:
    bits.insert(1, f"{race.race_date:%A, %d %B %Y}")
if any(r.scratched for r in race.runners):
    bits.append(f"{sum(1 for r in race.runners if r.scratched)} scratched")
st.caption(" · ".join(bits))

with st.container(horizontal=True):
    st.metric("Selection", f"{top.tab}. {top.name}", border=True)
    st.metric("Win probability", f"{top.p_final*100:.1f}%", border=True)
    st.metric("Fair price", f"${top.fair:.2f}",
              delta=(f"{top.ev*100:+.0f}% EV at {fmt_odds(top.odds)}"
                     if top.ev is not None else None), border=True)
    if fav:
        st.metric("Market favourite", f"{fav.tab}. {fav.name}",
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
    df = pd.DataFrame([{
        "No.": r.tab, "Runner": r.name, "Odds": r.odds,
        "Win %": r.p_final * 100, "Top 2 %": r.p_top2 * 100,
        "Place %": r.p_top3 * 100, "Fair $": r.fair,
        "EV %": (r.ev * 100) if r.ev is not None else None,
        "Model %": r.p_model * 100,
        "Market %": (r.p_market * 100) if r.p_market is not None else None,
    } for r in rated])
    st.dataframe(df, hide_index=True, column_config={
        "Odds": st.column_config.NumberColumn(format="$%.2f"),
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
                long.append({"Runner": f"{r.tab}. {r.name}",
                             "Source": "Form model", "Probability": r.p_model * 100})
                if r.p_market is not None:
                    long.append({"Runner": f"{r.tab}. {r.name}",
                                 "Source": "Market", "Probability": r.p_market * 100})
            chart = alt.Chart(pd.DataFrame(long)).mark_bar().encode(
                x=alt.X("Probability:Q", title="Win probability (%)"),
                y=alt.Y("Runner:N", sort=[f"{r.tab}. {r.name}" for r in rated],
                        title=None),
                yOffset="Source:N", color=alt.Color("Source:N", title=None),
                tooltip=["Runner", "Source",
                         alt.Tooltip("Probability:Q", format=".1f")],
            ).properties(height=24 * len(rated) + 40)
            st.altair_chart(chart)
            st.caption("Where the two disagree is where the model is taking a "
                       "position. That is also where it can be wrong.")

    with right:
        with st.container(border=True):
            st.markdown("**Value**")
            overlays = [r for r in rated if r.ev is not None and r.ev > 0]
            # An overlay where the MODEL rates a runner 5x shorter than the
            # market is far more likely to be the model being wrong. Measure the
            # model's own disagreement, not the blend's: p_final is already 62%
            # market, which compresses the ratio to ~3x and hides the outliers.
            EXTREME = 5.0
            sane = [r for r in overlays
                    if r.p_market and r.p_model < EXTREME * r.p_market]
            wild = [r for r in overlays if r not in sane]
            if sane:
                for r in sorted(sane, key=lambda z: -z.ev):
                    st.write(f"**{r.tab}. {r.name}** — {money(r.odds)} against a "
                             f"fair {money(r.fair)} · **{r.ev*100:+.1f}% EV** "
                             f"· {r.used_runs} usable runs")
            else:
                st.write("No runner is priced above its modelled chance within a "
                         "sane margin. On this race the model has no bet.")
            if wild:
                st.markdown("**Extreme disagreements — treat as model error**")
                for r in sorted(wild, key=lambda z: -z.ev):
                    st.write(f"{r.tab}. {r.name} — {money(r.odds)}, model says "
                             f"{money(r.fair)} (the form model rates it "
                             f"{r.p_model/r.p_market:.0f}x shorter than the "
                             f"market) · {r.used_runs} usable runs")
                st.caption(
                    "The model rating a runner several times shorter than the "
                    "market usually means it has over-credited a couple of races "
                    "at a flattering trip, not that it has found something. It "
                    "has **no opposition-strength adjustment**, so beating a "
                    "weak field looks the same as beating a strong one.")
            st.caption("An overlay is the model disagreeing with the market. "
                       "On this model's short record its value picks have not "
                       "won — treat them as a hypothesis, not a tip.")
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
        head = (f"{r.tab}. {r.name} — {r.p_final*100:.1f}%  ·  "
                f"{fmt_odds(r.odds)}  ·  rating {r.rating:+.2f}L")
        with st.expander(head):
            a, b = st.columns([1, 2])
            with a:
                st.markdown("**Rating terms**")
                st.dataframe(
                    pd.DataFrame([{"Term": k, "Lengths": v}
                                  for k, v in r.terms.items()]),
                    hide_index=True,
                    column_config={"Lengths": st.column_config.NumberColumn(
                        format="%+.2f")})
                meta = [f"{r.used_runs} of {len(r.runner.runs)} runs usable",
                        f"evidence {r.evidence:.2f}"]
                if race.code == GREYHOUND:
                    meta.append(f"box {r.tab} (effective {r.eff_box}, "
                                f"{r.gap_inside} vacant inside)")
                if race.code == THOROUGHBRED:
                    meta.append(f"{r.runner.weight}kg, barrier {r.runner.barrier}")
                    meta.append(f"jockey {r.runner.jockey}")
                if race.code == HARNESS:
                    meta.append(f"**{r.runner.dq_count} non-finishes** in "
                                f"{len(r.runner.runs)}")
                    meta.append(f"driver {r.runner.driver}")
                meta.append(f"trainer {r.runner.trainer}")
                st.caption(" · ".join(meta))
            with b:
                st.markdown("**Parsed run history**")
                st.dataframe(runs_frame(r, race.code), hide_index=True,
                             column_config={
                                 "Margin": st.column_config.NumberColumn(
                                     "Margin", format="%.1fL",
                                     help="In lengths. Negative = won by that "
                                          "margin. Blank = no result."),
                                 "SP": st.column_config.NumberColumn(format="$%.2f")})
                st.caption(" · ".join(
                    f"**{k}** {v[0]}: {v[1]}-{v[2]}-{v[3]}"
                    for k, v in r.runner.records.items()))

# --- diagnostics --------------------------------------------------------------

with tab_diag:
    if run_sens:
        with st.spinner("Re-rating the race with jittered parameters…"):
            tally = rating.sensitivity(race, p, draws=300)
        with st.container(border=True):
            st.markdown("**How often each runner rates top, over 300 random "
                        "parameter draws**")
            st.dataframe(
                pd.DataFrame([{"Runner": k, "Top-rated %": v * 100}
                              for k, v in tally.items()]),
                hide_index=True,
                column_config={"Top-rated %": st.column_config.ProgressColumn(
                    "Top-rated %", format="%.1f%%", min_value=0.0, max_value=100.0)})
            best = max(tally.values()) if tally else 0
            st.caption(
                "A selection near 100% is a property of the form."
                if best > 0.9 else
                "This selection is **not** robust to the parameter choice — the "
                "race is closer than the headline number suggests.")

    with st.container(border=True):
        st.markdown("**Rating terms across the field (lengths)**")
        keys = list(rated[0].terms)
        st.dataframe(
            pd.DataFrame([{"Runner": f"{r.tab}. {r.name}",
                           **{k: r.terms.get(k, 0.0) for k in keys},
                           "TOTAL": r.rating} for r in rated]),
            hide_index=True,
            column_config={k: st.column_config.NumberColumn(format="%+.2f")
                           for k in keys + ["TOTAL"]})

    if race.code == HARNESS:
        with st.container(border=True):
            st.markdown("**Non-finishes (DQG / disqualified)**")
            st.dataframe(pd.DataFrame([{
                "No.": r.tab, "Runner": r.name,
                "Non-finishes": r.runner.dq_count,
                "Runs shown": len(r.runner.runs),
                "Rate %": r.runner.dq_rate * 100,
                "Penalty (L)": r.terms.get("reliability", 0.0),
            } for r in rated]).sort_values("Rate %", ascending=False),
                hide_index=True, column_config={
                    "Rate %": st.column_config.NumberColumn(format="%.0f%%"),
                    "Penalty (L)": st.column_config.NumberColumn(format="%+.2f")})
            st.caption("A break in gait is not a result, so those runs are "
                       "excluded from the margin average and scored here instead.")

    with st.container(border=True):
        st.markdown("**Parse report**")
        st.write(f"Code: `{race.code}` · track code inferred from the run "
                 f"tables: `{race.track_code() or 'unknown'}`")
        susp = [(r.name, run) for r in race.field_ for run in r.runs
                if run.field_size_suspect]
        if susp:
            st.write(f"{len(susp)} run(s) with an impossible `X of Y` field size, "
                     f"imputed as {p.impute_field_size}:")
            st.dataframe(pd.DataFrame(
                [{"Runner": n, "Date": run.run_date, "Trk": run.track,
                  "Shown": f"{run.pos} of ?"} for n, run in susp]), hide_index=True)
        for w in race.warnings:
            st.write("• " + w)
        if not race.warnings and not susp:
            st.write("No parser warnings.")

# --- method -------------------------------------------------------------------

with tab_method:
    st.markdown("""
### What it does

Each runner's last-10 runs become a **weighted average beaten margin**, in
lengths at today's distance. Each past run is weighted by **recency**
(`exp(−days/τ)`), **distance relevance** (`exp(−((d−today)/σ)²)`), **track shape**
(straight tracks have no first turn, so that form does not transfer to or from a
circle track) and **surface** — a separate axis, so straight turf form is
discounted on both counts.

That margin plus `class`, `conversion`, `distance`, `course`, `surface`,
`going` and `layoff` gives a rating in lengths. A softmax turns ratings into
race-conditional win probabilities, blended in log space with the market after
a power de-vig.

### What changes per code

| | Greyhound | Thoroughbred | Harness |
|---|---|---|---|
| Margins | lengths | lengths | **metres**, ÷2.5 |
| Extra terms | early speed, box | **weight**, barrier | **reliability (DQG)** |
| Past-run adjustment | — | weight carried vs today | distance handicap credited |
| Typical spread | 2.4L | 4.0L | 9.0L |

**Thoroughbred.** A handicapper's own margins are not comparable across its runs
until they are put on the same weight. A run under 56kg when today's mark is
61.5kg flatters it by about 3.8 lengths at the default scale.

**Harness.** A `DQG` — broke gait, disqualified — is **not a result**, and R&S
print a sentinel margin of `99m` against it. Those runs are dropped from the
margin average and the *rate* is scored separately, because on a French trot
page it is the most predictive thing there: a horse that breaks in seven of ten
starts is not a 4-length problem, it is an unreliable one. Runs off a `+25m`
distance handicap get credit for the extra ground.

### Things it gets wrong

**There is no opposition-strength adjustment.** Margins are scaled for distance
and field size but not for the quality of the field beaten, so a runner beaten
1L in a weak race outrates one beaten 5L in a strong one. Across the greyhound
races this was built on, the runner with the best weighted margin finished 3rd
and 4th. `form` is the weakest column and everything else sits on it. The real
fix is fitting abilities jointly across many races.

**The record is three greyhound races.** `python backtest.py` scores them. There
is **no scored result yet for thoroughbred or harness** — those ratings are
untested, and the sensible way to read them today is as a structured summary of
the form, not as a proven edge.

### Sanity notes

- The runner's **name precedes** the tag block; the **trainer comes last**. A
  reader that grabs the last capitalised line reports the trainer (greyhound),
  the **weight** (thoroughbred) or the **driver** (harness) instead.
- R&S ship different run-table column sets **on the same page** — at Cabourg
  some runners carry a `Draw` column and some do not — so every table is read
  through its own header row.
- R&S emit impossible finishing lines such as `6 of 4`; those field sizes are
  discarded and imputed, and Diagnostics lists them.
- Greyhound sectionals are measured to a different marker at each track and
  distance, so only runs at **today's track and distance** are compared.
""")
    st.caption("For free and confidential support call 1800 858 858 or visit "
               "gamblinghelponline.org.au")
