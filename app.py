"""RacingScorePredictor - the Universal Horse-Racing Scoring Framework, applied.

Paste a Racing & Sports Enhanced Form page; every runner is scored 0-100 across
the framework's fifteen weighted categories and the field is ranked.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

import rs_parser as parser
import scoring as sc

st.set_page_config(page_title="RacingScorePredictor",
                   page_icon=":material/insights:",
                   layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
.block-container {padding-top: 1.2rem; padding-bottom: 3rem;}
div[data-testid="stMetric"] {border:1px solid rgba(128,128,128,.22);
  border-radius:.7rem; padding:.55rem .8rem;}
.pick {border:1px solid rgba(46,204,113,.55); background:rgba(46,204,113,.10);
  border-radius:.8rem; padding:.8rem 1.1rem; margin:.4rem 0 .9rem;}
.muted {opacity:.65; font-size:.78rem; letter-spacing:.06em;}
</style>
""", unsafe_allow_html=True)

st.title("RacingScorePredictor")
st.caption("Universal Horse-Racing Scoring Framework (Reference V2) — fifteen "
           "weighted categories, adapted to today's distance and surface, "
           "rescaled to 100 over whatever evidence the page actually carries.")

# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.header("Scoring", divider="gray")
    profile = st.radio(
        "Output profile", ["overall", "win", "place"],
        format_func=lambda s: {"overall": "Overall (balanced)",
                               "win": "Win (peak ability, class, pace)",
                               "place": "Place (consistency, suitability)"}[s],
        help="Section 19 of the framework recommends three views of the same "
             "categories. Win leans on peak ability, class and pace; Place on "
             "consistency, distance/going suitability and head-to-head.")
    race_type = st.selectbox(
        "Race type", ["auto", "handicap", "wfa", "group", "maiden"],
        help="Section 17. Weight-for-age cuts the weight category from 6% to "
             "about 2.5%; Group races lift ability and class. 'auto' reads it "
             "from the race header.")
    spread = st.slider(
        "Probability spread", 5.0, 30.0, float(sc.DEFAULT_SPREAD), 1.0,
        help="Converts scores to percentages. A LOWER number makes the top "
             "pick look more certain. Nothing here is calibrated against "
             "results, so this is a presentation choice, not a forecast.")
    st.divider()
    show_unavailable = st.toggle("Show unavailable categories", value=True,
                                 help="Categories with no evidence on the page. "
                                      "They carry no weight either way.")
    st.caption("Odds are parsed and shown but never scored — the rating stays "
               "independent of the market by design.")

paste_tab, scores_tab, detail_tab, weights_tab, method_tab = st.tabs(
    ["1 · Paste", "2 · Scores", "3 · Category detail", "4 · Weights", "Method"])

st.session_state.setdefault("parsed", None)

# ------------------------------------------------------------------- paste
with paste_tab:
    st.subheader("Paste the Racing & Sports Enhanced Form page")
    st.caption("Open the race's **Enhanced Form** page, select all (Ctrl+A), "
               "copy, and paste below. Works on Australian, UK/Irish and South "
               "African pages.")
    txt = st.text_area("Form page", height=260, label_visibility="collapsed",
                       placeholder="Paste the whole page here…")
    if st.button("Parse & score", type="primary", icon=":material/play_arrow:"):
        if not txt.strip():
            st.warning("Nothing pasted yet.")
        else:
            try:
                header, runners, warns = parser.parse(txt)
                st.session_state["parsed"] = (header, runners, warns)
            except Exception as exc:                       # noqa: BLE001
                st.session_state["parsed"] = None
                st.error(f"Could not parse that page — {type(exc).__name__}: {exc}")

    if st.session_state["parsed"]:
        header, runners, warns = st.session_state["parsed"]
        active = [r for r in runners if not r.get("scratched")]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Runners", f"{len(active)}",
                  delta=f"{len(runners) - len(active)} scratched"
                  if len(runners) != len(active) else None, delta_color="off")
        c2.metric("Distance", f"{header.get('distance_m') or '?'}m")
        c3.metric("Surface", f"{header.get('surface') or '?'}")
        c4.metric("Going", f"{header.get('going') or '?'}")
        st.success(f"**{header.get('track','?')} — Race "
                   f"{header.get('race_no','?')}** · {header.get('date','')}")
        for w in warns:
            st.warning(w, icon=":material/warning:")
        with st.expander("What was read from the page"):
            st.dataframe(pd.DataFrame([{
                "Tab": r.get("tab"), "Horse": r.get("horse"),
                "Wt": r.get("wt"), "Claim": r.get("claim"), "BP": r.get("bp"),
                "OHR": r.get("ohr"), "OHR from": r.get("ohr_source"),
                "Jockey": r.get("jockey"), "JRat": r.get("jrat"),
                "Trainer": r.get("trainer"), "TRat": r.get("trat"),
                "Runs read": len(r.get("recent_runs") or []),
                "H2H": len(r.get("h2h") or []),
                "Days since": r.get("dslr"),
                "Odds": r.get("tab_odds"),
                "Scratched": r.get("scratched"),
            } for r in runners]), hide_index=True)

# ------------------------------------------------------------------ scores
def _result():
    if not st.session_state["parsed"]:
        return None
    header, runners, _ = st.session_state["parsed"]
    rt = None if race_type == "auto" else race_type
    return header, sc.score_race(runners, header, profile=profile, race_type=rt)


with scores_tab:
    got = _result()
    if not got:
        st.info("Paste a race in **1 · Paste** first.", icon=":material/info:")
    elif not got[1]["rows"]:
        st.error("Fewer than two active runners — nothing to rank.")
    else:
        header, res = got
        rows = res["rows"]
        probs = sc.win_probabilities(rows, spread=spread)
        runners_by_tab = {r["tab"]: r for r in st.session_state["parsed"][1]}

        top = rows[0]
        st.markdown(
            f'<div class="pick"><div class="muted">TOP RATED — '
            f'{profile.upper()} PROFILE</div>'
            f'<h2 style="margin:.15rem 0">#{top["tab"]} · {top["horse"]}</h2>'
            f'<b>Score {top["final"]:.1f}</b> · Field index '
            f'{top["field_index"]:.0f} · Confidence {top["confidence"]:.0f}% · '
            f'{top["style"]}</div>', unsafe_allow_html=True)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Pace pressure index", f"{res['context']['ppi']:.2f}",
                  help="1.0 per Leader + 0.7 per On-Pace + 0.35 per Prominent. "
                       "High pressure upgrades closers.")
        c2.metric("Likely leaders", f"{res['context']['n_leaders']}")
        c3.metric("Distance band", res["band"])
        c4.metric("Race type", res["race_type"].upper())

        table = pd.DataFrame([{
            "Rank": r["rank"], "Tab": r["tab"], "Horse": r["horse"],
            "Score": r["final"], "Field index": r["field_index"],
            "Model %": 100 * p, "Confidence %": r["confidence"],
            "Style": r["style"],
            "Weight covered": r["available_weight"],
            "Odds": runners_by_tab.get(r["tab"], {}).get("tab_odds"),
        } for r, p in zip(rows, probs)])
        table["Odds"] = table["Odds"].apply(
            lambda v: np.nan if v is None or float(v) >= 900 else float(v))

        st.dataframe(
            table, hide_index=True, height=min(60 + 36 * len(table), 520),
            column_config={
                "Score": st.column_config.NumberColumn(format="%.1f"),
                "Field index": st.column_config.ProgressColumn(
                    "Field index", format="%.0f", min_value=0.0, max_value=100.0),
                "Model %": st.column_config.NumberColumn(format="%.1f%%"),
                "Confidence %": st.column_config.ProgressColumn(
                    "Confidence", format="%.0f%%", min_value=0.0, max_value=100.0),
                "Weight covered": st.column_config.NumberColumn(
                    "Weight covered", format="%.0f/100",
                    help="How much of the 100-point framework had evidence for "
                         "this horse. The score is rescaled over exactly this."),
                "Odds": st.column_config.NumberColumn(format="$%.2f"),
            })
        st.caption(
            "**Score** is 100 × Σ(category contributions) ÷ Σ(available "
            "weights) — so a horse missing a category is rescaled, never "
            "penalised. **Field index** puts the top horse on 100. "
            "**Model %** is an uncalibrated transform of the score, adjustable "
            "in the sidebar; it is a ranking expressed as percentages, not a "
            "probability forecast.")

        if len(rows) >= 3:
            st.markdown("#### Where the top three differ")
            keys = [k for k, _, _ in sc.CATEGORIES]
            comp = pd.DataFrame(
                {r["horse"]: [r["categories"][k]["score"] for k in keys]
                 for r in rows[:3]}, index=[sc.LABELS[k] for k in keys])
            comp["Spread"] = comp.max(axis=1) - comp.min(axis=1)
            comp = comp.sort_values("Spread", ascending=False)
            comp.insert(0, "Category", comp.index)
            # Progress columns rather than a Styler gradient: background_gradient
            # needs matplotlib, which is a heavy dependency to add for shading.
            st.dataframe(
                comp, hide_index=True, height=380,
                column_config={c: st.column_config.ProgressColumn(
                    c, format="%.1f", min_value=0.0, max_value=10.0)
                    for c in comp.columns if c not in ("Category", "Spread")}
                | {"Spread": st.column_config.NumberColumn(format="%.1f")})
            st.caption("Category scores out of 10, ordered by how much the top "
                       "three disagree. This is where the ranking is decided.")

# ---------------------------------------------------------------- detail
with detail_tab:
    got = _result()
    if not got or not got[1]["rows"]:
        st.info("Paste a race in **1 · Paste** first.", icon=":material/info:")
    else:
        header, res = got
        rows = res["rows"]
        pick = st.selectbox(
            "Runner", [r["tab"] for r in rows],
            format_func=lambda t: next(
                f"#{r['tab']} {r['horse']}  —  score {r['final']:.1f}"
                for r in rows if r["tab"] == t))
        row = next(r for r in rows if r["tab"] == pick)

        c1, c2, c3 = st.columns(3)
        c1.metric("Final score", f"{row['final']:.1f}")
        c2.metric("Field index", f"{row['field_index']:.0f}")
        c3.metric("Data confidence", f"{row['confidence']:.0f}%")

        recs = []
        for k, label, base in sc.CATEGORIES:
            c = row["categories"][k]
            if not c["available"] and not show_unavailable:
                continue
            recs.append({
                "Category": label, "Base wt": base,
                "Adjusted wt": c["weight"],
                "Available": "yes" if c["available"] else "no",
                "Score /10": c["score"] if c["available"] else np.nan,
                "Contribution": c["contribution"] if c["available"] else np.nan,
                "Evidence": "; ".join(c["notes"]) if c["notes"] else "—",
            })
        st.dataframe(
            pd.DataFrame(recs), hide_index=True, height=560,
            column_config={
                "Base wt": st.column_config.NumberColumn(format="%.0f"),
                "Adjusted wt": st.column_config.NumberColumn(format="%.2f"),
                "Score /10": st.column_config.ProgressColumn(
                    "Score /10", format="%.1f", min_value=0.0, max_value=10.0),
                "Contribution": st.column_config.NumberColumn(format="%.2f"),
                "Evidence": st.column_config.TextColumn(width="large"),
            })
        st.caption(
            f"Contributions total **{sum(c['contribution'] for c in row['categories'].values()):.2f}** "
            f"over **{row['available_weight']:.2f}** available weight → "
            f"100 × {sum(c['contribution'] for c in row['categories'].values()):.2f} ÷ "
            f"{row['available_weight']:.2f} = **{row['final']:.1f}**.")

# ---------------------------------------------------------------- weights
with weights_tab:
    got = _result()
    if not got:
        st.info("Paste a race in **1 · Paste** first.", icon=":material/info:")
    else:
        header, res = got
        st.subheader("How today's weights were derived")
        st.caption(f"Distance band **{res['band']}** · surface "
                   f"**{res['surface_key']}** · race type "
                   f"**{res['race_type']}** · profile **{profile}**. "
                   "Base weights are multiplied by each factor, then "
                   "renormalised so the total is exactly 100.")
        dm = sc.DISTANCE_MULT[res["band"]]
        sm = sc.SURFACE_MULT.get(res["surface_key"], {})
        rm = sc.RACE_TYPE_MULT.get(res["race_type"], {})
        pm = sc.PROFILES.get(profile, {})
        wdf = pd.DataFrame([{
            "Category": lbl, "Base": base,
            "Distance ×": dm.get(k, 1.0), "Surface ×": sm.get(k, 1.0),
            "Race type ×": rm.get(k, 1.0), "Profile ×": pm.get(k, 1.0),
            "Adjusted": res["weights"][k],
        } for k, lbl, base in sc.CATEGORIES])
        st.dataframe(wdf, hide_index=True, height=560, column_config={
            "Base": st.column_config.NumberColumn(format="%.0f"),
            "Distance ×": st.column_config.NumberColumn(format="%.2f"),
            "Surface ×": st.column_config.NumberColumn(format="%.2f"),
            "Race type ×": st.column_config.NumberColumn(format="%.2f"),
            "Profile ×": st.column_config.NumberColumn(format="%.2f"),
            "Adjusted": st.column_config.NumberColumn(format="%.2f"),
        })
        st.metric("Adjusted weights total", f"{wdf['Adjusted'].sum():.1f}")

# ----------------------------------------------------------------- method
with method_tab:
    st.subheader("How this works")
    st.markdown(f"""
### The calculation

For each horse and each of the fifteen categories:

1. **Category score** 0–10 from the form page.
2. **Adjusted weight** = base weight × distance multiplier × surface multiplier
   × race-type multiplier (× profile), renormalised so the set totals 100.
3. **Available flag** — 1 if the page carried evidence, 0 if not.
4. **Contribution** = (score ÷ 10) × adjusted weight, for available categories.
5. **Final score** = 100 × Σ contributions ÷ Σ available adjusted weights.
6. **Field index** = 100 × horse score ÷ best score in the field.
7. **Confidence** — computed separately from career starts, how much of the
   framework had evidence, and recency.

### The four rules that shape it

**Missing means unknown, not poor.** A category with no evidence leaves the
denominator entirely. A horse that has never raced on today's going is not
punished for it — the framework is explicit that untested is neutral, not zero,
and the *Weight covered* column shows exactly how much of the 100 points each
horse was actually judged on.

**Field-relative thinking.** Ratings, jockey and trainer ratings, weight and
draw are scored against today's opponents, not an absolute scale.

**No double counting.** Class takes the single strongest applicable piece of
evidence. A higher-class win does not also collect same-class win and place
points.

**The market stays out.** Odds are parsed and displayed, never scored. The
rating is independent of the price by construction, which is the only way it can
later be compared against one.

### Base weights

| Category | Weight | | Category | Weight |
|---|---|---|---|---|
| Recent Form | 15 | | Weight / Claim | 6 |
| Ability / Ratings / Speed | 14 | | Direct H2H | 6 |
| Pace / Race Shape | 10 | | Course / Track | 4 |
| Class / Opposition | 9 | | Barrier / Draw | 3 |
| Distance / Stamina | 9 | | Fitness / Preparation | 3 |
| Surface / Going | 8 | | Jockey | 3 |
| Sectionals / Efficiency | 6 | | Trainer | 2 |
| | | | Trip / Gear / Stewards | 2 |

### What the page cannot give

Three sub-parameters are marked unavailable rather than guessed, because the
form page does not carry them: **true sectional times** (L600/L400/L200),
**trials and workouts**, and **track configuration**. Their weight is removed
and the rest rescale. Roughly **85 of the 100 base points** were scoreable for
every runner on the sample race, with Class and Head-to-Head the two that
sometimes go missing.

### What this is not

**Nothing here has been validated against results.** The framework itself is
explicit that its weights are *"a rational baseline for testing, not universal
scientific constants"*, and it asks you to freeze the rules, then track winner
rank, top-3 and top-4 recall and calibration over many completed races across
several countries before trusting them.

This app is a faithful implementation of that stated method. It is the thing you
would validate — not evidence that the method works. The **Model %** column in
particular is an uncalibrated transform of the score, not a probability
forecast.

---
*Prediction is probabilistic decision support, not a guaranteed outcome. Gamble
responsibly — Gambling Help 1800 858 858,
[gamblinghelponline.org.au](https://www.gamblinghelponline.org.au).*
""")
