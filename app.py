"""AustraliaPdfHorseRacing — form-guide PDFs in, ranked runners out."""
from __future__ import annotations

import glob
import os

import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="AustraliaPdfHorseRacing",
                   page_icon=":material/sports_score:", layout="wide")

# Streamlit Cloud pulls new files but keeps already-imported modules in
# sys.modules, so a deploy can run new app code against stale helpers. This has
# bitten three of these apps. Fail loudly with the fix rather than serving
# quietly wrong numbers.
try:
    import features as F
    import past_form as PF
    import pdf_parser as PP
    import predict as PR
    import train as T
    for _m, _attr in [(PP, "parse_meeting"), (PF, "parse_past_runs"),
                      (F, "build_upcoming"), (PR, "score_race"),
                      (T, "place_probabilities")]:
        if not hasattr(_m, _attr):
            raise ImportError(f"{_m.__name__} is missing {_attr}")
except ImportError as exc:
    st.error(
        f"**The app is running against a stale copy of its own modules** "
        f"({exc}).\n\nOn Streamlit Cloud: **Manage app → ⋮ → Reboot app**. "
        "A rerun is not enough — the modules are cached in `sys.modules`.",
        icon=":material/error:")
    st.stop()

import joblib


@st.cache_resource
def load_bundle():
    return joblib.load("model_bundle.joblib")


@st.cache_data(show_spinner=False)
def parse_uploads(blobs: list[tuple[str, bytes]]):
    fields, pasts, problems = [], [], []
    for name, data in blobs:
        tmp = f"_upload_{abs(hash(name)) % 10**9}.pdf"
        with open(tmp, "wb") as fh:
            fh.write(data)
        try:
            f = PP.parse_meeting(tmp)
            p = PF.parse_past_runs(tmp)
            if f.empty:
                problems.append(f"{name}: no runners were found")
            else:
                f["file"] = name
                fields.append(f)
            if not p.empty:
                pasts.append(p)
        except Exception as exc:                              # noqa: BLE001
            problems.append(f"{name}: {type(exc).__name__} — {exc}")
        finally:
            try:
                os.remove(tmp)
            except OSError:
                pass
    field = pd.concat(fields, ignore_index=True) if fields else pd.DataFrame()
    past = pd.concat(pasts, ignore_index=True) if pasts else pd.DataFrame()
    if not field.empty:
        problems.extend(PP.warnings_for(field))
    return field, past, problems


B = load_bundle()
V = B["validation"]

# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.title("AustraliaPdfHorseRacing")
    st.caption("Racing & Sports Australian meeting form guides, as PDF.")

    ups = st.file_uploader("Meeting PDFs", type=["pdf"],
                           accept_multiple_files=True)
    use_sample = st.toggle(
        "Use the bundled sample meetings", value=False,
        help="Darwin and Kununurra on 29 Aug 2026 and Carnarvon on 30 Aug — "
             "shipped with the app so it can be tried without a file to hand.")
    places = st.select_slider("Places paid", [2, 3, 4], value=3)
    st.divider()

    st.metric("Top pick wins", f"{100*V['pick_win']:.1f}%",
              f"{100*(V['pick_win']-V['random_win']):+.1f} pts vs a random pick")
    st.metric("Top pick places", f"{100*V['pick_place']:.1f}%",
              f"{100*(V['pick_place']-V['random_place']):+.1f} pts")
    st.caption(f"Out of sample on {V['n_races']:,} held-out races. "
               f"The starting price does better: "
               f"{100*V['market_win']:.1f}% / {100*V['market_place']:.1f}%.")

pred_tab, field_tab, val_tab, method_tab = st.tabs(
    ["Predictions", "The field", "How good is it", "Method"])

SAMPLES = sorted(glob.glob("sample_data/*.pdf"))

if not ups and not use_sample:
    with pred_tab:
        st.info("Upload one or more meeting PDFs to begin.",
                icon=":material/upload_file:")
        st.markdown(
            f"""
Each file is a Racing & Sports meeting form guide. The app reads two things
from it: the **field table** for every race, and every runner's **past-run
history** — about 400 past runs per meeting, each with a finishing position,
a field size and a starting price.

The model was fitted on **{B['n_training_rows']:,} past runs** from
{B['n_horses']:,} horses between {B['date_range'][0]} and {B['date_range'][1]}.
It never sees the price of the race it is predicting, because these files do
not contain one.
""")
    st.stop()

if ups:
    blobs = [(u.name, u.getvalue()) for u in ups]
else:
    blobs = [(p.rsplit("/", 1)[-1], open(p, "rb").read()) for p in SAMPLES]
field, past, problems = parse_uploads(blobs)

if field.empty:
    with pred_tab:
        st.error("Nothing could be read from those files.",
                 icon=":material/error:")
        for p in problems:
            st.write("• ", p)
    st.stop()

# past runs from the bundle's training corpus are not available at runtime, so
# a horse is scored on whatever history its own PDF carries
races = (field[["race_id", "track", "date", "race_no", "distance",
                "race_name", "field_size"]]
         .drop_duplicates("race_id")
         .sort_values(["track", "race_no"]))

# ------------------------------------------------------------- predictions
with pred_tab:
    c1, c2 = st.columns([3, 2])
    with c1:
        # a dict, not a positional lookup: Streamlit re-formats whatever value
        # the widget is holding, and after the uploaded files change that can
        # be a race_id no longer in the frame
        label = {r.race_id: f"{r.track} R{r.race_no}  ·  {r.distance}m  ·  "
                            f"{r.field_size} runners"
                 for r in races.itertuples()}
        choice = st.selectbox("Race", list(races.race_id),
                              format_func=lambda k: label.get(k, str(k)))
    row = races[races.race_id == choice].iloc[0]
    with c2:
        st.caption(f"**{row.race_name or '—'}**  \n{row.date} · {row.distance}m")

    runners = field[field.race_id == choice]
    t = PR.score_race(runners, past, B, places=places, sims=12000)

    band = t.attrs["band"]
    conf = t.attrs["confidence"]
    tone = {"high": st.success, "medium": st.warning, "low": st.info}[conf]
    top = t.iloc[0]
    tone(
        f"**Selection: {top.horse}** (tab {int(top.tab)}, {top.jockey or 'jockey TBA'})"
        f" — **{conf} confidence**\n\n"
        f"It is {t.attrs['gap']:.2f} score-units clear of the next runner. "
        f"Across **{band['n']} held-out races** where the top pick was that far "
        f"clear, it won **{100*band['win']:.1f}%** "
        f"(95% CI {100*band['win_ci'][0]:.1f}–{100*band['win_ci'][1]:.1f}%) "
        f"and placed **{100*band['place']:.1f}%**, in fields averaging "
        f"{band['avg_field']:.1f} runners.",
        icon=":material/target:")

    st.markdown("**Why this runner**")
    for r in t.attrs["reasons"]:
        st.markdown(f"- {r}")
    st.caption("Each line is a term of the fitted linear predictor, measured "
               "against this race's own average. The number is how much it "
               "moved the score.")

    if t.attrs["no_history"]:
        st.warning(
            f"{t.attrs['no_history']} of {len(t)} runners have **no past runs** "
            "in this file — first starters, or horses whose history did not "
            "parse. They are scored on the field median, which is a guess.",
            icon=":material/help:")

    st.dataframe(
        t[["Rank", "tab", "horse", "jockey", "trainer", "n_history",
           "score", "Win %", "Place %"]],
        width="stretch", hide_index=True,
        column_config={
            "tab": st.column_config.NumberColumn("Tab", width="small"),
            "horse": "Horse", "jockey": "Jockey", "trainer": "Trainer",
            "n_history": st.column_config.NumberColumn(
                "Runs on file", width="small",
                help="Past runs available for this horse in the uploaded PDF."),
            "score": st.column_config.NumberColumn("Score", format="%.2f"),
            "Win %": st.column_config.ProgressColumn(
                "Win %", format="%.1f%%", min_value=0.0,
                max_value=float(max(t["Win %"].max(), 1))),
            "Place %": st.column_config.NumberColumn("Place %", format="%.1f%%"),
        })

    st.caption(
        f"Win percentages are a softmax over the {len(t)} runners. The scale "
        "was fitted on held-out races that showed only about "
        f"{V['avg_visible']:.1f} of their runners each, so the **ranking** is "
        "measured but the spread between the percentages is an extrapolation. "
        "Treat the order as the output and the numbers as indicative.")

    with st.expander("Form comments and gear (context only — not in the model)"):
        ctx = t[["Rank", "tab", "horse", "gear", "comment"]]
        st.dataframe(ctx, width="stretch", hide_index=True)
        st.caption(
            "These are the one thing these PDFs carry that the spreadsheet "
            "exports do not. They are shown because they are useful to read, "
            "and excluded from the model because they did not survive testing: "
            "blinkers going on looked worth +0.069 of a finishing percentile "
            "(p=0.001), but horses get blinkers after a bad run, and once the "
            "previous run is controlled for the effect is +0.017 (p=0.48).")

# ------------------------------------------------------------- the field
with field_tab:
    st.subheader(f"{len(races)} races · {len(field)} runners")
    if problems:
        with st.expander(f"{len(problems)} parsing note(s)", expanded=False):
            for p in problems:
                st.write("• ", p)
    st.dataframe(
        field[["track", "race_no", "distance", "tab", "horse", "age", "sex",
               "weight", "barrier", "jockey", "claim", "trainer",
               "form_figures", "career_wins", "career_places", "career_starts",
               "days_since_run", "gear"]],
        width="stretch", hide_index=True, height=420)
    st.caption(
        "**Weight** is the allotted weight, before the apprentice claim — "
        "subtract `claim` for the weight actually carried. **Barrier** is the "
        "adjusted draw and can repeat within a race when a runner has been "
        "scratched: these are pre-acceptance guides, so the field shown can be "
        "larger than the field that starts.")

    if not past.empty:
        st.subheader("Past runs read from these files")
        st.dataframe(
            past[["horse", "date", "track", "distance", "finish",
                  "past_field_size", "sp", "was_favourite", "margin",
                  "jockey", "going"]].sort_values(["horse", "date"]),
            width="stretch", hide_index=True, height=300)
        st.caption(f"{len(past):,} past runs. `sp` is the decimal starting "
                   "price — the PDF stores it as odds-to-one, so $1.30 appears "
                   "in the file as `0.3F`.")

# ------------------------------------------------------- how good is it
with val_tab:
    st.subheader("What this model is actually worth")
    st.markdown(f"""
Everything below is **out of sample**: the model is refitted five times on data
before a date and scored on races after it. A random split would let a horse's
later runs train the model that predicts its earlier ones.

### Picking a runner

On **{V['n_races']:,} held-out races** in fields averaging
{V['avg_field']:.1f} runners:

| | top pick wins | top pick places |
|---|---|---|
| **this model** | **{100*V['pick_win']:.1f}%** | **{100*V['pick_place']:.1f}%** |
| the starting price | {100*V['market_win']:.1f}% | {100*V['market_place']:.1f}% |
| a random pick from the same horses | {100*V['random_win']:.1f}% | {100*V['random_place']:.1f}% |

The model recovers roughly
**{100*(V['pick_win']-V['random_win'])/(V['market_win']-V['random_win']):.0f}%
of the market's edge over a random pick, without ever seeing a price.** It does
not beat the market and is not meant to — these files contain no odds, so the
model's job is to be useful when you have no price, not to beat one.

### Probabilities

| | log-loss |
|---|---|
| this model | {V['logloss']:.4f} |
| the starting price | {V['market_logloss']:.4f} |
| no information at all | {V['chance_logloss']:.4f} |

Rank correlation between the model's score and the actual finishing percentile
is **{V['spearman']:+.3f}**.

### Confidence
""")
    bands = pd.DataFrame(B["confidence_bands"])
    bands["win %"] = (100 * bands.win).round(1)
    bands["95% CI"] = bands.win_ci.apply(
        lambda c: f"{100*c[0]:.1f} – {100*c[1]:.1f}")
    bands["place %"] = (100 * bands.place).round(1)
    st.dataframe(bands[["band", "gap_lo", "n", "win %", "95% CI", "place %"]],
                 width="stretch", hide_index=True,
                 column_config={"gap_lo": "gap at least", "n": "races"})
    st.caption(
        "Confidence is set by how far clear the top pick is, because that is "
        "the one thing whose payoff can be measured here. It is not set by the "
        "win percentage: that is a softmax over the whole field, while the "
        f"held-out races show only about {V['avg_visible']:.1f} of their "
        "runners each.")

    st.subheader("Walk-forward folds")
    st.dataframe(pd.DataFrame(B["folds"]), width="stretch", hide_index=True)

# ------------------------------------------------------------------ method
with method_tab:
    st.markdown(f"""
### What is read from the PDF

Two things. The **field table** — tab, form figures, horse, age and sex,
weight, barrier, jockey and claim, trainer, career record, prize money, runs
this preparation, days since the last run and last win. And every runner's
**past-run history**, which is where the model's training data comes from:
each past run carries a finishing position out of a known field size, the date,
track, distance, going, margin, race time, sectional, the jockey, the weight,
the barrier, the gear change, the first three finishers, the running positions
and **the starting price**.

Two layout traps had to be handled. The tab number is sometimes on its own line
and sometimes glued to the form figures — matching only the glued form found 4
of 9 runners at Strathalbyn. And the form figures are glued to the horse name
with no separator, so the split is made on case; matching them
case-insensitively turned `x7036` + `Dance Dance Dance` into `x7036D` +
`ance Dance Dance`.

The prices needed care too. `Odds 0.3F` is decimal odds **minus one**, with `F`
marking favourite. Read literally it turns a $1.30 winner into a 0.3 shot. The
check that caught it was calibration: with the correction, the actual win rate
tracks the implied probability across every price band.

### The model

A ridge regression on **finishing percentile** — where a horse finished as a
fraction of its field, so a 3rd of 8 and a 4th of 11 are comparable.

Three targets were tried on identical held-out groups:

| | log-loss | top-1 |
|---|---|---|
| **ridge on finishing percentile** | **0.9998** | **53.3%** |
| rank-ordered conditional logit | 1.0164 | 50.5% |
| gradient boosting | 1.1016 | 47.4% |

The conditional logit is the textbook choice and it lost. The reason is the
shape of this data: a past race is visible only through the runners entered
again this weekend, so a within-race ranking likelihood can use just the 60% of
rows that land in a group of two or more, while a percentile target uses all
{B['n_training_rows']:,}.

Features are built from each horse's runs **strictly before** the run being
predicted. Race-constant columns — field size, distance, prize money — are
deliberately excluded: they cannot be identified from within-race comparisons.
Leaving field size in is how a phantom became the largest coefficient in an
earlier version, because the race key was merging divided races and the model
learned to tell the two divisions apart.

### What was tried and rejected

- **Per-fold temperature calibration.** Made it worse (1.0245 against 1.0077);
  the fitted scale swung between 0.19 and 1.31 because each fold's calibration
  slice was too small. One temperature on the pooled walk-forward predictions
  is used instead.
- **Within-race centring of features.** No effect (1.1322 against 1.1313).
- **Gear changes.** Blinkers going on looked worth +0.069 of a finishing
  percentile at p=0.001. But horses about to get blinkers had a *worse*
  previous run (0.557 against 0.465), and controlling for it leaves +0.017 at
  p=0.48. It was regression to the mean.

### What this does not do

It does not beat the market, and it has no price to beat — these files carry no
odds for the upcoming race. It is a fundamentals-only model for the case where
you have a form guide and nothing else.

The win percentages rest on an assumption worth naming: the scale was fitted on
partial fields and applied to full ones, which is valid under independence of
irrelevant alternatives but is not separately verified here. The ranking is
measured; the spread between percentages is not.

The field tables are **pre-acceptance**, so they can list more runners than
start, and barriers can repeat where a runner has been scratched.

---

*Probabilistic decision support, not a guaranteed outcome. Gamble responsibly —
Gambling Help 1800 858 858,
[gamblinghelponline.org.au](https://www.gamblinghelponline.org.au).*
""")
