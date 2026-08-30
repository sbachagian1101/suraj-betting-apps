"""HorsePredictorPro — an ensemble race model, validated walk-forward.

Upload a Racing & Sports race sheet and get win and place probabilities from a
blend of a conditional logit, a gradient-boosted classifier, a LambdaRank model
and the de-vigged market price.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

import data as D
import models as M
import predict as P

st.set_page_config(page_title="HorsePredictorPro", page_icon=":material/query_stats:",
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


@st.cache_resource(show_spinner="Loading the model…")
def bundle():
    return P.load_bundle()


# Streamlit Cloud pulls new code but keeps already-imported modules in
# sys.modules, so a commit that ADDS a name to a helper module can leave the new
# app.py running against the old module. It fails deep inside whichever tab uses
# the new symbol, as an opaque AttributeError, long after the page has rendered
# fine. Checking up front turns it into one actionable sentence.
_REQUIRED = {"data": ("read_race_file", "jockey_columns", "build_features",
                      "shin_devig", "market_probability", "prepare"),
             "predict": ("load_bundle", "score_race")}
_stale = [f"{m}.{n}" for m, names in _REQUIRED.items()
          for n in names if not hasattr({"data": D, "predict": P}[m], n)]
if _stale:
    st.error(
        "**This deployment is running stale code.** Missing `"
        + "`, `".join(_stale)
        + "`, which the current `app.py` needs. Streamlit Cloud pulled the new "
          "files but kept the old modules in memory. Fix it with "
          "**Manage app → ⋮ → Reboot app**.", icon=":material/error:")
    st.stop()

B = bundle()
V = B["validation"]
if "jockey_validation" not in B:
    st.error(
        "**The saved model is out of date** — it has no jockey-only model. "
        "Re-run `python train.py` and commit the new `model_bundle.joblib`.",
        icon=":material/error:")
    st.stop()
J = B["jockey_validation"]      # used by the sidebar and the Method tab

st.title("HorsePredictorPro")
st.caption(f"Conditional logit + gradient boosting + LambdaRank, blended with "
           f"the de-vigged market. Trained on {B['n_races']:,} races "
           f"({B['n_runners']:,} runners), {B['date_range'][0]}–{B['date_range'][1]}.")

with st.sidebar:
    st.header("Settings", divider="gray")
    feature_set = st.radio(
        "What the model may look at",
        ["all", "jockey"],
        format_func=lambda k: {"all": "Everything (217 features)",
                               "jockey": "Jockey only (33 columns)"}[k],
        help="Jockey-only sees the rider's record and nothing about the horse: "
             "earnings, starts, wins, places, strike rate and ROI over the last "
             "100 rides, 12 months, this season and last, plus the apprentice "
             "claim. It is a much weaker model — see the metrics below.")
    use_market = st.toggle(
        "Use the market price", value=True,
        help="On: blend the model with the de-vigged book — the most accurate "
             "setting. Off: fundamentals only, so the rating is independent of "
             "the price and can be compared against it honestly.")
    min_edge = st.slider(
        "Flag value at edge above", 0.0, 0.50, 0.0, 0.05, format="%.2f",
        help="Edge = model probability × your odds − 1. The validated strategy "
             "used edge > 0.")
    sims = st.select_slider("Place simulations", [5000, 20000, 50000], 20000)
    st.divider()
    if feature_set == "jockey":
        st.metric("Top pick wins", f"{100*J['top1']:.1f}%",
                  f"{100*(J['top1']-J['market_top1']):+.1f} pts vs market",
                  delta_color="inverse")
        st.metric("A dart throw", f"{100*J['dart_throw_top1']:.1f}%",
                  help="The jockey columns roughly double a random pick — real "
                       "signal, but far behind the market's "
                       f"{100*J['market_top1']:.1f}%.")
        st.caption("Jockey-only. Weaker than the full model on every measure; "
                   "the numbers are on the Validation tab.")
    else:
        st.metric("Log-loss vs market", f"{-V['logloss_edge']:+.4f}",
                  help="Negative is better. Beat the market in "
                       f"{V['folds_positive_logloss']} walk-forward folds.")
        st.metric("Value-bet ROI", f"{100*V['value_roi']:+.1f}%",
                  f"{V['folds_positive_roi']} folds positive")
        st.caption("Walk-forward, out of sample. Full detail on the "
                   "Validation tab.")

pred_tab, detail_tab, valid_tab, method_tab = st.tabs(
    ["1 · Predict", "2 · Model detail", "3 · Validation", "Method"])

st.session_state.setdefault("race", None)

with pred_tab:
    st.subheader("Upload a race sheet")
    st.caption("A Racing & Sports single-race export "
               "(`<date>-<track>-rNN.xlsx` or `.csv`) — the 128-column format "
               "with `Best Fixed Odds`.")
    up = st.file_uploader("Race file", type=["xlsx", "csv"],
                          label_visibility="collapsed")
    if up is not None:
        try:
            st.session_state["race"] = D.read_race_file(up, up.name)
        except Exception as exc:                             # noqa: BLE001
            st.session_state["race"] = None
            st.error(f"Could not read that file — {type(exc).__name__}: {exc}")

    race = st.session_state["race"]
    if race is None:
        st.info("Upload a race to begin.", icon=":material/upload_file:")
    else:
        try:
            t = P.score_race(race, B, use_market=use_market, sims=int(sims),
                             feature_set=feature_set)
        except Exception as exc:                             # noqa: BLE001
            st.error(f"Could not score that race — {type(exc).__name__}: {exc}")
            t = None

        if t is not None:
            top = t.iloc[0]
            st.markdown(
                '<div class="pick"><div class="muted">TOP RATED</div>'
                '<h2 style="margin:.15rem 0">#{} · {}</h2>'
                '<b>Win {:.1f}%</b> · Place {:.1f}% · Fair ${:.2f}'
                '{}</div>'.format(
                    int(top["Num"]) if pd.notna(top["Num"]) else 0, top["Horse"],
                    top["Win %"], top["Place %"], top["Fair $"],
                    f' · market ${top["Odds"]:.2f}' if np.isfinite(top["Odds"]) else ""),
                unsafe_allow_html=True)

            if feature_set == "jockey":
                st.warning(
                    f"**Jockey-only mode.** The model is looking at "
                    f"{J['n_columns']} columns about the rider and nothing "
                    f"about the horse. On {J['test_races']} held-out races its "
                    f"top pick won **{100*J['top1']:.1f}%** — against "
                    f"{100*J['dart_throw_top1']:.1f}% for a dart throw, so the "
                    f"jockey data is genuinely informative, but against "
                    f"**{100*J['market_top1']:.1f}%** for the market and "
                    f"**{100*J['full_top1']:.1f}%** for the full model. "
                    f"Blending it with the market put "
                    f"{100*J['market_weight_when_blended']:.0f}% of the weight "
                    "on the market: it adds nothing on top of the price.",
                    icon=":material/visibility_off:")

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Runners", len(t))
            c2.metric("Places paid", t.attrs["places_paid"])
            c3.metric("Mode", "Blend" if "market" in t.attrs["mode"] else "Fundamentals")
            ov = t.attrs["overround"]
            c4.metric("Book", f"{ov:.3f}" if ov else "—",
                      f"{100*(ov-1):.1f}% margin" if ov else "incomplete",
                      delta_color="inverse")
            if not t.attrs["has_book"]:
                st.warning(
                    f"Only {t.attrs['priced_runners']} of {len(t)} runners are "
                    "priced, so the market could not be de-vigged and the model "
                    "fell back to **fundamentals only**. That is the weaker "
                    "setting — on held-out races it scored 1.9361 log-loss "
                    "against the blend's 1.8111 — and it is **not** the "
                    "configuration the value results were measured in. Treat any "
                    "edge below with more caution than usual.",
                    icon=":material/warning:")
            if ov and ov > 1.25:
                st.warning(
                    f"This book carries a **{100*(ov-1):.0f}% margin**. The "
                    "validated edge was measured on books averaging 12.9% — at "
                    "this price nothing here is likely to be value.",
                    icon=":material/warning:")

            show = t.drop(columns=[c for c in t.columns if c.startswith("_")])
            st.dataframe(
                show, hide_index=True, height=min(60 + 36 * len(show), 520),
                column_config={
                    "Odds": st.column_config.NumberColumn(format="$%.2f"),
                    "Win %": st.column_config.ProgressColumn(
                        "Win %", format="%.1f%%", min_value=0.0,
                        max_value=float(max(show["Win %"].max(), 1))),
                    "Place %": st.column_config.NumberColumn(format="%.1f%%"),
                    "Market %": st.column_config.NumberColumn(format="%.1f%%"),
                    "Fair $": st.column_config.NumberColumn(format="$%.2f"),
                    "Edge": st.column_config.NumberColumn(format="%+.2f"),
                })

            val = t[t["Edge"] > min_edge].dropna(subset=["Edge"])
            if len(val):
                names = " · ".join(
                    f'#{int(r["Num"])} {r["Horse"]} (${r["Odds"]:.2f}, '
                    f'edge {r["Edge"]:+.2f})' for _, r in val.iterrows())
                st.success(f"**Value at edge > {min_edge:.2f}:** {names}",
                           icon=":material/trending_up:")
                if t.attrs["validated_config"]:
                    st.caption(
                        f"The validated strategy backed every runner with a "
                        f"positive edge: {V['value_bets']:,} bets, "
                        f"{100*V['value_strike']:.1f}% strike, ROI "
                        f"{100*V['value_roi']:+.1f}% (95% CI "
                        f"{100*V['value_roi_ci'][0]:+.1f}% to "
                        f"{100*V['value_roi_ci'][1]:+.1f}%). These are longshots "
                        f"— expect one winner in nine and long losing runs.")
                else:
                    st.caption(
                        "⚠️ These edges come from the **fundamentals-only** "
                        "model, because the market term is unavailable or "
                        "switched off. The +22.6% ROI was measured with the "
                        "market in the blend and does **not** transfer to this "
                        "setting. Without a market anchor the model is at its "
                        "weakest on exactly these long prices.")
            else:
                st.info(f"No runner clears an edge of {min_edge:.2f}. "
                        "That is the common case and a real answer.",
                        icon=":material/info:")

with detail_tab:
    race = st.session_state["race"]
    if race is None:
        st.info("Upload a race in **1 · Predict** first.", icon=":material/info:")
    else:
        t = P.score_race(race, B, use_market=use_market, sims=5000,
                         feature_set=feature_set)
        st.subheader("What each model thinks")
        st.caption("The three fundamental models disagree by design — that is "
                   "why blending them helps. Where they agree, the blend is "
                   "confident; where they scatter, it is not.")
        comp = t[["Num", "Horse", "_logit", "_gbm", "_rank", "Market %", "Win %"]]
        comp = comp.rename(columns={"_logit": "Conditional logit",
                                    "_gbm": "Gradient boosting",
                                    "_rank": "LambdaRank",
                                    "Win %": "Blend"})
        st.dataframe(comp, hide_index=True, column_config={
            c: st.column_config.NumberColumn(format="%.1f%%")
            for c in ("Conditional logit", "Gradient boosting", "LambdaRank",
                      "Market %", "Blend")})
        w = B["weights"] if use_market else B["weights_fundamentals"]
        labels = (["Conditional logit", "Gradient boosting", "LambdaRank", "Market"]
                  if use_market else
                  ["Conditional logit", "Gradient boosting", "LambdaRank"])
        st.markdown("**Blend weights in use** — chosen on a validation period the "
                    "models were not fitted on:")
        st.dataframe(pd.DataFrame({"Model": labels, "Weight": w}),
                     hide_index=True, column_config={
                         "Weight": st.column_config.ProgressColumn(
                             "Weight", format="%.2f", min_value=0.0, max_value=1.0)})
        st.caption(
            "LambdaRank takes zero weight here. It ranks well but is not "
            "calibrated, and the blend is chosen on log-loss — so the two "
            "probability models plus the market crowd it out. It is kept "
            "because the weight is re-chosen on every refit and it earns a "
            "share in some periods.")

with valid_tab:
    st.subheader("How it was tested")
    st.markdown(f"""
Split by **date**, never at random: train → validate (blend weights only) →
test. Runners from one race can never straddle a split, and no model ever sees
the future of a jockey's season.

### Accuracy on {V['test_races']} untouched races

| | log-loss | vs market | top-1 |
|---|---|---|---|
| market, Shin de-vigged | {V['market_logloss']:.4f} | — | {100*V['market_top1']:.1f}% |
| **the blend** | **{V['blend_logloss']:.4f}** | **−{V['logloss_edge']:.4f}** | {100*V['blend_top1']:.1f}% |

The edge is **{V['logloss_edge']:+.4f}** log-loss, 95% CI
{V['edge_ci'][0]:+.4f} to {V['edge_ci'][1]:+.4f}, and it was positive in
**{V['folds_positive_logloss']}** walk-forward folds.

**Note the top-1 column.** The blend picks the winner *less* often than the
favourite ({100*V['blend_top1']:.1f}% against {100*V['market_top1']:.1f}%). Its
gain is in pricing the whole field, not in naming the winner — which is exactly
why the value strategy bets across the card rather than backing its top pick.

### Money

Backing every runner with a positive edge, refitting each fold:

| | |
|---|---|
| bets | {V['value_bets']:,} |
| strike rate | {100*V['value_strike']:.1f}% (about one in nine) |
| ROI | **{100*V['value_roi']:+.1f}%**, 95% CI {100*V['value_roi_ci'][0]:+.1f}% to {100*V['value_roi_ci'][1]:+.1f}% |
| folds positive | {V['folds_positive_roi']} |

### The three things that would kill it

**Price slippage.** These bets average about $22. The edge survives a 5% worse
price ({100*V['roi_at_5pct_worse']:+.1f}%) and a 10% worse price
({100*V['roi_at_10pct_worse']:+.1f}%), but at 20% worse it is
{100*V['roi_at_20pct_worse']:+.1f}% — gone. If you cannot reliably take the
quoted best price, you do not have this edge.

**Concentration.** Removing the five best-priced winners from each fold turns
three of the five folds negative. A small number of results carry the return.

**Variance.** One winner in nine, with a longest losing run of 48 bets and a
worst drawdown of 190 units at flat stakes in the validation. That is the
normal shape of this strategy, not a malfunction.

### Jockey-only mode

A second, deliberately blinkered model that sees **{J["n_columns"]} columns** about the
rider and nothing about the horse: earnings, starts, wins, places, strike rate
and ROI over the last 100 rides, 12 months, this season and last season, plus
the apprentice claim.

| | top-1 | top-3 | log-loss |
|---|---|---|---|
| a dart throw | {100*J["dart_throw_top1"]:.1f}% | — | — |
| **jockey only** | **{100*J["top1"]:.1f}%** | {100*J["top3"]:.1f}% | {J["logloss"]:.4f} |
| the market | {100*J["market_top1"]:.1f}% | {100*J["market_top3"]:.1f}% | {J["market_logloss"]:.4f} |
| everything + market | {100*J["full_top1"]:.1f}% | — | {J["full_logloss"]:.4f} |

The jockey columns roughly **double a random pick**, so they carry real signal.
They are also comfortably behind both the market and the full model, and when
the jockey model was blended with the market the fitted weight on the market was
**{100*J["market_weight_when_blended"]:.0f}%** — the rider's record adds nothing on top of the price.

The strongest columns, by a distance, are the **ROI figures** — this season,
12 months, last season and last 100 — ahead of strike rates and average
earnings. ROI captures whether a jockey outperforms the prices they ride at,
which is closer to skill than a raw win rate is.

### What was NOT done

No result was used to select the features, the model family or the blend method.
The blend weights are the only tuned quantity and they are chosen on a period
the models were not fitted on. Nothing here has been tested on live racing.
""")

with method_tab:
    st.subheader("The models")
    st.markdown("""
Four different mathematical objects. They are combined because they optimise
genuinely different criteria — an ensemble of near-identical models is just a
slower single model.

**Conditional logit** (Bolton & Chapman, 1986). The likelihood that matches the
question: exactly one horse wins, so a runner's probability is a softmax over
its own field. Fitted by maximising the grouped log-likelihood directly with an
L2 penalty. Linear, so it cannot overfit the way a tree can, and calibrated by
construction. It takes the largest fundamental weight.

**Gradient-boosted classifier.** Binary win/lose per runner, renormalised within
the race. Non-linear, catches interactions the logit cannot.

**LambdaRank.** Optimises the *order* within a race rather than a per-runner
probability, so its mistakes are shaped differently from the other two.

**Plackett–Luce.** Not fitted — given win probabilities it samples whole
finishing orders (Gumbel top-k) to get place probabilities. Sampling an order
respects the fact that exactly one horse can be first, which independent
per-horse draws get wrong.

**Shin (1993) de-vig**, solved by bisection, converts the book to probabilities.
The usual fixed-point iteration oscillates instead of converging on books with a
big favourite.

The blend is a **weighted geometric mean in log space** — the form Benter (1994)
uses for combining a fundamental model with the market.

### Features

Every column is turned into a **within-race z-score** *and* a **within-race
rank**, because a rating of 82 means nothing on its own and everything relative
to today's opponents. Ranks sit alongside z-scores because prize money and
earnings are heavily skewed and a rank is immune to that. Columns that are
constant within a race (distance, prize) are detected and passed through raw
rather than normalised to zero.

---
*Prediction is probabilistic decision support, not a guaranteed outcome. Gamble
responsibly — Gambling Help 1800 858 858,
[gamblinghelponline.org.au](https://www.gamblinghelponline.org.au).*
""")
