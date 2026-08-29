"""SoccerPredict — a tuned, calibrated team-strength model that knows its place.

Upload season match data, pick a fixture, get 1X2 / BTTS / Over-Under 2.5
probabilities from a Dixon–Coles model tuned on your own league — shown next to
the bookmaker, and honest about which of the two is sharper.
"""
from __future__ import annotations

import glob
import io
import os

import numpy as np
import pandas as pd
import streamlit as st

import assess as A
import soccer_data as sd
import soccer_model as sm

st.set_page_config(page_title="SoccerPredict", page_icon=":material/sports_soccer:",
                   layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
.block-container {padding-top: 1.2rem; padding-bottom: 3rem;}
div[data-testid="stMetric"] {border:1px solid rgba(128,128,128,.22);
  border-radius:.7rem; padding:.55rem .8rem;}
.pick {border:1px solid rgba(52,152,219,.55); background:rgba(52,152,219,.10);
  border-radius:.8rem; padding:.8rem 1.1rem; margin:.4rem 0 .9rem;}
.muted {opacity:.65; font-size:.78rem; letter-spacing:.06em;}
</style>
""", unsafe_allow_html=True)

_HERE = os.path.dirname(os.path.abspath(__file__))


@st.cache_data(show_spinner="Reading match data…")
def load_uploads(payloads):
    return sd.load_frames([io.BytesIO(b) for _, b in payloads],
                          [n for n, _ in payloads])


@st.cache_data(show_spinner="Reading bundled data…")
def load_sample():
    return sd.load_frames(sorted(glob.glob(os.path.join(_HERE, "sample_data", "*.csv"))))


@st.cache_data(show_spinner="Searching settings for this league…")
def tune_cached(df):
    return A.tune(df)


@st.cache_data(show_spinner="Re-fitting before every matchday…")
def backtest(df, params):
    return sm.walk_forward(df, **params)


st.title("SoccerPredict")
st.caption("A Dixon–Coles team-strength model, tuned on your league and scored "
           "against the bookmaker. It sits **behind the market** — this app is "
           "built to show you by how much, and where.")

with st.sidebar:
    st.header("Data", divider="gray")
    ups = st.file_uploader("Season CSVs (one per season)", type=["csv"],
                           accept_multiple_files=True,
                           help="FootyStats-style export. Two to three seasons "
                                "is the useful range — a third measured worth "
                                "0.0002 log-loss, a fourth less.")
    use_sample = st.checkbox("Use bundled data (Dutch Eerste Divisie, 3 seasons)",
                             value=not ups)
    st.divider()
    st.header("Model", divider="gray")
    auto = st.toggle("Tune on this league", value=True,
                     help="Searches 27 settings on an inner time split. The "
                          "shipped defaults were tuned on a 10-team league and "
                          "rank 23rd of 27 on a 20-team one.")

if ups:
    df, notes = load_uploads([(u.name, u.getvalue()) for u in ups])
elif use_sample:
    df, notes = load_sample()
else:
    df, notes = pd.DataFrame(), ["Upload at least one season CSV, or tick the "
                                 "bundled-data box."]

params, tune_table = A.default_params(), pd.DataFrame()
if not df.empty and auto:
    params, tune_table = tune_cached(df)

with st.sidebar:
    if not df.empty:
        st.caption("**Settings in use**")
        st.dataframe(
            pd.DataFrame({"setting": ["time decay ξ", "shrinkage",
                                      "weight on xG", "weight on SoT"],
                          "value": [params["xi"], params["reg"],
                                    params["w_xg"], params["w_sot"]]}),
            hide_index=True, width="stretch")

data_tab, predict_tab, backtest_tab, method_tab = st.tabs(
    ["1 · Data", "2 · Predict", "3 · How good is it", "Method"])

# --------------------------------------------------------------------- data
with data_tab:
    if df.empty:
        st.info("Upload season CSVs, or tick the bundled-data box in the sidebar.",
                icon=":material/upload_file:")
        for n in notes:
            st.warning(n)
    else:
        for n in notes:
            st.info(n, icon=":material/check_circle:")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Completed matches", f"{len(df):,}")
        c2.metric("Teams", int(pd.concat([df.home_team_name,
                                          df.away_team_name]).nunique()))
        c3.metric("Goals per match", f"{(df.hg + df.ag).mean():.2f}")
        c4.metric("Home win rate", f"{(df.hg > df.ag).mean():.1%}")
        st.caption(f"{df.date.min().date()} to {df.date.max().date()}")

        if not tune_table.empty:
            st.markdown("### Settings searched on this league")
            best = tune_table.iloc[0]
            dflt = tune_table[(tune_table.xi == sm.XI) & (tune_table.reg == sm.REG)
                              & (tune_table.w_xg == sm.W_XG)]
            c1, c2 = st.columns(2)
            c1.metric("Best setting found", f"{best.log_loss:.4f}",
                      "validation log-loss", delta_color="off")
            if len(dflt):
                rank = int(dflt.index[0]) + 1
                c2.metric("Where the shipped defaults rank",
                          f"{rank} of {len(tune_table)}",
                          f"{dflt.iloc[0].log_loss - best.log_loss:+.4f} worse",
                          delta_color="inverse")
            st.dataframe(tune_table.head(10), hide_index=True, column_config={
                "log_loss": st.column_config.NumberColumn("validation log-loss",
                                                          format="%.4f"),
                "xi": st.column_config.NumberColumn(format="%.4f"),
                "reg": st.column_config.NumberColumn(format="%.3f")})
            spread = float(tune_table.log_loss.max() - tune_table.log_loss.min())
            st.caption(
                f"The whole grid spans {spread:.4f} log-loss and the top few "
                "settings sit inside the noise, so treat the winner as *a "
                "reasonable setting*, not the optimum. Tuning gains about "
                "0.0025 — real, but nowhere near enough to close the gap to the "
                "market.")

        with st.expander("The matches"):
            st.dataframe(
                df[["date", "home_team_name", "hg", "ag", "away_team_name"]]
                .sort_values("date", ascending=False),
                hide_index=True, height=420)

# ------------------------------------------------------------------ predict
with predict_tab:
    if df.empty:
        st.info("Load data first.", icon=":material/info:")
    else:
        teams = sd.teams_of(df)
        c1, c2 = st.columns(2)
        home = c1.selectbox("Home team", teams, index=0)
        away = c2.selectbox("Away team", [t for t in teams if t != home], index=0)

        st.markdown("**The bookmaker's prices** — the comparison this app is "
                    "built around. Leave at 1.00 if you do not have them.")
        o1, o2, o3 = st.columns(3)
        odds_h = o1.number_input(f"1 · {home}", 1.0, 200.0, 1.0, 0.05)
        odds_d = o2.number_input("X · Draw", 1.0, 200.0, 1.0, 0.05)
        odds_a = o3.number_input(f"2 · {away}", 1.0, 200.0, 1.0, 0.05)

        if st.button("Predict", type="primary", icon=":material/play_arrow:"):
            st.session_state["fixture"] = (home, away)

        if st.session_state.get("fixture") == (home, away):
            try:
                model = sm.fit(df, **params)
                p = sm.predict(model, home, away)
            except KeyError as exc:
                st.error(f"No history for {exc}. A team needs matches in the "
                         "loaded seasons before it can be rated.")
                p = None
            except ValueError as exc:
                st.error(f"Could not fit the model — {exc}")
                p = None

            if p:
                lh, la = p["lambda_home"], p["lambda_away"]
                probs = np.array([p["home_win"], p["draw"], p["away_win"]])
                best = ["Home win", "Draw", "Away win"][int(probs.argmax())]
                st.markdown(
                    f'<div class="pick"><div class="muted">MODEL VIEW</div>'
                    f'<h2 style="margin:.15rem 0">{home} {lh:.2f} – {la:.2f} '
                    f'{away}</h2>Most likely: <b>{best} '
                    f'({100*probs.max():.1f}%)</b> · most likely score '
                    f'<b>{p["top_scorelines"][0][0]}–{p["top_scorelines"][0][1]}</b> '
                    f'({100*p["top_scorelines"][0][2]:.1f}%)</div>',
                    unsafe_allow_html=True)

                odds = (odds_h, odds_d, odds_a)
                have_odds = min(odds) > 1.0
                fl = A.flag(p, odds if have_odds else None)
                mk = sm.devig_1x2(*odds) if fl else [float("nan")] * 3
                rec = A.recommend(p, model, odds if have_odds else None)

                # ---- the recommendation -------------------------------
                tone = {"high": "#2ecc71", "medium": "#3498db",
                        "low": "#f1c40f"}[rec["confidence"]]
                ev_line = ""
                if rec["ev"] is not None:
                    ev_line = (f' · at ${rec["odds"]:.2f} the expected return is '
                               f'<b>{100*rec["ev"]:+.1f}%</b>')
                st.markdown(
                    f'<div style="border:1px solid {tone}; background:{tone}1a; '
                    f'border-radius:.8rem; padding:.9rem 1.2rem; margin:.5rem 0 .3rem">'
                    f'<div class="muted">RECOMMENDATION · CONFIDENCE '
                    f'{rec["confidence"].upper()}</div>'
                    f'<h2 style="margin:.15rem 0">{rec["selection"]}</h2>'
                    f'<b>{100*rec["probability"]:.1f}%</b> · fair price '
                    f'<b>${rec["fair_odds"]:.2f}</b>{ev_line}</div>',
                    unsafe_allow_html=True)

                st.caption(A.CONFIDENCE_TEXT[rec["confidence"]])

                with st.expander("**Why** — the reasoning behind this call",
                                 expanded=True):
                    for r_ in rec["reasons"]:
                        st.markdown(f"- {r_}")
                    b = rec["bucket"]
                    if b:
                        span = (f"{100*b['ci'][0]:+.0f}% to "
                                f"{100*b['ci'][1]:+.0f}%")
                        if b["conclusive"]:
                            st.markdown(
                                f"- **What this has been worth:** flat-staking "
                                f"every pick in this bucket over {b['n']} "
                                f"matches **lost {abs(100*b['roi']):.1f}%** "
                                f"(95% CI {span} — clear of zero, so that is a "
                                "real loss, not noise).")
                        else:
                            st.markdown(
                                f"- **What this has been worth:** {b['n']} "
                                f"matches, point estimate "
                                f"{100*b['roi']:+.1f}%, but the interval spans "
                                f"{span} — it crosses zero, so this bucket "
                                "establishes nothing either way. Do not read "
                                "the point estimate as a result.")
                    st.markdown(
                        f"- **The honest bottom line:** across every bucket "
                        f"tested, flat-staking lost money. Backing the market "
                        f"favourite every week returned "
                        f"{100*A.FAVOURITE_ROI['roi']:+.1f}% against a "
                        f"{100*(A.FAVOURITE_ROI['overround']-1):.1f}% book "
                        "margin. This is a **view**, not an edge — the "
                        "recommendation names the most likely outcome, which is "
                        "not the same as a bet worth making.")

                st.divider()
                st.markdown("### The full picture")

                for col, label, pm, pk in zip(
                        st.columns(3),
                        [f"1 · {home}", "X · Draw", f"2 · {away}"], probs, mk):
                    col.metric(label, f"{100*pm:.1f}%",
                               f"market {100*pk:.1f}%" if fl else "no price",
                               delta_color="off")

                if fl:
                    box = {"aligned": st.success, "watch": st.info,
                           "caution": st.warning}[fl["level"]]
                    box(f"**{fl['level'].upper()} — biggest gap "
                        f"{100*fl['gap']:.1f} points on “{fl['outcome']}”** "
                        f"(model {100*fl['model_p']:.1f}%, market "
                        f"{100*fl['market_p']:.1f}%). {A.FLAG_TEXT[fl['level']]}",
                        icon=":material/flag:")
                    if not fl["same_favourite"]:
                        st.warning(
                            "The model and the market name **different "
                            "favourites**. On this league's history the market "
                            "won those arguments — see tab 3.",
                            icon=":material/warning:")
                else:
                    st.info("Enter the three prices to see where the model "
                            "disagrees. Without them there is nothing to check "
                            "it against.", icon=":material/info:")

                st.markdown("### Other markets")
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("BTTS yes", f"{100*p['btts_yes']:.1f}%")
                m2.metric("BTTS no", f"{100*p['btts_no']:.1f}%")
                m3.metric("Over 2.5", f"{100*p['over_25']:.1f}%")
                m4.metric("Under 2.5", f"{100*p['under_25']:.1f}%")
                st.caption(
                    "⚠️ On the bundled league these carried **no signal over the "
                    "base rate** — 0.6677 on over/under 2.5 against a base rate "
                    "of 0.6622, i.e. worse than always predicting the league "
                    "average. Shown for completeness, not as a recommendation.")

                st.markdown("### Most likely scorelines")
                sl = pd.DataFrame([{"Score": f"{i}–{j}",
                                    "Probability %": round(100 * v, 1)}
                                   for i, j, v in p["top_scorelines"]])
                st.dataframe(sl, hide_index=True, column_config={
                    "Probability %": st.column_config.ProgressColumn(
                        "Probability %", format="%.1f%%", min_value=0.0,
                        max_value=float(sl["Probability %"].max()))})

# ----------------------------------------------------------------- backtest
with backtest_tab:
    if df.empty:
        st.info("Load data first.", icon=":material/info:")
    else:
        st.caption("Walk-forward: re-fitted before every matchday, predicting "
                   "only that day, so no match contributes to its own prediction.")
        if st.button("Run the backtest", type="primary",
                     icon=":material/query_stats:"):
            st.session_state["bt"] = backtest(df, params)

        bt = st.session_state.get("bt")
        if bt is None:
            st.info("Run the backtest to see how the model actually does.",
                    icon=":material/info:")
        elif bt.empty:
            st.error("The backtest produced no predictions — not enough history.")
        else:
            ev = sm.evaluate(bt)
            d = A.with_market(bt)
            gap = A.overall_gap(d)

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Matches scored", f"{ev['n']:,}")
            c2.metric("Model log-loss", f"{ev['logloss_1x2']:.4f}")
            c3.metric("League base rates", f"{ev['logloss_baserate']:.4f}",
                      f"{ev['logloss_1x2'] - ev['logloss_baserate']:+.4f}",
                      delta_color="inverse")
            if gap:
                c4.metric("Bookmaker", f"{gap['market']:.4f}",
                          f"{gap['gap']:+.4f}", delta_color="inverse")

            if gap:
                verdict = ("**behind the market**" if gap["market_ahead"]
                           else "not separable from the market")
                st.warning(
                    f"Over {gap['n']:,} matches with a full book the model is "
                    f"{verdict}: {gap['model']:.4f} against {gap['market']:.4f}, "
                    f"a gap of **{gap['gap']:+.4f}** (95% CI {gap['ci_low']:+.4f} "
                    f"to {gap['ci_high']:+.4f}). It does beat the league base "
                    f"rate of {ev['logloss_baserate']:.4f}, so it is learning "
                    "something real — just less than the market already knows.",
                    icon=":material/info:")

                st.markdown("### Where the model is weakest")
                dt = A.disagreement_table(d)
                if not dt.empty:
                    st.dataframe(dt, hide_index=True, column_config={
                        "model": st.column_config.NumberColumn(format="%.4f"),
                        "market": st.column_config.NumberColumn(format="%.4f"),
                        "model_minus_market": st.column_config.NumberColumn(
                            "model − market", format="%+.4f",
                            help="Positive means the model is worse."),
                        "ci_low": st.column_config.NumberColumn("CI low",
                                                                format="%+.4f"),
                        "ci_high": st.column_config.NumberColumn("CI high",
                                                                 format="%+.4f"),
                        "model_worse": st.column_config.CheckboxColumn(
                            "significant")})
                    st.markdown(
                        "**Read `model − market` downwards.** It grows with "
                        "disagreement: the further the model strays from the "
                        "market, the worse it does. That is why the flag on the "
                        "Predict tab treats a large gap as a warning about the "
                        "model, not a tip about the match.")

                pc = A.pick_conflict(d)
                if pc.get("conflicts"):
                    st.markdown("### When they name different favourites")
                    k1, k2, k3 = st.columns(3)
                    k1.metric("Matches",
                              f"{pc['conflicts']} ({100*pc['share']:.0f}%)")
                    k2.metric("Model right", f"{100*pc['model_right']:.1f}%")
                    k3.metric("Market right", f"{100*pc['market_right']:.1f}%",
                              f"{100*(pc['market_right']-pc['model_right']):+.1f} pts")

            st.markdown("### Is it calibrated?")
            ct = A.calibration_table(bt)
            st.dataframe(ct, hide_index=True, column_config={
                "predicted": st.column_config.NumberColumn(format="%.3f"),
                "actual": st.column_config.NumberColumn(format="%.3f"),
                "error": st.column_config.NumberColumn("actual − predicted",
                                                       format="%+.3f")})
            st.caption(
                f"Mean absolute calibration error "
                f"**{A.calibration_error(bt):.4f}**, weighted by how many "
                "forecasts fall in each band. Calibration and accuracy are "
                "different things: this model is honest about its own "
                "uncertainty even though the market out-predicts it.")

            st.markdown("### Other markets")
            base_o = float(bt.o25.mean())
            bo = float(-(bt.o25 * np.log(base_o)
                         + (1 - bt.o25) * np.log(1 - base_o)).mean())
            g1, g2, g3 = st.columns(3)
            g1.metric("BTTS log-loss", f"{ev['logloss_btts']:.4f}")
            g2.metric("Over/Under 2.5", f"{ev['logloss_o25']:.4f}")
            g3.metric("Over 2.5 base rate", f"{bo:.4f}",
                      f"{ev['logloss_o25'] - bo:+.4f}", delta_color="inverse")
            st.caption(
                f"Compare against the **base rate**, not a coin flip: always "
                f"predicting this league's {100*base_o:.1f}% over-2.5 rate "
                f"scores {bo:.4f}. A coin flip (0.6931) is the wrong benchmark "
                "and flatters these numbers.")

# ------------------------------------------------------------------- method
with method_tab:
    st.subheader("What this is, and what it is not")
    st.markdown("""
### The model

**Dixon–Coles** team strengths. Each team gets an attack and a defence
parameter, and a fixture's expected goals are

```
λ_home = exp(μ + attack_home + defence_away + γ)
λ_away = exp(μ + attack_away + defence_home)
```

with `γ` the home advantage **fitted from your data**, not assumed. Attack is
always measured against the specific opponent's defence, so a team gets no
credit for feasting on the league's worst side. A low-score correction (`τ`)
repairs the 0–0 / 1–0 / 0–1 / 1–1 cells that independent Poissons misprice —
which is where football lives.

The response is a blend of goals, xG and shots on target: goals are what
happened, the other two are less noisy measures of how a team played.

### Tuned per league, because the defaults were not yours

The shipped defaults were tuned on a **ten-team** Latvian league. On a
**twenty-team** Dutch one they rank **23rd of 27**. Every league now gets its own
search on an inner time split. The gain is real but small — about **0.0025
log-loss** — and the top settings sit inside the noise, so the winner is *a
reasonable setting*, not the optimum.

### It sits behind the market, and the app says so everywhere

On three seasons of Dutch Eerste Divisie:

| | 1X2 log-loss |
|---|---|
| league base rates | 1.0557 |
| **this model** | **≈1.021** |
| the bookmaker | 1.0008 |

The model captures roughly two-thirds of the available signal. The gap to the
bookmaker is about **+0.02 with a confidence interval clear of zero** — real,
not noise.

### The flag is inverted, and that is the whole point

The intuitive idea is that where the model departs from the market it has found
something. **Measured, the opposite holds.** Split the matches by how far apart
the two are and the model's disadvantage *grows* with the disagreement; where
the two name different favourites the market has been right far more often.

So a large gap is evidence that **the model** is wrong — usually because the
market has absorbed team news the model cannot see. The flag says "look closer",
never "back this". Tab 3 recomputes that evidence from whatever data you load,
rather than taking it on faith.

### Measured and rejected

- **More seasons.** A third season of the same league was worth **0.0002**
  log-loss. A fourth would be worth less.
- **Other leagues.** Home advantage ranges 0.164 to 0.415 across nine leagues
  and the low-score correction flips sign. Pooling would make this league worse.
- **Blending with the market.** Choosing the weight on a validation half put
  **zero** weight on the model.
- **The feed's own pre-match PPG/xG columns.** Adding them made it worse —
  1.0455 against 1.0348 for the model alone.
- **The goals markets.** Over/under 2.5 scored 0.6677 against a 0.6622 base
  rate: worse than always predicting the league average.

### What would actually help

Not more rows. In order: **closing odds** rather than pre-match prices — sharper,
and the movement between them is itself signal; **lineups and injuries**, absent
from this feed entirely and probably the largest missing variable in football
modelling; and for a promoted side, data from the division it came up from.

---
*Prediction is probabilistic decision support, not a guaranteed outcome. Gamble
responsibly — Gambling Help 1800 858 858,
[gamblinghelponline.org.au](https://www.gamblinghelponline.org.au).*
""")
