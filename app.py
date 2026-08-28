"""SoccerPredict - upload season match data, pick a fixture, get 1X2 / BTTS / O-U 2.5."""
from __future__ import annotations

import glob
import os

import numpy as np
import pandas as pd
import streamlit as st

import soccer_data as sd
import soccer_model as sm

st.set_page_config(page_title="SoccerPredict", page_icon="⚽",
                   layout="wide", initial_sidebar_state="expanded")
st.markdown('''
<style>
.betgreen {border:1px solid rgba(46,204,113,.75); background:rgba(46,204,113,.16);
  border-radius:.8rem; padding:.85rem 1.1rem; margin:.5rem 0 .2rem;}
.betyellow {border:1px solid rgba(241,196,15,.85); background:rgba(241,196,15,.16);
  border-radius:.8rem; padding:.85rem 1.1rem; margin:.5rem 0 .2rem;}
.bethead {font-size:1.35rem; font-weight:700; margin:0 0 .2rem;}
.betwhy {opacity:.85; font-size:.9rem;}
</style>
''', unsafe_allow_html=True)
st.markdown("""
<style>
.block-container {padding-top: 1.15rem; padding-bottom: 3rem;}
div[data-testid="stMetric"] {border:1px solid rgba(128,128,128,.22);border-radius:.7rem;padding:.55rem .8rem;}
.hero {padding:1rem 1.1rem;border-radius:.85rem;background:linear-gradient(135deg,rgba(16,185,129,.18),rgba(0,0,0,0));border:1px solid rgba(128,128,128,.22);margin-bottom:1rem;}
.pick {padding:1rem 1.2rem;border:1px solid rgba(46,204,113,.35);border-radius:.8rem;background:rgba(46,204,113,.08);margin:.4rem 0 1rem 0;}
.muted {opacity:.72;font-size:.9rem;}
</style>
""", unsafe_allow_html=True)

SAMPLE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sample_data")


@st.cache_data(show_spinner=False)
def load_sample():
    files = sorted(glob.glob(os.path.join(SAMPLE_DIR, "*.csv")))
    if not files:
        return pd.DataFrame(), ["No bundled sample data found."]
    return sd.load_frames(files, names=[os.path.basename(f) for f in files])


@st.cache_data(show_spinner=False)
def load_uploads(payloads: list[tuple[str, bytes]]):
    import io as _io
    return sd.load_frames([_io.BytesIO(b) for _, b in payloads],
                          names=[n for n, _ in payloads])


@st.cache_data(show_spinner=False)
def fit_cached(df: pd.DataFrame, xi: float, reg: float, w_xg: float, w_sot: float):
    return sm.fit(df, xi=xi, reg=reg, w_xg=w_xg, w_sot=w_sot)


@st.cache_data(show_spinner=False)
def backtest_cached(df: pd.DataFrame, xi: float, reg: float, w_xg: float, w_sot: float):
    bt = sm.walk_forward(df, xi=xi, reg=reg, w_xg=w_xg, w_sot=w_sot)
    return bt, sm.evaluate(bt)


st.markdown(
    '<div class="hero"><h1 style="margin:0">⚽ SoccerPredict</h1>'
    '<div class="muted">Upload season match data · Dixon–Coles team strengths from '
    'goals, xG and shots on target · 1X2, BTTS and Over/Under 2.5</div></div>',
    unsafe_allow_html=True)

with st.sidebar:
    st.header("Data source")
    uploads = st.file_uploader(
        "Season CSVs (upload one file per season)", type=["csv"], accept_multiple_files=True,
        help="FootyStats-style match export. Upload as many seasons as you have; "
             "three is a good balance of sample size and relevance.")
    use_sample = st.checkbox("Use bundled sample data (Latvian Virsliga 2024–2026)",
                             value=not uploads)
    st.divider()
    st.header("Model settings")
    xi = st.slider("Time decay ξ (per day)", 0.0, 0.004, float(sm.XI), 0.0001, format="%.4f",
                   help="Weights each match by exp(-ξ × days ago). The backtest preferred "
                        "a very light decay — in a small league, old matches still inform.")
    reg = st.slider("Strength shrinkage", 0.0, 0.05, float(sm.REG), 0.001, format="%.3f",
                    help="Pulls attack/defence toward league average. Protects newly "
                         "promoted teams with few matches.")
    w_xg = st.slider("Weight on xG", 0.0, 0.6, float(sm.W_XG), 0.05)
    w_sot = st.slider("Weight on shots on target", 0.0, 0.6, float(sm.W_SOT), 0.05)
    if w_xg + w_sot > 0.9:
        st.warning("xG + SoT weights above 0.9 leave almost no weight on actual goals.")
        w_sot = min(w_sot, 0.9 - w_xg)
    st.caption(f"Actual goals therefore carry weight **{max(0.0, 1 - w_xg - w_sot):.2f}**.")
    st.divider()
    st.header("Bet signal")
    bet_threshold = st.slider(
        "Minimum win probability", 0.30, 0.80, float(sm.BET_MIN_WIN_PROB), 0.01,
        format="%.2f",
        help="A signal needs the side's 1X2 probability strictly above this AND "
             "the two most likely scorelines to agree. Green: both top scores "
             "are wins for that side. Yellow: the top score is a win but the "
             "second is a draw.")
    st.divider()
    st.caption("Predictions are probabilistic decision support, not a guaranteed outcome.")

if uploads:
    df, notes = load_uploads([(u.name, u.getvalue()) for u in uploads])
elif use_sample:
    df, notes = load_sample()
else:
    df, notes = pd.DataFrame(), ["Upload at least one season CSV, or tick the sample-data box."]

data_tab, predict_tab, ratings_tab, backtest_tab, method_tab = st.tabs(
    ["1 · Data", "2 · Predict", "3 · Team Ratings", "4 · Backtest", "Method"])

with data_tab:
    st.subheader("Loaded data")
    for n in notes:
        (st.error if n.startswith("❌") else st.info)(n)
    if df.empty:
        st.stop()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Completed matches", len(df))
    c2.metric("Teams", len(sd.teams_of(df)))
    c3.metric("Seasons", df["season"].nunique())
    c4.metric("Goals per match", f"{(df['hg'] + df['ag']).mean():.2f}")
    st.caption(f"Date range **{df['date'].min():%d %b %Y} → {df['date'].max():%d %b %Y}**. "
               f"Home wins {np.mean(df.hg > df.ag):.1%} · draws {np.mean(df.hg == df.ag):.1%} · "
               f"away wins {np.mean(df.hg < df.ag):.1%} · "
               f"BTTS {np.mean((df.hg > 0) & (df.ag > 0)):.1%} · "
               f"Over 2.5 {np.mean(df.hg + df.ag > 2.5):.1%}.")
    st.markdown("**Matches per season**")
    tmp = df.assign(total=df["hg"] + df["ag"])
    per = (tmp.groupby("season")
              .agg(matches=("total", "size"), teams=("home_team_name", "nunique"),
                   goals_per_match=("total", "mean"),
                   btts_rate=("total", lambda s: np.nan))
              .reset_index())
    per["goals_per_match"] = per["goals_per_match"].round(2)
    per["btts_rate"] = (tmp.assign(b=(tmp.hg > 0) & (tmp.ag > 0))
                          .groupby("season")["b"].mean().round(3).values)
    st.dataframe(per, width="stretch", hide_index=True)
    with st.expander("Preview the parsed matches"):
        cols = ["date", "season", "home_team_name", "hg", "ag", "away_team_name",
                "team_a_xg", "team_b_xg", "home_team_shots_on_target",
                "away_team_shots_on_target"]
        st.dataframe(df[[c for c in cols if c in df.columns]].tail(200),
                     width="stretch", hide_index=True)

try:
    model = fit_cached(df, xi, reg, w_xg, w_sot)
except Exception as exc:                                       # noqa: BLE001
    with predict_tab:
        st.error(f"Could not fit a model: {exc}")
    st.stop()

with predict_tab:
    teams = model.teams
    st.subheader("Select the fixture")
    c1, c2 = st.columns(2)
    home = c1.selectbox("Home team", teams,
                        index=teams.index("Auda") if "Auda" in teams else 0)
    away_options = [t for t in teams if t != home]
    default_away = away_options.index("Liepāja") if "Liepāja" in away_options else 0
    away = c2.selectbox("Away team", away_options, index=default_away)

    with st.expander("Optional: enter bookmaker odds to compare against the market"):
        st.caption("Leave blank to see the pure model. The backtest shows the market is "
                   "sharper than the model on 1X2, so treat disagreement as a flag to "
                   "investigate, not as free money.")
        o1, o2, o3 = st.columns(3)
        odds_h = o1.number_input("Home win odds", 1.0, 100.0, 1.0, 0.01)
        odds_d = o2.number_input("Draw odds", 1.0, 100.0, 1.0, 0.01)
        odds_a = o3.number_input("Away win odds", 1.0, 100.0, 1.0, 0.01)
        w_market = st.slider("Weight on market in the blend", 0.0, 1.0, 0.5, 0.05)

    if st.button("Analyse & predict ▶", type="primary", width="stretch"):
        st.session_state["pred"] = sm.predict(model, home, away)
        st.session_state["pred_pair"] = (home, away)

    pred = st.session_state.get("pred")
    if pred and st.session_state.get("pred_pair") == (home, away):
        lh, la = pred["lambda_home"], pred["lambda_away"]
        probs = np.array([pred["home_win"], pred["draw"], pred["away_win"]])
        blended = None
        if min(odds_h, odds_d, odds_a) > 1.0:
            mkt = sm.devig_1x2(odds_h, odds_d, odds_a)
            blended = sm.blend(probs, mkt, w_market)

        best = ["Home win", "Draw", "Away win"][int(probs.argmax())]
        st.markdown(
            f'<div class="pick"><div class="muted">MODEL VIEW</div>'
            f'<h2 style="margin:.15rem 0">{home} {lh:.2f} – {la:.2f} {away}</h2>'
            f'Most likely outcome: <b>{best} ({100*probs.max():.1f}%)</b> · '
            f'most likely score <b>{pred["top_scorelines"][0][0]}–{pred["top_scorelines"][0][1]}</b> '
            f'({100*pred["top_scorelines"][0][2]:.1f}%)</div>', unsafe_allow_html=True)

        sig = sm.bet_signal(pred, min_win_prob=float(bet_threshold))
        if sig:
            cls = "betgreen" if sig["level"] == sm.BET_GREEN else "betyellow"
            tick = "\u2705" if sig["level"] == sm.BET_GREEN else "\u26a0\ufe0f"
            st.markdown(
                f'<div class="{cls}"><div class="bethead">{tick} Bet '
                f'{sig["team"]}</div>'
                f'<div class="betwhy">{sm.bet_signal_text(sig)}</div></div>',
                unsafe_allow_html=True)
            M = sm.BET_MEASURED
            if sig["level"] == sm.BET_GREEN:
                st.caption(
                    f"On {M['matches']} walk-forward matches from the bundled "
                    f"seasons, green fired {M['green_n']} times and the named "
                    f"team won **{100*M['green_won']:.1f}%** of them — against "
                    f"{100*M['over45_won']:.1f}% for backing every side the "
                    f"model rates over 45%.")
            else:
                st.caption(
                    f"**Yellow is a caution, not a weaker green.** On "
                    f"{M['matches']} walk-forward matches it fired "
                    f"{M['yellow_n']} times and the named team won only "
                    f"**{100*M['yellow_won']:.1f}%** — below the "
                    f"{100*M['yellow_model_said']:.1f}% the model itself gave "
                    f"them, and **{100*M['yellow_drew']:.0f}%** of these matches "
                    f"were drawn against {100*M['over45_drew']:.0f}% for >45% "
                    f"sides generally. The draw in second place is a real "
                    f"warning. n={M['yellow_n']} — treat it as a flag to look "
                    f"closer, not a bet.")
        elif max(pred["home_win"], pred["away_win"]) > float(bet_threshold):
            st.info(
                f"No bet signal: a side is above "
                f"{100*float(bet_threshold):.0f}% but the two most likely "
                f"scorelines do not confirm it. A draw as the *most* likely "
                f"score is the distribution disagreeing with the 1X2 number.")

        st.markdown("### 1X2")
        cols = st.columns(3)
        for col, label, p, o in zip(cols, [f"1 · {home}", "X · Draw", f"2 · {away}"],
                                    probs, [odds_h, odds_d, odds_a]):
            col.metric(label, f"{100*p:.1f}%", f"fair ${1/max(p,1e-9):.2f}")
        if blended is not None:
            st.markdown("**Model vs market**")
            mkt = sm.devig_1x2(odds_h, odds_d, odds_a)
            t = pd.DataFrame({
                "Outcome": [f"1 · {home}", "X · Draw", f"2 · {away}"],
                "Model %": 100 * probs, "Market %": 100 * mkt,
                "Blend %": 100 * blended,
                "Your odds": [odds_h, odds_d, odds_a],
                "Model fair $": 1 / np.clip(probs, 1e-9, None),
                "Edge (model)": probs * np.array([odds_h, odds_d, odds_a]) - 1,
            })
            st.dataframe(t, width="stretch", hide_index=True, column_config={
                "Model %": st.column_config.NumberColumn(format="%.1f%%"),
                "Market %": st.column_config.NumberColumn(format="%.1f%%"),
                "Blend %": st.column_config.NumberColumn(format="%.1f%%"),
                "Model fair $": st.column_config.NumberColumn(format="$%.2f"),
                "Edge (model)": st.column_config.NumberColumn(format="%+.3f"),
            })
            st.caption("Edge = model probability × your odds − 1. Positive means the model "
                       "thinks the price is too big. The backtest says the market usually "
                       "wins these arguments.")

        st.markdown("### Both teams to score / Over–Under 2.5")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("BTTS Yes", f"{100*pred['btts_yes']:.1f}%", f"fair ${1/max(pred['btts_yes'],1e-9):.2f}")
        c2.metric("BTTS No", f"{100*pred['btts_no']:.1f}%", f"fair ${1/max(pred['btts_no'],1e-9):.2f}")
        c3.metric("Over 2.5", f"{100*pred['over_25']:.1f}%", f"fair ${1/max(pred['over_25'],1e-9):.2f}")
        c4.metric("Under 2.5", f"{100*pred['under_25']:.1f}%", f"fair ${1/max(pred['under_25'],1e-9):.2f}")
        st.caption(f"Also: Over 1.5 {100*pred['over_15']:.1f}% · Over 3.5 {100*pred['over_35']:.1f}% · "
                   f"{home} clean sheet {100*pred['home_cs']:.1f}% · "
                   f"{away} clean sheet {100*pred['away_cs']:.1f}%.")

        st.markdown("### Most likely scorelines")
        sl = pd.DataFrame([{"Score": f"{i}–{j}", "Probability %": round(100 * p, 1)}
                           for i, j, p in pred["top_scorelines"]])
        st.dataframe(sl, width="stretch", hide_index=True, column_config={
            "Probability %": st.column_config.ProgressColumn(
                "Probability %", format="%.1f%%", min_value=0.0,
                max_value=float(sl["Probability %"].max()))})

        with st.expander("Full score matrix (rows = home goals, columns = away goals)"):
            m = pred["matrix"][:7, :7]
            mm = pd.DataFrame((100 * m).round(2),
                              index=[f"{home} {i}" for i in range(7)],
                              columns=[f"{away} {j}" for j in range(7)])
            st.dataframe(mm, width="stretch")

        st.markdown("### Why")
        i, j = model.index[home], model.index[away]
        why = pd.DataFrame([
            {"Component": "Attack index", home: float(np.exp(model.attack[i])),
             away: float(np.exp(model.attack[j])),
             "Reads as": "1.00 = league-average attack; higher scores more"},
            {"Component": "Defence index", home: float(np.exp(model.defence[i])),
             away: float(np.exp(model.defence[j])),
             "Reads as": "1.00 = league-average defence; LOWER concedes fewer"},
            {"Component": "Expected goals this match", home: lh, away: la,
             "Reads as": "Attack × opponent defence × home advantage"},
        ])
        st.dataframe(why, width="stretch", hide_index=True, column_config={
            home: st.column_config.NumberColumn(format="%.3f"),
            away: st.column_config.NumberColumn(format="%.3f")})
        st.caption(
            f"Home advantage in this data is worth **{model.home_advantage_goals:+.2f} goals** "
            f"to the home side. Low-score correction ρ = {model.rho:+.4f}. "
            f"Fitted on {model.n_matches} matches with response weights "
            f"goals {model.weights['goals']:.2f} / xG {model.weights['xg']:.2f} / "
            f"SoT {model.weights['sot']:.2f}.")

with ratings_tab:
    st.subheader("Team ratings")
    st.caption("Attack and defence indexes come from the fitted model — they already "
               "account for opponent quality and home advantage, unlike raw goals per game.")
    tt = sd.team_table(df, model)
    st.dataframe(tt, width="stretch", hide_index=True, column_config={
        "Attack idx": st.column_config.NumberColumn(format="%.3f"),
        "Defence idx": st.column_config.NumberColumn(format="%.3f"),
    })
    st.caption("**Attack idx** > 1 scores more than the league average; "
               "**Defence idx** < 1 concedes fewer. **xG diff/g** is goals minus xG per game: "
               "strongly positive suggests finishing above expectation, which often regresses.")

with backtest_tab:
    st.subheader("Walk-forward backtest")
    st.caption("Re-fits the model before every matchday and predicts that day's fixtures. "
               "No match ever contributes to its own prediction, so these numbers are an "
               "honest estimate of live performance.")
    if st.button("Run backtest", type="primary"):
        with st.spinner("Re-fitting the model for every matchday…"):
            bt, ev = backtest_cached(df, xi, reg, w_xg, w_sot)
        st.session_state["bt"] = (bt, ev)
    got = st.session_state.get("bt")
    if got:
        bt, ev = got
        if not ev:
            st.warning("Not enough data to backtest.")
        else:
            st.markdown(f"**{ev['n']} out-of-sample predictions**")
            rows = [
                {"Model": "This model", "1X2 log-loss": ev["logloss_1x2"],
                 "1X2 RPS": ev["rps_1x2"], "1X2 accuracy": 100 * ev["acc_1x2"]},
                {"Model": "League base rates", "1X2 log-loss": ev["logloss_baserate"],
                 "1X2 RPS": ev["rps_baserate"], "1X2 accuracy": np.nan},
            ]
            if "market_logloss" in ev:
                rows.insert(1, {"Model": "Bookmaker (de-vigged)",
                                "1X2 log-loss": ev["market_logloss"],
                                "1X2 RPS": ev["market_rps"],
                                "1X2 accuracy": 100 * ev["market_acc"]})
            st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True,
                         column_config={
                             "1X2 log-loss": st.column_config.NumberColumn(format="%.4f"),
                             "1X2 RPS": st.column_config.NumberColumn(format="%.4f"),
                             "1X2 accuracy": st.column_config.NumberColumn(format="%.1f%%")})
            c1, c2 = st.columns(2)
            c1.metric("BTTS log-loss", f"{ev['logloss_btts']:.4f}",
                      f"accuracy {ev['acc_btts']:.1%}")
            c2.metric("Over 2.5 log-loss", f"{ev['logloss_o25']:.4f}",
                      f"accuracy {ev['acc_o25']:.1%}")
            st.caption("Lower log-loss and RPS are better. A coin flip scores 0.6931 on the "
                       "binary markets; uninformative 1X2 scores 1.0986.")

            st.markdown("**Calibration** — do things predicted at 30% happen 30% of the time?")
            P = bt[["p_H", "p_D", "p_A"]].values
            o = np.zeros_like(P)
            for k, r in enumerate(bt["result"]):
                o[k, {"H": 0, "D": 1, "A": 2}[r]] = 1
            fp, fo = P.ravel(), o.ravel()
            edges = [0, .1, .2, .3, .4, .5, .6, .75, 1.01]
            cal = []
            for lo, hi in zip(edges[:-1], edges[1:]):
                msk = (fp >= lo) & (fp < hi)
                if msk.sum() >= 10:
                    cal.append({"Predicted band": f"{lo:.0%}–{hi:.0%}", "n": int(msk.sum()),
                                "Mean predicted": 100 * float(fp[msk].mean()),
                                "Actual rate": 100 * float(fo[msk].mean())})
            st.dataframe(pd.DataFrame(cal), width="stretch", hide_index=True,
                         column_config={
                             "Mean predicted": st.column_config.NumberColumn(format="%.1f%%"),
                             "Actual rate": st.column_config.NumberColumn(format="%.1f%%")})
            with st.expander("Every backtested match"):
                st.dataframe(bt, width="stretch", hide_index=True)

with method_tab:
    st.markdown("""
### What the model does

**The data it uses.** From each completed match: final score, **xG for each side**,
**shots on target**, the date, and the two team names. Bookmaker odds are read only
to *benchmark* the model in the backtest — they never feed the fit. Rows that are not
`complete` (unplayed fixtures, abandonments) are excluded.

**Two data traps handled explicitly.** Shot and foul columns use `-1` to mean
"not recorded" — averaging that in would quietly drag a team's rating down, so it is
treated as missing. And three matches in the sample carry `0.00` xG for *both* sides,
one of them a 6–1: that is missing data, not a real goalless-chance game, so the xG
is dropped rather than believed.

### The method

A **Dixon–Coles** team-strength model. Every team gets an attack and a defence
parameter, and a fixture's expected goals are

```
λ_home = exp(μ + attack_home + defence_away + γ)
λ_away = exp(μ + attack_away + defence_home)
```

where `γ` is home advantage, **fitted from your data** rather than assumed. Because
attack is always measured against the specific opponent's defence, a team is not
credited for scoring freely against the league's worst side.

Three refinements, each chosen by backtest rather than taste:

1. **The response is a blend**, not just goals: 50% goals, 25% xG, 25% shots-on-target
   converted to a goals scale at the league's own conversion rate. Goals are what
   happened; xG and shots on target are less noisy measures of how a team actually
   played. In testing the blend beat pure goals on *all three* markets.
2. **Light time decay.** Matches are weighted `exp(-ξ × days ago)`. Tuning preferred a
   *very* small ξ — in a ten-team league, aggressive recency-chasing throws away more
   signal than it gains.
3. **Low-score correction.** Independent Poissons misprice 0–0, 1–0, 0–1 and 1–1, which
   is exactly where football lives. The Dixon–Coles `τ` adjustment is fitted on the real
   scorelines.

Newly promoted teams with few matches are pulled toward the league average by an L2
shrinkage term, so a two-game hot streak does not make a team a title favourite.

### Getting from expected goals to the markets

The two λ values build a **joint score matrix** (0–10 goals each side, τ-corrected).
Every market is then a sum over cells of that one matrix:

- **1X2** — cells where home > away, home = away, home < away
- **BTTS** — all cells with both scores ≥ 1
- **Over/Under 2.5** — cells where the total is ≥ 3 versus ≤ 2

Because they all come from a single distribution, the three markets are **mutually
consistent** by construction — you can never get a 1X2 that contradicts the Over/Under.

### Honesty about accuracy

The **Backtest** tab re-fits the model before every matchday and predicts that day's
games, so no match contributes to its own prediction. On the bundled Latvian data
(429 out-of-sample matches) the model reached **61.5% 1X2 accuracy**, log-loss
**0.867** — comfortably better than league base rates (1.060), and modestly behind the
bookmaker's closing odds (0.844).

That last gap is the honest headline: **the market is sharper than this model on 1X2.**
Blending the two did not beat the market alone in testing. Use the model where the
market is absent or slow, and treat a large model-vs-market disagreement as a prompt
to look closer, not as free money.

A post-hoc probability calibration layer was tried and **rejected** — it improved
in-sample but failed out-of-sample, so it is not shipped.

Over/Under 2.5 (log-loss 0.676) carries more signal than BTTS (0.686), which is only
modestly better than a coin flip at 0.693. Weight your confidence accordingly.

---
*Predictions are probabilistic decision support, not a guaranteed outcome.
Gamble responsibly — Gambling Help 1800 858 858.*
""")
