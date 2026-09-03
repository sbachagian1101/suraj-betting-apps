"""FootyStats xG -- Streamlit front end, one model per league.

Predict   price one fixture against the bookmaker's odds
Fixtures  price every unplayed fixture in the file at once
Ratings   current attack / defence strengths
Backtest  what the model scored out of sample, next to the market
Help      what the numbers mean and what they are worth

Every probability on screen comes from `model.py`; nothing is hand-tuned.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

import model as M

st.set_page_config(page_title="FootyStats xG", page_icon=":material/sports_soccer:", layout="wide")

# Streamlit can keep serving a cached copy of `model` after an edit adds a new
# name to it. Fail in one sentence at the top rather than halfway down a page.
_REQUIRED = ["discover_leagues", "parse_filename", "league_label", "load_matches", "fit",
             "predict", "walk_forward", "summarise", "calibration", "flat_stake_roi"]
_missing = [n for n in _REQUIRED if not hasattr(M, n)]
if _missing:
    st.error(f"This server is running a stale copy of model.py (missing {', '.join(_missing)}). "
             "Stop and restart `streamlit run app.py`.")
    st.stop()

UPLOAD_DIR = Path(__file__).parent / "data" / "uploads"


# ---------------------------------------------------------------------------
# Data + model (cached)
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_league(files: tuple[str, ...], mtimes: tuple[float, ...]) -> pd.DataFrame:
    # mtimes are only there to bust the cache when an upload replaces a file
    return M.load_matches([Path(p) for p in files])


@st.cache_resource(show_spinner=False)
def fitted(league: str, n_rows: int, last_ts: int, halflife: float, use_xg: bool) -> M.Fit:
    # cache key is league + data fingerprint + settings; the frame comes from load_league
    return M.fit(st.session_state["matches"], params=M.Params(halflife_days=halflife, use_xg=use_xg))


@st.cache_data(show_spinner=False)
def backtest(league: str, n_rows: int, last_ts: int, halflife: float, use_xg: bool) -> pd.DataFrame:
    return M.walk_forward(st.session_state["matches"],
                          params=M.Params(halflife_days=halflife, use_xg=use_xg))


def pct(x: float) -> str:
    return f"{100 * x:.1f}%"


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.title("FootyStats xG")
    st.caption("Time-weighted Poisson on match xG, Dixon-Coles corrected. "
               "One model per league, fitted from FootyStats match exports.")

    st.subheader("Data")
    up = st.file_uploader("Add FootyStats CSVs", type="csv", accept_multiple_files=True,
                          help="FootyStats > League > Download. Keep the original file names: the league "
                               "and season are read from them. Only *matches* files feed the model; "
                               "teams/players files are accepted and ignored.")
    if up:
        UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        ignored = []
        for f in up:
            info = M.parse_filename(f.name)
            if info is None:
                st.warning(f"Skipped {f.name}: not a FootyStats export name.")
                continue
            (UPLOAD_DIR / f.name).write_bytes(f.getvalue())
            if info["kind"] != "matches":
                ignored.append(f.name)
        if ignored:
            st.caption(f"Saved but not used: {len(ignored)} non-matches file"
                       f"{'s' if len(ignored) > 1 else ''}.")

    leagues = M.discover_leagues([M.DATA_DIR, UPLOAD_DIR])
    if not leagues:
        st.error("No matches CSVs found. Upload a FootyStats matches export.")
        st.stop()
    keys = sorted(leagues)
    remembered = st.session_state.get("league")
    league = st.selectbox("League", keys, format_func=M.league_label,
                          index=keys.index(remembered) if remembered in keys else 0)
    st.session_state["league"] = league

    st.subheader("Model")
    use_xg = st.toggle("Fit to xG (off = goals)", value=True,
                       help="xG scored slightly better in the Bulgarian backtest; the gap is small. "
                            "Matches without recorded xG use goals either way.")
    halflife = st.slider("Half-life (days)", 20, 365, 60, 10,
                         help="A match this old counts half as much as one played today.")

files = leagues[league]
try:
    matches = load_league(tuple(str(p) for p in files), tuple(p.stat().st_mtime for p in files))
except Exception as exc:  # bad upload
    st.error(f"Could not read the {M.league_label(league)} data: {exc}")
    st.stop()
st.session_state["matches"] = matches
done = M.completed(matches)
upcoming = M.fixtures(matches)
fp = (league, len(matches), int(matches.timestamp.max()), float(halflife), bool(use_xg))

with st.sidebar:
    xg_cov = done.has_xg.mean() if len(done) else 0.0
    n_seasons = done.season.nunique()
    st.caption(f"{len(done)} completed matches over {n_seasons} season{'s' if n_seasons != 1 else ''} "
               f"({', '.join(sorted(done.season.unique()))}); last result {done.kickoff.max():%d %b %Y}. "
               f"{len(upcoming)} unplayed fixtures. xG recorded for {xg_cov:.0%} of matches.")
    if xg_cov < 0.75:
        st.warning("xG is missing for many matches in this league; those are fitted on goals instead.")

fit = fitted(*fp)
teams = fit.teams

st.header(M.league_label(league))
tab_predict, tab_fixtures, tab_ratings, tab_backtest, tab_help = st.tabs(
    ["Predict", "Fixtures", "Ratings", "Backtest", "Help"])


# ---------------------------------------------------------------------------
# Predict
# ---------------------------------------------------------------------------
with tab_predict:
    with_odds = upcoming[upcoming.odds_h > 1].copy()
    with_odds["label"] = [f"{k:%a %d %b %H:%M} GMT  ·  {h} v {a}"
                          for k, h, a in zip(with_odds.kickoff, with_odds.home, with_odds.away)]
    options = ["Pick teams manually"] + with_odds.label.tolist()
    choice = st.selectbox("Fixture", options, index=min(1, len(options) - 1), key=f"fx_{league}")

    if choice == "Pick teams manually":
        c1, c2 = st.columns(2)
        home = c1.selectbox("Home", teams, index=0, key=f"home_{league}")
        away = c2.selectbox("Away", teams, index=min(1, len(teams) - 1), key=f"away_{league}")
        oh, od, oa = 0.0, 0.0, 0.0
    else:
        row = with_odds[with_odds.label == choice].iloc[0]
        home, away = row.home, row.away
        oh, od, oa = float(row.odds_h), float(row.odds_d), float(row.odds_a)

    st.markdown("**Bookmaker odds** (optional, decimal)")
    o1, o2, o3 = st.columns(3)
    oh = o1.number_input(f"{home} win", 1.0, 100.0, oh if oh > 1 else 1.0, 0.01, key=f"oh{league}{home}{away}")
    od = o2.number_input("Draw", 1.0, 100.0, od if od > 1 else 1.0, 0.01, key=f"od{league}{home}{away}")
    oa = o3.number_input(f"{away} win", 1.0, 100.0, oa if oa > 1 else 1.0, 0.01, key=f"oa{league}{home}{away}")
    mk = M.market_probs(oh, od, oa) if min(oh, od, oa) > 1 else None

    if home == away:
        st.warning("Pick two different teams.")
        st.stop()

    pr = M.predict(fit, home, away)
    p = pr["probs"]
    fair = M.fair_odds(p)

    st.subheader(f"{home} v {away}")
    st.caption(f"Expected goals {pr['lam_home']:.2f} – {pr['lam_away']:.2f}. "
               f"Fitted on {fit.n_matches} matches; {home} has {fit.matches_by_team.get(home, 0)} in the sample, "
               f"{away} has {fit.matches_by_team.get(away, 0)}.")

    with st.container(horizontal=True):
        for j, name in enumerate([home, "Draw", away]):
            delta = None
            if mk is not None:
                delta = f"{100 * (p[j] - mk[j]):+.1f} pts vs market"
            st.metric(f"{name}", pct(p[j]), delta, border=True,
                      help=f"Model fair odds {fair[j]:.2f}"
                           + (f" · bookmaker {[oh, od, oa][j]:.2f} (fair {1 / mk[j]:.2f})" if mk is not None else ""))

    if mk is not None:
        edges = p - mk
        best = int(np.argmax(edges))
        ovr = M.overround(oh, od, oa)
        st.caption(f"Bookmaker overround {100 * (ovr - 1):.1f}%. Deltas are model minus the bookmaker's "
                   f"*fair* (overround-removed) probability.")
        ev = p * np.array([oh, od, oa]) - 1
        if ev.max() > 0:
            k = M.kelly(p[best], [oh, od, oa][best])
            st.warning(f"The model prices **{[home, 'Draw', away][best]}** above the book "
                       f"(nominal EV {100 * ev[best]:+.1f}% per unit; full Kelly would say {100 * k:.1f}% of bank). "
                       "Treat this as a prompt to find out what the market knows, not as a bet: in the "
                       "Bulgarian walk-forward backtest, backing every such gap lost about 16% of turnover. "
                       "Run the Backtest tab for this league.")
        else:
            st.success("No positive expected value at these prices. The model and the bookmaker agree "
                       "within the margin, so there is no bet here.")

    c1, c2, c3 = st.columns(3)
    with c1, st.container(border=True):
        st.markdown("**Goal lines**")
        st.dataframe(pd.DataFrame({
            "line": ["Over 1.5", "Over 2.5", "Over 3.5", "Both teams score",
                     f"{home} clean sheet", f"{away} clean sheet"],
            "probability": [pct(pr["over15"]), pct(pr["over25"]), pct(pr["over35"]), pct(pr["btts"]),
                            pct(pr["home_clean_sheet"]), pct(pr["away_clean_sheet"])],
        }), hide_index=True)
    with c2, st.container(border=True):
        st.markdown("**Most likely scorelines**")
        st.dataframe(pd.DataFrame({"score": [f"{i}-{j}" for (i, j), _ in pr["top_scores"]],
                                   "probability": [pct(v) for _, v in pr["top_scores"]]}), hide_index=True)
    with c3, st.container(border=True):
        st.markdown("**Head to head**")
        h2h = M.head_to_head(matches, home, away)
        if len(h2h):
            st.dataframe(h2h, hide_index=True)
        else:
            st.caption("No previous meetings in the data.")

    c1, c2 = st.columns(2)
    for col, team in ((c1, home), (c2, away)):
        with col, st.container(border=True):
            st.markdown(f"**{team} — last six**")
            st.dataframe(M.recent_form(matches, team, 6), hide_index=True)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
with tab_fixtures:
    st.subheader("Unplayed fixtures priced by the model")
    if len(upcoming) == 0:
        st.caption("No unplayed fixtures in this league's files.")
    else:
        n_show = st.slider("Fixtures to show", 5, 60, 20, 5, key=f"nshow_{league}")
        rows = []
        for _, r in upcoming.head(n_show).iterrows():
            pr = M.predict(fit, r.home, r.away)
            mk = M.market_probs(r.odds_h, r.odds_d, r.odds_a)
            rows.append({
                "kickoff (GMT)": r.kickoff.strftime("%a %d %b %H:%M"), "home": r.home, "away": r.away,
                "xG home": pr["lam_home"], "xG away": pr["lam_away"],
                "P(home)": pr["probs"][0], "P(draw)": pr["probs"][1], "P(away)": pr["probs"][2],
                "P(over 2.5)": pr["over25"],
                "book home": r.odds_h if r.odds_h > 1 else np.nan,
                "book draw": r.odds_d if r.odds_d > 1 else np.nan,
                "book away": r.odds_a if r.odds_a > 1 else np.nan,
                "best edge": (pr["probs"] - mk).max() if mk is not None else np.nan,
            })
        fx = pd.DataFrame(rows)
        st.dataframe(fx, hide_index=True, column_config={
            "xG home": st.column_config.NumberColumn(format="%.2f"),
            "xG away": st.column_config.NumberColumn(format="%.2f"),
            "P(home)": st.column_config.ProgressColumn(format="percent", min_value=0, max_value=1),
            "P(draw)": st.column_config.ProgressColumn(format="percent", min_value=0, max_value=1),
            "P(away)": st.column_config.ProgressColumn(format="percent", min_value=0, max_value=1),
            "P(over 2.5)": st.column_config.NumberColumn(format="percent"),
            "best edge": st.column_config.NumberColumn(format="percent",
                                                       help="Largest (model − fair bookmaker) probability gap"),
        })
        st.caption("FootyStats only carries odds for the next few days of fixtures; blanks mean no price yet.")


# ---------------------------------------------------------------------------
# Ratings
# ---------------------------------------------------------------------------
with tab_ratings:
    st.subheader("Team strengths")
    st.caption(f"As of {fit.as_of:%d %b %Y}. Attack = goals expected per match against an average defence "
               f"at a neutral venue; defence = goals expected to concede against an average attack. "
               f"Home advantage adds {100 * (np.exp(fit.home_adv) - 1):.0f}% to the home side's expected goals.")
    rt = M.ratings_table(fit)
    st.dataframe(rt, hide_index=True, column_config={
        "attack": st.column_config.NumberColumn(format="%.2f"),
        "defence": st.column_config.NumberColumn(format="%.2f"),
        "rating": st.column_config.NumberColumn(format="%+.2f", help="attack minus defence"),
        "matches": st.column_config.NumberColumn(help="matches in the fitted sample (all weights)"),
    })
    st.caption("Teams new to the division this season have few matches and sit close to league average "
               "by construction — that is the ridge prior, not a judgement. Teams from earlier seasons "
               "that have since left the division still appear, with stale ratings.")


# ---------------------------------------------------------------------------
# Backtest
# ---------------------------------------------------------------------------
with tab_backtest:
    st.subheader("Out-of-sample record")
    st.markdown("Every completed match from the 91st onward is priced using only games that finished on "
                "earlier days, with the settings in the sidebar. Lower log loss and Brier are better; "
                "the bookmaker's fair price is the bar to clear.")
    if len(done) < 120:
        st.warning("Fewer than 120 completed matches: the backtest will be too short to mean much.")
    if st.button("Run backtest", type="primary", icon=":material/play_arrow:"):
        st.session_state["bt_key"] = fp
    if st.session_state.get("bt_key") == fp:
        with st.spinner("Refitting the model match-day by match-day…"):
            bt = backtest(*fp)
        if bt.k_h.notna().sum() == 0:
            st.error("No bookmaker odds in the backtest window, so there is nothing to compare against.")
        else:
            summ = M.summarise(bt)
            st.dataframe(summ, column_config={
                "log loss": st.column_config.NumberColumn(format="%.4f"),
                "Brier": st.column_config.NumberColumn(format="%.4f"),
                "accuracy": st.column_config.NumberColumn(format="percent"),
            })
            gap = summ.loc["Model", "log loss"] - summ.loc["Bookmaker (fair)", "log loss"]
            if gap > 0:
                st.warning(f"The bookmaker beats the model by {gap:.3f} log loss. Treat the model as a "
                           "second opinion and a sanity check on prices, not as a source of edge.")
            else:
                st.success(f"The model beats the bookmaker by {-gap:.3f} log loss on this sample. "
                           "Check the calibration and P&L below before trusting it.")

            c1, c2 = st.columns(2)
            with c1, st.container(border=True):
                st.markdown("**Calibration — model**")
                st.dataframe(M.calibration(bt, "m"), hide_index=True, column_config={
                    "predicted": st.column_config.NumberColumn(format="percent"),
                    "actual": st.column_config.NumberColumn(format="percent")})
            with c2, st.container(border=True):
                st.markdown("**Calibration — bookmaker**")
                st.dataframe(M.calibration(bt, "k"), hide_index=True, column_config={
                    "predicted": st.column_config.NumberColumn(format="percent"),
                    "actual": st.column_config.NumberColumn(format="percent")})

            with st.container(border=True):
                st.markdown("**Flat-stake P&L if you had backed every model edge**")
                min_edge = st.slider("Minimum edge (model − fair book)", 0.0, 0.15, 0.03, 0.01,
                                     format="%.2f", key=f"edge_{league}")
                st.dataframe(M.flat_stake_roi(bt, min_edge), hide_index=True, column_config={
                    "strike": st.column_config.NumberColumn(format="percent"),
                    "ROI": st.column_config.NumberColumn(format="percent"),
                    "P&L (units)": st.column_config.NumberColumn(format="%+.1f")})
                st.caption("Recorded odds are FootyStats' pre-match prices, not closing prices, and no "
                           "commission is deducted. Small samples: a run of 40 bets says little.")

            with st.expander("Every priced match"):
                show = bt.copy()
                show["kickoff"] = show.kickoff.dt.strftime("%d %b %Y")
                show["result"] = show.result.map(dict(enumerate(M.OUTCOMES)))
                st.dataframe(show, hide_index=True)
    else:
        st.caption("Press the button. Takes about ten seconds.")


# ---------------------------------------------------------------------------
# Help
# ---------------------------------------------------------------------------
with tab_help:
    st.markdown("""
### What this is
A goals model for any league you have FootyStats match exports for. Each league is fitted on
its own; pick it in the sidebar. Each team gets an attack and a defence strength; the league
gets a scoring rate and a home advantage. A match is priced as two Poisson goal counts (with
the Dixon-Coles tweak for 0-0, 1-0, 0-1 and 1-1), which gives 1X2, goal lines, and scoreline
probabilities.

Older matches are down-weighted with a half-life, so the ratings follow the current side, and
the strengths are shrunk toward league average so a promoted club with three games does not
get an extreme rating.

### What it is worth
Honest answer: **it is a credible second opinion, not an edge.** In the Bulgarian
walk-forward backtest the bookmaker's own prices, with the margin removed, score better than
the model on log loss, and mixing the two does not improve on the bookmaker alone. That is
normal for leagues like these — the market already contains team news, motivation and lineups
that no results file can see. Run the Backtest tab for each league you load.

What the model *is* good for:
- Spotting a price that is a long way from a sane goals-based estimate, so you can look for
  the reason (injuries, a dead rubber, a B-team resting players) before betting.
- Goal-line and scoreline probabilities, which FootyStats does not price consistently.
- Fixtures the book has not priced yet.

### Adding leagues and updating data
Download the *matches* CSV for a league and season from FootyStats and drop it in the
sidebar uploader, keeping FootyStats' file name (`<league>-matches-<y1>-to-<y2>-stats.csv`).
The league and season are read from the name. Upload the newer file for the same season and
it replaces the older rows. Teams and players files are accepted so you can drop a whole
download in at once, but they are not used: team files are season totals already implied by
the matches, and the players files in the Bulgarian zip were empty.

### What the data does not contain
B-teams (CSKA Sofia II, Astana II, Tobol II, …) change personnel week to week depending on
the senior squad; that is exactly the kind of thing the market knows and this model cannot.
Where xG is missing for a match — about half the 2026 Kazakh games — the model uses goals for
that match; the sidebar shows the coverage.

### Columns deliberately not used
`home_ppg` and `away_ppg` in the FootyStats export are season-to-date points per game
**including** the match on that row, so they leak the result. The `Pre-Match PPG` and
pre-match xG columns are fine but add nothing once the model has the match history.
""")
