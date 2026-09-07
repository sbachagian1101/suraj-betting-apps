"""Rendering helpers shared by both pages."""
from __future__ import annotations

import pandas as pd
import streamlit as st

import model as M
import soccerway as SW
from pipeline import Analysis


def form_badges(form: str) -> str:
    colour = {"W": "green", "D": "gray", "L": "red"}
    return " ".join(f":{colour[c]}-badge[**{c}**]" for c in form) if form else "—"


def pct(x: float) -> str:
    return f"{x:.0%}"


def records_table(records: list[SW.MatchRecord], weights: list[float]) -> pd.DataFrame:
    rows = []
    for r, w in zip(records, weights):
        rows.append({
            "Date": r.match.kickoff.strftime("%d %b"),
            "Competition": r.match.competition.split(":")[-1].strip(),
            "Opponent": r.opponent,
            "H/A": r.side,
            "Score": r.score,
            "Res": r.result,
            "xG": r.xg_for,
            "xGA": r.xg_against,
            "Rating": r.team_rating,
            "Opp rating": r.opponent_rating,
            "Weight": w,
        })
    return pd.DataFrame(rows)


def team_panel(prof: M.TeamProfile, la: M.LineupAssessment | None, venue: str):
    st.subheader(f"{prof.name}  ·  {venue}")
    st.markdown(f"Form, last {prof.n} (latest first): {form_badges(prof.form)} &nbsp; **{prof.points} pts**")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Goals for / game", f"{prof.goals_for:.2f}")
    c2.metric("Goals against / game", f"{prof.goals_against:.2f}")
    c3.metric("xG / game", f"{prof.xg_for:.2f}" if prof.xg_for is not None else "n/a")
    c4.metric("xGA / game", f"{prof.xg_against:.2f}" if prof.xg_against is not None else "n/a")
    if prof.records:
        st.dataframe(
            records_table(prof.records, prof.weights), hide_index=True, width="stretch",
            column_config={
                "xG": st.column_config.NumberColumn(format="%.2f"),
                "xGA": st.column_config.NumberColumn(format="%.2f"),
                "Rating": st.column_config.NumberColumn(format="%.1f"),
                "Opp rating": st.column_config.NumberColumn(format="%.1f"),
                "Weight": st.column_config.NumberColumn(format="%.2f"),
            })
    else:
        st.info("No finished matches found for this team.")

    if la:
        tag = ":green-badge[Confirmed XI]" if la.source == "today" else ":orange-badge[Probable XI from last match]"
        st.markdown(f"**Line-up** {tag} &nbsp; {la.formation or ''}")
        st.dataframe(pd.DataFrame(la.rows), hide_index=True, width="stretch",
                     column_config={"Avg rating": st.column_config.NumberColumn(format="%.2f")})
        c1, c2 = st.columns(2)
        c1.metric(f"XI avg rating (last {prof.n_lineup})",
                  f"{la.lineup_rating:.2f}" if la.lineup_rating is not None else "n/a",
                  delta=f"{la.gap:+.2f} vs team avg" if la.lineup_rating is not None else None)
        c2.metric(f"Team avg rating (last {prof.n_lineup})",
                  f"{la.team_recent_rating:.2f}" if la.team_recent_rating is not None else "n/a")
        if la.missing:
            st.caption("Regulars not starting: " + ", ".join(
                f"{h.name} ({h.avg:.1f})" if h.avg else h.name for h in la.missing))
    else:
        st.info("No line-up available.")


def fixture_line(A: Analysis) -> str:
    f = A.fixture
    if not f:
        return "No upcoming fixture listed between these teams."
    when = f"{SW.fmt_mu(f.kickoff, with_date=True)} Mauritius ({f.kickoff.strftime('%H:%M')} UTC)"
    score = (f" · {f.home_score}-{f.away_score}"
             if f.stage in ("2", "3") and f.home_score is not None else "")
    return f"{f.competition} · {when} · **{f.status}**{score} · [match page]({f.url})"


def prediction_row(A: Analysis):
    pred, odds = A.pred, A.odds
    imp = M.implied(odds) if odds else None
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric(f"{A.prof_h.name} win", pct(pred.p_home),
              delta=f"{pred.p_home - imp['home']:+.1%} vs book" if imp else None)
    c2.metric("Draw", pct(pred.p_draw),
              delta=f"{pred.p_draw - imp['draw']:+.1%} vs book" if imp else None)
    c3.metric(f"{A.prof_a.name} win", pct(pred.p_away),
              delta=f"{pred.p_away - imp['away']:+.1%} vs book" if imp else None)
    c4.metric("Expected score", f"{pred.exp_home:.2f} – {pred.exp_away:.2f}")
    c5.metric("Most likely", f"{pred.top_scores[0][0]}", delta=pct(pred.top_scores[0][1]),
              delta_color="off")
    if odds and imp:
        st.caption(f"bet365 1X2: {odds['home']:.2f} / {odds['draw']:.2f} / {odds['away']:.2f} "
                   f"(opened {odds['home_open']:.2f} / {odds['draw_open']:.2f} / {odds['away_open']:.2f}). "
                   f"Implied after removing the {imp['overround']:.1%} margin: "
                   f"{imp['home']:.0%} / {imp['draw']:.0%} / {imp['away']:.0%}.")


def bet_banner(A: Analysis):
    """Big black-on-lime call when the model's edge over the book sits in the
    bet band; a quiet one-liner otherwise."""
    sig = M.bet_signal(A.pred, A.odds)
    if sig is None:
        st.caption("No odds for this match yet, so no bet signal.")
        return
    name = {"home": A.prof_h.name, "draw": "the draw", "away": A.prof_a.name}[sig["side"]]
    detail = (f"{name}: model {sig['model']:.0%} v book {sig['book']:.0%} "
              f"(gap {sig['gap']:+.1%}, odds {sig['odds']:.2f})")
    if sig["bet"]:
        st.markdown(
            f'<div style="background:#c6ff00;color:#000;font-weight:900;font-size:2.4rem;'
            f'line-height:1.15;padding:0.7rem 1.2rem;border-radius:0.6rem;text-align:center;'
            f'letter-spacing:0.04em;margin:0.4rem 0 0.2rem 0">{sig["label"]}</div>'
            f'<div style="text-align:center;color:#000;font-size:1.05rem;font-weight:600;'
            f'margin-bottom:0.6rem">{detail}</div>',
            unsafe_allow_html=True)
    else:
        st.caption(f"**NO BET** · biggest gap is {detail}: " + "; ".join(sig["reasons"]) +
                   f". A bet needs the model above {M.BET_MIN_PROB:.0%} on that side and a gap "
                   f"between {M.BET_MIN_EDGE:.0%} and {M.BET_MAX_EDGE:.0%}.")


def insights_list(A: Analysis):
    for line in M.insights(A.prof_h, A.prof_a, A.pred, A.la_h, A.la_a, A.odds):
        st.markdown(f"- {line}")


def scoreline_section(A: Analysis):
    pred = A.pred
    c1, c2 = st.columns([1, 2])
    with c1:
        st.markdown("**Top scorelines**")
        top_df = pd.DataFrame([(s, round(p * 100, 1)) for s, p in pred.top_scores],
                              columns=["Score", "Prob %"])
        st.dataframe(top_df, hide_index=True, width="stretch",
                     column_config={"Prob %": st.column_config.NumberColumn(format="%.1f%%")})
        st.metric("Over 2.5 goals", pct(pred.p_over25))
        st.metric("Both teams score", pct(pred.p_btts))
    with c2:
        st.markdown(f"**Score grid** (rows {A.prof_h.name}, columns {A.prof_a.name})")
        g = pd.DataFrame(pred.grid).iloc[:6, :6]
        g.index = [str(i) for i in g.index]
        g.columns = [str(c) for c in g.columns]
        top = float(g.values.max()) or 1.0

        def _shade(v):
            a = min(max(v / top, 0.0), 1.0)
            return f"background-color: rgba(46, 160, 67, {0.08 + 0.72 * a:.2f})"

        st.dataframe(g.style.format("{:.1%}").map(_shade), width="stretch")


def methodology(A: Analysis):
    pred = A.pred
    st.markdown(f"""
* **Attack rate** = {M.XG_WEIGHT:.0%} xG-for + {1 - M.XG_WEIGHT:.0%} goals-for per game over the
  last {A.n} matches; **defence rate** likewise from xGA and goals-against.
  Matches are **recency-weighted**: match *k* back counts {M.DECAY}^k, so the newest match
  has weight 1 and the {A.n}th has {M.DECAY ** (A.n - 1):.2f}
  (effective sample {A.prof_h.n_eff:.1f} games).
* Both rates are shrunk toward a league average of {M.LEAGUE_AVG} goals with a prior weight of
  {M.PRIOR_GAMES:g} games, against that effective sample size.
* **Player ratings and the line-up comparison use a shorter window** of the last {A.n_lineup}
  matches, because who is in form changes faster than a team's underlying strength.
* Expected goals: home = attack × opponent defence ÷ average × {M.HOME_ADV};
  away likewise × {M.AWAY_ADV}.
* **Line-up adjustment**: each starter's average Soccerway rating over those matches is compared
  with the team's average rating; the gap scales attack by exp({M.RATING_K} × gap) and the
  opponent's attack by exp(−{M.RATING_K_OPP} × gap). Starters with no recent rating are neutral.
  Here the multipliers were {pred.home_mult:.3f} (home) and {pred.away_mult:.3f} (away).
* Scorelines from an independent Poisson grid up to {M.MAX_GOALS} goals each.
* Only matches that kicked off **before** this fixture count as history, so a live or finished
  fixture never feeds its own prediction.

**Caveat.** The book has beaten every short-form model we have graded, so read the
model-vs-book gap as a prompt to look closer, not as an edge.
""")


def render_full(A: Analysis):
    """Two-team page layout: header, prediction, insights, panels, expanders."""
    st.title(A.title)
    st.markdown(fixture_line(A))
    prediction_row(A)
    bet_banner(A)
    st.subheader("Insights")
    insights_list(A)
    left, right = st.columns(2)
    with left:
        team_panel(A.prof_h, A.la_h, "Home")
    with right:
        team_panel(A.prof_a, A.la_a, "Away")
    with st.expander("Scoreline probabilities"):
        scoreline_section(A)
    with st.expander("How the prediction is built"):
        methodology(A)


def render_compact(A: Analysis):
    """Inside an expander (no nested expanders allowed): tabs instead."""
    st.markdown(fixture_line(A))
    prediction_row(A)
    bet_banner(A)
    t1, t2, t3, t4 = st.tabs(["Insights", "Teams & line-ups", "Scorelines", "Method"])
    with t1:
        insights_list(A)
    with t2:
        left, right = st.columns(2)
        with left:
            team_panel(A.prof_h, A.la_h, "Home")
        with right:
            team_panel(A.prof_a, A.la_a, "Away")
    with t3:
        scoreline_section(A)
    with t4:
        methodology(A)
