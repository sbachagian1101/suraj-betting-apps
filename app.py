"""Soccerway two-team predictor.

Paste two Soccerway team links, the app pulls each side's last N matches
(form, xG, xGA, player ratings), finds today's fixture between them, reads the
published line-up, and turns it into a Poisson prediction with insights.
"""
from __future__ import annotations

import importlib

import pandas as pd
import streamlit as st

st.set_page_config(page_title="Soccerway Predictor", page_icon="⚽", layout="wide")

# Streamlit Cloud sometimes redeploys a new app.py against a cached old helper
# module. Check for every name we need BEFORE importing anything from them, so
# the failure is one actionable sentence instead of a redacted traceback.
_NEEDED = {
    "soccerway": ["parse_team_url", "team_results", "find_fixture", "match_stats",
                  "match_lineups", "match_odds", "build_record", "clean_name",
                  "SoccerwayError", "MatchRecord"],
    "model": ["profile", "assess_lineup", "predict", "implied", "insights",
              "LineupAssessment", "TeamProfile", "XG_WEIGHT", "LEAGUE_AVG",
              "PRIOR_GAMES", "HOME_ADV", "AWAY_ADV", "RATING_K", "RATING_K_OPP", "MAX_GOALS"],
}
_missing = []
for _mod, _names in _NEEDED.items():
    try:
        _m = importlib.import_module(_mod)
    except Exception as _exc:                      # noqa: BLE001
        _missing.append(f"{_mod} (import failed: {_exc})")
        continue
    _missing += [f"{_mod}.{n}" for n in _names if not hasattr(_m, n)]
if _missing:
    st.error("**This deployment is running stale code.** Missing: `" + "`, `".join(_missing)
             + "`. Streamlit Cloud pulled the new files but kept an old module in memory. "
             "Fix it with **Manage app → ⋮ → Reboot app**.", icon=":material/error:")
    st.stop()

import model as M          # noqa: E402
import soccerway as SW     # noqa: E402

EXAMPLE_A = "https://us.soccerway.com/team/groningen/MBUGcjb9/"
EXAMPLE_B = "https://us.soccerway.com/team/twente/dhOKTHGA/"


# --------------------------------------------------------------------------- #
# Cached fetchers
# --------------------------------------------------------------------------- #
@st.cache_data(ttl=600, show_spinner=False)
def _results(slug: str, team_id: str):
    return SW.team_results(slug, team_id)


@st.cache_data(ttl=300, show_spinner=False)
def _fixture(slug: str, team_id: str, other_id: str):
    return SW.find_fixture(slug, team_id, other_id)


@st.cache_data(ttl=6 * 3600, show_spinner=False)
def _finished_match(event_id: str):
    """Stats + line-ups for a finished match (immutable, cache long)."""
    try:
        stats = SW.match_stats(event_id)
    except SW.SoccerwayError:
        stats = {}
    try:
        lineups = SW.match_lineups(event_id)
    except SW.SoccerwayError:
        lineups = {}
    return stats, lineups


@st.cache_data(ttl=90, show_spinner=False)
def _live_lineups(event_id: str):
    try:
        return SW.match_lineups(event_id)
    except SW.SoccerwayError:
        return {}


@st.cache_data(ttl=90, show_spinner=False)
def _odds(event_id: str):
    return SW.match_odds(event_id)


# --------------------------------------------------------------------------- #
# Pipeline
# --------------------------------------------------------------------------- #
def load_team(url: str, n: int, progress) -> tuple[str, str, str, list[SW.MatchRecord]]:
    slug, team_id = SW.parse_team_url(url)
    results = _results(slug, team_id)[:n]
    name = ""
    records = []
    for i, m in enumerate(results):
        progress(f"{slug}: match {i + 1} of {len(results)}")
        stats, lineups = _finished_match(m.id)
        rec = SW.build_record(m, team_id, stats, lineups)
        records.append(rec)
        name = m.home_name if rec.side == "H" else m.away_name
    if not name:
        name = slug.replace("-", " ").title()
    return slug, team_id, name, records


# --------------------------------------------------------------------------- #
# UI helpers
# --------------------------------------------------------------------------- #
def form_badges(form: str) -> str:
    colour = {"W": "green", "D": "gray", "L": "red"}
    return " ".join(f":{colour[c]}-badge[**{c}**]" for c in form) if form else "—"


def records_table(records: list[SW.MatchRecord]) -> pd.DataFrame:
    rows = []
    for r in records:
        rows.append({
            "Date": r.match.kickoff.strftime("%d %b"),
            "Opponent": r.opponent,
            "H/A": r.side,
            "Score": r.score,
            "Res": r.result,
            "xG": r.xg_for,
            "xGA": r.xg_against,
            "Rating": r.team_rating,
            "Opp rating": r.opponent_rating,
            "Formation": r.formation,
        })
    return pd.DataFrame(rows)


def pct(x: float) -> str:
    return f"{x:.0%}"


def team_panel(prof: M.TeamProfile, la: M.LineupAssessment | None, venue: str):
    st.subheader(f"{prof.name}  ·  {venue}")
    st.markdown(f"Form (latest first): {form_badges(prof.form)} &nbsp; **{prof.points} pts**")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Goals for / game", f"{prof.goals_for:.2f}")
    c2.metric("Goals against / game", f"{prof.goals_against:.2f}")
    c3.metric("xG / game", f"{prof.xg_for:.2f}" if prof.xg_for is not None else "n/a")
    c4.metric("xGA / game", f"{prof.xg_against:.2f}" if prof.xg_against is not None else "n/a")
    if prof.records:
        st.dataframe(
            records_table(prof.records), hide_index=True, width="stretch",
            column_config={
                "xG": st.column_config.NumberColumn(format="%.2f"),
                "xGA": st.column_config.NumberColumn(format="%.2f"),
                "Rating": st.column_config.NumberColumn(format="%.1f"),
                "Opp rating": st.column_config.NumberColumn(format="%.1f"),
            })
    else:
        st.info("No finished matches found for this team.")

    if la:
        tag = ":green-badge[Confirmed XI]" if la.source == "today" else ":orange-badge[Probable XI from last match]"
        st.markdown(f"**Line-up** {tag} &nbsp; {la.formation or ''}")
        st.dataframe(pd.DataFrame(la.rows), hide_index=True, width="stretch",
                     column_config={"Avg rating": st.column_config.NumberColumn(format="%.2f")})
        c1, c2 = st.columns(2)
        c1.metric("XI avg rating (last N)",
                  f"{la.lineup_rating:.2f}" if la.lineup_rating is not None else "n/a",
                  delta=f"{la.gap:+.2f} vs team avg" if la.lineup_rating is not None else None)
        c2.metric("Team avg rating (last N)",
                  f"{la.team_recent_rating:.2f}" if la.team_recent_rating is not None else "n/a")
        if la.missing:
            st.caption("Regulars not starting: " + ", ".join(
                f"{h.name} ({h.avg:.1f})" if h.avg else h.name for h in la.missing))
    else:
        st.info("No line-up available.")


# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.title("⚽ Soccerway Predictor")
    st.caption("Paste the two Soccerway team pages. The app reads each side's last N "
               "matches, finds their fixture, and predicts it from today's line-up.")
    url_a = st.text_input("Team A link", value=EXAMPLE_A)
    url_b = st.text_input("Team B link", value=EXAMPLE_B)
    n = st.slider("Matches to use", 1, 10, 3)
    manual_home = st.radio("If no fixture is listed, home team is", ["Team A", "Team B"],
                           horizontal=True)
    go = st.button("Analyse", type="primary", width="stretch")
    st.caption("Line-ups appear on Soccerway roughly an hour before kick-off. "
               "Before that the last match's XI is used and labelled as such.")

if not go and "result" not in st.session_state:
    st.markdown("### Paste two team links in the sidebar and press **Analyse**.")
    st.markdown(
        "The app pulls, for each team, the last N finished matches with result, xG, xGA, "
        "formation and every starter's rating, then the fixture between them with the "
        "published line-up and bet365 odds. The prediction is a Poisson model on xG-blended "
        "attack and defence rates, adjusted by how today's XI has rated recently.")
    st.stop()

# --------------------------------------------------------------------------- #
# Run
# --------------------------------------------------------------------------- #
if go:
    status = st.status("Fetching Soccerway data…", expanded=True)
    try:
        slug_a, id_a, name_a, recs_a = load_team(url_a, n, status.write)
        slug_b, id_b, name_b, recs_b = load_team(url_b, n, status.write)
        if id_a == id_b:
            raise SW.SoccerwayError("Both links point to the same team.")
        status.write("Looking for the fixture between them…")
        fixture = _fixture(slug_a, id_a, id_b)
        live = _live_lineups(fixture.id) if fixture else {}
        odds = _odds(fixture.id) if fixture else None
    except SW.SoccerwayError as exc:
        status.update(label="Failed", state="error")
        st.error(str(exc))
        st.stop()
    status.update(label="Data loaded", state="complete", expanded=False)

    if fixture:
        a_is_home = fixture.home_id == id_a
    else:
        a_is_home = manual_home == "Team A"
    if a_is_home:
        home = (id_a, name_a, recs_a)
        away = (id_b, name_b, recs_b)
    else:
        home = (id_b, name_b, recs_b)
        away = (id_a, name_a, recs_a)

    prof_h = M.profile(home[1], home[2])
    prof_a = M.profile(away[1], away[2])

    def _assess(records, side, prof):
        tl = live.get(side) if live else None
        if tl and tl.published:
            return M.assess_lineup(tl.starters, "today", tl.formation, prof)
        if records and records[0].starters:
            return M.assess_lineup(records[0].starters, "last match", records[0].formation, prof)
        return None

    la_h = _assess(home[2], "HOME", prof_h)
    la_a = _assess(away[2], "AWAY", prof_a)
    pred = M.predict(prof_h, prof_a, la_h, la_a)
    st.session_state["result"] = dict(fixture=fixture, prof_h=prof_h, prof_a=prof_a,
                                      la_h=la_h, la_a=la_a, pred=pred, odds=odds, n=n)

R = st.session_state["result"]
fixture, prof_h, prof_a, la_h, la_a, pred, odds = (
    R["fixture"], R["prof_h"], R["prof_a"], R["la_h"], R["la_a"], R["pred"], R["odds"])

# --------------------------------------------------------------------------- #
# Header
# --------------------------------------------------------------------------- #
st.title(f"{prof_h.name} v {prof_a.name}")
if fixture:
    when = fixture.kickoff.strftime("%a %d %b %Y, %H:%M UTC")
    state = "LIVE" if fixture.live else ("Finished" if fixture.finished else "Scheduled")
    score = (f" · currently {fixture.home_score}-{fixture.away_score}"
             if fixture.live and fixture.home_score is not None else "")
    st.markdown(f"{fixture.competition} · {when} · **{state}**{score} · "
                f"[match page]({fixture.url})")
else:
    st.warning("Soccerway lists no upcoming fixture between these two teams. "
               "Home side taken from the sidebar; no line-up or odds available, "
               "so the last match's XI is used.")

# --------------------------------------------------------------------------- #
# Prediction
# --------------------------------------------------------------------------- #
imp = M.implied(odds) if odds else None
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric(f"{prof_h.name} win", pct(pred.p_home),
          delta=f"{pred.p_home - imp['home']:+.1%} vs book" if imp else None)
c2.metric("Draw", pct(pred.p_draw),
          delta=f"{pred.p_draw - imp['draw']:+.1%} vs book" if imp else None)
c3.metric(f"{prof_a.name} win", pct(pred.p_away),
          delta=f"{pred.p_away - imp['away']:+.1%} vs book" if imp else None)
c4.metric("Expected score", f"{pred.exp_home:.2f} – {pred.exp_away:.2f}")
c5.metric("Most likely", f"{pred.top_scores[0][0]}", delta=pct(pred.top_scores[0][1]),
          delta_color="off")

if odds:
    st.caption(f"bet365 1X2: {odds['home']:.2f} / {odds['draw']:.2f} / {odds['away']:.2f} "
               f"(opened {odds['home_open']:.2f} / {odds['draw_open']:.2f} / {odds['away_open']:.2f}). "
               f"Implied after removing the {imp['overround']:.1%} margin: "
               f"{imp['home']:.0%} / {imp['draw']:.0%} / {imp['away']:.0%}.")

# --------------------------------------------------------------------------- #
# Insights
# --------------------------------------------------------------------------- #
st.subheader("Insights")
for line in M.insights(prof_h, prof_a, pred, la_h, la_a, odds):
    st.markdown(f"- {line}")

# --------------------------------------------------------------------------- #
# Team panels
# --------------------------------------------------------------------------- #
left, right = st.columns(2)
with left:
    team_panel(prof_h, la_h, "Home")
with right:
    team_panel(prof_a, la_a, "Away")

# --------------------------------------------------------------------------- #
# Detail
# --------------------------------------------------------------------------- #
with st.expander("Scoreline probabilities"):
    c1, c2 = st.columns([1, 2])
    with c1:
        st.markdown("**Top scorelines**")
        st.dataframe(pd.DataFrame(pred.top_scores, columns=["Score", "Prob"]),
                     hide_index=True, width="stretch",
                     column_config={"Prob": st.column_config.NumberColumn(format="%.1%")})
        st.metric("Over 2.5 goals", pct(pred.p_over25))
        st.metric("Both teams score", pct(pred.p_btts))
    with c2:
        st.markdown(f"**Score grid** (rows {prof_h.name}, columns {prof_a.name})")
        g = pd.DataFrame(pred.grid).iloc[:6, :6]
        g.index = [str(i) for i in g.index]
        g.columns = [str(c) for c in g.columns]
        top = float(g.values.max()) or 1.0

        def _shade(v):
            # green tint scaled to the most likely score; no matplotlib needed
            a = min(max(v / top, 0.0), 1.0)
            return f"background-color: rgba(46, 160, 67, {0.08 + 0.72 * a:.2f})"

        st.dataframe(g.style.format("{:.1%}").map(_shade), width="stretch")

with st.expander("How the prediction is built"):
    st.markdown(f"""
* **Attack rate** = {M.XG_WEIGHT:.0%} xG-for + {1 - M.XG_WEIGHT:.0%} goals-for per game over the
  last {R['n']} matches; **defence rate** likewise from xGA and goals-against.
* Both rates are shrunk toward a league average of {M.LEAGUE_AVG} goals with a prior weight of
  {M.PRIOR_GAMES:g} games, because {R['n']} matches is a small sample.
* Expected goals: home = attack × opponent defence ÷ average × {M.HOME_ADV};
  away likewise × {M.AWAY_ADV}.
* **Line-up adjustment**: each starter's average Soccerway rating over those matches is compared
  with the team's average rating; the gap scales attack by exp({M.RATING_K} × gap) and the
  opponent's attack by exp(−{M.RATING_K_OPP} × gap). Starters with no recent rating are neutral.
  Today the multipliers were {pred.home_mult:.3f} (home) and {pred.away_mult:.3f} (away).
* Scorelines from an independent Poisson grid up to {M.MAX_GOALS} goals each.
* Data: Soccerway team pages (results, fixtures), the match stats feed (xG), the line-up feed
  (formation, ratings) and bet365 pre-match odds. Nothing is stored between runs.

**Caveat.** With only a few matches the numbers swing hard on one freak game. The book has
beaten every short-form model we have graded, so read the model-vs-book gap as a prompt to
look closer, not as an edge.
""")
