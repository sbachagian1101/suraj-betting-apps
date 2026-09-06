"""Page 1: paste two team links, predict their next meeting."""
import streamlit as st

import model as M
import soccerway as SW
import ui
from pipeline import analyse_pair

EXAMPLE_A = "https://us.soccerway.com/team/groningen/MBUGcjb9/"
EXAMPLE_B = "https://us.soccerway.com/team/twente/dhOKTHGA/"

with st.sidebar:
    st.header("Two teams")
    st.caption("Paste two Soccerway or Flashscore team pages. The app reads each side's "
               "recent matches, finds their fixture, and predicts it from today's line-up.")
    url_a = st.text_input("Team A link", value=EXAMPLE_A,
                          help="Soccerway or Flashscore team page; both share the same team ids.")
    url_b = st.text_input("Team B link", value=EXAMPLE_B)
    n_rates = st.slider("Matches for xG / goals rates", 1, 15, M.N_RATES,
                        help=f"Recency-weighted: each older match counts {M.DECAY:.0%} of the one before it.")
    n_lineup = st.slider("Matches for player ratings / line-up", 1, 10, M.N_LINEUP,
                         help="Shorter window: who is in form changes faster than team strength.")
    manual_home = st.radio("If no fixture is listed, home team is", ["Team A", "Team B"],
                           horizontal=True)
    go = st.button("Analyse", type="primary", width="stretch")
    st.caption("Line-ups appear on Soccerway roughly an hour before kick-off. "
               "Before that the last match's XI is used and labelled as such.")

if go:
    status = st.status("Fetching Soccerway data…", expanded=True)
    try:
        A = analyse_pair(url_a, url_b, n_rates, n_lineup,
                         a_is_home_if_no_fixture=(manual_home == "Team A"),
                         progress=status.write)
    except SW.SoccerwayError as exc:
        status.update(label="Failed", state="error")
        st.error(str(exc))
        st.stop()
    status.update(label="Data loaded", state="complete", expanded=False)
    st.session_state["pair_result"] = A

if "pair_result" not in st.session_state:
    st.markdown("### Paste two team links in the sidebar and press **Analyse**.")
    st.markdown(
        "The app pulls, for each team, the recent finished matches with result, xG, xGA, "
        "formation and every starter's rating, then the fixture between them with the "
        "published line-up and bet365 odds. The prediction is a Poisson model on "
        "recency-weighted, xG-blended attack and defence rates, adjusted by how today's XI "
        "has rated recently.")
    st.stop()

A = st.session_state["pair_result"]
if not A.fixture:
    st.warning("Soccerway lists no upcoming fixture between these two teams. "
               "Home side taken from the sidebar; no line-up or odds available, "
               "so the last match's XI is used.")
ui.render_full(A)
