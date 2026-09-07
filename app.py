"""Soccerway Predictor: entry point.

Two pages:
* Two teams  - paste two team links, predict their next meeting.
* League day - paste league names, predict every match of the day in them.
"""
import importlib

import streamlit as st

st.set_page_config(page_title="Soccerway Predictor", page_icon="⚽", layout="wide")

# Streamlit Cloud sometimes redeploys a new entry file against a cached old
# helper module. Check for every name we need BEFORE importing anything from
# them, so the failure is one actionable sentence instead of a redacted traceback.
_NEEDED = {
    "soccerway": ["parse_team_url", "team_results", "team_fixtures", "match_stats",
                  "match_lineups", "match_odds", "build_record", "clean_name",
                  "daily_matches", "select_leagues", "history_before", "to_mu", "fmt_mu",
                  "SoccerwayError", "MatchRecord"],
    "model": ["profile", "assess_lineup", "predict", "implied", "insights",
              "recency_weights", "LineupAssessment", "TeamProfile",
              "XG_WEIGHT", "LEAGUE_AVG", "PRIOR_GAMES", "HOME_ADV", "AWAY_ADV",
              "RATING_K", "RATING_K_OPP", "MAX_GOALS", "DECAY", "N_RATES", "N_LINEUP",
              "bet_signal", "BET_MIN_EDGE", "BET_MAX_EDGE", "BET_MIN_PROB"],
    "pipeline": ["analyse_pair", "analyse_fixture", "daily_cached", "latest_match",
                 "on_day", "DayIndex", "Analysis"],
    "ui": ["render_full", "render_compact", "bet_banner", "prediction_row", "insights_list",
           "team_panel", "methodology"],
    "board": ["BOARD", "refresh_minutes", "Board"],
    "alerts": ["imminent", "started", "new_alerts", "beep_wav", "odds_colour", "PURPLE", "ROSE"],
    "board_ui": ["render", "board_frame", "style_board", "starts_in"],
    "charts": ["model_vs_book", "outcome_pie", "score_heatmap", "xg_history", "rating_bars"],
}
EXPECTED_MODEL_BUILD = "2026-09-07.12-5"   # must equal model.BUILD; bump both together
_missing = []
for _mod, _names in _NEEDED.items():
    try:
        _m = importlib.import_module(_mod)
    except Exception as _exc:                      # noqa: BLE001
        _missing.append(f"{_mod} (import failed: {_exc})")
        continue
    _missing += [f"{_mod}.{n}" for n in _names if not hasattr(_m, n)]
    if _mod == "model" and getattr(_m, "BUILD", None) != EXPECTED_MODEL_BUILD:
        _missing.append(f"model.BUILD {getattr(_m, 'BUILD', None)!r} != {EXPECTED_MODEL_BUILD!r}")
if _missing:
    st.error("**This deployment is running stale code.** Missing: `" + "`, `".join(_missing)
             + "`. Streamlit Cloud pulled the new files but kept an old module in memory. "
             "Fix it with **Manage app → ⋮ → Reboot app**.", icon=":material/error:")
    st.stop()

with st.sidebar:
    st.title("⚽ Soccerway Predictor")

pg = st.navigation([
    st.Page("views/two_teams.py", title="Two teams", icon=":material/sports_soccer:", default=True),
    st.Page("views/league_day.py", title="League day", icon=":material/calendar_today:"),
])
pg.run()
