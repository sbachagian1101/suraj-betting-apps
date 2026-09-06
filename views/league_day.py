"""Page 2: paste league names, predict every match of the day in those leagues."""
from datetime import datetime, timedelta, timezone

import pandas as pd
import streamlit as st

import model as M
import soccerway as SW
import ui
from pipeline import analyse_fixture, daily_cached

DEFAULT_LEAGUES = "ENGLAND: Premier League\nFRANCE: Ligue 1\nGERMANY: Bundesliga"
STATUS_OPTIONS = ["Scheduled", "Live", "Finished"]

today_utc = datetime.now(timezone.utc).date()

with st.sidebar:
    st.header("League day")
    st.caption("One league per line, as Flashscore/Soccerway name them. The app finds every "
               "match of that day in those leagues and predicts each one.")
    leagues_text = st.text_area("Leagues", value=DEFAULT_LEAGUES, height=120)
    day = st.date_input("Match day", value=today_utc,
                        min_value=today_utc - timedelta(days=7),
                        max_value=today_utc + timedelta(days=7))
    tz = st.slider("Your UTC offset (hours)", -12, 14, 4,
                   help="Decides which matches count as that day. Mauritius is +4.")
    statuses = st.multiselect("Include matches that are", STATUS_OPTIONS, default=STATUS_OPTIONS)
    n_rates = st.slider("Matches for xG / goals rates", 1, 15, M.N_RATES)
    n_lineup = st.slider("Matches for player ratings / line-up", 1, 10, M.N_LINEUP)
    go = st.button("Run all matches", type="primary", width="stretch")
    st.caption("Roughly 10 seconds per match. A full Saturday across three leagues is "
               "25 to 30 matches, so expect a few minutes; rows appear as they finish.")

if go:
    offset = (day - today_utc).days
    queries = [ln for ln in leagues_text.splitlines() if ln.strip()]
    try:
        all_matches = daily_cached(offset, tz)
    except SW.SoccerwayError as exc:
        st.error(str(exc))
        st.stop()
    found, missing = SW.select_leagues(queries, all_matches)
    if missing:
        st.warning("Could not match: " + "; ".join(missing) +
                   ". Use the Flashscore wording, e.g. 'ENGLAND: Premier League'.")
    wanted = set(found.values())
    matches = [m for m in all_matches if m.competition in wanted and m.status in statuses]
    matches.sort(key=lambda m: (m.competition, m.kickoff))
    if not matches:
        st.info("No matches found for those leagues on that day with the chosen statuses.")
        st.stop()
    st.markdown("Matched leagues: " + ", ".join(f"**{v}**" for v in dict.fromkeys(found.values())))

    results: list[tuple[SW.Match, object]] = []
    bar = st.progress(0.0, text=f"0 of {len(matches)} matches")
    table_slot = st.empty()
    for i, m in enumerate(matches, 1):
        bar.progress((i - 1) / len(matches), text=f"{m.home_name} v {m.away_name} ({i} of {len(matches)})")
        try:
            results.append((m, analyse_fixture(m, n_rates, n_lineup)))
        except Exception as exc:                      # noqa: BLE001 - keep going
            results.append((m, exc))
        table_slot.caption(f"{i} of {len(matches)} done")
    bar.progress(1.0, text=f"Done: {len(matches)} matches")
    table_slot.empty()
    st.session_state["league_results"] = dict(results=results, day=day, n=n_rates, n_lineup=n_lineup)

if "league_results" not in st.session_state:
    st.markdown("### Paste league names in the sidebar and press **Run all matches**.")
    st.markdown(
        "For every match of that day in those leagues the app fetches both teams' recent "
        "matches (results, xG, xGA, line-ups and ratings), today's line-up if published, and "
        "bet365 odds, and predicts the match. You get a summary table, a CSV download, and "
        "the full breakdown for each match below it. Only matches played **before** each "
        "fixture count as its history, so a finished match is still predicted pre-match "
        "and shown next to its actual score.")
    st.stop()

R = st.session_state["league_results"]
results = R["results"]

# --------------------------------------------------------------------------- #
# Summary table
# --------------------------------------------------------------------------- #
rows = []
for m, A in results:
    base = {
        "Kick-off (UTC)": m.kickoff.strftime("%H:%M"),
        "League": m.competition,
        "Home": m.home_name, "Away": m.away_name,
        "Status": m.status,
        "Actual": (f"{m.home_score}-{m.away_score}"
                   if m.stage in ("2", "3") and m.home_score is not None else ""),
    }
    if isinstance(A, Exception):
        rows.append(base | {"Error": str(A)[:80]})
        continue
    p, o = A.pred, A.odds
    imp = M.implied(o) if o else None
    edge = ""
    if imp:
        gaps = {"H": p.p_home - imp["home"], "D": p.p_draw - imp["draw"], "A": p.p_away - imp["away"]}
        k = max(gaps, key=gaps.get)
        edge = f"{k} {gaps[k]:+.0%}"
    rows.append(base | {
        "Home %": round(p.p_home * 100), "Draw %": round(p.p_draw * 100), "Away %": round(p.p_away * 100),
        "Exp score": f"{p.exp_home:.2f}-{p.exp_away:.2f}",
        "Likely": p.top_scores[0][0],
        "O2.5 %": round(p.p_over25 * 100),
        "Odds H/D/A": f"{o['home']:.2f}/{o['draw']:.2f}/{o['away']:.2f}" if o else "",
        "Biggest gap vs book": edge,
        "XI": (("Confirmed" if A.la_h and A.la_h.source == "today" else "Probable") + " / " +
               ("Confirmed" if A.la_a and A.la_a.source == "today" else "Probable")),
        "Error": "",
    })
df = pd.DataFrame(rows)

n_ok = sum(1 for _, A in results if not isinstance(A, Exception))
st.title(f"League day · {R['day'].strftime('%a %d %b %Y')}")
st.caption(f"{n_ok} of {len(results)} matches predicted · rates window {R['n']} · "
           f"line-up window {R['n_lineup']} · sorted by league and kick-off, not by edge.")
st.dataframe(df, hide_index=True, width="stretch")
st.download_button("Download CSV", df.to_csv(index=False).encode("utf-8"),
                   file_name=f"league-day-{R['day']}.csv", mime="text/csv")

# --------------------------------------------------------------------------- #
# Per-match detail
# --------------------------------------------------------------------------- #
st.subheader("Match by match")
for m, A in results:
    label = f"{m.kickoff.strftime('%H:%M')} · {m.home_name} v {m.away_name}"
    if isinstance(A, Exception):
        with st.expander(f"{label} · failed"):
            st.error(str(A))
        continue
    p = A.pred
    label += f" · {p.p_home:.0%} / {p.p_draw:.0%} / {p.p_away:.0%} · {p.top_scores[0][0]}"
    if m.stage in ("2", "3") and m.home_score is not None:
        label += f" · actual {m.home_score}-{m.away_score}"
    with st.expander(label):
        ui.render_compact(A)
