"""Live board tab: tiered auto-refresh table with kick-off alerts and click-through charts.

Rendered by views/league_day.py inside its "Live board" tab. Kept in its own
module so the pieces (frame builder, styler, alerts) are importable and testable.
"""
from __future__ import annotations

import time
from datetime import datetime, timezone

import pandas as pd
import streamlit as st

import alerts
import charts
import model as M
import soccerway as SW
import ui
from board import BOARD, refresh_minutes
from pipeline import DayIndex

DEFAULT_LEAGUES = "ENGLAND: Premier League\nFRANCE: Ligue 1\nGERMANY: Bundesliga"
FAST_REDRAW = "1s"        # while a match is within 5 minutes: alternate colours = blink
SLOW_REDRAW = "30s"


def select_board_matches(matches: list[SW.Match], leagues: set[str], horizon_hours: float,
                         now: datetime) -> list[SW.Match]:
    """Matches to put on the board: every live or finished match of the day in the
    chosen leagues, plus scheduled ones kicking off within the horizon."""
    out = []
    for m in matches:
        if m.competition not in leagues:
            continue
        if m.stage in ("2", "3"):
            out.append(m)
        elif m.stage == "1" and 0 <= (m.kickoff - now).total_seconds() <= horizon_hours * 3600:
            out.append(m)
    return sorted(out, key=lambda x: x.kickoff)


HOURS = [f"{h:02d}:00" for h in range(25)]          # 00:00 .. 24:00 (Mauritius time)


def in_window(m: SW.Match, from_hour: int, to_hour: int) -> bool:
    """Kick-off (Mauritius time) between from_hour:00 and to_hour:00 inclusive."""
    t = SW.to_mu(m.kickoff)
    minutes = t.hour * 60 + t.minute
    return from_hour * 60 <= minutes <= to_hour * 60


def sidebar_filters(board_leagues: list[str]) -> tuple[set[str], int, int]:
    """Sidebar controls: which of the board's leagues to show, and the time window.
    Returns (leagues shown, from hour, to hour)."""
    with st.sidebar:
        st.markdown("**Live board filters**")
        removed = st.session_state.setdefault("board_removed_leagues", set())
        default = [l for l in board_leagues if l not in removed]
        if board_leagues:
            shown = st.multiselect("Leagues on the board (close the ones you do not want)",
                                   board_leagues, default=default,
                                   help="Removing a league here only hides its rows; the worker "
                                        "keeps tracking it. Re-add it any time.")
            st.session_state["board_removed_leagues"] = set(board_leagues) - set(shown)
        else:
            shown = []
            st.caption("No leagues on the board yet.")
        c1, c2 = st.columns(2)
        f = c1.selectbox("From (MU)", HOURS[:-1], index=0, key="board_from")
        t = c2.selectbox("To (MU)", HOURS[1:], index=23, key="board_to")
        from_hour, to_hour = int(f[:2]), int(t[:2])
        if to_hour <= from_hour:
            st.warning("'To' must be after 'From'; showing the whole day.")
            from_hour, to_hour = 0, 24
    return set(shown), from_hour, to_hour


def starts_in(m: SW.Match, now: datetime) -> str:
    if m.stage == "3":
        return "FT"
    if m.stage == "2":
        return "LIVE"
    secs = (m.kickoff - now).total_seconds()
    if secs <= 0:
        return "kicking off"
    h, rem = divmod(int(secs), 3600)
    mn, sc = divmod(rem, 60)
    if h:
        return f"{h}h {mn:02d}m"
    if mn >= 5:
        return f"{mn}m"
    return f"{mn}m {sc:02d}s"


def xi_text(la) -> str:
    """'6.37 (-0.03)': the XI's average rating over the line-up window and its gap
    to the team average; blank when there is no rated line-up."""
    if la is None or la.lineup_rating is None:
        return ""
    tag = "" if la.source == "today" else " ~"          # ~ = probable XI from the last match
    return f"{la.lineup_rating:.2f} ({la.gap:+.2f}){tag}"


def board_frame(rows, now: datetime) -> pd.DataFrame:
    out = []
    for r in rows:
        m = r.match
        country, league = DayIndex.split(m.competition)
        nan = float("nan")
        every = refresh_minutes(m.kickoff, now)
        rec = {
            "id": m.id,
            "Start (UTC)": m.kickoff.strftime("%H:%M"),
            "Start (MU)": SW.fmt_mu(m.kickoff),
            "Starts in": starts_in(m, now),
            "Country": country, "League": league,
            "Match": f"{m.home_name} v {m.away_name}",
            "Score": (f"{m.home_score}-{m.away_score}"
                      if m.stage in ("2", "3") and m.home_score is not None else ""),
            "Home %": nan, "Draw %": nan, "Away %": nan,
            "vs book H": nan, "vs book D": nan, "vs book A": nan,
            "Selection / prediction": r.error[:40] if r.error else "…computing",
            "Home XI (last N)": "", "Away XI (last N)": "",
            "Odds H": nan, "Odds D": nan, "Odds A": nan,
            "Updated": r.computed_at.strftime("%H:%M") if r.computed_at else "",
            "Refresh": f"every {every} min" if every else "frozen",
        }
        A = r.analysis
        if A is not None:
            p = A.pred
            imp = M.implied(A.odds) if A.odds else None
            sig = M.bet_signal(p, A.odds)
            rec.update({
                "Home %": round(p.p_home * 100), "Draw %": round(p.p_draw * 100),
                "Away %": round(p.p_away * 100),
                "vs book H": round((p.p_home - imp["home"]) * 100, 1) if imp else nan,
                "vs book D": round((p.p_draw - imp["draw"]) * 100, 1) if imp else nan,
                "vs book A": round((p.p_away - imp["away"]) * 100, 1) if imp else nan,
                "Selection / prediction": sig["label"] if sig else "no odds",
                "Home XI (last N)": xi_text(A.la_h),
                "Away XI (last N)": xi_text(A.la_a),
                "Odds H": A.odds["home"] if A.odds else nan,
                "Odds D": A.odds["draw"] if A.odds else nan,
                "Odds A": A.odds["away"] if A.odds else nan,
            })
        out.append(rec)
    df = pd.DataFrame(out)
    # Streamlit's grid prints missing numbers as "None" whatever the Styler says,
    # so hand it display strings for the numeric columns.
    for c in ("Home %", "Draw %", "Away %"):
        df[c] = df[c].map(lambda v: "" if pd.isna(v) else f"{v:.0f}")
    for c in ("vs book H", "vs book D", "vs book A"):
        df[c] = df[c].map(lambda v: "" if pd.isna(v) else f"{v:+.1f}%")
    for c in ("Odds H", "Odds D", "Odds A"):
        df[c] = df[c].map(lambda v: "" if pd.isna(v) else f"{v:.2f}")
    return df


def style_board(df: pd.DataFrame, blink_ids: set[str], blink_on: bool, started_ids: set[str]):
    """Row colour first (rose for started, purple on alternate redraws for imminent),
    then the cell colours for gaps, odds and the selection, which sit on top."""
    def row_style(row):
        rid = row.get("id")
        if rid in started_ids:
            css = f"background-color: {alerts.ROSE};"
        elif rid in blink_ids and blink_on:
            css = f"background-color: {alerts.PURPLE};"
        else:
            css = ""
        return [css] * len(row)

    def pred_style(v):
        if isinstance(v, str) and v.startswith("BET "):
            return "background-color: #c6ff00; color: #000; font-weight: 800;"
        return ""

    def pct_style(v):
        try:
            return "background-color: #c6ff00; color: #000;" if float(v) > M.BET_MIN_PROB * 100 else ""
        except ValueError:
            return ""

    def gap_style(v):
        try:
            x = float(str(v).rstrip("%"))
        except ValueError:
            return ""
        if x > 0:
            return "background-color: #c6ff00; color: #000;"
        return "color: #999;" if x < 0 else ""

    sty = df.style.apply(row_style, axis=1)
    sty = sty.map(pred_style, subset=["Selection / prediction"])
    sty = sty.map(pct_style, subset=["Home %", "Draw %", "Away %"])
    sty = sty.map(gap_style, subset=["vs book H", "vs book D", "vs book A"])
    sty = sty.map(alerts.odds_colour, subset=["Odds H", "Odds D", "Odds A"])
    return sty


BELL_CSS = """
<style>
@keyframes swbell { 0%,100% { transform: rotate(0); } 25% { transform: rotate(18deg); } 75% { transform: rotate(-18deg); } }
.sw-bell { display:inline-block; font-size:1.6rem; animation: swbell 0.8s infinite; transform-origin: top center; }
.sw-bell-box { background:#e9d5ff; border-radius:0.5rem; padding:0.4rem 0.8rem; font-weight:700; }
</style>
"""


def render(D, n_rates: int, n_lineup: int) -> None:
    if not D:
        st.markdown("Press **Fetch Today Matches** on the *Pick a match* tab first, then choose the "
                    "leagues to keep on the board.")
        return
    idx: DayIndex = D["index"]
    all_leagues = sorted({m.competition for m in idx.matches})
    upcoming = {m.competition for m in idx.matches if m.stage == "1"}
    typed = SW.select_leagues(DEFAULT_LEAGUES.splitlines(), idx.matches, strict=True)[0].values()
    default = [l for l in dict.fromkeys(typed) if l in upcoming]
    if not default:
        skip = ("U19", "U20", "U21", "U23", "Women", "Amateur", "Reserve", "Youth", "Regional",
                "Group", "Division 2", "Division 3", "Liga 2", "Liga 3", " B", " C", "Torneo")
        default = [l for l in all_leagues if l in upcoming and not any(k in l for k in skip)][:5]
    c1, c2, c3 = st.columns([3, 1, 1])
    chosen_leagues = c1.multiselect(
        "Leagues on the board", all_leagues, default=default, key="board_leagues",
        help="Every scheduled match of the day in these leagues is tracked.")
    horizon = c2.slider("Only matches within (hours)", 1, 24, 24, key="board_horizon")
    only_odds = c2.toggle("Only matches with odds", value=True, key="board_only_odds",
                          help="Hides matches the book has not priced. They appear "
                               "automatically once odds turn up on a refresh.")
    start = c3.button("Start / update board", type="primary", width="stretch", key="board_start")
    stop = c3.button("Stop board", width="stretch", key="board_stop")
    st.caption("Refresh: every 60 min beyond 3 h from kick-off, 30 min within 3 h, 15 min within "
               "1 h, 5 min within 30 min. Started matches are frozen; score and status keep "
               "updating every 5 min. The table redraws every 30 s, and once a second while a "
               "match is inside 5 minutes of kick-off: its row blinks purple and the bell beeps "
               "once; started matches turn rose. Click a row for the charts.")
    if start:
        now = datetime.now(timezone.utc)
        picked = select_board_matches(idx.matches, set(chosen_leagues), horizon, now)
        # matches skipped at fetch time for having no odds: never computed, never shown,
        # but re-checked every 30 min and promoted if the book prices them
        skipped = select_board_matches(D.get("unpriced", []), set(chosen_leagues), horizon, now)
        if not picked and not skipped:
            st.warning("No matches in those leagues inside that horizon.")
        elif not picked:
            st.warning(f"None of the {len(skipped)} matches in those leagues has odds yet. "
                       "They are re-checked every 30 minutes and appear once priced.")
            BOARD.configure([], n_rates, n_lineup, D["offset"], D["tz"], candidates=skipped)
        else:
            BOARD.configure(picked, n_rates, n_lineup, D["offset"], D["tz"], candidates=skipped)
    if stop:
        BOARD.stop()

    st.session_state.setdefault("board_notified", set())
    now0 = datetime.now(timezone.utc)
    snapshot0 = BOARD.snapshot()
    fast = any(alerts.imminent(r.match, now0) for r in snapshot0)
    st.session_state["board_fast"] = fast
    leagues_shown, from_hour, to_hour = sidebar_filters(
        sorted({r.match.competition for r in snapshot0}))

    @st.fragment(run_every=FAST_REDRAW if fast else SLOW_REDRAW)
    def board_table():
        now = datetime.now(timezone.utc)
        rows = BOARD.snapshot()
        S = BOARD.status()
        if not rows:
            st.info("The board is empty. Pick leagues and press **Start / update board**.")
            return
        priced = [r for r in rows if r.analysis is not None and r.analysis.odds]
        shown = priced if only_odds else rows
        shown = [r for r in shown if r.match.competition in leagues_shown
                 and in_window(r.match, from_hour, to_hour)]
        shown_matches = [r.match for r in shown]
        imminent_ids = {m.id for m in shown_matches if alerts.imminent(m, now)}
        started_ids = {m.id for m in shown_matches if alerts.started(m, now)}
        # switch the redraw cadence when the first match enters (or the last leaves) the window
        if bool(imminent_ids) != st.session_state.get("board_fast", False):
            st.session_state["board_fast"] = bool(imminent_ids)
            st.rerun(scope="app")

        fresh = alerts.new_alerts(shown_matches, now, st.session_state["board_notified"])
        if fresh:
            st.session_state["board_notified"].update(m.id for m in fresh)
            st.audio(alerts.beep_wav(), format="audio/wav", autoplay=True)
        if imminent_ids:
            names = ", ".join(f"{m.home_name} v {m.away_name} ({starts_in(m, now)})"
                              for m in shown_matches if m.id in imminent_ids)
            st.markdown(BELL_CSS + f'<div class="sw-bell-box"><span class="sw-bell">🔔</span> '
                        f'Kick-off within 5 minutes: {names}</div>', unsafe_allow_html=True)

        m1, m2, m3, m4, m5, m6, m7 = st.columns(7)
        m1.metric("Matches", S["total"])
        m2.metric("Predicted", S["computed"])
        m3.metric("With odds", len(priced))
        m4.metric("Imminent", len(imminent_ids))
        m5.metric("Errors", S["errors"])
        m6.metric("Worker", "running" if S["running"] else "stopped")
        nxt = S["next_due"]
        m7.metric("Next refresh", f"in {max(0, int((nxt - now).total_seconds() // 60))} min" if nxt else "—")
        done, total = S["computed"], S["total"]
        st.progress(done / total if total else 0.0,
                    text=f"{done} out of {total} matches computed ({done / total:.0%})"
                         + (f" · computing {S['busy_with']}…" if S["busy_with"] else
                            (" · all done, refreshing on schedule" if done >= total else ""))
                         + (f" · {S['skipped']} without odds skipped, re-checked every 30 min"
                            if S.get("skipped") else ""))
        if not shown:
            st.info("Nothing to show with the current filters: the odds toggle, the leagues "
                    "shown and the time window in the sidebar. Unpriced matches appear as "
                    "soon as the book prices them.")
            return
        df = board_frame(shown, now)
        blink_on = int(time.time()) % 2 == 0
        ev = st.dataframe(
            style_board(df, imminent_ids, blink_on, started_ids), hide_index=True, width="stretch",
            height=min(520, 60 + 36 * len(df)),
            on_select="rerun", selection_mode="single-row", key="board_table",
            column_config={"id": None})
        sel = ev.selection.rows if ev and ev.selection else []
        if sel:
            new_id = df.iloc[sel[0]]["id"]
            if st.session_state.get("board_sel") != new_id:
                st.session_state["board_sel"] = new_id
                st.rerun(scope="app")

    @st.fragment(run_every=SLOW_REDRAW)
    def board_detail():
        now = datetime.now(timezone.utc)
        chosen_id = st.session_state.get("board_sel")
        r = BOARD.row(chosen_id) if chosen_id else None
        if r is None:
            st.caption("Select a row to see the charts and insights for that match.")
            return
        st.divider()
        if r.analysis is None:
            st.info(f"{r.match.home_name} v {r.match.away_name}: not computed yet"
                    + (f" ({r.error})" if r.error else "."))
            return
        A = r.analysis
        st.subheader(f"{A.title} · {starts_in(r.match, now)}")
        ui.prediction_row(A)
        ui.bet_banner(A)
        g1, g2 = st.columns(2)
        g1.plotly_chart(charts.model_vs_book(A), width="stretch")
        g2.plotly_chart(charts.outcome_pie(A), width="stretch")
        g3, g4 = st.columns(2)
        g3.plotly_chart(charts.score_heatmap(A), width="stretch")
        g4.plotly_chart(charts.xg_history(A), width="stretch")
        rb = charts.rating_bars(A)
        if rb is not None:
            st.plotly_chart(rb, width="stretch")
        t1, t2, t3 = st.tabs(["Insights", "Teams & line-ups", "Method"])
        with t1:
            ui.insights_list(A)
        with t2:
            l, rr = st.columns(2)
            with l:
                ui.team_panel(A.prof_h, A.la_h, "Home")
            with rr:
                ui.team_panel(A.prof_a, A.la_a, "Away")
        with t3:
            ui.methodology(A)
        with st.expander("Worker log"):
            for line in BOARD.status()["log"]:
                st.text(line)

    board_table()
    board_detail()
