"""Page 2: today's matches from Flashscore.

Two ways in:
* **Pick a match** - press *Fetch Today Matches*, then choose country -> league
  -> match from dropdowns filled from the day's feed. The prediction panel can
  refresh itself every minute so odds, line-ups and the live score stay current.
* **Whole day** - paste league names, predict every match of the day in them,
  get a table plus one expander per match.
"""
from datetime import datetime, timedelta, timezone

import pandas as pd
import streamlit as st

import board_ui
import model as M
import soccerway as SW
import ui
from pipeline import DayIndex, analyse_fixture, daily_cached, latest_match, on_day, priced_matches

DEFAULT_LEAGUES = "ENGLAND: Premier League\nFRANCE: Ligue 1\nGERMANY: Bundesliga"
STATUS_OPTIONS = ["Scheduled", "Live", "Finished"]
REFRESH_SECONDS = 60

today_utc = datetime.now(timezone.utc).date()

with st.sidebar:
    st.header("League day")
    day = st.date_input("Match day", value=today_utc,
                        min_value=today_utc - timedelta(days=7),
                        max_value=today_utc + timedelta(days=7))
    tz = st.slider("Your UTC offset (hours)", -12, 14, 4,
                   help="Decides which matches count as that day. Mauritius is +4.")
    n_rates = st.slider("Matches for xG / goals rates", 1, 15, M.N_RATES,
                        help=f"Recency-weighted: each older match counts {M.DECAY:.0%} of the one before it.")
    n_lineup = st.slider("Matches for player ratings / line-up", 1, 10, M.N_LINEUP,
                         help="Shorter window: who is in form changes faster than team strength.")
    st.caption("Line-ups appear on Soccerway roughly an hour before kick-off. "
               "Before that the last match's XI is used and labelled as such.")

offset = (day - today_utc).days
tab_pick, tab_all, tab_board = st.tabs(["Pick a match", "Whole day", "Live board"])


def match_label(m: SW.Match) -> str:
    s = f"{SW.fmt_mu(m.kickoff)} MU ({m.kickoff.strftime('%H:%M')} UTC) · {m.home_name} v {m.away_name}"
    if m.stage in ("2", "3") and m.home_score is not None:
        s += f" · {m.home_score}-{m.away_score} ({m.status})"
    elif m.status != "Scheduled":
        s += f" · {m.status}"
    return s


# =========================================================================== #
# Tab 1: pick a match
# =========================================================================== #
with tab_pick:
    c1, c2 = st.columns([1, 3])
    fetch = c1.button("Fetch Today Matches", type="primary", width="stretch", key="fetch_btn",
                      help="Reads every football match of the chosen day from Flashscore.")
    c2.caption(f"Match day **{day.strftime('%a %d %b %Y')}** (UTC{tz:+d}). "
               "Change the day or offset in the sidebar, then fetch again.")
    if fetch:
        try:
            all_ms = on_day(daily_cached(offset, tz, refresh=True), day, tz)
        except SW.SoccerwayError as exc:
            st.error(str(exc))
            st.stop()
        # Odds sweep first: only matches the book has priced are kept anywhere in
        # the app. One cheap request per match, in parallel.
        bar = st.progress(0.0, text=f"Checking odds for {len(all_ms)} matches…")
        priced, unpriced = priced_matches(
            all_ms, progress=lambda d, n: bar.progress(d / n, text=f"Checking odds: {d} of {n} matches"))
        bar.empty()
        st.session_state["day_index"] = dict(index=DayIndex(priced), day=day, offset=offset, tz=tz,
                                             fetched=datetime.now(timezone.utc),
                                             unpriced=unpriced, total=len(all_ms))
        st.session_state.pop("picked", None)

    D = st.session_state.get("day_index")
    if not D:
        st.markdown("Press **Fetch Today Matches** to load the day's fixtures, then pick "
                    "country, league and match from the dropdowns. Matches the book has not "
                    "priced are skipped entirely.")
    else:
        idx: DayIndex = D["index"]
        st.caption(f"{D.get('total', len(idx.matches))} matches on the feed, "
                   f"**{len(idx.matches)} with odds** in {len(idx.countries())} countries "
                   f"({len(D.get('unpriced', []))} without odds skipped), fetched "
                   f"{D['fetched'].strftime('%H:%M:%S')} UTC for {D['day'].strftime('%a %d %b')}.")
        countries = idx.countries()
        d1, d2, d3 = st.columns([1, 1.4, 2.2])
        country = d1.selectbox("Country", countries, index=None, placeholder="Choose a country",
                               key="country_sel")
        leagues = idx.leagues(country) if country else []
        league = d2.selectbox("League", leagues, index=None, placeholder="Choose a league",
                              disabled=not country, key="league_sel")
        fixtures = idx.fixtures(country, league) if (country and league) else []
        by_label: dict[str, SW.Match] = {}
        for m in fixtures:
            lab = match_label(m)
            if lab in by_label:                       # same teams, same minute: disambiguate
                lab += f" [{m.id}]"
            by_label[lab] = m
        chosen_label = d3.selectbox("Match (Home v Away)", list(by_label), index=None,
                                    placeholder="Choose a match", disabled=not league,
                                    key="match_sel")
        chosen = by_label.get(chosen_label) if chosen_label else None
        b1, b2 = st.columns([1, 3])
        predict = b1.button("Predict this match", type="primary", width="stretch",
                            disabled=chosen is None, key="predict_btn")
        auto = b2.toggle(f"Keep updating every {REFRESH_SECONDS} s (score, odds, line-ups)",
                         value=False, disabled=chosen is None)
        if predict and chosen is not None:
            st.session_state["picked"] = dict(match=chosen, offset=D["offset"], tz=D["tz"],
                                              n=n_rates, n_lineup=n_lineup)

        P = st.session_state.get("picked")
        if P and chosen is not None and P["match"].id != chosen.id:
            st.info("You changed the match. Press **Predict this match** to run it.")
        elif P:
            def _panel():
                m = latest_match(P["match"], P["offset"], P["tz"])
                with st.spinner("Fetching…", show_time=True):
                    try:
                        A = analyse_fixture(m, P["n"], P["n_lineup"])
                    except SW.SoccerwayError as exc:
                        st.error(str(exc))
                        return
                st.caption(f"Updated {datetime.now(timezone.utc).strftime('%H:%M:%S')} UTC"
                           + (" · auto-refresh on" if auto else ""))
                ui.render_full(A)

            # A fragment with run_every re-renders only this panel; team histories are
            # cached so a refresh costs a few requests (day feed, line-ups, odds).
            st.fragment(run_every=f"{REFRESH_SECONDS}s" if auto else None)(_panel)()


# =========================================================================== #
# Tab 2: whole day
# =========================================================================== #
with tab_all:
    l1, l2 = st.columns([2, 1])
    leagues_text = l1.text_area("Leagues, one per line, as Flashscore names them",
                                value=DEFAULT_LEAGUES, height=110)
    statuses = l2.multiselect("Include matches that are", STATUS_OPTIONS, default=STATUS_OPTIONS)
    go = l2.button("Run all matches", type="primary", width="stretch", key="run_all_btn")
    st.caption("Roughly 10 seconds per match. A full Saturday across three leagues is "
               "25 to 30 matches, so expect a few minutes; the table appears when all are done.")

    if go:
        queries = [ln for ln in leagues_text.splitlines() if ln.strip()]
        try:
            all_matches = on_day(daily_cached(offset, tz), day, tz)
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
        # skip anything the book has not priced: no time spent computing it
        bar = st.progress(0.0, text=f"Checking odds for {len(matches)} matches…")
        matches, skipped = priced_matches(
            matches, progress=lambda d, n: bar.progress(d / n, text=f"Checking odds: {d} of {n}"))
        bar.empty()
        if skipped:
            st.caption(f"{len(skipped)} matches without odds skipped.")
        if not matches:
            st.info("No priced matches found for those leagues on that day with the chosen statuses.")
            st.stop()
        st.markdown("Matched leagues: " + ", ".join(f"**{v}**" for v in dict.fromkeys(found.values())))

        results: list[tuple[SW.Match, object]] = []
        bar = st.progress(0.0, text=f"0 of {len(matches)} matches")
        for i, m in enumerate(matches, 1):
            bar.progress((i - 1) / len(matches),
                         text=f"{m.home_name} v {m.away_name} ({i} of {len(matches)})")
            try:
                results.append((m, analyse_fixture(m, n_rates, n_lineup)))
            except Exception as exc:                      # noqa: BLE001 - keep going
                results.append((m, exc))
        bar.progress(1.0, text=f"Done: {len(matches)} matches")
        st.session_state["league_results"] = dict(results=results, day=day, n=n_rates,
                                                  n_lineup=n_lineup)

    R = st.session_state.get("league_results")
    if not R:
        st.markdown(
            "For every match of that day in those leagues the app fetches both teams' recent "
            "matches (results, xG, xGA, line-ups and ratings), today's line-up if published, and "
            "bet365 odds, and predicts the match. You get a summary table, a CSV download, and "
            "the full breakdown for each match below it. Only matches played **before** each "
            "fixture count as its history, so a finished match is still predicted pre-match "
            "and shown next to its actual score.")
    else:
        results = R["results"]
        rows = []
        for m, A in results:
            base = {
                "Kick-off (UTC)": m.kickoff.strftime("%H:%M"),
                "Kick-off (MU)": SW.fmt_mu(m.kickoff),
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
            sig = M.bet_signal(p, o)
            edge = f"{sig['side'][0].upper()} {sig['gap']:+.0%}" if sig else ""
            rows.append(base | {
                "Bet": sig["label"] if sig else "",
                "Home %": round(p.p_home * 100), "Draw %": round(p.p_draw * 100),
                "Away %": round(p.p_away * 100),
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
        st.subheader(f"League day · {R['day'].strftime('%a %d %b %Y')}")
        st.caption(f"{n_ok} of {len(results)} matches predicted · rates window {R['n']} · "
                   f"line-up window {R['n_lineup']} · sorted by league and kick-off, not by edge.")
        st.dataframe(df, hide_index=True, width="stretch")
        st.download_button("Download CSV", df.to_csv(index=False).encode("utf-8"),
                           file_name=f"league-day-{R['day']}.csv", mime="text/csv")

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
            sig = M.bet_signal(p, A.odds)
            if sig and sig["bet"]:
                label += f" · 🟢 {sig['label']}"
            with st.expander(label):
                ui.render_compact(A)


# =========================================================================== #
# Tab 3: live board (see board_ui.py)
# =========================================================================== #
with tab_board:
    board_ui.render(st.session_state.get("day_index"), n_rates, n_lineup)
