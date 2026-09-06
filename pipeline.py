"""Fetch-and-predict pipeline shared by both pages.

Keeps its own small thread-safe TTL cache instead of ``st.cache_data`` so the
fetching can run in worker threads (Streamlit's cache is not meant to be
called from threads without a script context).
"""
from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Callable, Optional

import model as M
import soccerway as SW

_cache: dict[tuple, tuple[float, object]] = {}
_lock = threading.Lock()

TTL_RESULTS = 600
TTL_FINISHED = 6 * 3600
TTL_LIVE = 90


def _cached(key: tuple, ttl: float, fn: Callable):
    now = time.time()
    with _lock:
        hit = _cache.get(key)
        if hit and hit[0] > now:
            return hit[1]
    value = fn()
    with _lock:
        _cache[key] = (now + ttl, value)
    return value


def clear_cache() -> None:
    with _lock:
        _cache.clear()


def results_cached(slug: str, team_id: str) -> list[SW.Match]:
    return _cached(("results", team_id), TTL_RESULTS, lambda: SW.team_results(slug, team_id))


def fixtures_cached(slug: str, team_id: str) -> list[SW.Match]:
    return _cached(("fixtures", team_id), TTL_RESULTS, lambda: SW.team_fixtures(slug, team_id))


def finished_match_cached(event_id: str) -> tuple[dict, dict]:
    def fetch():
        try:
            stats = SW.match_stats(event_id)
        except SW.SoccerwayError:
            stats = {}
        try:
            lineups = SW.match_lineups(event_id)
        except SW.SoccerwayError:
            lineups = {}
        return stats, lineups
    return _cached(("finished", event_id), TTL_FINISHED, fetch)


def live_lineups_cached(event_id: str) -> dict:
    def fetch():
        try:
            return SW.match_lineups(event_id)
        except SW.SoccerwayError:
            return {}
    return _cached(("lineups", event_id), TTL_LIVE, fetch)


def odds_cached(event_id: str) -> Optional[dict]:
    return _cached(("odds", event_id), TTL_LIVE, lambda: SW.match_odds(event_id))


def daily_cached(day_offset: int, tz_hours: int) -> list[SW.Match]:
    return _cached(("daily", day_offset, tz_hours), 120,
                   lambda: SW.daily_matches(day_offset, tz_hours))


# --------------------------------------------------------------------------- #
@dataclass
class TeamData:
    slug: str
    team_id: str
    name: str
    records: list[SW.MatchRecord]


def load_team(slug: str, team_id: str, n: int, before: Optional[SW.Match] = None,
              workers: int = 3) -> TeamData:
    """Last ``n`` finished matches (before ``before`` if given) with stats and line-ups."""
    results = results_cached(slug, team_id)
    if before is not None:
        results = SW.history_before(results, before)
    results = results[:n]
    with ThreadPoolExecutor(max_workers=workers) as ex:
        pairs = list(ex.map(finished_match_cached, [m.id for m in results]))
    records, name = [], ""
    for m, (stats, lineups) in zip(results, pairs):
        rec = SW.build_record(m, team_id, stats, lineups)
        records.append(rec)
        name = name or (m.home_name if rec.side == "H" else m.away_name)
    return TeamData(slug, team_id, name or slug.replace("-", " ").title(), records)


@dataclass
class Analysis:
    fixture: Optional[SW.Match]
    home: TeamData
    away: TeamData
    prof_h: M.TeamProfile
    prof_a: M.TeamProfile
    la_h: Optional[M.LineupAssessment]
    la_a: Optional[M.LineupAssessment]
    pred: M.Prediction
    odds: Optional[dict]
    n: int
    n_lineup: int

    @property
    def title(self) -> str:
        return f"{self.prof_h.name} v {self.prof_a.name}"


def _assess(records, side, prof, live):
    tl = live.get(side) if live else None
    if tl and tl.published:
        return M.assess_lineup(tl.starters, "today", tl.formation, prof)
    if records and records[0].starters:
        return M.assess_lineup(records[0].starters, "last match", records[0].formation, prof)
    return None


def analyse(home: TeamData, away: TeamData, fixture: Optional[SW.Match],
            n_rates: int, n_lineup: int) -> Analysis:
    live = live_lineups_cached(fixture.id) if fixture else {}
    odds = odds_cached(fixture.id) if fixture else None
    prof_h = M.profile(home.name, home.records, n_rates, n_lineup)
    prof_a = M.profile(away.name, away.records, n_rates, n_lineup)
    la_h = _assess(home.records, "HOME", prof_h, live)
    la_a = _assess(away.records, "AWAY", prof_a, live)
    pred = M.predict(prof_h, prof_a, la_h, la_a)
    return Analysis(fixture, home, away, prof_h, prof_a, la_h, la_a, pred, odds, n_rates, n_lineup)


def _load_both(h: tuple[str, str], a: tuple[str, str], n: int,
               before: Optional[SW.Match]) -> tuple[TeamData, TeamData]:
    with ThreadPoolExecutor(max_workers=2) as ex:
        fh = ex.submit(load_team, h[0], h[1], n, before)
        fa = ex.submit(load_team, a[0], a[1], n, before)
        return fh.result(), fa.result()


def analyse_pair(url_a: str, url_b: str, n_rates: int, n_lineup: int,
                 a_is_home_if_no_fixture: bool = True,
                 progress: Optional[Callable[[str], None]] = None) -> Analysis:
    """Two pasted team links -> Analysis of their next meeting."""
    say = progress or (lambda s: None)
    slug_a, id_a = SW.parse_team_url(url_a)
    slug_b, id_b = SW.parse_team_url(url_b)
    if id_a == id_b:
        raise SW.SoccerwayError("Both links point to the same team.")
    say("Looking for the fixture between them…")
    fixture = next((m for m in fixtures_cached(slug_a, id_a) if id_b in (m.home_id, m.away_id)), None)
    a_home = fixture.home_id == id_a if fixture else a_is_home_if_no_fixture
    h, a = ((slug_a, id_a), (slug_b, id_b)) if a_home else ((slug_b, id_b), (slug_a, id_a))
    say("Fetching both teams' recent matches…")
    home, away = _load_both(h, a, max(n_rates, n_lineup), fixture)
    say("Building the prediction…")
    return analyse(home, away, fixture, n_rates, n_lineup)


def analyse_fixture(match: SW.Match, n_rates: int, n_lineup: int) -> Analysis:
    """A match from the daily list -> Analysis (history strictly before kick-off)."""
    home, away = _load_both((match.home_slug or match.home_name.lower(), match.home_id),
                            (match.away_slug or match.away_name.lower(), match.away_id),
                            max(n_rates, n_lineup), match)
    return analyse(home, away, match, n_rates, n_lineup)
