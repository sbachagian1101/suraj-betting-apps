"""Live board: keeps predictions for a set of matches fresh on a tiered schedule.

One background thread lives inside the Streamlit server process (the same
way the Steamers collector does), so it keeps refreshing for as long as the
app is open, across page reloads and fragment reruns. The UI only reads
snapshots; it never computes.

Refresh tiers, measured to kick-off:
    more than 3 h away   every 60 min
    within 3 h           every 30 min
    within 1 h           every 15 min
    within 30 min        every 5 min
Once a match has started its prediction is frozen and only the status and
score keep updating from the day feed (every LIST_REFRESH seconds).
"""
from __future__ import annotations

import threading
import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Callable, Optional

import soccerway as SW
from pipeline import Analysis, analyse_fixture, daily_cached

TIERS = [(30, 5), (60, 15), (180, 30)]      # (minutes to kick-off <=, refresh every minutes)
BEYOND_MINUTES = 60
LIST_REFRESH = 300                          # seconds between day-feed status refreshes
LOOP_SLEEP = 5


def refresh_minutes(kickoff: datetime, now: datetime) -> Optional[int]:
    """Minutes between refreshes for a match this far from kick-off; None once started."""
    mins = (kickoff - now).total_seconds() / 60
    if mins <= 0:
        return None
    for limit, every in TIERS:
        if mins <= limit:
            return every
    return BEYOND_MINUTES


@dataclass
class Row:
    match: SW.Match
    analysis: Optional[Analysis] = None
    error: str = ""
    computed_at: Optional[datetime] = None
    next_due: Optional[datetime] = None
    refreshes: int = 0

    @property
    def started(self) -> bool:
        return self.match.stage in ("2", "3")


@dataclass
class Board:
    compute: Callable[[SW.Match, int, int], Analysis] = analyse_fixture
    rows: dict[str, Row] = field(default_factory=dict)
    n_rates: int = 12
    n_lineup: int = 5
    day_offset: int = 0
    tz_hours: int = 0
    running: bool = False
    busy_with: str = ""
    last_list_refresh: Optional[datetime] = None
    log: list[str] = field(default_factory=list)
    _lock: threading.Lock = field(default_factory=threading.Lock)
    _thread: Optional[threading.Thread] = None

    # ------------------------------------------------------------------ control
    def configure(self, matches: list[SW.Match], n_rates: int, n_lineup: int,
                  day_offset: int, tz_hours: int) -> None:
        """Set the matches to track. Existing predictions for the same ids are kept
        unless the windows changed."""
        now = datetime.now(timezone.utc)
        with self._lock:
            reset = (n_rates, n_lineup) != (self.n_rates, self.n_lineup)
            self.n_rates, self.n_lineup = n_rates, n_lineup
            self.day_offset, self.tz_hours = day_offset, tz_hours
            new: dict[str, Row] = {}
            for m in sorted(matches, key=lambda x: x.kickoff):
                old = self.rows.get(m.id)
                if old and not reset:
                    old.match = m
                    new[m.id] = old
                else:
                    new[m.id] = Row(match=m, next_due=now)
            # an update never drops a match that has already started, as long as
            # its league is still on the board
            comps = {m.competition for m in matches}
            for mid, old in self.rows.items():
                if mid not in new and old.started and old.match.competition in comps and not reset:
                    new[mid] = old
            self.rows = dict(sorted(new.items(), key=lambda kv: kv[1].match.kickoff))
            self._log(f"board set to {len(new)} matches (windows {n_rates}/{n_lineup})")
        self.start()

    def start(self) -> None:
        with self._lock:
            if self._thread and self._thread.is_alive():
                self.running = True
                return
            self.running = True
            self._thread = threading.Thread(target=self._loop, name="soccerway-board", daemon=True)
            self._thread.start()

    def stop(self) -> None:
        with self._lock:
            self.running = False
            self._log("board stopped")

    def clear(self) -> None:
        self.stop()
        with self._lock:
            self.rows = {}

    # ------------------------------------------------------------------ reading
    def snapshot(self) -> list[Row]:
        with self._lock:
            return sorted(self.rows.values(), key=lambda r: r.match.kickoff)

    def row(self, match_id: str) -> Optional[Row]:
        with self._lock:
            return self.rows.get(match_id)

    def status(self) -> dict:
        with self._lock:
            rows = list(self.rows.values())
            return dict(
                running=self.running and bool(self._thread and self._thread.is_alive()),
                total=len(rows),
                computed=sum(1 for r in rows if r.analysis is not None),
                errors=sum(1 for r in rows if r.error),
                busy_with=self.busy_with,
                next_due=min((r.next_due for r in rows if r.next_due and not r.started), default=None),
                last_list_refresh=self.last_list_refresh,
                log=list(self.log[-8:]),
            )

    # ------------------------------------------------------------------ worker
    def _log(self, msg: str) -> None:
        self.log.append(f"{datetime.now(timezone.utc):%H:%M:%S} {msg}")
        del self.log[:-50]

    def _due(self, now: datetime) -> Optional[Row]:
        """Next row to compute: anything never computed (including live and
        finished matches, predicted once from their pre-match history), then
        scheduled rows whose refresh is due. Started rows are never refreshed."""
        with self._lock:
            cands = [r for r in self.rows.values()
                     if (r.analysis is None and not r.error)
                     or (not r.started and r.next_due is not None and r.next_due <= now)]
        cands.sort(key=lambda r: (r.analysis is not None, r.match.kickoff))   # never-computed first
        return cands[0] if cands else None

    def _refresh_list(self, now: datetime) -> None:
        try:
            latest = {m.id: m for m in daily_cached(self.day_offset, self.tz_hours, refresh=True)}
        except Exception as exc:                                   # noqa: BLE001
            self._log(f"day feed refresh failed: {exc}")
            return
        with self._lock:
            for r in self.rows.values():
                if r.match.id in latest:
                    r.match = latest[r.match.id]
            self.last_list_refresh = now

    def _loop(self) -> None:
        while True:
            with self._lock:
                if not self.running:
                    break
            now = datetime.now(timezone.utc)
            if self.last_list_refresh is None or (now - self.last_list_refresh).total_seconds() >= LIST_REFRESH:
                self._refresh_list(now)
            row = self._due(now)
            if row is None:
                time.sleep(LOOP_SLEEP)
                continue
            self.busy_with = f"{row.match.home_name} v {row.match.away_name}"
            try:
                A = self.compute(row.match, self.n_rates, self.n_lineup)
                with self._lock:
                    row.analysis, row.error = A, ""
                    row.computed_at = datetime.now(timezone.utc)
                    row.refreshes += 1
            except Exception as exc:                               # noqa: BLE001
                with self._lock:
                    row.error = f"{type(exc).__name__}: {exc}"[:200]
                self._log(f"{self.busy_with}: {row.error}")
                traceback.print_exc()
            finally:
                every = refresh_minutes(row.match.kickoff, datetime.now(timezone.utc))
                with self._lock:
                    row.next_due = (datetime.now(timezone.utc) + timedelta(minutes=every)) if every else None
                self.busy_with = ""


BOARD = Board()
