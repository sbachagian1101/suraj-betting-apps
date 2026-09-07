"""Paste -> identity -> country feeds (in parallel) -> merged race card -> model."""
from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from typing import Callable, Optional

import model as M
import rs_paste
from common import MARKET_FIELDS, RaceCard, SourceError, merge_cards
from sources import betfair, hkjc, hrn, ladbrokes, pmu, sportinglife, winningform

BUILD = "2026-09-07c"      # bumped with every change; app.py refuses a stale copy

#: which adapters serve which country, in merge order (first = base when there is no paste field)
PLAN: dict[str, list] = {
    "AUS": [ladbrokes, betfair],
    "FR": [pmu, ladbrokes, betfair],
    "UK": [sportinglife, ladbrokes, betfair],
    "IRE": [sportinglife, ladbrokes, betfair],
    "HK": [hkjc, betfair],
    "SA": [winningform, betfair],
    "USA": [hrn, ladbrokes, betfair],
}
#: adapters whose prices should override the paste's "best odds" column
MARKET_AUTHORITY = {ladbrokes.NAME, pmu.NAME, sportinglife.NAME}


@dataclass
class SourceStatus:
    name: str
    ok: bool
    detail: str = ""
    seconds: float = 0.0
    runners: int = 0


@dataclass
class Analysis:
    identity: rs_paste.Identity
    card: RaceCard
    rows: list[M.Rated]
    sim: Optional[M.SimResult]
    notes: list[str]
    insights: list[str]
    status: list[SourceStatus] = field(default_factory=list)
    params: M.Params = field(default_factory=M.Params)
    seconds: float = 0.0


def _call(mod, day: date, venue: str, race_no: int, country: str, start_utc) -> RaceCard:
    if mod is betfair:
        return betfair.fetch(day, venue, race_no, country, start_utc)
    if mod is sportinglife:
        return sportinglife.fetch(day, venue, race_no, country)
    return mod.fetch(day, venue, race_no, country)


def local_start_utc(idn: rs_paste.Identity) -> Optional[datetime]:
    """Best-effort UTC start from the paste's local time, for Betfair market matching."""
    if not idn.start_time or not idn.race_date:
        return None
    tz = {"AUS": "Australia/Sydney", "FR": "Europe/Paris", "UK": "Europe/London",
          "IRE": "Europe/Dublin", "HK": "Asia/Hong_Kong", "SA": "Africa/Johannesburg",
          "USA": "America/New_York"}.get(idn.country)
    if not tz:
        return None
    try:
        from zoneinfo import ZoneInfo
        hh, mm = idn.start_time.split(":")
        return datetime(idn.race_date.year, idn.race_date.month, idn.race_date.day, int(hh), int(mm),
                        tzinfo=ZoneInfo(tz)).astimezone(timezone.utc)
    except Exception:  # noqa: BLE001
        return None


def gather(idn: rs_paste.Identity, paste_card: RaceCard, timeout: float = 75.0,
           progress: Optional[Callable[[str], None]] = None) -> tuple[RaceCard, list[SourceStatus]]:
    mods = PLAN.get(idn.country, [])
    status: list[SourceStatus] = []
    results: dict[str, RaceCard] = {}
    start_utc = local_start_utc(idn)
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=max(1, len(mods))) as pool:
        futs = {pool.submit(_call, m, idn.race_date, idn.venue, idn.race_no, idn.country, start_utc): m
                for m in mods}
        for fut in as_completed(futs, timeout=timeout):
            m = futs[fut]
            try:
                card = fut.result()
                results[m.NAME] = card
                status.append(SourceStatus(m.NAME, True, f"{len(card.field_)} runners",
                                           time.time() - t0, len(card.field_)))
                if progress:
                    progress(f"{m.NAME}: {len(card.field_)} runners")
            except SourceError as exc:
                status.append(SourceStatus(m.NAME, False, str(exc), time.time() - t0))
                if progress:
                    progress(f"{m.NAME}: {exc}")
            except Exception as exc:  # noqa: BLE001 - a layout change must not kill the page
                status.append(SourceStatus(m.NAME, False, f"{type(exc).__name__}: {exc}", time.time() - t0))
                if progress:
                    progress(f"{m.NAME}: failed ({type(exc).__name__})")
    # Base: the paste when it carries runners, else the first successful adapter in plan order.
    base = paste_card if paste_card.entries else None
    for m in mods:
        c = results.get(m.NAME)
        if c is None:
            continue
        if base is None:
            base = c
            if paste_card.sources:
                # keep the paste's header facts (they are the user's own view of the race)
                merge_cards(base, paste_card)
            continue
        prefer = MARKET_FIELDS if m.NAME in MARKET_AUTHORITY else ()
        merge_cards(base, c, prefer_fields=prefer)
    if base is None:
        base = paste_card
    base.country = base.country or idn.country
    base.venue = base.venue or idn.venue
    base.race_no = base.race_no or idn.race_no
    base.race_date = base.race_date or idn.race_date
    return base, status


def analyse(raw: str, country: str = "", venue: str = "", race_no: Optional[int] = None,
            race_day: Optional[date] = None, params: Optional[M.Params] = None,
            progress: Optional[Callable[[str], None]] = None, speed_raw: str = "") -> Analysis:
    t0 = time.time()
    idn, paste_card = rs_paste.parse_paste(raw or "")
    # The Speed Map page may be pasted in its own box or appended to the main paste.
    speed_rows: dict[int, rs_paste.SpeedRow] = {}
    speed_note = ""
    for text in (speed_raw or "", raw or ""):
        if text and rs_paste.has_speed_map(text):
            speed_rows = rs_paste.parse_speed_map(text)
            if speed_rows:
                sid = rs_paste.identity(text)
                if sid.venue and idn.venue and (sid.venue != idn.venue or (sid.race_no and idn.race_no
                                                                          and sid.race_no != idn.race_no)):
                    speed_note = (f"The Speed Map paste is for {sid.venue} race {sid.race_no}, not "
                                  f"{idn.venue} race {idn.race_no} - ignored.")
                    speed_rows = {}
                elif not idn.complete:
                    # a speed-map-only paste still identifies the race
                    idn.country = idn.country or sid.country
                    idn.venue = idn.venue or sid.venue
                    idn.race_no = idn.race_no or sid.race_no
                    idn.race_date = idn.race_date or sid.race_date
                break
    # typed fields fill whatever the paste did not carry
    idn.country = idn.country or country
    idn.venue = idn.venue or venue
    idn.race_no = idn.race_no or race_no
    idn.race_date = idn.race_date or race_day
    if not idn.complete:
        missing = [k for k, v in (("country", idn.country), ("venue", idn.venue),
                                  ("race number", idn.race_no), ("date", idn.race_date)) if not v]
        raise ValueError("Cannot identify the race: missing " + ", ".join(missing))
    paste_card.country = idn.country
    card, status = gather(idn, paste_card, progress=progress)
    if speed_rows:
        n = rs_paste.apply_speed_map(card, speed_rows)
        card.notes.append(f"Speed Map paste: AES/AFS pace values attached to {n} of "
                          f"{len(speed_rows)} runners; barriers taken from it.")
        if progress:
            progress(f"Speed Map paste: {n} runners matched")
    if speed_note:
        card.warnings.append(speed_note)
    p = params or M.defaults_for(idn.country)
    rows, notes = M.rate(card, p)
    sim = M.simulate(rows, p) if rows else None
    notes = notes + [n for n in card.notes if n not in notes]
    ins = M.insights(card, rows, notes) if rows else notes
    return Analysis(identity=idn, card=card, rows=rows, sim=sim, notes=notes, insights=ins,
                    status=status, params=p, seconds=time.time() - t0)


def rerate(a: Analysis, params: M.Params) -> Analysis:
    """Re-run the model on an already-fetched card (slider changes)."""
    rows, notes = M.rate(a.card, params)
    notes = notes + [n for n in a.card.notes if n not in notes]
    sim = M.simulate(rows, params) if rows else None
    return Analysis(identity=a.identity, card=a.card, rows=rows, sim=sim, notes=notes,
                    insights=M.insights(a.card, rows, notes) if rows else notes,
                    status=a.status, params=params, seconds=a.seconds)
