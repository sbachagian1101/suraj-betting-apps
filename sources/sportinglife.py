"""Sporting Life racecard JSON (UK and Ireland, plus some French cards).

``/api/horse-racing/racing/racecards/{date}`` lists meetings with going and
surface; ``/api/horse-racing/race/{id}`` carries each ride's draw, weight,
official rating, headgear, jockey claim, trainer, jockey, Sky Bet price with
history, a commentary line, lifetime stats and the horse's previous results.
Previous results carry position and field size but not beaten lengths, so the
model estimates a margin from the finishing position and flags it as such.
"""
from __future__ import annotations

import re
from datetime import date
from typing import Optional

from common import (Entry, Fluc, PastRun, RaceCard, SourceError, cached, frac_to_decimal,
                    get_json, imperial_to_m, venue_matches)

NAME = "Sporting Life"
BASE = "https://www.sportinglife.com/api/horse-racing"
COUNTRY_CODES = {"UK": {"ENG", "SCOT", "WAL", "GB", "UK"}, "IRE": {"EIRE", "IRE", "IRL"},
                 "FR": {"FR", "FRA"}, "USA": {"USA", "US"}, "HK": {"HK"}, "SA": {"SA", "RSA", "ZAF"}}


def racecards(day: date) -> list[dict]:
    return cached(("sl_cards", day.isoformat()), 600,
                  lambda: get_json(f"{BASE}/racing/racecards/{day.isoformat()}") or [])


def find_race(day: date, venue: str, race_no: int, country: str,
              start_time: str = "") -> Optional[tuple[dict, dict]]:
    codes = COUNTRY_CODES.get(country, set())
    for m in racecards(day):
        ms = m.get("meeting_summary") or {}
        course = ms.get("course") or {}
        cc = ((course.get("country") or {}).get("short_name") or "").upper()
        if codes and cc not in codes and country in ("UK", "IRE"):
            # ENG/SCOT/WAL vs EIRE - keep strict so Perth (Scotland) != Perth (AUS)
            continue
        if not venue_matches(course.get("name", ""), venue):
            continue
        races = sorted(m.get("races", []), key=lambda r: r.get("time") or "")
        if start_time:
            for r in races:
                if (r.get("time") or "")[:5] == start_time[:5]:
                    return m, r
        if 1 <= race_no <= len(races):
            return m, races[race_no - 1]
    return None


def stones_to_kg(text: str) -> Optional[float]:
    m = re.match(r"^\s*(\d+)-(\d+)\s*$", text or "")
    if not m:
        return None
    return round((int(m.group(1)) * 14 + int(m.group(2))) * 0.45359237, 1)


def estimated_margin(pos: Optional[int], field_size: Optional[int]) -> Optional[float]:
    """No beaten lengths in this feed: a typical flat-race spacing by position."""
    if not pos:
        return None
    if pos == 1:
        return -0.75
    return round(1.4 * (pos - 1) ** 0.9, 2)


def _going_surface(g: str) -> str:
    g = (g or "").lower()
    if "standard" in g or "all-weather" in g or "all weather" in g:
        return "AW"
    return "TURF"


def _runs(horse: dict, race_day: date) -> list[PastRun]:
    out: list[PastRun] = []
    for r in horse.get("previous_results") or []:
        try:
            d = date.fromisoformat(r.get("date", ""))
        except ValueError:
            d = None
        pos = r.get("position")
        cas = r.get("casualty") or {}
        run = PastRun(run_date=d, days_ago=(race_day - d).days if d else None,
                      track=r.get("course_shortcode") or r.get("course_name", ""),
                      race_name=r.get("race_name", ""), race_class=str(r.get("race_class") or ""),
                      dist_m=imperial_to_m(r.get("distance", "")), going=r.get("going", ""),
                      surface=_going_surface(r.get("going", "")), pos=pos if pos else None,
                      field_size=r.get("runner_count"), weight=stones_to_kg(r.get("weight", "")),
                      sp=frac_to_decimal(r.get("odds", "")), comment=r.get("ride_description", ""),
                      rating=r.get("bha"), currency="GBP")
        if not pos or cas:
            run.non_finish = True
            run.comment = (cas.get("name") or cas.get("type") or "") + " " + run.comment
        else:
            run.margin_l = estimated_margin(pos, run.field_size)
        out.append(run)
    return out


def card_from_race(rj: dict, meeting: dict | None, race_day: date, country: str) -> RaceCard:
    rs = rj.get("race_summary") or {}
    ms = (meeting or {}).get("meeting_summary") or {}
    card = RaceCard(country=country, venue=rs.get("course_name", ""), name=rs.get("name", ""),
                    dist_m=imperial_to_m(rs.get("distance", "")),
                    surface=((rs.get("course_surface") or {}).get("surface") or ms.get("surface_summary") or "").upper(),
                    going=ms.get("going", ""), race_class=f"Class {rs.get('race_class')}" if rs.get("race_class") else "",
                    start_time=(rs.get("time") or "")[:5], race_date=race_day, currency="GBP",
                    sources=[NAME])
    if rs.get("age"):
        card.extras["age"] = rs["age"]
    if rs.get("verdict"):
        card.extras["preview"] = re.sub(r"<[^>]+>", "", rs["verdict"])
    if rj.get("betting_forecast"):
        card.extras["forecast"] = rj["betting_forecast"]
    prizes = rj.get("prizes") or {}
    plist = prizes.get("prize") if isinstance(prizes, dict) else prizes
    vals = []
    for p in plist or []:
        raw = str(p.get("prize", "") if isinstance(p, dict) else p)
        m = re.search(r"([\d.]+)", raw.replace(",", ""))
        if m:
            vals.append(float(m.group(1)))
    if vals:
        card.prize = max(vals)
    for ride in rj.get("rides", []):
        horse = ride.get("horse") or {}
        e = Entry(number=ride.get("cloth_number"), name=horse.get("name", ""),
                  barrier=ride.get("draw_number") or None, weight=stones_to_kg(ride.get("handicap", "")),
                  jockey=(ride.get("jockey") or {}).get("name", ""),
                  trainer=(ride.get("trainer") or {}).get("name", ""), age=horse.get("age"),
                  sex=str((horse.get("sex") or {}).get("type", "") if isinstance(horse.get("sex"), dict)
                          else horse.get("sex") or "").upper(),
                  official_rating=ride.get("official_rating"),
                  last_run_days=horse.get("last_ran_days"),
                  form_string=((horse.get("formsummary") or {}).get("display_text") or "").replace("-", ""),
                  comment=ride.get("commentary") or "", sources=[NAME],
                  scratched=(ride.get("ride_status") or "RUNNER") != "RUNNER")
        hg = ride.get("headgear") or []
        e.gear = ", ".join(h.get("name", "") for h in hg if isinstance(h, dict))
        if ride.get("jockey_claim"):
            e.extras["claim_lb"] = ride["jockey_claim"]
        stats = ride.get("horse_lifetime_stats") or []
        if stats:
            s = stats[0]
            try:
                e.career = (int(s.get("run_count") or 0), int(s.get("win_count") or 0),
                            int(s.get("place_count") or 0), 0)
            except (TypeError, ValueError):
                pass
        bet = ride.get("betting") or {}
        e.odds = frac_to_decimal(bet.get("current_odds", ""))
        hist = bet.get("historical_odds") or []
        for h in hist:
            if isinstance(h, dict):
                px = frac_to_decimal(str(h.get("odds") or h.get("price") or ""))
                if px:
                    e.flucs.append(Fluc(None, px))
        if e.flucs and e.odds_open is None:
            e.odds_open = e.flucs[0].price
        e.extras["insights"] = [i.get("type", "") for i in ride.get("insights") or [] if isinstance(i, dict)]
        e.runs = _runs(horse, race_day)
        if e.runs and e.last_run_days is None:
            days = [r.days_ago for r in e.runs if r.days_ago is not None]
            e.last_run_days = min(days) if days else None
        card.entries.append(e)
    return card


def fetch(day: date, venue: str, race_no: int, country: str = "UK", start_time: str = "") -> RaceCard:
    hit = find_race(day, venue, race_no, country, start_time)
    if not hit:
        raise SourceError(f"{NAME}: no {venue} race {race_no} on {day}")
    m, r = hit
    rid = (r.get("race_summary_reference") or {}).get("id")
    rj = cached(("sl_race", rid), 120, lambda: get_json(f"{BASE}/race/{rid}") or {})
    if not rj.get("rides"):
        raise SourceError(f"{NAME}: race {rid} has no rides yet")
    card = card_from_race(rj, m, day, country)
    card.race_no = race_no
    return card
