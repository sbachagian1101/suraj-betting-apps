"""Ladbrokes Australia affiliate feed (no login).

Covers Australian meetings fully and prices most UK, Irish, US and French
meetings.  Per runner it carries barrier, weight, jockey, trainer, gear, the
last-20 form string, the jockey's and trainer's recent records, a speed-map
label with settling position, and every price fluctuation with a timestamp,
which is what the steamer/drifter analysis is built on.
"""
from __future__ import annotations

from datetime import date, datetime, timezone
from typing import Optional

from common import (Entry, Fluc, RaceCard, SourceError, cached, get_json,
                    venue_matches)

NAME = "Ladbrokes AU"
BASE = "https://api.ladbrokes.com.au"
COUNTRY_CODES = {"AUS": {"AUS"}, "UK": {"UK", "GBR", "GB"}, "IRE": {"IRE", "IRL"},
                 "USA": {"USA", "US"}, "FR": {"FRA", "FR"}, "HK": {"HK", "HKG"},
                 "SA": {"ZAF", "RSA", "SA"}}


def meetings(day: date) -> list[dict]:
    def fetch():
        data = get_json(f"{BASE}/affiliates/v1/racing/meetings",
                        {"date_from": day.isoformat(), "date_to": day.isoformat(), "enc": "json"})
        return (data or {}).get("data", {}).get("meetings", []) or []
    return cached(("lb_meetings", day.isoformat()), 600, fetch)


def find_race(day: date, venue: str, race_no: int, country: str) -> Optional[dict]:
    codes = COUNTRY_CODES.get(country, {country})
    best = None
    for m in meetings(day):
        if m.get("category") != "T":
            continue
        if m.get("country") not in codes and country:
            continue
        if not venue_matches(m.get("name", ""), venue):
            continue
        for r in m.get("races", []):
            if int(r.get("race_number") or 0) == race_no:
                best = {"meeting": m, "race": r}
                return best
    return best


def _stats(d: dict | None) -> Optional[tuple[int, int, int]]:
    if not d:
        return None
    s = int(d.get("number_of_starts") or 0)
    w = int(d.get("number_of_wins") or 0)
    p = w + int(d.get("number_of_seconds") or 0) + int(d.get("number_of_thirds") or 0)
    if not p:
        p = int(d.get("number_of_placings") or 0)
    return (s, w, p) if s else None


def _ts(s: str | None) -> Optional[datetime]:
    if not s:
        return None
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00")[:26] + "+00:00"
                                      if "+" not in s.replace("Z", "") else s.replace("Z", "+00:00"))
    except ValueError:
        try:
            return datetime.strptime(s[:19], "%Y-%m-%dT%H:%M:%S").replace(tzinfo=timezone.utc)
        except ValueError:
            return None


def card_from_event(ev: dict, meeting: dict | None = None) -> RaceCard:
    race = ev.get("race") or {}
    import re as _re
    going = str(race.get("track_condition") or "")
    going = _re.sub(r"([A-Za-z])(\d)", r"\1 \2", going)          # "Good4" -> "Good 4"
    card = RaceCard(venue=race.get("meeting_name", ""), race_no=race.get("race_number"),
                    name=race.get("description", ""), dist_m=race.get("distance"),
                    going=going, weather=race.get("weather", ""),
                    rail=race.get("rail_position", ""), sources=[NAME])
    st = race.get("advertised_start")
    if st:
        card.extras["start_utc"] = datetime.fromtimestamp(int(st), tz=timezone.utc)
    if race.get("comment"):
        card.extras["preview"] = race["comment"]
    pm = race.get("prize_monies") or {}
    if isinstance(pm, dict):
        tot = pm.get("total_value") or pm.get("total")
        try:
            card.prize = float(tot) if tot else None
        except (TypeError, ValueError):
            pass
    tips = race.get("tips") or []
    tip_rank = {t: i + 1 for i, t in enumerate(tips) if isinstance(t, str)}
    fav = (ev.get("favourite") or {}).get("entrant_id")
    for r in ev.get("runners", []):
        e = Entry(number=r.get("runner_number"), name=r.get("name", ""), barrier=r.get("barrier"),
                  jockey=r.get("jockey") or "", trainer=r.get("trainer_name") or "",
                  age=r.get("age"), sex=r.get("sex") or "", scratched=bool(r.get("is_scratched")),
                  form_string=r.get("last_twenty_starts") or "", gear=r.get("gear") or "",
                  comment=r.get("form_comment") or "", sources=[NAME])
        w = (r.get("weight") or {}).get("allocated") or (r.get("weight") or {}).get("total")
        try:
            e.weight = float(w) if w else None
        except (TypeError, ValueError):
            pass
        try:
            e.prize_money = float(r.get("prize_money")) if r.get("prize_money") else None
        except (TypeError, ValueError):
            pass
        odds = r.get("odds") or {}
        if odds.get("fixed_win") and odds["fixed_win"] > 1:
            e.odds = float(odds["fixed_win"])
        if odds.get("fixed_place") and odds["fixed_place"] > 1:
            e.place_odds = float(odds["fixed_place"])
        fw = r.get("flucs_with_timestamp") or {}
        op = (fw.get("open") or {}).get("fluc")
        if op and op > 1:
            e.odds_open = float(op)
        elif r.get("flucs"):
            e.odds_open = float(r["flucs"][0]) if r["flucs"][0] and r["flucs"][0] > 1 else None
        for f in fw.get("last_six") or []:
            if f.get("fluc") and f["fluc"] > 1:
                e.flucs.append(Fluc(_ts(f.get("timestamp")), float(f["fluc"])))
        if not e.flucs and r.get("flucs"):
            e.flucs = [Fluc(None, float(x)) for x in r["flucs"] if x and x > 1]
        sm = r.get("speedmap") or {}
        e.speed_label = sm.get("label") or ""
        if sm.get("settling_lengths") is not None:
            try:
                e.settling = float(sm["settling_lengths"])
            except (TypeError, ValueError):
                pass
        jp = r.get("jockey_past_performances") or {}
        e.jockey_stats = _stats(jp.get("last_50_starts"))
        e.trainer_stats = _stats(jp.get("trainer"))
        pp = r.get("past_performances") or {}
        e.combo_stats = _stats(pp.get("jockey"))
        if r.get("entrant_id") in tip_rank:
            e.tip_rank = tip_rank[r["entrant_id"]]
        e.extras["ladbrokes_favourite"] = r.get("entrant_id") == fav
        e.extras["ladbrokes_mover"] = bool(r.get("mover"))
        e.extras["class_level"] = r.get("class_level") or ""
        card.entries.append(e)
    if meeting:
        card.extras["state"] = meeting.get("state")
    return card


def fetch(day: date, venue: str, race_no: int, country: str = "AUS") -> RaceCard:
    hit = find_race(day, venue, race_no, country)
    if not hit:
        raise SourceError(f"{NAME}: no {venue} race {race_no} on {day}")
    rid = hit["race"]["id"]
    ev = cached(("lb_event", rid), 60,
                lambda: get_json(f"{BASE}/affiliates/v1/racing/events/{rid}", {"enc": "json"}))
    data = (ev or {}).get("data") or {}
    if not data.get("runners"):
        raise SourceError(f"{NAME}: race found but no runners returned")
    card = card_from_event(data, hit["meeting"])
    card.race_date = day
    return card
