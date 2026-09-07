"""Betfair Exchange read-only website feed (no account needed).

The JSON the betfair.com exchange pages call for logged-out visitors.  Used for
the countries whose own sites carry no prices (Hong Kong, South Africa) and as a
second opinion elsewhere.  Discovery: navigation facet search for horse-racing
WIN markets in a time window, then a by-event call for country, venue, start
time and cloth numbers; prices come from the by-market call only.
"""
from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from typing import Optional

from common import (Entry, RaceCard, SourceError, cached, get_json, post_json,
                    venue_matches)

NAME = "Betfair Exchange"
APP_KEY = "nzIFcwyWhrlwYMrh"          # public key embedded in the betfair.com website
NAV = "https://www.betfair.com/www/sports/navigation/facet/v1/search"
ERO = "https://ero.betfair.com/www/sports/exchange/readonly/v1"
COUNTRY = {"AUS": "AU", "FR": "FR", "UK": "GB", "IRE": "IE", "HK": "HK", "SA": "ZA", "USA": "US"}
COMMON = {"_ak": APP_KEY, "alt": "json", "currencyCode": "GBP", "locale": "en"}


def _iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.000Z")


def _event_ids(day: date) -> list[int]:
    start = datetime.combine(day, datetime.min.time(), tzinfo=timezone.utc) - timedelta(hours=14)
    end = start + timedelta(hours=14 + 24 + 12)
    body = {
        "filter": {
            "marketBettingTypes": ["ODDS"], "productTypes": ["EXCHANGE"], "marketTypeCodes": ["WIN"],
            "selectBy": "RANK", "contentGroup": {"language": "en", "regionCode": "UK"},
            "maxResults": 0, "eventTypeIds": [7],
            "marketStartTime": {"from": _iso(start), "to": _iso(end)},
        },
        "facets": [{"type": "EVENT_TYPE", "skipValues": 0, "maxValues": 10,
                    "next": {"type": "EVENT", "skipValues": 0, "maxValues": 400}}],
        "currencyCode": "GBP", "locale": "en_GB",
    }
    data = post_json(NAV, body, params={"_ak": APP_KEY, "alt": "json"})
    ids: list[int] = []
    for et in data.get("facets", [{}])[0].get("values", []):
        for ev in (et.get("next") or {}).get("values", []):
            ids.append(int(ev["key"]["eventId"]))
    return ids


def _byevent(event_ids) -> dict:
    return get_json(f"{ERO}/byevent", {**COMMON, "eventIds": ",".join(map(str, event_ids)),
                                        "types": "EVENT,MARKET_DESCRIPTION,RUNNER_DESCRIPTION,RUNNER_METADATA"},
                    retries=0) or {}


def markets(day: date) -> list[dict]:
    """Every WIN market in the window: {market_id, country, venue, start, name, runners}."""
    def fetch():
        ids = _event_ids(day)
        out: list[dict] = []
        for i in range(0, len(ids), 3):
            chunk = ids[i:i + 3]
            try:
                payloads = [_byevent(chunk)]
            except SourceError:
                payloads = []
                for one in chunk:
                    try:
                        payloads.append(_byevent([one]))
                    except SourceError:
                        continue
            for data in payloads:
                for et in data.get("eventTypes", []):
                    for node in et.get("eventNodes", []):
                        ev = node.get("event") or {}
                        for m in node.get("marketNodes", []):
                            d = m.get("description") or {}
                            if d.get("marketType") != "WIN":
                                continue
                            runners = {}
                            for r in m.get("runners", []):
                                meta = (r.get("description") or {}).get("metadata") or {}
                                num = meta.get("CLOTH_NUMBER")
                                runners[str(r["selectionId"])] = {
                                    "name": (r.get("description") or {}).get("runnerName", ""),
                                    "number": int(num) if num and str(num).isdigit() else None}
                            out.append({"market_id": m["marketId"], "country": ev.get("countryCode"),
                                        "venue": ev.get("eventName", ""), "name": d.get("marketName", ""),
                                        "start": datetime.fromisoformat(d["marketTime"].replace("Z", "+00:00")),
                                        "runners": runners})
        out.sort(key=lambda x: x["start"])
        return out
    return cached(("bf_markets", day.isoformat()), 600, fetch)


def find_market(day: date, venue: str, race_no: int, country: str,
                start_utc: Optional[datetime] = None) -> Optional[dict]:
    cc = COUNTRY.get(country)
    cands = [m for m in markets(day) if (not cc or m["country"] == cc) and venue_matches(m["venue"], venue)]
    if not cands:
        return None
    if start_utc is not None:
        best = min(cands, key=lambda m: abs((m["start"] - start_utc).total_seconds()))
        if abs((best["start"] - start_utc).total_seconds()) <= 600:
            return best
    # Betfair names markets 'R2 1200m Mdn' in AU; elsewhere fall back to time order
    for m in cands:
        if m["name"].upper().startswith(f"R{race_no} "):
            return m
    # group by meeting date (local), then ordinal
    by_day: dict[date, list[dict]] = {}
    for m in cands:
        by_day.setdefault(m["start"].date(), []).append(m)
    for _, ms in sorted(by_day.items()):
        if 1 <= race_no <= len(ms):
            return ms[race_no - 1]
    return None


def prices(market: dict) -> dict:
    data = get_json(f"{ERO}/bymarket", {**COMMON, "marketIds": market["market_id"],
                                         "rollupLimit": 10, "rollupModel": "STAKE",
                                         "types": "MARKET_STATE,RUNNER_DESCRIPTION,RUNNER_STATE,RUNNER_EXCHANGE_PRICES_BEST"})
    out = {}
    for et in (data or {}).get("eventTypes", []):
        for node in et.get("eventNodes", []):
            for m in node.get("marketNodes", []):
                if m.get("marketId") != market["market_id"]:
                    continue
                for r in m.get("runners", []):
                    sel = str(r["selectionId"])
                    ex = r.get("exchange") or {}
                    st = r.get("state") or {}
                    back = (ex.get("availableToBack") or [{}])[0].get("price")
                    lay = (ex.get("availableToLay") or [{}])[0].get("price")
                    info = market["runners"].get(sel, {})
                    out[sel] = {"name": info.get("name") or (r.get("description") or {}).get("runnerName", ""),
                                "number": info.get("number"), "back": back, "lay": lay,
                                "ltp": st.get("lastPriceTraded"),
                                "active": st.get("status") in (None, "ACTIVE")}
    return out


def fetch(day: date, venue: str, race_no: int, country: str,
          start_utc: Optional[datetime] = None) -> RaceCard:
    mk = find_market(day, venue, race_no, country, start_utc)
    if not mk:
        raise SourceError(f"{NAME}: no {venue} market for race {race_no}")
    px = prices(mk)
    card = RaceCard(country=country, venue=mk["venue"], race_no=race_no, race_date=day,
                    sources=[NAME], extras={"betfair_market": mk["market_id"],
                                            "start_utc": mk["start"]})
    for sel, p in px.items():
        e = Entry(number=p["number"], name=p["name"], sources=[NAME], scratched=not p["active"])
        back, lay, ltp = p["back"], p["lay"], p["ltp"]
        # An empty market shows 1.01 to back and 1000 to lay; a mid of that is fiction.
        # Prefer the last traded price, then a tight back/lay spread, else nothing.
        mid = None
        if ltp and ltp > 1:
            mid = ltp
        elif back and lay and back > 1 and lay / back <= 1.25:
            mid = 2.0 / (1.0 / back + 1.0 / lay)
        if mid and mid > 1:
            e.exchange_odds = round(mid, 2)
            e.extras["betfair"] = {"back": back, "lay": lay, "ltp": ltp}
        card.entries.append(e)
    if not card.entries:
        raise SourceError(f"{NAME}: market {mk['market_id']} returned no runners")
    return card
