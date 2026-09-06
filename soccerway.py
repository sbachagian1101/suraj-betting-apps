"""Data layer for the Soccerway (Flashscore engine) predictor.

Everything here talks to the public endpoints that the Soccerway web pages
themselves use. No login, no browser, no API key.

* Team results / fixtures are embedded in the team page HTML as a
  ``~``-separated feed of ``KEY÷value¬`` records.
* Per-match stats (xG, shots, possession) come from the ``df_st_1_<id>`` feed.
* Line-ups, formations and player ratings come from a GraphQL endpoint
  (``_hash=dlie2``).
* Pre-match 1X2 odds come from the odds GraphQL endpoint (``_hash=ope2``).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

import requests

PROJECT_ID = "2035"                       # Soccerway (U.S.) project on the Flashscore engine
SITE = "https://us.soccerway.com"
FEED_HOST = f"https://global.flashscore.ninja/{PROJECT_ID}/x/feed"
GRAPHQL = f"https://{PROJECT_ID}.ds.lsapp.eu/pq_graphql"
ODDS_GRAPHQL = "https://global.ds.lsapp.eu/odds/pq_graphql"
BOOKMAKER_ID = "549"                      # bet365 (the site's default book)

HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                   "(KHTML, like Gecko) Chrome/128.0 Safari/537.36"),
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": SITE + "/",
    "Origin": SITE,
    "x-fsign": "SW9D1eZo",
}
TIMEOUT = 25

REC_SEP = "~"
FIELD_SEP = "¬"    # ¬
KV_SEP = "÷"       # ÷

TEAM_URL_RE = re.compile(r"soccerway\.com/team/([^/?#]+)/([A-Za-z0-9]{8})/?", re.I)


class SoccerwayError(RuntimeError):
    pass


# --------------------------------------------------------------------------- #
# Small data classes
# --------------------------------------------------------------------------- #
@dataclass
class Match:
    id: str
    kickoff: datetime
    competition: str
    home_id: str
    home_name: str
    away_id: str
    away_name: str
    home_score: Optional[int] = None
    away_score: Optional[int] = None
    stage: str = ""                     # AB code: 1 scheduled, 2 live, 3 finished

    @property
    def finished(self) -> bool:
        return self.stage == "3" and self.home_score is not None and self.away_score is not None

    @property
    def live(self) -> bool:
        return self.stage == "2"

    @property
    def url(self) -> str:
        return f"{SITE}/game/?mid={self.id}"

    def side_of(self, team_id: str) -> str:
        if team_id == self.home_id:
            return "H"
        if team_id == self.away_id:
            return "A"
        raise ValueError(f"{team_id} did not play in match {self.id}")


@dataclass
class Player:
    id: str
    name: str
    number: str
    role: str
    rating: Optional[float]
    starter: bool


@dataclass
class TeamLineup:
    side: str                            # "HOME" / "AWAY"
    name: str
    formation: str
    average_rating: Optional[float]
    players: list[Player] = field(default_factory=list)
    coach: str = ""

    @property
    def starters(self) -> list[Player]:
        return [p for p in self.players if p.starter]

    @property
    def published(self) -> bool:
        return len(self.starters) >= 11


# --------------------------------------------------------------------------- #
# HTTP helpers
# --------------------------------------------------------------------------- #
def _get(url: str, params: dict | None = None) -> requests.Response:
    try:
        r = requests.get(url, params=params, headers=HEADERS, timeout=TIMEOUT)
    except requests.RequestException as exc:
        raise SoccerwayError(f"Network error fetching {url}: {exc}") from exc
    if r.status_code != 200:
        raise SoccerwayError(f"HTTP {r.status_code} fetching {url}")
    return r


def parse_team_url(url: str) -> tuple[str, str]:
    """Return (slug, team_id) from a Soccerway team link."""
    m = TEAM_URL_RE.search(url.strip())
    if not m:
        raise SoccerwayError(
            "That does not look like a Soccerway team link. Expected something like "
            "https://us.soccerway.com/team/groningen/MBUGcjb9/")
    return m.group(1), m.group(2)


COUNTRY_TAG_RE = re.compile(r"\s*\([A-Za-z]{2,4}\)\s*$")


def clean_name(name: str) -> str:
    """Drop the '(Ned)' style country tag European fixtures carry."""
    return COUNTRY_TAG_RE.sub("", name).strip()


def parse_feed(text: str) -> list[dict]:
    """Split a Flashscore feed into a list of dicts."""
    records = []
    for chunk in text.split(REC_SEP):
        rec = {}
        for pair in chunk.split(FIELD_SEP):
            if KV_SEP in pair:
                k, v = pair.split(KV_SEP, 1)
                rec[k] = v
        if rec:
            records.append(rec)
    return records


# --------------------------------------------------------------------------- #
# Team results / fixtures (embedded in the team page)
# --------------------------------------------------------------------------- #
def _matches_from_html(html: str, team_id: str) -> list[Match]:
    out: list[Match] = []
    competition = ""
    for rec in parse_feed(html):
        if "ZA" in rec:
            competition = rec["ZA"]
        if "AA" not in rec or "PX" not in rec or "PY" not in rec:
            continue
        if team_id not in (rec["PX"], rec["PY"]):
            continue
        try:
            kickoff = datetime.fromtimestamp(int(rec["AD"]), tz=timezone.utc)
        except (KeyError, ValueError):
            continue

        def _int(key):
            v = rec.get(key, "")
            return int(v) if v.isdigit() else None

        out.append(Match(
            id=rec["AA"], kickoff=kickoff, competition=competition,
            home_id=rec["PX"], home_name=clean_name(rec.get("AE", "")),
            away_id=rec["PY"], away_name=clean_name(rec.get("AF", "")),
            home_score=_int("AG"), away_score=_int("AH"),
            stage=rec.get("AB", ""),
        ))
    # de-duplicate (the page can embed the same match twice)
    seen, uniq = set(), []
    for m in out:
        if m.id not in seen:
            seen.add(m.id)
            uniq.append(m)
    return uniq


def team_results(slug: str, team_id: str) -> list[Match]:
    """Finished matches, most recent first."""
    html = _get(f"{SITE}/team/{slug}/{team_id}/results/").text
    ms = [m for m in _matches_from_html(html, team_id) if m.finished]
    ms.sort(key=lambda m: m.kickoff, reverse=True)
    return ms


def team_fixtures(slug: str, team_id: str) -> list[Match]:
    """Upcoming (or in-play) matches, soonest first."""
    html = _get(f"{SITE}/team/{slug}/{team_id}/fixtures/").text
    ms = [m for m in _matches_from_html(html, team_id) if not m.finished]
    ms.sort(key=lambda m: m.kickoff)
    return ms


def find_fixture(slug_a: str, id_a: str, id_b: str) -> Optional[Match]:
    """The next match between the two teams, if Soccerway lists one."""
    for m in team_fixtures(slug_a, id_a):
        if id_b in (m.home_id, m.away_id):
            return m
    return None


# --------------------------------------------------------------------------- #
# Match stats (xG etc.)
# --------------------------------------------------------------------------- #
def _num(s: str) -> Optional[float]:
    m = re.match(r"\s*([0-9]+(?:\.[0-9]+)?)", s or "")
    return float(m.group(1)) if m else None


def match_stats(event_id: str) -> dict:
    """Full-time stats as {label: (home, away)} with numeric parsing.

    Returns an empty dict when the provider has no stats for the match.
    """
    text = _get(f"{FEED_HOST}/df_st_1_{event_id}").text
    stats: dict[str, tuple[Optional[float], Optional[float]]] = {}
    section = None
    for rec in parse_feed(text):
        if "SE" in rec:
            section = rec["SE"]
        if section != "Game":
            continue
        if "SG" in rec:
            stats[rec["SG"]] = (_num(rec.get("SH", "")), _num(rec.get("SI", "")))
    return stats


def xg_from_stats(stats: dict) -> tuple[Optional[float], Optional[float]]:
    for label, val in stats.items():
        if label.lower().startswith("expected goals"):
            return val
    return (None, None)


# --------------------------------------------------------------------------- #
# Line-ups and ratings
# --------------------------------------------------------------------------- #
def match_lineups(event_id: str) -> dict[str, TeamLineup]:
    """{"HOME": TeamLineup, "AWAY": TeamLineup}. Missing sides are omitted."""
    r = _get(GRAPHQL, {"_hash": "dlie2", "eventId": event_id, "projectId": PROJECT_ID})
    try:
        ev = r.json()["data"]["findEventById"]
    except (ValueError, KeyError, TypeError) as exc:
        raise SoccerwayError(f"Unexpected line-up payload for {event_id}") from exc
    out: dict[str, TeamLineup] = {}
    for ep in (ev or {}).get("eventParticipants") or []:
        side = (ep.get("type") or {}).get("side", "")
        lineup = ep.get("lineup") or {}
        starters = set()
        for g in lineup.get("groups") or []:
            if g.get("groupType") == "STARTERS":
                starters.update(g.get("playerIds") or [])
        players = []
        for p in lineup.get("players") or []:
            rating = p.get("rating") or {}
            titles = [r.get("title", "") for r in (p.get("playerRoles") or [])]
            captain = "Captain" in titles
            position = next((t for t in titles if t and t != "Captain"), "")
            players.append(Player(
                id=p.get("id", ""),
                name=(p.get("listName") or p.get("fieldName") or "") + (" (C)" if captain else ""),
                number=p.get("number") or "",
                role=position,
                rating=_num(str(rating.get("value", ""))) if rating else None,
                starter=p.get("id") in starters,
            ))
        coaches = (lineup.get("coaches") or {}).get("players") or []
        out[side] = TeamLineup(
            side=side,
            name=ep.get("name", ""),
            formation=((lineup.get("formation") or {}).get("name") or ""),
            average_rating=_num(str(ep.get("averageRating") or "")),
            players=players,
            coach=coaches[0].get("listName", "") if coaches else "",
        )
    return out


# --------------------------------------------------------------------------- #
# Odds
# --------------------------------------------------------------------------- #
def match_odds(event_id: str) -> Optional[dict]:
    """Pre-match 1X2 odds from the site's default bookmaker, or None."""
    try:
        r = _get(ODDS_GRAPHQL, {"_hash": "ope2", "eventId": event_id,
                                "bookmakerId": BOOKMAKER_ID,
                                "betType": "HOME_DRAW_AWAY", "betScope": "FULL_TIME"})
        d = r.json()["data"]["findPrematchOddsForBookmaker"]
    except (SoccerwayError, ValueError, KeyError, TypeError):
        return None
    if not d:
        return None
    out = {}
    for k in ("home", "draw", "away"):
        item = d.get(k) or {}
        out[k] = _num(str(item.get("value", "")))
        out[k + "_open"] = _num(str(item.get("opening", "")))
    if not all(out[k] for k in ("home", "draw", "away")):
        return None
    return out


# --------------------------------------------------------------------------- #
# One past match seen from a given team's point of view
# --------------------------------------------------------------------------- #
@dataclass
class MatchRecord:
    match: Match
    side: str                          # "H" or "A"
    goals_for: int
    goals_against: int
    xg_for: Optional[float]
    xg_against: Optional[float]
    team_rating: Optional[float]
    formation: str
    starters: list[Player]
    opponent_rating: Optional[float]

    @property
    def result(self) -> str:
        if self.goals_for > self.goals_against:
            return "W"
        if self.goals_for < self.goals_against:
            return "L"
        return "D"

    @property
    def opponent(self) -> str:
        return self.match.away_name if self.side == "H" else self.match.home_name

    @property
    def score(self) -> str:
        return f"{self.match.home_score}-{self.match.away_score}"


def build_record(match: Match, team_id: str, stats: dict, lineups: dict) -> MatchRecord:
    side = match.side_of(team_id)
    xg_h, xg_a = xg_from_stats(stats)
    own = lineups.get("HOME" if side == "H" else "AWAY")
    opp = lineups.get("AWAY" if side == "H" else "HOME")
    return MatchRecord(
        match=match, side=side,
        goals_for=match.home_score if side == "H" else match.away_score,
        goals_against=match.away_score if side == "H" else match.home_score,
        xg_for=xg_h if side == "H" else xg_a,
        xg_against=xg_a if side == "H" else xg_h,
        team_rating=own.average_rating if own else None,
        formation=own.formation if own else "",
        starters=own.starters if own else [],
        opponent_rating=opp.average_rating if opp else None,
    )
