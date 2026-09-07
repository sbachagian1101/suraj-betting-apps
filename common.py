"""Shared types, HTTP session, cache and name matching for every source.

Every country adapter returns the same ``RaceCard`` / ``Entry`` / ``PastRun``
shapes so the model never has to know where a number came from.  Fields a
source cannot supply stay ``None``; the model treats a missing value as *no
evidence*, never as zero ability.
"""
from __future__ import annotations

import re
import threading
import time
import unicodedata
from dataclasses import dataclass, field
from datetime import date, datetime
from difflib import SequenceMatcher
from typing import Callable, Optional

import requests

UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/128.0 Safari/537.36")

COUNTRIES = {
    "Australia": "AUS", "France": "FR", "United Kingdom": "UK", "Ireland": "IRE",
    "Hong Kong": "HK", "South Africa": "SA", "USA": "USA",
}
CODE_TO_COUNTRY = {v: k for k, v in COUNTRIES.items()}
CURRENCY = {"AUS": "AUD", "FR": "EUR", "UK": "GBP", "IRE": "EUR", "HK": "HKD",
            "SA": "ZAR", "USA": "USD"}


class SourceError(Exception):
    """A source could not deliver (network, layout change, race not found)."""


# --- HTTP ---------------------------------------------------------------------

_session = requests.Session()
_session.headers.update({"User-Agent": UA, "Accept-Language": "en-GB,en;q=0.9",
                         "Accept": "application/json, text/html;q=0.9, */*;q=0.8"})


def get_text(url: str, params: dict | None = None, timeout: int = 30, retries: int = 1,
             **kw) -> str:
    last: Exception | None = None
    for attempt in range(retries + 1):
        try:
            r = _session.get(url, params=params, timeout=timeout, **kw)
            r.raise_for_status()
            return r.text
        except requests.RequestException as exc:
            last = exc
            time.sleep(1.0 * (attempt + 1))
    raise SourceError(f"GET {url} failed: {last}")


def get_json(url: str, params: dict | None = None, timeout: int = 30, retries: int = 1, **kw):
    last: Exception | None = None
    for attempt in range(retries + 1):
        try:
            r = _session.get(url, params=params, timeout=timeout, **kw)
            r.raise_for_status()
            if not r.content.strip():
                return None
            return r.json()
        except (requests.RequestException, ValueError) as exc:
            last = exc
            time.sleep(1.0 * (attempt + 1))
    raise SourceError(f"GET {url} failed: {last}")


def post_json(url: str, body: dict, params: dict | None = None, timeout: int = 30):
    try:
        r = _session.post(url, params=params, json=body, timeout=timeout)
        r.raise_for_status()
        return r.json()
    except (requests.RequestException, ValueError) as exc:
        raise SourceError(f"POST {url} failed: {exc}") from exc


# --- tiny TTL cache (thread-safe, usable from worker threads) ------------------

_cache: dict[tuple, tuple[float, object]] = {}
_lock = threading.Lock()


def cached(key: tuple, ttl: float, fn: Callable):
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


# --- names --------------------------------------------------------------------

def ascii_fold(text: str) -> str:
    return unicodedata.normalize("NFKD", text or "").encode("ascii", "ignore").decode()


def norm_name(name: str) -> str:
    """Normalise a horse or venue name for cross-source matching: fold accents,
    drop country tags like (IRE)/(FR)/(AUS), upper-case, collapse punctuation."""
    text = ascii_fold(name).upper()
    text = re.sub(r"\(.*?\)", " ", text)
    text = re.sub(r"[^A-Z0-9 ]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def similarity(a: str, b: str) -> float:
    a, b = norm_name(a), norm_name(b)
    if not a or not b:
        return 0.0
    if a == b:
        return 1.0
    if a.startswith(b) or b.startswith(a):
        return 0.95
    return SequenceMatcher(None, a, b).ratio()


def venue_matches(a: str, b: str) -> bool:
    """Loose venue equality: 'Grafton' ~ 'Grafton (NSW)', 'Saint-Cloud' ~ 'SAINT-CLOUD',
    'Deauville' ~ 'DEAUVILLE', 'Happy Valley' ~ 'HV'."""
    na, nb = norm_name(a), norm_name(b)
    if not na or not nb:
        return False
    if na == nb or na.startswith(nb) or nb.startswith(na):
        return True
    # first word match for 'Wolverhampton (AW)' vs 'Wolverhampton'
    if na.split()[0] == nb.split()[0] and len(na.split()[0]) >= 5:
        return True
    return SequenceMatcher(None, na, nb).ratio() >= 0.86


# --- data model ---------------------------------------------------------------

@dataclass
class PastRun:
    run_date: Optional[date] = None
    days_ago: Optional[int] = None
    track: str = ""
    race_name: str = ""
    race_class: str = ""
    prize: Optional[float] = None       # in the race's own currency
    currency: str = ""
    dist_m: Optional[int] = None
    surface: str = ""                   # TURF / AW / DIRT / SAND
    going: str = ""
    pos: Optional[int] = None
    field_size: Optional[int] = None
    margin_l: Optional[float] = None    # lengths beaten (+), or won by (-)
    weight: Optional[float] = None
    barrier: Optional[int] = None
    jockey: str = ""
    sp: Optional[float] = None          # decimal starting price
    time_s: Optional[float] = None
    sectional: Optional[float] = None   # last-600m or source-specific split, seconds
    running_pos: str = ""
    comment: str = ""
    non_finish: bool = False
    rating: Optional[float] = None      # official/handicap rating carried that day

    @property
    def usable(self) -> bool:
        return (not self.non_finish and self.margin_l is not None
                and self.dist_m is not None and self.days_ago is not None)


@dataclass
class Fluc:
    when: Optional[datetime]
    price: float


@dataclass
class Entry:
    number: Optional[int] = None
    name: str = ""
    barrier: Optional[int] = None
    weight: Optional[float] = None
    jockey: str = ""
    trainer: str = ""
    age: Optional[int] = None
    sex: str = ""
    scratched: bool = False
    career: tuple[int, int, int, int] = (0, 0, 0, 0)   # starts, wins, 2nds, 3rds
    prize_money: Optional[float] = None
    form_string: str = ""
    last_run_days: Optional[int] = None
    official_rating: Optional[float] = None            # OR / HK rating / WF merit
    records: dict[str, tuple[int, int, int, int]] = field(default_factory=dict)
    runs: list[PastRun] = field(default_factory=list)
    odds: Optional[float] = None
    odds_open: Optional[float] = None
    place_odds: Optional[float] = None
    flucs: list[Fluc] = field(default_factory=list)
    exchange_odds: Optional[float] = None
    jockey_stats: Optional[tuple[int, int, int]] = None    # starts, wins, places
    trainer_stats: Optional[tuple[int, int, int]] = None
    combo_stats: Optional[tuple[int, int, int]] = None
    speed_label: str = ""                                  # Leader / On pace / Midfield / Backmarker
    settling: Optional[float] = None                       # lengths off the lead early
    gear: str = ""
    comment: str = ""
    tip_rank: Optional[int] = None
    sectional_600: Optional[float] = None                  # last-600m best recent, seconds
    sources: list[str] = field(default_factory=list)
    extras: dict = field(default_factory=dict)

    @property
    def starts(self) -> int:
        return self.career[0]

    @property
    def wins(self) -> int:
        return self.career[1]

    @property
    def places(self) -> int:
        return self.career[1] + self.career[2] + self.career[3]

    def record(self, key: str) -> tuple[int, int, int, int]:
        return self.records.get(key, (0, 0, 0, 0))

    @property
    def move_pct(self) -> Optional[float]:
        """Price move since opening, as a fraction: +0.25 = drifted 25%, -0.2 = firmed."""
        if self.odds and self.odds_open and self.odds_open > 1:
            return self.odds / self.odds_open - 1.0
        return None


@dataclass
class RaceCard:
    country: str = ""
    venue: str = ""
    race_no: Optional[int] = None
    race_date: Optional[date] = None
    name: str = ""
    dist_m: Optional[int] = None
    surface: str = ""
    going: str = ""
    race_class: str = ""
    prize: Optional[float] = None
    currency: str = ""
    start_time: str = ""
    weather: str = ""
    rail: str = ""
    entries: list[Entry] = field(default_factory=list)
    sources: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    extras: dict = field(default_factory=dict)

    @property
    def field_(self) -> list[Entry]:
        return sorted((e for e in self.entries if not e.scratched),
                      key=lambda e: (e.number is None, e.number or 0))

    def find(self, number: int | None, name: str) -> Optional[Entry]:
        if number is not None:
            for e in self.entries:
                if e.number == number:
                    return e
        best, score = None, 0.0
        for e in self.entries:
            s = similarity(e.name, name)
            if s > score:
                best, score = e, s
        return best if score >= 0.85 else None


# --- merging ------------------------------------------------------------------

_SCALAR_FIELDS = ("barrier", "weight", "jockey", "trainer", "age", "sex", "prize_money",
                  "form_string", "last_run_days", "official_rating", "odds", "odds_open",
                  "place_odds", "exchange_odds", "jockey_stats", "trainer_stats", "combo_stats",
                  "speed_label", "settling", "gear", "comment", "tip_rank", "sectional_600")


MARKET_FIELDS = ("odds", "odds_open", "place_odds", "flucs")


def merge_entry(base: Entry, other: Entry, prefer_other: bool = False,
                prefer_fields: tuple[str, ...] = ()) -> Entry:
    """Fill gaps in ``base`` from ``other`` (or overwrite when ``prefer_other``;
    ``prefer_fields`` names fields that ``other`` always wins)."""
    for f in _SCALAR_FIELDS:
        b, o = getattr(base, f), getattr(other, f)
        if o not in (None, "", ()) and (prefer_other or f in prefer_fields or b in (None, "", ())):
            setattr(base, f, o)
    if other.career[0] and (prefer_other or not base.career[0]):
        base.career = other.career
    for k, v in other.records.items():
        if prefer_other or k not in base.records:
            base.records[k] = v
    if other.runs and (prefer_other or len(other.runs) > len(base.runs)):
        base.runs = other.runs
    if other.flucs and (prefer_other or "flucs" in prefer_fields or not base.flucs):
        base.flucs = other.flucs
    base.scratched = base.scratched or other.scratched
    for s in other.sources:
        if s not in base.sources:
            base.sources.append(s)
    for k, v in other.extras.items():
        base.extras.setdefault(k, v)
    if not base.name and other.name:
        base.name = other.name
    if base.number is None and other.number is not None:
        base.number = other.number
    return base


def merge_cards(base: RaceCard, other: RaceCard, prefer_other: bool = False,
                prefer_fields: tuple[str, ...] = ()) -> RaceCard:
    """Join two views of the same race: by saddlecloth number first, then by name."""
    for f in ("name", "dist_m", "surface", "going", "race_class", "prize", "currency",
              "start_time", "weather", "rail", "race_date", "venue"):
        b, o = getattr(base, f), getattr(other, f)
        if o not in (None, "") and (prefer_other or b in (None, "")):
            setattr(base, f, o)
    unmatched: list[Entry] = []
    for e in other.entries:
        tgt = base.find(e.number, e.name)
        if tgt is not None and (tgt.number == e.number or e.number is None or tgt.number is None
                                or similarity(tgt.name, e.name) >= 0.85):
            merge_entry(tgt, e, prefer_other, prefer_fields)
        else:
            unmatched.append(e)
    if unmatched:
        if not base.entries:
            base.entries.extend(unmatched)
        else:
            base.warnings.append(
                f"{other.sources[0] if other.sources else 'source'}: no match in the field for "
                + ", ".join(f"{u.number or '?'} {u.name}" for u in unmatched[:6]))
    for s in other.sources:
        if s not in base.sources:
            base.sources.append(s)
    base.notes.extend(other.notes)
    base.warnings.extend(other.warnings)
    for k, v in other.extras.items():
        base.extras.setdefault(k, v)
    return base


# --- small parsers shared by adapters ------------------------------------------

def frac_to_decimal(text: str) -> Optional[float]:
    """'7/2' -> 4.5, 'EVS' -> 2.0, '4.5' -> 4.5, '8/10F' -> 1.8."""
    t = (text or "").strip().upper().replace("F", "").replace("J", "").strip()
    if not t or t in ("SP", "-", "NR"):
        return None
    if t in ("EVS", "EVENS", "1/1"):
        return 2.0
    m = re.match(r"^(\d+(?:\.\d+)?)\s*/\s*(\d+(?:\.\d+)?)$", t)
    if m:
        return 1.0 + float(m.group(1)) / float(m.group(2))
    try:
        v = float(t.replace("$", ""))
        return v if v > 1.0 else None
    except ValueError:
        return None


def imperial_to_m(text: str) -> Optional[int]:
    """'6f 12y' -> 1218, '1m 2f' -> 2012, '1m1f207y' -> 2000 (rounded to 1 m)."""
    t = (text or "").lower().replace(" ", "")
    if not t:
        return None
    m = re.match(r"^(\d{3,4})m$", t)
    if m:
        return int(m.group(1))
    miles = furlongs = yards = 0.0
    mm = re.search(r"(\d+(?:\.\d+)?)m(?![a-z])", t)
    fm = re.search(r"(\d+(?:\.\d+)?)f", t)
    ym = re.search(r"(\d+)y", t)
    if mm:
        miles = float(mm.group(1))
    if fm:
        furlongs = float(fm.group(1))
    if ym:
        yards = float(ym.group(1))
    if not (mm or fm or ym):
        return None
    return int(round(miles * 1609.344 + furlongs * 201.168 + yards * 0.9144))


def lengths_from_text(text: str) -> Optional[float]:
    """'2 1/4' -> 2.25, 'nk' -> 0.3, 'hd' -> 0.2, 'shd' -> 0.1, 'nse' -> 0.05, 'dist' -> 30."""
    t = (text or "").strip().lower()
    if not t or t in ("-", "---", "--"):
        return None
    named = {"nse": 0.05, "nose": 0.05, "shd": 0.1, "sh": 0.1, "shead": 0.1, "hd": 0.2,
             "head": 0.2, "snk": 0.25, "nk": 0.3, "neck": 0.3, "dist": 30.0, "dht": 0.0}
    if t in named:
        return named[t]
    m = re.match(r"^(\d+)?(?:\s*-?\s*(\d)/(\d))?$", t)
    if m and (m.group(1) or m.group(2)):
        whole = float(m.group(1) or 0)
        frac = float(m.group(2)) / float(m.group(3)) if m.group(2) else 0.0
        return whole + frac
    try:
        return float(t.replace("l", ""))
    except ValueError:
        return None


def days_between(later: Optional[date], earlier: Optional[date]) -> Optional[int]:
    if later and earlier:
        return (later - earlier).days
    return None
