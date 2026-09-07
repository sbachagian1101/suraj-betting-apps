"""Read a pasted Racing & Sports page (Full Fields, or just its header).

The paste is the app's *race identity*: the breadcrumb links carry the country,
venue, date and race number (``/thoroughbred/australia/grafton/2026-09-07/R2``),
so those never have to be typed.  When the paste also contains the runner
blocks, the vendored ``rs_parser`` reads each runner's last-10 run table and
that becomes the richest form source we have for Australia.

Two paste flavours arrive: plain select-all text, and a Markdown export with
``* [Race 2](https://...)`` bullets and pipe tables.  Both are normalised here.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date
from typing import Optional

import rs_parser
from common import CURRENCY, Entry, PastRun, RaceCard, days_between

COUNTRY_SLUGS = {
    "australia": "AUS", "france": "FR", "united-kingdom": "UK", "uk": "UK",
    "great-britain": "UK", "england": "UK", "scotland": "UK", "wales": "UK",
    "ireland": "IRE", "hong-kong": "HK", "south-africa": "SA", "usa": "USA",
    "united-states": "USA", "america": "USA",
}
_RE_LINK = re.compile(r"\[([^\]]*)\]\((https?://[^)\s]+)\)")
_RE_URL = re.compile(r"https?://(?:www\.)?racingandsports\.com(?:\.au)?/form-guide/"
                     r"(?P<code>thoroughbred|greyhound|harness)/(?P<country>[a-z\-]+)/"
                     r"(?P<venue>[a-z0-9\-]+)/(?P<date>\d{4}-\d{2}-\d{2})(?:/R(?P<race>\d+))?",
                     re.I)
_RE_LONGDATE = re.compile(r"([A-Z][a-z]+day),\s*(\d{1,2})(?:st|nd|rd|th)?\s+([A-Z][a-z]+)\s+(\d{4})")
_RE_TITLE = re.compile(r"^(.+?)\s+Form Guide\s*\(Race\s*(\d+)\)", re.I)
_RE_TIME = re.compile(r"^(\d{1,2}):(\d{2})$")
_RE_TYPE = re.compile(r"Type:\s*(.+?)\s+(?:Fastest Time:|SOT:|$)", re.I)
_RE_WT = re.compile(r"WT:\s*([\d.]+)\s*kg", re.I)
_RE_PRIZE = re.compile(r"^\s*(AUD|EUR|NZD|GBP|USD|HKD|ZAR)\s*[€$£R]?\s*([\d,.]+)\s*(k?)\s*$", re.I)
_RE_SOT = re.compile(r"SOT:\s*([A-Z0-9]+)", re.I)
SURFACES = ("ALL WEATHER", "TURF", "SAND", "DIRT", "SYNTHETIC", "POLYTRACK", "FIBRESAND",
            "TAPETA", "AW")


@dataclass
class Identity:
    country: str = ""          # our code: AUS / FR / UK / IRE / HK / SA / USA
    venue: str = ""
    race_no: Optional[int] = None
    race_date: Optional[date] = None
    race_name: str = ""
    start_time: str = ""
    dist_m: Optional[int] = None
    surface: str = ""
    going: str = ""
    going_rating: str = ""
    race_type: str = ""
    prize: Optional[float] = None
    currency: str = ""
    weight_min: Optional[float] = None
    code: str = "thoroughbred"
    urls: list[str] = field(default_factory=list)

    @property
    def complete(self) -> bool:
        return bool(self.country and self.venue and self.race_no and self.race_date)


def clean_markdown(raw: str) -> str:
    """Markdown export -> the plain text the select-all copy would have given."""
    text = raw.replace("\r\n", "\n").replace("\r", "\n")
    text = _RE_LINK.sub(lambda m: m.group(1), text)
    out = []
    for ln in text.split("\n"):
        s = ln
        s = re.sub(r"^\s*[\*\-•]\s+", "", s)                  # bullets
        s = re.sub(r"^\s*#{1,6}\s+", "", s)                  # headings
        s = s.replace("**", "").replace("\\|", "|")
        if s.strip().startswith("|") and s.strip().endswith("|"):
            cells = s.strip()[1:-1].split("|")
            if all(re.match(r"^\s*:?-{2,}:?\s*$", c) for c in cells):
                continue                                    # |---|---| separator
            s = "\t".join(c.strip() for c in cells)
        out.append(s)
    return "\n".join(out)


def parse_distance_line(line: str) -> tuple[Optional[int], str, str, str]:
    """'1200m TURF GOOD 4' -> (1200, 'TURF', 'GOOD', '4');  '2750m SAND STANDARD' ->
    (2750, 'SAND', 'STANDARD', '');  '1m1f207y TURF GOOD' -> (2000, 'TURF', 'GOOD', '').
    Parsed, never pattern-matched: an unknown going word must not fail the line."""
    s = line.strip().upper()
    m = re.match(r"^(\d{3,4})M\b(.*)$", s)
    if m:
        dist = int(m.group(1))
        rest = m.group(2).strip()
    else:
        m = re.match(r"^((?:\d+M)?(?:\d+F)?(?:\d+Y)?)\s+(.*)$", s)
        if not m or not m.group(1):
            return None, "", "", ""
        from common import imperial_to_m
        dist = imperial_to_m(m.group(1))
        rest = m.group(2).strip()
    surface = ""
    for w in SURFACES:
        if rest.startswith(w):
            surface = "AW" if w in ("ALL WEATHER", "AW", "SYNTHETIC", "POLYTRACK",
                                    "FIBRESAND", "TAPETA") else w
            rest = rest[len(w):].strip()
            break
    parts = rest.split()
    going = parts[0] if parts else ""
    rating = parts[1] if len(parts) > 1 and re.match(r"^\d+$", parts[1]) else ""
    return dist, surface, going, rating


def identity(raw: str) -> Identity:
    """Race identity from the breadcrumb URLs and the page header."""
    idn = Identity()
    text = raw.replace("\r\n", "\n")
    for m in _RE_URL.finditer(text):
        idn.urls.append(m.group(0))
        idn.code = m.group("code").lower()
        c = COUNTRY_SLUGS.get(m.group("country").lower(), "")
        if c and not idn.country:
            idn.country = c
        if not idn.venue:
            idn.venue = m.group("venue").replace("-", " ").title()
        if idn.race_date is None:
            try:
                idn.race_date = date.fromisoformat(m.group("date"))
            except ValueError:
                pass
        if m.group("race") and idn.race_no is None:
            idn.race_no = int(m.group("race"))
    lines = [ln.strip() for ln in clean_markdown(text).split("\n")]
    seen_time = False
    for i, s in enumerate(lines[:200]):
        if not s:
            continue
        t = _RE_TITLE.match(s)
        if t:
            idn.venue = idn.venue or t.group(1).strip()
            idn.race_no = idn.race_no or int(t.group(2))
        d = _RE_LONGDATE.search(s)
        if d and idn.race_date is None and d.group(3)[:3].title() in rs_parser.MONTHS:
            idn.race_date = date(int(d.group(4)), rs_parser.MONTHS[d.group(3)[:3].title()],
                                 int(d.group(2)))
        tm = _RE_TIME.match(s)
        if tm and not idn.start_time:
            idn.start_time = f"{int(tm.group(1)):02d}:{tm.group(2)}"
            seen_time = True
            continue
        if seen_time and not idn.race_name:
            if s.lower() in ("(local)", "local"):
                continue
            if re.search(r"[A-Za-z]{3}", s) and "Type:" not in s and "WT:" not in s:
                idn.race_name = s
                continue
        ty = _RE_TYPE.search(s)
        if ty and not idn.race_type:
            idn.race_type = ty.group(1).strip()
        w = _RE_WT.search(s)
        if w and idn.weight_min is None:
            try:
                idn.weight_min = float(w.group(1))
            except ValueError:
                pass
        if idn.dist_m is None:
            dist, surf, going, rating = parse_distance_line(s)
            if dist and (surf or going):
                idn.dist_m, idn.surface, idn.going, idn.going_rating = dist, surf, going, rating
        pz = _RE_PRIZE.match(s)
        if pz and idn.prize is None:
            try:
                val = float(pz.group(2).replace(",", ""))
                if pz.group(3).lower() == "k":
                    val *= 1000
                idn.prize, idn.currency = val, pz.group(1).upper()
            except ValueError:
                pass
        if idn.going == "" and _RE_SOT.search(s):
            idn.going_rating = idn.going_rating or _RE_SOT.search(s).group(1)
    if idn.country and not idn.currency:
        idn.currency = CURRENCY.get(idn.country, "")
    return idn


def _run_from_rs(run: rs_parser.Run, race_day: Optional[date]) -> PastRun:
    prize = run.prize_value
    return PastRun(
        run_date=run.run_date, days_ago=run.days_ago if run.days_ago is not None
        else days_between(race_day, run.run_date),
        track=run.track, race_class=run.race_class,
        prize=prize[1] if prize else None, currency=prize[0] if prize else "",
        dist_m=run.dist_m,
        surface={"T": "TURF", "AW": "AW", "S": "SAND", "D": "DIRT"}.get(run.surface, run.surface),
        going=run.going, pos=run.pos, field_size=run.field_size, margin_l=run.margin,
        weight=run.weight, barrier=run.box, jockey=run.jockey, sp=run.sp,
        sectional=run.sectional, non_finish=run.disqualified,
        comment=run.beat_or_beaten_by,
    )


def to_card(race: rs_parser.Race, idn: Identity) -> RaceCard:
    card = RaceCard(country=idn.country, venue=idn.venue or race.track,
                    race_no=idn.race_no or race.race_no, race_date=idn.race_date or race.race_date,
                    name=idn.race_name, dist_m=idn.dist_m or race.dist_m,
                    surface=idn.surface or race.surface, going=idn.going or race.going,
                    race_class=idn.race_type or race.grade, prize=idn.prize,
                    currency=idn.currency, start_time=idn.start_time,
                    sources=["Racing & Sports paste"])
    if race.runners:
        card.warnings = list(race.warnings)
    else:
        # a header-only paste is the normal case: the runners come from the feeds
        card.sources = ["Racing & Sports paste (header)"] if idn.urls or idn.race_name else []
        card.notes.append("The paste carried the race header only; runners and form come from the feeds.")
    if idn.going_rating:
        card.extras["going_rating"] = idn.going_rating
    for r in race.runners:
        sex = {"GELDING": "G", "HORSE": "H", "MARE": "M", "FILLY": "F", "COLT": "C"}.get(
            (r.sex or "").upper(), (r.sex or "")[:1].upper())
        e = Entry(number=r.tab, name=r.name.title() if r.name.isupper() else r.name,
                  barrier=r.barrier, weight=r.weight, jockey=r.jockey.title(),
                  trainer=r.trainer.title(), age=r.age, sex=sex, scratched=r.scratched,
                  form_string=r.form_string, odds=r.odds, sources=["Racing & Sports paste"])
        c = r.record("Career")
        if c[0]:
            e.career = c
        e.records = dict(r.records)
        e.runs = [_run_from_rs(run, card.race_date) for run in r.runs]
        if r.fluc_pct is not None and r.odds:
            e.odds_open = r.odds / (1 + r.fluc_pct / 100.0) if r.fluc_pct > -100 else None
        days = [run.days_ago for run in e.runs if run.days_ago is not None]
        if days:
            e.last_run_days = min(days)
        e.extras["tags"] = r.tags
        card.entries.append(e)
    return card


def parse_paste(raw: str) -> tuple[Identity, RaceCard]:
    idn = identity(raw)
    race = rs_parser.parse(clean_markdown(raw))
    card = to_card(race, idn)
    return idn, card
