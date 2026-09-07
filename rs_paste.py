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

import rs_enhanced
import rs_parser
from common import CURRENCY, Entry, PastRun, RaceCard, days_between, nice_name

BUILD = "2026-09-07c"      # bumped with every change; app.py refuses a stale copy

COUNTRY_NAMES = {
    "Australia": "AUS", "France": "FR", "United Kingdom": "UK", "Great Britain": "UK",
    "England": "UK", "Scotland": "UK", "Wales": "UK", "Ireland": "IRE", "Hong Kong": "HK",
    "South Africa": "SA", "USA": "USA", "United States": "USA",
}
_RE_CRUMB = re.compile(r"Form Guide(?:Greyhound|Thoroughbred|Harness)(" + "|".join(
    re.escape(k) for k in sorted(COUNTRY_NAMES, key=len, reverse=True)) + r")(.+?) RacesRace\s*(\d+)")
_RE_SPEED_TITLE = re.compile(r"(?m)^.*Form Guide\s*\(Race\s*\d+\)\s*\|\s*Speed Map.*$")

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
    # plain-text copies carry no URLs; the glued breadcrumb still names country, venue, race
    cb = _RE_CRUMB.search(text)
    if cb:
        idn.country = idn.country or COUNTRY_NAMES[cb.group(1)]
        idn.venue = idn.venue or cb.group(2).strip().title()
        idn.race_no = idn.race_no or int(cb.group(3))
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
        e = Entry(number=r.tab, name=nice_name(r.name),
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


# --- Speed Map page ------------------------------------------------------------

_SM_ROW = re.compile(r"(?m)^\s*(\d{1,2})\t+([A-Z][A-Z0-9' .\-()]+?)\s*$")
_SM_ROW_LOOSE = re.compile(r"(?m)^\s*(\d{1,2})\s{2,}([A-Z][A-Z0-9' .\-()]+?)\s*$")
_NUM = re.compile(r"^\$?(\d+(?:\.\d+)?)$")


@dataclass
class SpeedRow:
    tab: int
    name: str = ""
    weight: Optional[float] = None
    jockey: str = ""
    jr: Optional[float] = None       # R&S jockey rating
    bp: Optional[int] = None         # barrier after scratchings
    aes: Optional[float] = None      # average early speed (higher = faster early)
    afs: Optional[float] = None      # average finishing speed (higher = stronger late)


def has_speed_map(raw: str) -> bool:
    t = raw or ""
    return "Pace Values" in t or ("AES" in t and "AFS" in t)


def parse_speed_map(raw: str) -> dict[int, SpeedRow]:
    """Read the R&S Speed Map "Pace Values" table.

    Each runner spans three lines: ``1\\t\\tNAME``, the breeding line, then
    ``63.0\\tJOCKEY (a2kg)\\t4.1\\t5\\t17.0\\t17.6`` = WT, jockey, JR, BP, AES, AFS.
    Only the trailing numbers are trusted, so an empty jockey cell cannot shift
    the pairing.  Returns {} when the paste has no Pace Values table.
    """
    t = clean_markdown(raw or "")
    if "AES" not in t:
        return {}
    t = t[t.index("AES") - 200 if t.index("AES") > 200 else 0:]
    rows = list(_SM_ROW.finditer(t)) or list(_SM_ROW_LOOSE.finditer(t))
    out: dict[int, SpeedRow] = {}
    for idx, m in enumerate(rows):
        tab = int(m.group(1))
        end = rows[idx + 1].start() if idx + 1 < len(rows) else len(t)
        seg = t[m.end():end]
        for ln in seg.splitlines():
            cells = [c.strip() for c in re.split(r"\t|\s{2,}", ln)]
            nums = [c for c in cells if _NUM.match(c)]
            if len(nums) >= 4:
                aes, afs = float(nums[-2]), float(nums[-1])
                if 10.0 <= aes <= 22.0 and 10.0 <= afs <= 22.0:
                    row = SpeedRow(tab=tab, name=nice_name(m.group(2).strip()), aes=aes, afs=afs)
                    try:
                        row.bp = int(float(nums[-3]))
                        row.jr = float(nums[-4])
                    except ValueError:
                        pass
                    if len(nums) >= 5:
                        try:
                            row.weight = float(nums[0])
                        except ValueError:
                            pass
                    words = [c for c in cells if c and not _NUM.match(c)]
                    if words:
                        row.jockey = re.sub(r"\s*\([^)]*\)\s*$", "", words[-1]).strip().title()
                    out[tab] = row
                    break
    return out


def apply_speed_map(card: RaceCard, rows: dict[int, SpeedRow]) -> int:
    """Attach AES / AFS / JR to the card's entries by tab number; prefer the Speed
    Map's barrier (re-numbered after scratchings).  Returns how many matched."""
    n = 0
    for tab, row in rows.items():
        e = card.find(tab, row.name)
        if e is None:
            continue
        e.extras["aes"], e.extras["afs"], e.extras["jr"] = row.aes, row.afs, row.jr
        if row.bp:
            e.barrier = row.bp
        if row.weight and not e.weight:
            e.weight = row.weight
        if row.jockey and not e.jockey:
            e.jockey = row.jockey
        if "Racing & Sports Speed Map" not in e.sources:
            e.sources.append("Racing & Sports Speed Map")
        n += 1
    if n and "Racing & Sports Speed Map" not in card.sources:
        card.sources.append("Racing & Sports Speed Map")
    return n


# --- Enhanced Form page --------------------------------------------------------

def page_type(raw: str) -> str:
    """'enhanced', 'full', 'speed' or '' from the page title line."""
    m = re.search(r"Form Guide\s*\(Race\s*\d+\)\s*\|\s*([A-Za-z ]+)", raw or "")
    if not m:
        return ""
    t = m.group(1).strip().lower()
    if t.startswith("enhanced"):
        return "enhanced"
    if t.startswith("full"):
        return "full"
    if t.startswith("speed"):
        return "speed"
    return ""


def split_speed_map(raw: str) -> tuple[str, str]:
    """A paste that carries the Speed Map page after another page -> (form, speed)."""
    text = (raw or "").replace("\r\n", "\n")
    m = _RE_SPEED_TITLE.search(text)
    if not m:
        return text, ""
    # the breadcrumb line sits just above the title; keep it for identity()
    cut = text.rfind("\n", 0, max(m.start() - 1, 0))
    cut = text.rfind("\n", 0, cut) if cut > 0 else 0
    return text[:max(cut, 0)], text[max(cut, 0):]


_MONTHS_LONG = {m: i for i, m in enumerate(
    "Jan Feb Mar Apr May Jun Jul Aug Sep Oct Nov Dec".split(), start=1)}


def _date_from(text: str) -> Optional[date]:
    m = re.match(r"^\s*(\d{1,2})\s+([A-Za-z]{3})[a-z]*\s+(\d{4})", text or "")
    if not m or m.group(2).title() not in _MONTHS_LONG:
        return None
    try:
        return date(int(m.group(3)), _MONTHS_LONG[m.group(2).title()], int(m.group(1)))
    except ValueError:
        return None


_FILTER_KEYS = {"Car": "Career", "12m": "Last 12m", "Crs": "Course", "Dist": "Dist",
                "Crs & Dist": "C&D", "Firm": "Firm", "Good": "Good", "Soft": "Soft",
                "Heavy": "Heavy", "AW": "AW", "Turf": "Turf", "FU": "First Up", "2U": "Second Up"}


def _stats(win, plc, n) -> Optional[tuple[int, int, int]]:
    try:
        n = int(n or 0)
    except (TypeError, ValueError):
        return None
    if n <= 0 or win is None:
        return None
    return (n, int(round(float(win) * n)), int(round(float(plc or 0) * n)))


def enhanced_to_card(header: dict, runners: list[dict], idn: Identity, race_day: Optional[date]) -> RaceCard:
    card = RaceCard(country=idn.country, venue=idn.venue or header.get("track", ""),
                    race_no=idn.race_no or header.get("race_no"), race_date=race_day,
                    name=idn.race_name or header.get("race_name", ""),
                    dist_m=idn.dist_m or header.get("distance_m"),
                    surface=idn.surface or header.get("surface", ""), going=idn.going or header.get("going", ""),
                    race_class=idn.race_type or header.get("race_type", ""), prize=idn.prize,
                    currency=idn.currency, start_time=idn.start_time or header.get("time", ""),
                    sources=["Racing & Sports paste (Enhanced Form)"])
    for r in runners:
        detailed = bool(r.get("recent_runs")) or r.get("jky_n") is not None
        sex = {"GELDING": "G", "HORSE": "H", "MARE": "M", "FILLY": "F", "COLT": "C"}.get(
            str(r.get("sex") or "").upper(), str(r.get("sex") or "")[:1].upper())
        e = Entry(number=r.get("tab"), name=nice_name(str(r.get("horse", ""))), weight=r.get("wt") or None,
                  barrier=(r.get("bp_block") or r.get("bp") or None), jockey=str(r.get("jockey") or "").title(),
                  trainer=str(r.get("trainer") or "").title(), age=r.get("age") or None, sex=sex,
                  scratched=bool(r.get("scratched")), form_string=str(r.get("form") or ""),
                  gear="", sources=["Racing & Sports paste (Enhanced Form)"])
        px = r.get("tab_odds") or r.get("bf_odds")
        if px and 1.0 < float(px) < 900:
            e.odds = float(px)
            e.extras["price_book"] = r.get("price_book") or r.get("price_source") or ""
        if detailed:
            e.jockey_stats = _stats(r.get("jky_win"), r.get("jky_place"), r.get("jky_n"))
            e.trainer_stats = _stats(r.get("trn_win"), r.get("trn_place"), r.get("trn_n"))
            e.combo_stats = _stats(r.get("jt_win"), r.get("jt_place"), r.get("jt_n"))
            if r.get("dslr") is not None:
                e.last_run_days = int(r["dslr"])
            if r.get("ohr"):
                e.official_rating = float(r["ohr"])
            for key, label in _FILTER_KEYS.items():
                v = (r.get("filters") or {}).get(key)
                if v and len(v) == 3:
                    wins, places, starts = (int(x) for x in v)
                    e.records[label] = (starts, wins, places, 0)
            if e.records.get("Career"):
                e.career = e.records["Career"]
            pm = str((r.get("facts") or {}).get("Car PM") or "")
            mm = re.match(r"\$([\d.,]+)(k?)", pm)
            if mm:
                e.prize_money = float(mm.group(1).replace(",", "")) * (1000 if mm.group(2) else 1)
            settles = []
            for run in r.get("recent_runs") or []:
                d = _date_from(str(run.get("date") or ""))
                fin, fs = run.get("finish"), run.get("field_size")
                pr = PastRun(run_date=d, days_ago=run.get("days_ago") if run.get("days_ago") is not None
                             else days_between(race_day, d),
                             track=str(run.get("track") or ""), race_class=str(run.get("race_class") or ""),
                             prize=run.get("prize") or None, currency="AUD", dist_m=run.get("distance") or None,
                             surface={"T": "TURF", "AW": "AW", "S": "SAND", "D": "DIRT"}.get(
                                 str(run.get("surface") or "").upper(), str(run.get("surface") or "").upper()),
                             going={"G": "GOOD", "S": "SOFT", "H": "HEAVY", "F": "FIRM", "N": "STANDARD"}.get(
                                 str(run.get("going") or "").upper(), str(run.get("going") or "")),
                             pos=fin or None, field_size=fs or None, weight=run.get("weight") or None,
                             barrier=run.get("bp") or None, jockey=str(run.get("jockey") or "").title(),
                             sp=run.get("sp") or None, time_s=run.get("race_time_s") or None,
                             sectional=run.get("sec600_s") or None, comment=str(run.get("comment") or ""),
                             rating=run.get("ohr") or None)
                mg = run.get("margin")
                if fin and mg is not None:
                    pr.margin_l = -float(mg) if fin == 1 else float(mg)
                elif not fin:
                    pr.non_finish = True
                rp = [str(run.get(k)) for k in ("settle_pos", "pos_800", "turn_pos") if run.get(k)]
                pr.running_pos = " ".join(rp)
                if run.get("tempo"):
                    pr.comment = f"[{run['tempo']}] " + pr.comment
                if run.get("settle_pos") and fs and fs > 1:
                    settles.append((int(run["settle_pos"]) - 1) / (fs - 1))
                e.runs.append(pr)
            if settles:
                e.extras["settle_frac"] = sum(settles[:4]) / len(settles[:4])
            secs = [x.sectional for x in e.runs[:3] if x.sectional]
            if secs:
                e.sectional_600 = min(secs)
                e.extras["sectional_source"] = "R&S L600m"
            if e.runs and e.last_run_days is None:
                ds = [x.days_ago for x in e.runs if x.days_ago is not None]
                e.last_run_days = min(ds) if ds else None
        card.entries.append(e)
    return card


def parse_paste(raw: str) -> tuple[Identity, RaceCard]:
    form_text, _speed = split_speed_map(raw or "")
    idn = identity(form_text or raw or "")
    kind = page_type(form_text)
    if kind == "enhanced":
        header, runners, warns = rs_enhanced.parse(form_text)
        card = enhanced_to_card(header, runners, idn, idn.race_date)
        card.warnings = [w for w in warns if "Could not locate" in w]
        if not card.entries:
            card.notes.append("The Enhanced Form paste carried no field table; runners come from the feeds.")
        return idn, card
    race = rs_parser.parse(clean_markdown(form_text))
    card = to_card(race, idn)
    return idn, card
