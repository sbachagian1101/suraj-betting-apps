"""Parser for Racing & Sports greyhound **Full Fields** pages.

This is the page at .../Form Guide/Greyhound/<country>/<track>/Race N -> "Full Fields",
select-all + copy.  It is a different page from the *Enhanced Form* page that the
`greyhound` branch parses: Full Fields carries a per-runner **last-10 run table**
(FP / Marg / Date / Trk / Race / $R.PM / Dist / SOT / Box / SP / Sec.Time / Winner),
which is what this app's rating is built from.

Design rules (learned the hard way across the other racing apps):

* Read the run table through its **header row**, never by column position.
* **Keep empty cells** when splitting - a blank Sec.Time must not shift the columns.
* Panel labels are **glued to their values** (`Career17: 1 1 0`, `W% - P%6% - 12%`),
  so peel known labels in order rather than regexing loose numbers.
* Require only a box number + a name to emit a runner, and warn on a gap in the
  box sequence rather than silently shrinking the field.
* Never trust `X of Y`: R&S emit impossible pairs like `6 of 4`.  Flag them.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date, datetime

# --- track classification -----------------------------------------------------
# Straight tracks have no first turn, so railing ability and box speed do not
# transfer to or from circle tracks.  This is a *separate* axis from surface.
STRAIGHT_TRACKS: set[str] = {"RIST", "QSTR", "MURR", "HEAL"}

MONTHS = {m: i for i, m in enumerate(
    "Jan Feb Mar Apr May Jun Jul Aug Sep Oct Nov Dec".split(), start=1)}

_RECORD_LABELS = ("Career", "Course", "Last 12m", "Dist", "First Up", "Firm",
                  "Second Up", "Good", "AW", "Soft", "Turf", "Heavy", "Season")

_RE_FORM = re.compile(r"^[0-9xX]{3,8}$")
_RE_AGESEX = re.compile(r"^\d{1,2}[A-Z]{1,3}$")
_RE_BOX = re.compile(r"^\d{1,2}$")
_RE_ODDS = re.compile(r"^([+-]?\d+)%\s*\$?([\d.]+)?$")
_RE_PRICE = re.compile(r"\$([\d.]+)")
_RE_RECORD = re.compile(
    r"^(" + "|".join(re.escape(x) for x in _RECORD_LABELS) + r")(\d+):\s*(\d+)\s+(\d+)\s+(\d+)\s*$")
_RE_WP = re.compile(r"^W%\s*-\s*P%(\d+)%\s*-\s*(\d+)%\s*$")
_RE_BREED = re.compile(r"^(\d+)yo\s+([A-Z/]+)\s+([BD])\s*\|", re.I)
_RE_DISTLINE = re.compile(r"^(\d{3,4})m\s+([A-Z \-]+?)\s+(GOOD|FAST|SLOW|HEAVY|FIRM|SOFT)\s*$", re.I)
_RE_BREADCRUMB = re.compile(r"Greyhound([A-Z][a-z]+)(.+?)\s*RacesRace\s*(\d+)")
_RE_TITLE = re.compile(r"^(.+?)\s+Form Guide\s*\(Race\s*(\d+)\)")
_RE_LONGDATE = re.compile(r"([A-Z][a-z]+day),\s*(\d{1,2})(?:st|nd|rd|th)\s+([A-Z][a-z]+)\s+(\d{4})")
_RE_SHORTDATE = re.compile(r"^(\d{1,2})-([A-Za-z]{3})-(\d{4})$")
_RE_FP = re.compile(r"^(\d+)\s+of\s+(\d+)$")
_RE_MARGIN = re.compile(r"^([\d.]+)\s*L$", re.I)
_RE_TYPE = re.compile(r"^Type:\s*(.+?)\s+Fastest Time:", re.I)


@dataclass
class Run:
    """One past run from a runner's last-10 table."""
    pos: int | None = None
    field_size: int | None = None
    field_size_suspect: bool = False
    margin: float | None = None          # +ve = beaten by, -ve = won by
    run_date: date | None = None
    days_ago: int | None = None
    track: str = ""
    race_class: str = ""
    prize: str = ""
    dist_m: int | None = None
    surface: str = ""                    # AW / T / ...
    going: str = ""
    box: int | None = None
    sp: float | None = None
    sectional: float | None = None
    beat_or_beaten_by: str = ""

    @property
    def is_straight(self) -> bool:
        return self.track.upper() in STRAIGHT_TRACKS

    @property
    def is_foreign(self) -> bool:
        return "NZD" in self.prize.upper()

    @property
    def track_kind(self) -> str:
        if self.is_straight:
            return "straight"
        if self.is_foreign:
            return "foreign"
        return "circle"


@dataclass
class Runner:
    box: int | None = None
    name: str = ""
    trainer: str = ""
    form_string: str = ""
    odds: float | None = None
    fluc_pct: float | None = None
    scratched: bool = False
    age: int | None = None
    sex: str = ""
    sire: str = ""
    dam: str = ""
    tags: list[str] = field(default_factory=list)
    records: dict[str, tuple[int, int, int, int]] = field(default_factory=dict)
    win_pct: float | None = None
    place_pct: float | None = None
    runs: list[Run] = field(default_factory=list)

    def record(self, key: str) -> tuple[int, int, int, int]:
        """(starts, wins, seconds, thirds) for a panel label; zeros if absent."""
        return self.records.get(key, (0, 0, 0, 0))

    @property
    def career_starts(self) -> int:
        return self.record("Career")[0]

    @property
    def career_wins(self) -> int:
        return self.record("Career")[1]

    @property
    def career_places(self) -> tuple[int, int]:
        r = self.record("Career")
        return r[2], r[3]


@dataclass
class Race:
    track: str = ""
    race_no: int | None = None
    race_date: date | None = None
    dist_m: int | None = None
    surface: str = ""
    going: str = ""
    grade: str = ""
    prize: str = ""
    runners: list[Runner] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def field_(self) -> list[Runner]:
        """Non-scratched runners, in box order."""
        return sorted((r for r in self.runners if not r.scratched),
                      key=lambda r: (r.box is None, r.box))

    @property
    def is_straight(self) -> bool:
        return self.track_code().upper() in STRAIGHT_TRACKS

    def track_code(self) -> str:
        """Best-effort 4-letter code for this meeting, taken from the runners'
        own past runs (the header gives a full name, the run tables give codes)."""
        counts: dict[str, int] = {}
        for r in self.runners:
            for run in r.runs:
                if run.dist_m == self.dist_m and run.track:
                    counts[run.track] = counts.get(run.track, 0) + 1
        if counts:
            return max(counts, key=counts.get)
        return ""


# --- helpers ------------------------------------------------------------------

def _f(x: str) -> float | None:
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def _parse_short_date(s: str) -> date | None:
    m = _RE_SHORTDATE.match(s.strip())
    if not m:
        return None
    d, mon, y = m.groups()
    if mon[:3].title() not in MONTHS:
        return None
    try:
        return date(int(y), MONTHS[mon[:3].title()], int(d))
    except ValueError:
        return None


def _split_row(line: str) -> list[str]:
    """Split a table row, KEEPING empty cells."""
    if "\t" in line:
        return [c.strip() for c in line.split("\t")]
    return [c.strip() for c in re.split(r"\s{2,}", line)]


def _header_map(cells: list[str]) -> dict[str, int] | None:
    norm = [c.lower().replace(".", "").replace("$", "").replace("/", "").strip() for c in cells]
    if "fp" not in norm or "marg" not in norm:
        return None
    want = {"fp": "fp", "marg": "marg", "date": "date", "trk": "trk", "race": "race",
            "rpm": "prize", "dist": "dist", "sot": "sot", "box": "box", "sp": "sp",
            "sectime": "sec", "winner2nd": "winner"}
    out: dict[str, int] = {}
    for i, key in enumerate(norm):
        if key in want:
            out[want[key]] = i
    return out if "fp" in out else None


# --- run table ----------------------------------------------------------------

def _parse_run_row(cells: list[str], cmap: dict[str, int], race_day: date | None) -> Run | None:
    def cell(k: str) -> str:
        i = cmap.get(k)
        return cells[i] if i is not None and i < len(cells) else ""

    m = _RE_FP.match(cell("fp"))
    if not m:
        return None
    run = Run()
    run.pos, fs = int(m.group(1)), int(m.group(2))
    if run.pos > fs:
        # R&S emit impossible pairs such as "6 of 4"; do not trust either number's
        # pairing, keep the position and flag the field size as unusable.
        run.field_size, run.field_size_suspect = None, True
    else:
        run.field_size = fs

    mm = _RE_MARGIN.match(cell("marg"))
    if mm:
        val = float(mm.group(1))
        run.margin = -val if run.pos == 1 else val

    run.run_date = _parse_short_date(cell("date"))
    if run.run_date and race_day:
        run.days_ago = (race_day - run.run_date).days

    run.track = cell("trk").upper()
    run.race_class = cell("race")
    run.prize = cell("prize")

    dm = re.match(r"^(\d{3,4})m$", cell("dist"))
    if dm:
        run.dist_m = int(dm.group(1))

    sot = cell("sot").split()
    if sot:
        run.surface = sot[0].upper()
        if len(sot) > 1:
            run.going = sot[1].upper()

    if cell("box").isdigit():
        run.box = int(cell("box"))
    pm = _RE_PRICE.search(cell("sp"))
    if pm:
        run.sp = _f(pm.group(1))
    run.sectional = _f(cell("sec")) if cell("sec") else None
    run.winner_or_second = cell("winner")
    run.beat_or_beaten_by = cell("winner")
    return run


# --- runner blocks ------------------------------------------------------------

def _looks_like_runner_start(lines: list[str], i: int) -> bool:
    """A runner block opens with a form string on its own line, followed (after
    blanks) by an ALL-CAPS dog name."""
    if not _RE_FORM.match(lines[i].strip()):
        return False
    j = i + 1
    while j < len(lines) and not lines[j].strip():
        j += 1
    if j >= len(lines):
        return False
    nxt = lines[j].strip()
    if not nxt or nxt.lower() == "scratched":
        return False
    letters = [c for c in nxt if c.isalpha()]
    return bool(letters) and all(c.isupper() for c in letters) and len(nxt) >= 3


def _parse_runner(block: list[str], race_day: date | None,
                  warnings: list[str]) -> Runner | None:
    r = Runner()
    body = [ln.strip() for ln in block]
    r.form_string = body[0]

    idx = 1
    while idx < len(body) and not body[idx]:
        idx += 1
    if idx >= len(body):
        return None
    r.name = body[idx]
    idx += 1

    # tags -> A/S code -> box -> trainer.  The dog's NAME comes before the tags
    # and the TRAINER after the box: a reader who takes the last capitalised
    # line as the dog name will silently report the trainer instead.
    seen_agesex = False
    while idx < len(body):
        tok = body[idx]
        idx += 1
        if not tok:
            continue
        if not seen_agesex and _RE_AGESEX.match(tok):
            seen_agesex = True
            continue
        if not seen_agesex:
            if tok.isalpha() and len(tok) <= 5:
                r.tags.append(tok)
                continue
            break
        if _RE_BOX.match(tok):
            r.box = int(tok)
            continue
        r.trainer = tok
        break

    rest = body[idx:]
    if any(ln.lower() == "scratched" for ln in rest) or \
       any(ln.lower() == "scratched" for ln in body):
        r.scratched = True

    cmap: dict[str, int] | None = None
    for ln in rest:
        if not ln:
            continue

        if cmap is None:
            om = _RE_ODDS.match(ln)
            if om and om.group(2):
                r.fluc_pct = _f(om.group(1))
                r.odds = _f(om.group(2))
                continue
            rec = _RE_RECORD.match(ln)
            if rec:
                label, starts, w, s, t = rec.groups()
                r.records[label] = (int(starts), int(w), int(s), int(t))
                continue
            wp = _RE_WP.match(ln)
            if wp:
                r.win_pct, r.place_pct = float(wp.group(1)), float(wp.group(2))
                continue
            br = _RE_BREED.match(ln)
            if br:
                r.age, _, r.sex = int(br.group(1)), br.group(2), br.group(3)
                sm = re.search(r"Sire:\s*([^|]+)", ln)
                dm = re.search(r"Dam:\s*([^|]+)", ln)
                if sm:
                    r.sire = sm.group(1).strip()
                if dm:
                    r.dam = dm.group(1).strip()
                continue

        cells = _split_row(ln)
        hm = _header_map(cells)
        if hm:
            cmap = hm
            continue
        if cmap:
            run = _parse_run_row(cells, cmap, race_day)
            if run:
                r.runs.append(run)

    if r.name and r.box is None and not r.scratched:
        warnings.append(f"{r.name}: no box number found")
    return r


# --- header -------------------------------------------------------------------

def _parse_header(lines: list[str], race: Race) -> None:
    for ln in lines[:80]:
        s = ln.strip()
        if not s:
            continue
        if not race.track:
            m = _RE_TITLE.match(s)
            if m:
                race.track, race.race_no = m.group(1).strip(), int(m.group(2))
            else:
                m = _RE_BREADCRUMB.search(s)
                if m:
                    race.track, race.race_no = m.group(2).strip(), int(m.group(3))
        if race.race_date is None:
            m = _RE_LONGDATE.search(s)
            if m and m.group(3)[:3].title() in MONTHS:
                race.race_date = date(int(m.group(4)), MONTHS[m.group(3)[:3].title()],
                                      int(m.group(2)))
        if race.dist_m is None:
            m = _RE_DISTLINE.match(s)
            if m:
                race.dist_m = int(m.group(1))
                surf = m.group(2).strip().upper()
                race.surface = "AW" if "WEATHER" in surf else surf
                race.going = m.group(3).upper()
        if not race.grade:
            m = _RE_TYPE.match(s)
            if m:
                race.grade = m.group(1).strip()
        if not race.prize and re.match(r"^AUD \$[\d,]+$", s):
            race.prize = s


# --- entry point --------------------------------------------------------------

def parse(raw: str) -> Race:
    """Parse a pasted R&S greyhound Full Fields page into a `Race`."""
    text = raw.replace("\r\n", "\n").replace("\r", "\n")
    lines = text.split("\n")
    race = Race()
    _parse_header(lines, race)

    starts = [i for i in range(len(lines)) if _looks_like_runner_start(lines, i)]
    for k, i in enumerate(starts):
        end = starts[k + 1] if k + 1 < len(starts) else len(lines)
        runner = _parse_runner(lines[i:end], race.race_date, race.warnings)
        if runner and runner.name:
            race.runners.append(runner)

    seen: set[int] = set()
    dedup: list[Runner] = []
    for r in race.runners:
        if r.box is not None and r.box in seen:
            race.warnings.append(f"duplicate box {r.box} ({r.name}) - kept first")
            continue
        if r.box is not None:
            seen.add(r.box)
        dedup.append(r)
    race.runners = dedup

    if race.runners:
        boxes = sorted(b for b in (r.box for r in race.runners) if b is not None)
        if boxes:
            missing = [b for b in range(1, max(boxes) + 1) if b not in boxes]
            if missing:
                race.warnings.append(
                    "box sequence gap: no runner listed in box "
                    + ", ".join(map(str, missing)))
    else:
        race.warnings.append("no runners found - is this a Full Fields page?")

    if not race.field_:
        race.warnings.append("every runner is scratched")

    suspect = sum(1 for r in race.runners for run in r.runs if run.field_size_suspect)
    if suspect:
        race.warnings.append(
            f"{suspect} past run(s) show an impossible 'X of Y' field size "
            "(an R&S display artifact) - field size imputed for those")

    return race
