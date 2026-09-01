"""Parser for Racing & Sports **Full Fields** pages - greyhound, thoroughbred
and harness.

The Full Fields page is the one carrying each runner's **last-10 run table**
(FP / Marg / Date / Trk / Race / $R.PM / Dist / SOT / ... / SP / Winner).  It is
a different page from *Enhanced Form*, which the `greyhound` and `horse`
branches parse.

The three codes share a page skeleton but differ in three ways that matter:

* **Runner block layout.**  Greyhound runs `form -> name -> tags -> A/S -> box ->
  trainer`.  Thoroughbred and harness lead with the tab number and carry more
  columns after A/S (`Wgt, BP, Jockey, Trainer` and `Driver, Trainer`).  A
  reader that assumes one shape silently reports the weight or the driver as
  the trainer.
* **Margin units.**  Greyhound and thoroughbred margins are lengths (`7.9L`);
  harness margins are metres (`48.2m`).
* **Non-finishes.**  Harness carries `DQG` (disqualified - broke gait) with a
  sentinel margin of `99m`.  That is not a result and must never be read as one,
  but the *rate* of it is the most predictive thing on a French trot page.

Design rules, learned the hard way across the other racing apps:

* Read every table through its **header row**, never by column position.  The
  Cabourg page ships two different run-table column sets on the same page - some
  runners carry a `Draw` column and some do not.
* **Keep empty cells** when splitting - a blank Sec.Time or Draw must not shift
  the columns.
* Panel labels are **glued to their values** (`Career17: 1 1 0`,
  `W% - P%6% - 12%`), so peel known labels in order.
* Never trust `X of Y`: R&S emit impossible pairs like `6 of 4`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date

# --- codes --------------------------------------------------------------------

GREYHOUND = "greyhound"
THOROUGHBRED = "thoroughbred"
HARNESS = "harness"

#: what follows the A/S code in a runner block, per code
BLOCK_TAIL: dict[str, list[str]] = {
    GREYHOUND: ["box", "trainer"],
    THOROUGHBRED: ["weight", "barrier", "jockey", "trainer"],
    HARNESS: ["driver", "trainer"],
}

#: metres per horse/dog length, for converting harness margins
METRES_PER_LENGTH = 2.5

#: R&S sentinel margin meaning "no result" (harness)
SENTINEL_METRES = 99.0

# Straight tracks have no first turn, so railing ability and box speed do not
# transfer to or from a circle track.  Separate axis from surface.
STRAIGHT_TRACKS: set[str] = {"RIST", "QSTR", "MURR", "HEAL"}

#: race-surface word -> (run SOT code, records-panel label)
SURFACE_MAP = {
    "AW": ("AW", "AW"),
    "TURF": ("T", "Turf"),
    "SAND": ("S", None),
    "DIRT": ("D", None),
}
GOING_LABEL = {"GOOD": "Good", "SOFT": "Soft", "HEAVY": "Heavy", "FIRM": "Firm"}

MONTHS = {m: i for i, m in enumerate(
    "Jan Feb Mar Apr May Jun Jul Aug Sep Oct Nov Dec".split(), start=1)}

_RECORD_LABELS = ("Career", "Course", "Last 12m", "Dist", "First Up", "Firm",
                  "Second Up", "Good", "AW", "Soft", "Turf", "Heavy", "Season")

_RE_FORM = re.compile(r"^[0-9xX]{3,8}$")
_RE_AGESEX = re.compile(r"^\d{1,2}[A-Z]{1,3}$")
_RE_INT = re.compile(r"^\d{1,2}$")
_RE_WEIGHT = re.compile(r"^\d{2}(?:\.\d)?$")
_RE_ODDS = re.compile(r"^([+-]?\d+)%\s*\$?([\d.]+)?$")
_RE_PRICE = re.compile(r"\$([\d.]+)")
_RE_RECORD = re.compile(
    r"^(" + "|".join(re.escape(x) for x in _RECORD_LABELS) + r")(\d+):\s*(\d+)\s+(\d+)\s+(\d+)\s*$")
_RE_WP = re.compile(r"^W%\s*-\s*P%(\d+)%\s*-\s*(\d+)%\s*$")
_RE_BREED = re.compile(r"^(\d+)yo\s+([A-Z/]+)\s+(Gelding|Horse|Mare|Filly|Colt|B|D)\b", re.I)
_RE_DISTLINE = re.compile(
    r"^(\d{3,4})m\s+([A-Z \-]+?)\s+(GOOD|FAST|SLOW|HEAVY|FIRM|SOFT|DEAD|SLOPPY)\s*$", re.I)
_RE_CODE = re.compile(r"Form Guide(Greyhound|Thoroughbred|Harness)", re.I)
_RE_TITLE = re.compile(r"^(.+?)\s+Form Guide\s*\(Race\s*(\d+)\)")
_RE_BREADCRUMB = re.compile(r"RacesRace\s*(\d+)")
_RE_LONGDATE = re.compile(r"([A-Z][a-z]+day),\s*(\d{1,2})(?:st|nd|rd|th)\s+([A-Z][a-z]+)\s+(\d{4})")
_RE_SHORTDATE = re.compile(r"^(\d{1,2})-([A-Za-z]{3})-(\d{4})$")
_RE_FP = re.compile(r"^(\d+)\s+of\s+(\d+)$")
_RE_DQ = re.compile(r"^(DQ[A-Z]?|DSQ|BD|PU|FL|UR|RO|WD|NP)$", re.I)
_RE_MARGIN = re.compile(r"^([\d.]+)\s*(L|m)$", re.I)
_RE_TYPE = re.compile(r"Type:\s*(.+?)\s+Fastest Time:", re.I)
_RE_HANDICAP = re.compile(r"^(-?\d+)\s*m$", re.I)
_RE_PRIZE_HDR = re.compile(r"^\s*(?:AUD|EUR|NZD|GBP|USD)\s*[€$£]?[\d,]+$")


@dataclass
class Run:
    """One past run from a runner's last-10 table."""
    pos: int | None = None
    field_size: int | None = None
    field_size_suspect: bool = False
    disqualified: bool = False       # harness DQG / thoroughbred PU, BD, ...
    dq_code: str = ""
    margin: float | None = None      # in LENGTHS; +ve beaten by, -ve won by
    margin_raw: float | None = None  # as printed, before unit conversion
    margin_unit: str = ""            # "L" or "m"
    run_date: date | None = None
    days_ago: int | None = None
    track: str = ""
    race_class: str = ""
    prize: str = ""
    dist_m: int | None = None
    surface: str = ""
    going: str = ""
    box: int | None = None           # greyhound box / thoroughbred barrier
    weight: float | None = None      # thoroughbred kg carried
    jockey: str = ""
    handicap_m: float | None = None  # harness distance handicap (+25m = behind)
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

    @property
    def counts_as_form(self) -> bool:
        """A run usable in the beaten-margin average."""
        return (not self.disqualified and self.margin is not None
                and self.dist_m is not None and self.days_ago is not None)


@dataclass
class Runner:
    tab: int | None = None
    box: int | None = None
    name: str = ""
    trainer: str = ""
    jockey: str = ""
    driver: str = ""
    weight: float | None = None
    barrier: int | None = None
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

    def record(self, key: str | None) -> tuple[int, int, int, int]:
        if not key:
            return (0, 0, 0, 0)
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

    @property
    def handler(self) -> str:
        """Driver for harness, jockey for thoroughbred, empty for greyhound."""
        return self.driver or self.jockey

    @property
    def dq_count(self) -> int:
        return sum(1 for r in self.runs if r.disqualified)

    @property
    def dq_rate(self) -> float:
        return self.dq_count / len(self.runs) if self.runs else 0.0


@dataclass
class Race:
    code: str = GREYHOUND
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
        return sorted((r for r in self.runners if not r.scratched),
                      key=lambda r: (r.tab is None, r.tab))

    @property
    def is_straight(self) -> bool:
        return self.track_code().upper() in STRAIGHT_TRACKS

    @property
    def surface_code(self) -> str:
        return SURFACE_MAP.get(self.surface, ("", None))[0]

    @property
    def surface_record_label(self) -> str | None:
        return SURFACE_MAP.get(self.surface, ("", None))[1]

    @property
    def going_record_label(self) -> str | None:
        return GOING_LABEL.get(self.going)

    def track_code(self) -> str:
        """Short code for this meeting.  The header gives a full name, the run
        tables give codes, so infer the mapping.

        Frequency alone is not enough: greyhounds race at their local track most
        weeks, but thoroughbreds travel, so at Deauville the most common code at
        1600m is CTYA (Chantilly).  Prefer a code whose letters appear in order
        in the track's own name - DEA/DEAUVILLE, CBRG/CABOURG, HSHM/HORSHAM.
        """
        counts: dict[str, int] = {}
        for r in self.runners:
            for run in r.runs:
                if run.dist_m == self.dist_m and run.track:
                    counts[run.track] = counts.get(run.track, 0) + 1
        if not counts:
            return ""
        name = re.sub(r"[^A-Z]", "", self.track.upper())

        def is_subsequence(code: str) -> bool:
            it = iter(name)
            return all(ch in it for ch in code)

        matching = [c for c in counts if is_subsequence(c)]
        pool = matching or list(counts)
        return max(pool, key=lambda c: counts[c])


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
    norm = [c.lower().replace(".", "").replace("$", "").replace("/", "").strip()
            for c in cells]
    if "fp" not in norm or "marg" not in norm:
        return None
    want = {"fp": "fp", "marg": "marg", "date": "date", "trk": "trk", "race": "race",
            "rpm": "prize", "dist": "dist", "sot": "sot", "box": "box", "sp": "sp",
            "sectime": "sec", "winner2nd": "winner", "jockey": "jockey",
            "wt": "wt", "bp": "bp", "draw": "draw"}
    out: dict[str, int] = {}
    for i, key in enumerate(norm):
        if key in want and want[key] not in out:
            out[want[key]] = i
    return out if "fp" in out else None


# --- run table ----------------------------------------------------------------

def _parse_run_row(cells: list[str], cmap: dict[str, int],
                   race_day: date | None, code: str) -> Run | None:
    def cell(k: str) -> str:
        i = cmap.get(k)
        return cells[i] if i is not None and i < len(cells) else ""

    fp, marg = cell("fp"), cell("marg")
    if not fp and not marg:
        return None

    run = Run()
    m = _RE_FP.match(fp)
    if m:
        run.pos, fs = int(m.group(1)), int(m.group(2))
        if run.pos > fs:
            # R&S emit impossible pairs such as "6 of 4"; keep the position and
            # treat the field size as unusable rather than believing either.
            run.field_size, run.field_size_suspect = None, True
        else:
            run.field_size = fs
    elif _RE_DQ.match(fp):
        run.disqualified, run.dq_code = True, fp.upper()
    elif fp:
        return None
    else:
        # a blank FP alongside a sentinel margin is also a non-finish
        run.disqualified, run.dq_code = True, "NR"

    mm = _RE_MARGIN.match(marg)
    if mm:
        val, unit = float(mm.group(1)), mm.group(2).lower()
        run.margin_raw, run.margin_unit = val, unit
        if unit == "m" and abs(val - SENTINEL_METRES) < 1e-9:
            # 99m is R&S's "no result" sentinel, not a margin
            run.margin, run.disqualified = None, True
            run.dq_code = run.dq_code or "NR"
        else:
            lengths = val / METRES_PER_LENGTH if unit == "m" else val
            run.margin = -lengths if run.pos == 1 else lengths

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

    for key in ("box", "bp"):
        if cell(key).isdigit():
            run.box = int(cell(key))
            break
    run.weight = _f(cell("wt")) if cell("wt") else None
    run.jockey = re.sub(r"\s*\([^)]*\)\s*$", "", cell("jockey")).strip()

    hm = _RE_HANDICAP.match(cell("draw"))
    if hm:
        run.handicap_m = float(hm.group(1))

    pm = _RE_PRICE.search(cell("sp"))
    if pm:
        run.sp = _f(pm.group(1))
    run.sectional = _f(cell("sec")) if cell("sec") else None
    run.beat_or_beaten_by = re.sub(r"\s*\([\d.]+\)\s*$", "", cell("winner")).strip()
    return run


# --- runner blocks ------------------------------------------------------------

def _looks_like_runner_start(lines: list[str], i: int) -> bool:
    """A runner block opens with a form string on its own line, followed (after
    blanks) by an ALL-CAPS name."""
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


def _parse_runner(block: list[str], race: Race, tab_hint: int | None) -> Runner | None:
    r = Runner(tab=tab_hint)
    body = [ln.strip() for ln in block]
    r.form_string = body[0]

    idx = 1
    while idx < len(body) and not body[idx]:
        idx += 1
    if idx >= len(body):
        return None
    r.name = body[idx]
    idx += 1

    # tags, then the A/S code, then the per-code tail.  The NAME comes before the
    # tag block and the TRAINER at the end of the tail: a reader that takes the
    # last capitalised line reports the trainer (greyhound), the weight
    # (thoroughbred) or the driver (harness) instead.
    seen_agesex = False
    tail = list(BLOCK_TAIL[race.code])
    while idx < len(body):
        tok = body[idx]
        idx += 1
        if not tok:
            continue
        if tok.lower() == "chart" or tok.startswith("Chart with"):
            break
        if not seen_agesex:
            if _RE_AGESEX.match(tok):
                seen_agesex = True
            elif tok.isalpha() and len(tok) <= 5:
                r.tags.append(tok)
            else:
                break
            continue
        if not tail:
            break
        slot = tail[0]
        if slot == "box" and _RE_INT.match(tok):
            r.box = int(tok)
        elif slot == "weight" and _RE_WEIGHT.match(tok):
            r.weight = float(tok)
        elif slot == "barrier" and _RE_INT.match(tok):
            r.barrier = int(tok)
        elif slot in ("jockey", "driver", "trainer"):
            name = re.sub(r"\s*\([^)]*\)\s*$", "", tok).strip()
            setattr(r, slot, name)
        else:
            continue          # unexpected token for this slot; do not consume it
        tail.pop(0)

    if race.code == GREYHOUND:
        r.tab = r.tab if r.tab is not None else r.box
    r.box = r.box if r.box is not None else r.tab

    # NOTE: iterate the RAW lines from here on.  A table row may open with an
    # empty cell (a blank FP on a non-finish, a blank Draw), and stripping the
    # line deletes that cell and shifts every column left by one.
    rest_raw = list(block[idx:])
    if any(ln.lower() == "scratched" for ln in body):
        r.scratched = True

    cmap: dict[str, int] | None = None
    for raw_ln in rest_raw:
        ln = raw_ln.strip()
        if not ln:
            continue
        if cmap is None:
            om = _RE_ODDS.match(ln)
            if om and om.group(2):
                r.fluc_pct, r.odds = _f(om.group(1)), _f(om.group(2))
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
                r.age, r.sex = int(br.group(1)), br.group(3)
                sm = re.search(r"Sire:\s*([^|]+)", ln)
                dm = re.search(r"Dam:\s*([^|]+)", ln)
                if sm:
                    r.sire = sm.group(1).strip()
                if dm:
                    r.dam = dm.group(1).strip()
                continue

        cells = _split_row(raw_ln)
        hm = _header_map(cells)
        if hm:
            cmap = hm          # rebuilt per runner: column sets vary within a page
            continue
        if cmap:
            run = _parse_run_row(cells, cmap, race.race_date, race.code)
            if run:
                r.runs.append(run)
    return r


# --- header -------------------------------------------------------------------

def _parse_header(lines: list[str], race: Race) -> None:
    for ln in lines[:120]:
        s = ln.strip()
        if not s:
            continue
        m = _RE_CODE.search(s)
        if m:
            race.code = {"greyhound": GREYHOUND, "thoroughbred": THOROUGHBRED,
                         "harness": HARNESS}[m.group(1).lower()]
        if not race.track:
            t = _RE_TITLE.match(s)
            if t:
                race.track, race.race_no = t.group(1).strip(), int(t.group(2))
        if race.race_no is None:
            b = _RE_BREADCRUMB.search(s)
            if b:
                race.race_no = int(b.group(1))
        if race.race_date is None:
            d = _RE_LONGDATE.search(s)
            if d and d.group(3)[:3].title() in MONTHS:
                race.race_date = date(int(d.group(4)), MONTHS[d.group(3)[:3].title()],
                                      int(d.group(2)))
        if race.dist_m is None:
            dl = _RE_DISTLINE.match(s)
            if dl:
                race.dist_m = int(dl.group(1))
                surf = dl.group(2).strip().upper()
                race.surface = "AW" if "WEATHER" in surf else surf
                race.going = dl.group(3).upper()
        if not race.grade:
            ty = _RE_TYPE.search(s)
            if ty:
                race.grade = ty.group(1).strip()
        if not race.prize and _RE_PRIZE_HDR.match(s):
            race.prize = s.strip()


# --- entry point --------------------------------------------------------------

def parse(raw: str) -> Race:
    """Parse a pasted R&S Full Fields page into a `Race`."""
    text = raw.replace("\r\n", "\n").replace("\r", "\n")
    lines = text.split("\n")
    race = Race()
    _parse_header(lines, race)

    starts = [i for i in range(len(lines)) if _looks_like_runner_start(lines, i)]
    for k, i in enumerate(starts):
        end = starts[k + 1] if k + 1 < len(starts) else len(lines)
        # thoroughbred and harness lead the block with the tab number, on its own
        # line before the form string
        tab_hint = None
        j = i - 1
        while j >= 0 and not lines[j].strip():
            j -= 1
        if j >= 0 and _RE_INT.match(lines[j].strip()):
            tab_hint = int(lines[j].strip())
        runner = _parse_runner(lines[i:end], race, tab_hint)
        if runner and runner.name:
            race.runners.append(runner)

    seen: set[int] = set()
    dedup: list[Runner] = []
    for r in race.runners:
        if r.tab is not None and r.tab in seen:
            race.warnings.append(f"duplicate tab {r.tab} ({r.name}) - kept first")
            continue
        if r.tab is not None:
            seen.add(r.tab)
        dedup.append(r)
    race.runners = dedup

    if not race.runners:
        race.warnings.append("no runners found - is this a Full Fields page?")
    else:
        tabs = sorted(t for t in (r.tab for r in race.runners) if t is not None)
        if tabs:
            missing = [t for t in range(1, max(tabs) + 1) if t not in tabs]
            if missing:
                race.warnings.append(
                    "tab sequence gap: nothing listed for number "
                    + ", ".join(map(str, missing)))
    if not race.field_:
        race.warnings.append("every runner is scratched")

    suspect = sum(1 for r in race.runners for run in r.runs if run.field_size_suspect)
    if suspect:
        race.warnings.append(
            f"{suspect} past run(s) show an impossible 'X of Y' field size "
            "(an R&S display artifact) - field size imputed for those")

    if race.code == HARNESS:
        dq = sum(r.dq_count for r in race.field_)
        tot = sum(len(r.runs) for r in race.field_)
        if tot:
            race.warnings.append(
                f"{dq} of {tot} past runs in this field are non-finishes "
                f"(DQG / disqualified). They are excluded from the form average "
                f"and scored separately.")
    return race
