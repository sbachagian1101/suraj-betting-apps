"""Read a Racing & Sports Worksheets export.

One file per meeting. Races are separated by a blank line and a repeated
`Tab,Horse,...` header row. Columns:

    Tab   saddlecloth          RFS   runs from spell (FU = first up)
    DLS   days since last run  12m   twelve-month rating
    BRR   base race rating     FORM COND CONS BP JOCK JC  adjustments
    FR    final rating         EM    lengths behind the top-rated
    PER   R&S win percentage   DIV   R&S fair price

Scratchings carry the literal string `scr` in EM, PER and DIV rather than a
number, so every numeric field has to tolerate it.

Both a CSV upload and a pasted block are accepted; a paste may arrive
tab-separated, so the delimiter is sniffed rather than assumed.
"""
from __future__ import annotations

import csv
import io
import re
from dataclasses import dataclass, field

COLUMNS = ["Tab", "Horse", "RFS", "DLS", "12m", "BRR", "FORM", "COND",
           "CONS", "BP", "JOCK", "JC", "FR", "EM", "PER", "DIV"]
NUMERIC = ["DLS", "12m", "BRR", "FORM", "COND", "CONS", "BP", "JOCK", "JC",
           "FR", "EM", "PER"]

# Which calibration group a meeting belongs to. R&S percentages are confident
# by very different amounts in different jurisdictions, so this matters.
REGIONS = {
    "AUS": ["wodonga", "scone", "randwick", "flemington", "caulfield",
            "rosehill", "moonee", "eagle farm", "doomben", "ipswich",
            "gold coast", "sandown", "canterbury", "warwick farm",
            "murray bridge", "morphettville", "ascot", "belmont", "pakenham",
            "geelong", "bendigo", "cranbourne", "kembla", "newcastle",
            "gosford", "wyong", "toowoomba", "sunshine coast"],
    "NZ": ["ruakaka", "ellerslie", "te rapa", "trentham", "riccarton",
           "awapuni", "otaki", "matamata", "taupo", "tauranga", "pukekohe"],
    "UK": ["wolverhampton", "ripon", "lingfield", "brighton", "ascot uk",
           "newmarket", "york", "goodwood", "doncaster", "sandown park",
           "kempton", "chepstow", "bath", "salisbury", "haydock", "thirsk",
           "beverley", "catterick", "southwell", "newcastle uk", "epsom",
           "leicester", "nottingham", "yarmouth", "windsor", "chelmsford"],
    "IRE": ["gowran", "gowran-park", "curragh", "leopardstown", "naas",
            "navan", "punchestown", "galway", "listowel", "cork",
            "tipperary", "dundalk", "fairyhouse", "roscommon", "killarney"],
    "FR": ["deauville", "longchamp", "chantilly", "saint-cloud", "vincennes",
           "maisons", "compiegne", "clairefontaine", "lyon", "marseille"],
}


def region_for(meeting: str) -> str:
    m = (meeting or "").strip().lower()
    for reg, names in REGIONS.items():
        for n in names:
            if n in m:
                return reg
    return "OTHER"


@dataclass
class Runner:
    tab: int
    horse: str
    scratched: bool = False
    first_up: bool = False
    rfs: float | None = None
    dls: float = 0.0
    m12: float = 0.0
    brr: float = 0.0
    form: float = 0.0
    cond: float = 0.0
    cons: float = 0.0
    bp: float = 0.0
    jock: float = 0.0
    jc: float = 0.0
    fr: float = 0.0
    em: float = 0.0
    per: float = 0.0
    div: float | None = None


@dataclass
class Race:
    index: int
    runners: list = field(default_factory=list)

    @property
    def live(self):
        return [r for r in self.runners if not r.scratched]


@dataclass
class Meeting:
    name: str = ""
    region: str = "OTHER"
    races: list = field(default_factory=list)
    warnings: list = field(default_factory=list)


def _num(v, default=0.0):
    if v is None:
        return default
    s = str(v).strip().replace("$", "").replace(",", "").replace("%", "")
    if s == "" or s.lower() in ("scr", "fu", "-", "n/a", "na"):
        return default
    try:
        return float(s)
    except ValueError:
        return default


def _rows(text: str):
    """Yield row lists, sniffing comma vs tab."""
    sample = "\n".join(text.splitlines()[:12])
    delim = "\t" if sample.count("\t") > sample.count(",") else ","
    return list(csv.reader(io.StringIO(text), delimiter=delim))


def meeting_name_from_filename(name: str) -> str:
    """'Tuesday, 01st September 2026 - Ripon Races Worksheets (1).csv'."""
    n = re.sub(r"\.csv$", "", name or "", flags=re.I)
    if " - " in n:
        n = n.split(" - ", 1)[1]
    n = re.sub(r"\s*Races?\s*Worksheets?.*$", "", n, flags=re.I)
    n = re.sub(r"\s*\(\d+\)\s*$", "", n)
    return n.strip().replace("-", " ").title() or "Meeting"


def parse(text: str, meeting: str = "") -> Meeting:
    mt = Meeting(name=meeting or "Meeting")
    mt.region = region_for(mt.name)
    cur, idx = [], 0
    for row in _rows(text):
        if not row or not any(c.strip() for c in row):
            if cur:
                idx += 1
                mt.races.append(Race(idx, cur))
                cur = []
            continue
        head = row[0].strip()
        if head.lower() == "tab":
            if cur:
                idx += 1
                mt.races.append(Race(idx, cur))
                cur = []
            continue
        d = dict(zip(COLUMNS, [c.strip() for c in row]))
        digits = re.sub(r"\D", "", d.get("Tab", ""))
        if not digits:
            continue
        vals = {k.lower().replace("12m", "m12"): _num(d.get(k))
                for k in NUMERIC}
        r = Runner(
            tab=int(digits), horse=(d.get("Horse") or "").strip(),
            scratched=any((d.get(k, "") or "").strip().lower() == "scr"
                          for k in ("PER", "EM", "DIV")),
            first_up=(d.get("RFS", "") or "").strip().upper() == "FU",
            rfs=(None if (d.get("RFS", "") or "").strip().upper() == "FU"
                 else _num(d.get("RFS"), None)),
            div=_num(d.get("DIV"), None), **vals)
        if not r.horse:
            continue
        cur.append(r)
    if cur:
        idx += 1
        mt.races.append(Race(idx, cur))

    for rc in mt.races:
        if not rc.live:
            mt.warnings.append(f"Race {rc.index} has no unscratched runners.")
        tot = sum(r.per for r in rc.live)
        if rc.live and not (80.0 <= tot <= 120.0):
            mt.warnings.append(
                f"Race {rc.index}: the PER column sums to {tot:.0f}%, not "
                f"~100%. It is renormalised, but check the paste.")
    return mt
