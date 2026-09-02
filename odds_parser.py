"""Parse a pasted TABtouch (or similar) betting screen into runners.

The screen repeats one block per runner:

    8
    Add to Blackbook
    TORRITA
    Opening Win4.80Top Win4.80
    4.20          <- fixed win
    1.80          <- fixed place
    4.80Arrow     <- tote/second-source win
    2.10          <- tote/second-source place

Scratched runners replace the four price lines with a Scratched marker, and
the header carries meeting, race name, class, weather, going and distance.

Two traps this parser exists to avoid:

* The header line reads "Stewards Comments Overcast Soft5 1400m 8,9,12,2".
  That trailing number list looks exactly like a first-four result and is not
  one -- Ipswich showed "1,11,6,4" and the race was won by 6 from 11, 12, 2.
  It is a list of runners carrying stewards comments, and it is discarded.
* A runner number and a price are both bare numbers on their own line, so
  numbers are only read positionally after a name has been seen.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field


NOISE = {
    "add to blackbook", "winplace", "quinella", "exacta", "trifecta",
    "first 4", "first four", "double", "quaddie", "fd", "field",
    "add field to bet slip", "more betting options", "odds vs evens",
    "betting conditions", "rules", "mystery quick pick", "choose...",
    "win", "place", "racing", "sports", "sign up", "log in", "help",
    "media", "results", "blog", "races", "trots", "dogs", "all racing",
    "favourite nos", "mystery bet", "jackpots", "next 15", "next race",
    "go to next race", "tabtouch", "pool totals", "bet slip", "bet",
    "start betting!", "no active bets", "next races", "next jackpots",
    "all", "more", "chat now", "contact us",
}

# "Opening Win4.80Top Win4.80"  (either half may be absent)
OPEN_TOP = re.compile(
    r"Opening\s*Win\s*([\d.]+)\s*(?:Top\s*Win\s*([\d.]+))?", re.I)
# "CARLTON DRAUGHT ... Soft5 1400m" / "Fine Soft5 1700m"
COND = re.compile(
    r"\b(Firm|Good|Soft|Heavy|Synthetic|AW)\s*(\d+)?\b.*?\b(\d{3,4})\s*m\b",
    re.I)
PRICE = re.compile(r"^\$?([\d]+(?:\.[\d]+)?)\s*(?:Arrow)?$", re.I)
NUMBER = re.compile(r"^(\d{1,2})$")
NAME = re.compile(r"^[A-Z][A-Z'’\-. ]{2,}(?:\([A-Z0-9]{1,4}\))?"
                  r"(?:\s*\((?:EM\d|NZ|GB|IRE|USA|GER|FR|AUS|JPN)\))*\s*$")


@dataclass
class Runner:
    number: int
    name: str
    scratched: bool = False
    opening: float | None = None
    top: float | None = None
    fixed_win: float | None = None
    fixed_place: float | None = None
    tote_win: float | None = None
    tote_place: float | None = None

    @property
    def priced(self) -> bool:
        return (not self.scratched and self.fixed_win
                and self.fixed_win > 1.0)


@dataclass
class Race:
    meeting: str = ""
    title: str = ""
    race_class: str = ""
    going: str = ""
    distance: int | None = None
    weather: str = ""
    runners: list = field(default_factory=list)
    notes: list = field(default_factory=list)

    @property
    def active(self):
        return [r for r in self.runners if r.priced]

    @property
    def scratched(self):
        return [r for r in self.runners if r.scratched]


def _clean(text: str):
    for raw in text.replace("\r", "").split("\n"):
        s = raw.replace("\t", " ").strip()
        if not s:
            continue
        if s.lower().rstrip(":") in NOISE:
            continue
        yield s


def _is_name(s: str) -> bool:
    if not NAME.match(s):
        return False
    letters = [c for c in s if c.isalpha()]
    if len(letters) < 3:
        return False
    # a heading such as "CARLTON DRAUGHT CLASS ONE HANDICAP" is long; runner
    # names on this screen are short. Cut generously but not blindly.
    return len(s) <= 34


def parse(text: str) -> Race:
    race = Race()
    lines = list(_clean(text))

    # ---------------------------------------------------------- header block
    for s in lines[:40]:
        m = COND.search(s)
        if m and race.distance is None:
            race.going = (m.group(1).title()
                          + (m.group(2) or "")).strip()
            race.distance = int(m.group(3))
            w = re.match(r"^(?:Stewards Comments\s*)?([A-Za-z ]+?)\s+"
                         r"(?:Firm|Good|Soft|Heavy)", s, re.I)
            if w:
                race.weather = w.group(1).strip()
            # the trailing "8,9,12,2" is a stewards-comment runner list, NOT
            # a result. Record it as a note and never as finishing order.
            tail = re.search(r"\d{3,4}\s*m\s+([\d]+(?:\s*,\s*[\d]+)+)", s)
            if tail:
                race.notes.append(
                    "Header carries the runner list " + tail.group(1)
                    + " -- these are stewards comments, not a result.")
            break
    for s in lines[:12]:
        if re.match(r"^[A-Z]{1,3}\s+[A-Z][a-zA-Z' \-]+$", s):
            race.meeting = s
            break
    for s in lines[:14]:
        if len(s) > 12 and s.upper() == s and not COND.search(s) \
                and "STEWARD" not in s.upper() and not race.title:
            race.title = s
    for s in lines[:16]:
        if re.match(r"^(CL\d|BM\d+|MDN|HCP\d*|G[123]|LR|SW|OPEN)"
                    r"[\w \-]*$", s, re.I) and len(s) < 24:
            race.race_class = s
            break

    # --------------------------------------------------------- runner blocks
    i, pending = 0, None
    while i < len(lines):
        s = lines[i]
        nm = NUMBER.match(s)
        if nm:
            pending = int(nm.group(1))
            i += 1
            continue
        if pending is not None and _is_name(s):
            r = Runner(number=pending, name=s.strip())
            pending = None
            nums, j = [], i + 1
            while j < len(lines) and j < i + 9:
                t = lines[j]
                if re.search(r"scratch", t, re.I):
                    r.scratched = True
                    break
                mo = OPEN_TOP.search(t)
                if mo:
                    r.opening = float(mo.group(1))
                    r.top = float(mo.group(2)) if mo.group(2) else None
                    j += 1
                    continue
                # A bare runner number such as "12" also matches PRICE, so
                # the next runner has to be detected BEFORE prices are read
                # or its number is swallowed as a fifth price and the whole
                # runner disappears.
                if NUMBER.match(t) or _is_name(t) or len(nums) >= 4:
                    break
                mp = PRICE.match(t)
                if mp:
                    nums.append(float(mp.group(1)))
                    j += 1
                    continue
                j += 1
            if not r.scratched and len(nums) >= 2:
                r.fixed_win, r.fixed_place = nums[0], nums[1]
                if len(nums) >= 4:
                    r.tote_win, r.tote_place = nums[2], nums[3]
            if r.opening == 0.0:
                r.scratched = True
            race.runners.append(r)
            i = j
            continue
        i += 1

    seen, uniq = set(), []
    for r in race.runners:
        if r.number in seen:
            continue
        seen.add(r.number)
        uniq.append(r)
    race.runners = uniq
    return race
