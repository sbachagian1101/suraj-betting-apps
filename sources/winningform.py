"""Winning Form legacy racecards for South Africa (plain HTML, no login).

``cover.htm`` lists the next meetings and their race pages.  A race page has a
predicted-finish table (merit rating, mass, draw, jockey, trainer, weeks since
last run, last-run finish/lengths/distance, early bookmaker price), a
trainer-jockey combination table, and one block per horse with the jockey's and
trainer's last-30 records, wet/course/distance records, career record, stakes
and every recent run with date, course, going, class, distance, jockey, mass,
merit rating, draw-of-field, finish, lengths behind, winner, time, odds and a
comment.
"""
from __future__ import annotations

import re
from datetime import date, datetime
from typing import Optional

from bs4 import BeautifulSoup

from common import (Entry, PastRun, RaceCard, SourceError, cached, frac_to_decimal,
                    get_text, venue_matches)

NAME = "Winning Form"
BASE = "https://legacy.winningform.co.za/"
GOING = {"G": "GOOD", "Y": "YIELDING", "S": "SOFT", "H": "HEAVY", "F": "FIRM", "P": "POLY"}
_RE_MEETING = re.compile(r"^(.*?)\s*\((\w+)\)\s+([A-Z][a-z]+day),\s*(\d{1,2})\s+([A-Z][a-z]+)\s+(\d{4})")


def cover() -> list[dict]:
    """[{venue, code, date, races: {n: href}}] from cover.htm."""
    def fetch():
        html = get_text(BASE + "cover.htm")
        return parse_cover(html)
    return cached(("wf_cover",), 900, fetch)


def parse_cover(html: str) -> list[dict]:
    """The cover page is a sequence of ``<span class="rb2">Venue (CODE)<br>Weekday, D Month
    YYYY</span>`` headings, each followed by a table of ``Race N`` links."""
    soup = BeautifulSoup(html, "lxml")
    out: list[dict] = []
    for span in soup.find_all("span", class_="rb2"):
        t = re.sub(r"\s+", " ", span.get_text(" ", strip=True))
        m = _RE_MEETING.match(t)
        if not m:
            continue
        try:
            d = datetime.strptime(f"{m.group(4)} {m.group(5)} {m.group(6)}", "%d %B %Y").date()
        except ValueError:
            continue
        mt = {"venue": m.group(1).strip(), "code": m.group(2), "date": d, "races": {}}
        table = span.find_next("table")
        for a in (table.find_all("a", href=True) if table else []):
            rm = re.match(r"^Race\s+(\d+)$", a.get_text(" ", strip=True))
            if rm:
                mt["races"].setdefault(int(rm.group(1)), a["href"])
        if mt["races"]:
            out.append(mt)
    return out


def _f(s: str) -> Optional[float]:
    m = re.search(r"-?\d+(?:\.\d+)?", (s or "").replace(",", ""))
    return float(m.group(0)) if m else None


def _i(s: str) -> Optional[int]:
    m = re.search(r"\d+", s or "")
    return int(m.group(0)) if m else None


def _rec(s: str) -> tuple[int, int, int, int]:
    m = re.match(r"^\s*(\d+):(\d+)-(\d+)-(\d+)", s or "")
    if not m:
        return (0, 0, 0, 0)
    st, w, a, b = (int(x) for x in m.groups())
    return (st, w, a, b)


def _jt(s: str) -> tuple[str, Optional[tuple[int, int, int]]]:
    """'Callan MURRAY 56 * 30: 4 -3-3' -> ('Callan Murray', (30, 4, 10))."""
    m = re.match(r"^(.+?)\s+\d+\s*\*\s*(\d+):\s*(\d+)\s*-\s*(\d+)\s*-\s*(\d+)", s or "")
    if not m:
        return (s or "").strip().title(), None
    n, w, a, b = (int(x) for x in m.groups()[1:])
    return m.group(1).strip().title(), (n, w, w + a + b)


def parse_race(html: str, race_day: date) -> RaceCard:
    soup = BeautifulSoup(html, "lxml")
    card = RaceCard(country="SA", race_date=race_day, currency="ZAR", sources=[NAME])
    txt = re.sub(r"\s+", " ", soup.get_text(" ", strip=True))
    m = re.search(r"form guide\s+(.+?)\s+(\d{2}/\d{2}/\d{4})\s+(\d+)\s+(\d{1,2}\.\d{2})\s+(.+?)\s+(\d{3,4})\s+Metres", txt)
    if m:
        card.venue = m.group(1).strip()
        card.race_no = int(m.group(3))
        card.start_time = m.group(4).replace(".", ":")
        card.name = m.group(5).strip()
        card.dist_m = int(m.group(6))
    cm = re.search(r"Metres\s*\([A-Z]\)\s*(.+?)\s+(?:Apprentice|\(Course Record|R\s?\d)", txt)
    if cm:
        card.race_class = cm.group(1).strip()
    pm = re.search(r"\bR\s?(\d{4,7})\b", txt)
    if pm:
        card.prize = float(pm.group(1))
    card.surface = "AW" if re.search(r"Polytrack|Poly", card.venue or "", re.I) else "TURF"
    tables = soup.find_all("table")
    summary = None
    combos = None
    for t in tables:
        rows = t.find_all("tr")
        if len(rows) < 2:
            continue
        hdr = [c.get_text(" ", strip=True) for c in rows[1].find_all(["td", "th"])]
        if summary is None and "Merit Rated" in hdr and "Len Beh" in hdr:
            summary = (hdr, rows[2:])
        if combos is None and "Trainer" in hdr and "Jockey" in hdr and "Rns" in hdr:
            combos = rows[2:]
    entries: dict[int, Entry] = {}
    if summary:
        hdr, rows = summary
        for tr in rows:
            cells = [c.get_text(" ", strip=True) for c in tr.find_all("td")]
            if len(cells) < 6 or not cells[0].isdigit():
                continue
            e = Entry(number=int(cells[0]), name=cells[1].title(), sources=[NAME])
            if cells[2].startswith("*"):
                # first-run horse: No, Horse, **, Mass, Dr, Jockey, Trainer, ** note
                e.extras["first_run"] = True
                e.weight = _f(cells[3])
                e.barrier = _i(cells[4])
                e.jockey = cells[5].title() if len(cells) > 5 else ""
                e.trainer = cells[6].title() if len(cells) > 6 else ""
                entries[e.number] = e
                continue
            if len(cells) < 11:
                continue
            e.extras["predicted_lengths_behind"] = _f(cells[2])
            e.extras["speed_index"] = _i(cells[3])
            e.extras["predicted_time"] = _f(cells[4])
            e.official_rating = _f(cells[5])
            e.weight = _f(cells[6])
            e.barrier = _i(cells[7])
            e.gear = cells[8]
            e.jockey = cells[9].title()
            e.trainer = cells[10].title()
            if len(cells) > 11:
                wks = _i(cells[11])
                e.last_run_days = wks * 7 if wks is not None else None
            if len(cells) > 16:
                e.odds = frac_to_decimal(cells[16])
            entries[e.number] = e
    combo_stats: dict[int, tuple[int, int, int]] = {}
    if combos:
        for tr in combos:
            cells = [c.get_text(" ", strip=True) for c in tr.find_all("td")]
            for off in (0, 9):
                if len(cells) > off + 6 and cells[off].isdigit():
                    n, w, s2, t3 = (_i(cells[off + 3]) or 0, _i(cells[off + 4]) or 0,
                                    _i(cells[off + 5]) or 0, _i(cells[off + 6]) or 0)
                    combo_stats[int(cells[off])] = (n, w, w + s2 + t3)
    # per-horse blocks: a table whose first cell starts with "<no> <price> [merit]"
    for t in tables:
        rows = t.find_all("tr")
        if not rows:
            continue
        first = [c.get_text(" ", strip=True) for c in rows[0].find_all("td", recursive=False)]
        if not first:
            continue
        hm = re.match(r"^(\d{1,2})\s+(\S+)(?:\s+(\d+))?$", first[0])
        if not hm or len(first) < 2:
            continue
        num = int(hm.group(1))
        e = entries.get(num) or Entry(number=num, sources=[NAME])
        entries[num] = e
        if not e.name:
            nm = re.match(r"^([A-Z' .\-]+?)(?:\s*\(|\s+\d)", first[1])
            e.name = (nm.group(1) if nm else first[1]).strip().title()
        if e.odds is None:
            e.odds = frac_to_decimal(hm.group(2))
        if hm.group(3) and e.official_rating is None:
            e.official_rating = float(hm.group(3))
        am = re.search(r"(\d)\s*y\.o\.\s*([a-z]+)\s*([a-z])\.", first[1])
        if am:
            e.age, e.sex = int(am.group(1)), am.group(3).upper()
        # "61.0 4 6" = mass, draw, field size
        for c in first[2:9]:
            wm = re.match(r"^(\d{2}\.\d)\s+(\d{1,2})\s+(\d{1,2})$", c)
            if wm:
                e.weight = e.weight or float(wm.group(1))
                e.barrier = e.barrier or int(wm.group(2))
                e.extras["field_size"] = int(wm.group(3))
                break
        block = re.sub(r"\s+", " ", t.get_text(" ", strip=True))
        jt = re.search(r"([A-Z][a-zA-Z' .\-]+\s\d+\s\*\s\d+:\s*\d+\s*-\d+-\d+)\s+([A-Z][a-zA-Z' .\-/]+\s\d+\s\*\s\d+:\s*\d+\s*-\d+-\d+)", block)
        if jt:
            jn, js = _jt(jt.group(1))
            tn, ts = _jt(jt.group(2))
            e.jockey = e.jockey or jn
            e.trainer = e.trainer or tn
            e.jockey_stats, e.trainer_stats = js, ts
        for key, label in (("Wet", "Wet"), ("Crs", "Course"), ("Dst", "Dist"), ("C&D", "C&D")):
            rm = re.search(re.escape(key) + r":\s*(\d+:\d+-\d+-\d+)", block)
            if rm:
                e.records[label] = _rec(rm.group(1))
        tr_ = re.search(r"Tot Rns:\s*(\d+:\d+-\d+-\d+)", block)
        if tr_:
            e.career = _rec(tr_.group(1))
        sk = re.search(r"Stakes:\s*R\s*([\d,]+)", block)
        if sk:
            e.prize_money = _f(sk.group(1))
        if "FIRST RUN" in block.upper():
            e.extras["first_run"] = True
        # past runs: rows whose first cell is "(wks) yy.mm.dd"
        for tr in t.find_all("tr"):
            cells = [c.get_text(" ", strip=True) for c in tr.find_all("td")]
            if len(cells) < 14:
                continue
            dm = re.match(r"^\((\d+)\)\s*(\d{2})\.(\d{2})\.(\d{2})$", cells[0])
            if not dm:
                continue
            try:
                d = date(2000 + int(dm.group(2)), int(dm.group(3)), int(dm.group(4)))
            except ValueError:
                continue
            run = PastRun(run_date=d, days_ago=(race_day - d).days, track=cells[1],
                          going=GOING.get(cells[3], cells[3]), race_class=cells[4],
                          dist_m=_i(cells[6]), jockey=cells[7].title(), weight=_f(cells[8]),
                          currency="ZAR", surface="TURF")
            if cells[9]:
                run.rating = _f(cells[9])
            df = re.match(r"^(\d+)\s*-\s*(\d+)$", cells[11].replace(" ", ""))
            if df:
                run.barrier, run.field_size = int(df.group(1)), int(df.group(2))
            pos = _i(cells[12])
            lb = _f(cells[13])
            if pos:
                run.pos = pos
                run.margin_l = -0.5 if pos == 1 and (lb or 0) == 0 else lb
            else:
                run.non_finish = True
            if len(cells) > 14:
                run.comment = cells[14]
            if len(cells) > 15:
                run.time_s = _f(cells[15])
            if len(cells) > 17:
                run.sp = frac_to_decimal(cells[17])
            if len(cells) > 20:
                run.comment = cells[20] or run.comment
            e.runs.append(run)
        e.runs.sort(key=lambda r: r.run_date or date.min, reverse=True)
        if e.runs and e.last_run_days is None:
            e.last_run_days = e.runs[0].days_ago
    for num, st in combo_stats.items():
        if num in entries:
            entries[num].combo_stats = st
    card.entries = [entries[k] for k in sorted(entries)]
    if not card.entries:
        raise SourceError(f"{NAME}: no runners parsed")
    return card


def fetch(day: date, venue: str, race_no: int, country: str = "SA") -> RaceCard:
    for mt in cover():
        if mt["date"] == day and venue_matches(mt["venue"], venue):
            href = mt["races"].get(race_no)
            if not href:
                raise SourceError(f"{NAME}: {mt['venue']} has no race {race_no}")
            url = BASE + href.lstrip("/")
            html = cached(("wf_race", url), 600, lambda: get_text(url))
            card = parse_race(html, day)
            card.venue = card.venue or mt["venue"]
            card.race_no = card.race_no or race_no
            return card
    have = ", ".join(f"{m['venue']} {m['date']}" for m in cover())
    raise SourceError(f"{NAME}: no {venue} on {day} (listed: {have or 'nothing'})")
