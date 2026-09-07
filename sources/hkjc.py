"""Hong Kong Jockey Club racing information pages (server-rendered HTML).

* race card: ``/en-us/local/information/racecard?racedate=YYYY/MM/DD&racecourse=HV|ST&raceno=N``
  with rating, rating change, draw, weight (lb), declared body weight, jockey,
  trainer, gear, days since last run, season stakes and the last-6 form string;
* horse page: every past run with class, draw, rating, LBW, win odds, running
  positions and finish time;
* sectional page for a runner's latest race: split times and positions per 400 m.

Odds are not on these pages (the betting site is JavaScript-only); the pipeline
adds exchange prices from Betfair instead.
"""
from __future__ import annotations

import re
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime
from typing import Optional

from bs4 import BeautifulSoup

from common import (Entry, PastRun, RaceCard, SourceError, cached, get_text,
                    lengths_from_text, norm_name)

NAME = "HKJC"
BASE = "https://racing.hkjc.com"
COURSES = {"HAPPY VALLEY": "HV", "SHA TIN": "ST", "HV": "HV", "ST": "ST"}
LB_TO_KG = 0.45359237


def course_code(venue: str) -> Optional[str]:
    n = norm_name(venue)
    for k, v in COURSES.items():
        if n.startswith(k) or k.startswith(n):
            return v
    return None


def _table_with(soup: BeautifulSoup, must: tuple[str, ...]):
    for t in soup.find_all("table"):
        rows = t.find_all("tr")
        if len(rows) < 2:
            continue
        hdr = [c.get_text(" ", strip=True) for c in rows[0].find_all(["th", "td"])]
        if all(any(m in h for h in hdr) for m in must):
            return hdr, rows
    return None, []


def _int(s: str) -> Optional[int]:
    m = re.search(r"-?\d+", s or "")
    return int(m.group(0)) if m else None


def _float(s: str) -> Optional[float]:
    m = re.search(r"-?\d+(?:\.\d+)?", (s or "").replace(",", ""))
    return float(m.group(0)) if m else None


def _time_s(s: str) -> Optional[float]:
    m = re.match(r"^\s*(\d+)[:.](\d{2})\.(\d{2})\s*$", s or "")
    if m:
        return int(m.group(1)) * 60 + int(m.group(2)) + int(m.group(3)) / 100.0
    m = re.match(r"^\s*(\d+)\.(\d{2})\s*$", s or "")
    if m:
        return float(s)
    return None


def parse_racecard(html: str, race_day: date) -> RaceCard:
    soup = BeautifulSoup(html, "lxml")
    card = RaceCard(country="HK", race_date=race_day, currency="HKD", sources=[NAME])
    txt = re.sub(r"\s+", " ", soup.get_text(" ", strip=True))
    m = re.search(r"Race\s+(\d+)\s*-\s*([A-Z0-9' .&\-]+?)\s+([A-Z][a-z]+day,\s+[A-Z][a-z]+ \d{1,2}, \d{4}),\s*"
                  r"([A-Za-z ]+?),\s*(\d{1,2}:\d{2})\s*(.*?)Prize Money:\s*\$?([\d,]+),?\s*(?:Rating:\s*([\d\-]+),?)?\s*(Class \d|Group \d|Griffin[^,]*|[A-Z][A-Za-z ]+?)(?:\s+SETUP|\s+HORSE|$)",
                  txt)
    if m:
        card.race_no = int(m.group(1))
        card.name = m.group(2).strip().title()
        card.venue = m.group(4).strip()
        card.start_time = m.group(5)
        cond = m.group(6)
        dm = re.search(r"(\d{3,4})M", cond)
        card.dist_m = int(dm.group(1)) if dm else None
        card.surface = "AW" if "All Weather" in cond or "AWT" in cond else "TURF"
        cm = re.search(r'"([A-C+\d]+)"\s*Course', cond)
        if cm:
            card.rail = f'"{cm.group(1)}" course'
        card.prize = _float(m.group(7))
        card.race_class = (m.group(9) or "").strip()
        if m.group(8):
            card.extras["rating_band"] = m.group(8)
    gm = re.search(r"Going\s*:?\s*([A-Z][A-Z ]+?)(?:\s{2,}|\s+[A-Z][a-z])", txt)
    if gm:
        card.going = gm.group(1).strip()
    hdr, rows = _table_with(soup, ("Horse No", "Jockey", "Draw"))
    if not rows:
        raise SourceError(f"{NAME}: race card table not found")
    idx = {h: i for i, h in enumerate(hdr)}

    def cell(cells, key, default=""):
        for h, i in idx.items():
            if h.startswith(key) and i < len(cells):
                return cells[i]
        return default

    for tr in rows[1:]:
        tds = tr.find_all("td")
        cells = [c.get_text(" ", strip=True) for c in tds]
        if len(cells) < 8 or not cells[0].strip():
            continue
        num = _int(cells[0])
        name = cell(cells, "Horse", "")
        if idx.get("Horse") is not None:
            name = cells[idx["Horse"]] if idx["Horse"] < len(cells) else name
        e = Entry(number=num, name=name.title(), sources=[NAME])
        toks = [t for t in cell(cells, "Last 6").split("/") if t.strip()]
        # HKJC prints oldest -> latest; the model wants latest first, one char per run
        e.form_string = "".join((t if t.isdigit() and len(t) == 1 else ("0" if t.isdigit() else t[:1]))
                                for t in reversed(toks))
        wt = _float(cell(cells, "Wt."))
        e.weight = round(wt * LB_TO_KG, 1) if wt else None
        e.jockey = cell(cells, "Jockey")
        e.barrier = _int(cell(cells, "Draw"))
        e.trainer = cell(cells, "Trainer")
        e.official_rating = _float(cell(cells, "Rtg."))
        e.extras["rating_change"] = _int(cell(cells, "Rtg.+/-"))
        e.extras["body_weight"] = _int(cell(cells, "Horse Wt."))
        e.extras["body_weight_change"] = _int(cell(cells, "Wt.+/-"))
        e.extras["best_time"] = cell(cells, "Best Time")
        e.age = _int(cell(cells, "Age"))
        e.sex = cell(cells, "Sex").upper()
        e.prize_money = _float(cell(cells, "Season Stakes"))
        e.last_run_days = _int(cell(cells, "Days since"))
        e.gear = cell(cells, "Gear")
        if "Priority" in idx:
            e.extras["priority"] = cell(cells, "Priority")
        scratched = tr.get("class") and any("scratch" in c.lower() for c in tr.get("class"))
        e.scratched = bool(scratched) or "Scratched" in " ".join(cells) or (e.jockey == "" and e.barrier is None)
        for a in tr.find_all("a", href=True):
            if "horseid=" in a["href"].lower():
                e.extras["horse_url"] = BASE + a["href"] if a["href"].startswith("/") else a["href"]
                break
        card.entries.append(e)
    if not card.entries:
        raise SourceError(f"{NAME}: no runners on the race card")
    return card


def parse_horse(html: str, race_day: date) -> tuple[list[PastRun], tuple[int, int, int, int], dict]:
    soup = BeautifulSoup(html, "lxml")
    hdr, rows = _table_with(soup, ("Pla.", "Date", "Dist."))
    runs: list[PastRun] = []
    if rows:
        idx = {h: i for i, h in enumerate(hdr)}

        def cell(cells, key):
            for h, i in idx.items():
                if h.startswith(key) and i < len(cells):
                    return cells[i]
            return ""

        for tr in rows[1:]:
            cells = [c.get_text(" ", strip=True) for c in tr.find_all("td")]
            if len(cells) < 10:
                continue
            ds = cell(cells, "Date")
            try:
                d = datetime.strptime(ds, "%d/%m/%y").date()
            except ValueError:
                continue
            pla = cell(cells, "Pla.")
            pos = _int(pla)
            run = PastRun(run_date=d, days_ago=(race_day - d).days, track=cell(cells, "RC"),
                          race_name=cell(cells, "Race Index"),
                          dist_m=_int(cell(cells, "Dist.")), going=cell(cells, "G"),
                          race_class=cell(cells, "Race Class"), barrier=_int(cell(cells, "Dr.")),
                          rating=_float(cell(cells, "Rtg.")), jockey=cell(cells, "Jockey"),
                          sp=_float(cell(cells, "Win Odds")), running_pos=cell(cells, "Running"),
                          time_s=_time_s(cell(cells, "Finish Time")), currency="HKD",
                          surface="AW" if "AWT" in cell(cells, "RC") else "TURF")
            wt = _float(cell(cells, "Act. Wt."))
            run.weight = round(wt * LB_TO_KG, 1) if wt else None
            lbw = cell(cells, "LBW")
            if pos and re.match(r"^\d+", pla) and not re.search(r"[A-Z]{2,}", pla):
                run.pos = pos
                run.margin_l = -0.5 if pos == 1 else lengths_from_text(lbw)
                if run.margin_l is None:
                    run.margin_l = 1.5 * (pos - 1)
            else:
                run.non_finish = True
                run.comment = pla
            run.field_size = 12          # HK fields are 8-14; imputed, never printed
            runs.append(run)
    txt = re.sub(r"\s+", " ", soup.get_text(" ", strip=True))
    career = (0, 0, 0, 0)
    m = re.search(r"No\. of 1-2-3-Starts\*?\s*:?\s*(\d+)-(\d+)-(\d+)-(\d+)", txt)
    if m:
        w, s, t, st = (int(x) for x in m.groups())
        career = (st, w, s, t)
    extras = {}
    m = re.search(r"Total Stakes\*?\s*:?\s*\$?([\d,]+)", txt)
    if m:
        extras["total_stakes"] = _float(m.group(1))
    m = re.search(r"Current Rating\s*:?\s*(\d+)", txt)
    if m:
        extras["current_rating"] = int(m.group(1))
    m = re.search(r"Season Stakes\*?\s*:?\s*\$?([\d,]+)", txt)
    if m:
        extras["season_stakes"] = _float(m.group(1))
    return runs, career, extras


def parse_sectional(html: str) -> dict[str, dict]:
    """Horse name -> {'splits': [...], 'positions': [...], 'last_400': float, 'time': float}."""
    soup = BeautifulSoup(html, "lxml")
    out: dict[str, dict] = {}
    for t in soup.find_all("table"):
        rows = t.find_all("tr")
        hdr = " ".join(c.get_text(" ", strip=True) for c in rows[0].find_all(["th", "td"])) if rows else ""
        if "Running Position" not in hdr:
            continue
        for tr in rows[1:]:
            cells = [c.get_text(" ", strip=True) for c in tr.find_all("td")]
            if len(cells) < 5 or not cells[0].isdigit():
                continue
            name = re.sub(r"\s*\([A-Z]\d+\)\s*$", "", cells[2]).replace("\xa0", " ").strip()
            splits, positions = [], []
            for c in cells[3:-1]:
                parts = c.split()
                if not parts:
                    continue
                if parts[0].isdigit():
                    positions.append(int(parts[0]))
                nums = [float(x) for x in parts if re.match(r"^\d+\.\d+$", x)]
                if nums:
                    splits.append(nums[0])
            out[norm_name(name)] = {"splits": splits, "positions": positions,
                                    "last_400": splits[-1] if splits else None,
                                    "time": _time_s(cells[-1])}
    return out


def fetch(day: date, venue: str, race_no: int, country: str = "HK", with_history: bool = True) -> RaceCard:
    code = course_code(venue) or "ST"
    url = f"{BASE}/en-us/local/information/racecard"
    params = {"racedate": day.strftime("%Y/%m/%d"), "racecourse": code, "raceno": race_no}
    html = cached(("hk_card", params["racedate"], code, race_no), 300, lambda: get_text(url, params))
    card = parse_racecard(html, day)
    card.race_no = card.race_no or race_no
    card.venue = card.venue or venue
    if with_history:
        def load(e: Entry):
            u = e.extras.get("horse_url")
            if not u:
                return
            try:
                h = cached(("hk_horse", u), 3600, lambda: get_text(u))
                runs, career, ex = parse_horse(h, day)
                e.runs = runs
                if career[0]:
                    e.career = career
                e.extras.update(ex)
                if ex.get("total_stakes"):
                    e.prize_money = ex["total_stakes"]
            except SourceError as exc:
                card.warnings.append(f"{NAME}: {e.name}: {exc}")
        with ThreadPoolExecutor(max_workers=6) as pool:
            list(pool.map(load, card.entries))
        # Sectionals for each runner's latest run.  The sectional page is addressed by
        # date + race NUMBER of that day, while the horse page shows the season race
        # INDEX; the results page for (date, race 1) lists the day's races with their
        # indices, which is the cheapest reliable mapping.
        def race_no_for(day_: date, index: str) -> Optional[int]:
            rurl = f"{BASE}/racing/information/English/Racing/LocalResults.aspx"
            rp = {"RaceDate": day_.strftime("%Y/%m/%d")}
            try:
                rh = cached(("hk_res_day", rp["RaceDate"]), 86400, lambda: get_text(rurl, rp))
            except SourceError:
                return None
            mm = re.findall(r"RaceNo=(\d+)[^>]*>\s*(?:Race\s*)?(\d+)\s*<", rh)
            tx = re.sub(r"\s+", " ", BeautifulSoup(rh, "lxml").get_text(" ", strip=True))
            for m in re.finditer(r"RACE\s+(\d+)\s+\((\d+)\)", tx):
                if m.group(2) == index:
                    return int(m.group(1))
            return None

        def load_sec(e: Entry):
            last = next((r for r in e.runs if r.run_date and not r.non_finish), None)
            if not last or not last.race_name.isdigit():
                return
            no = race_no_for(last.run_date, last.race_name)
            if not no:
                return
            surl = f"{BASE}/racing/information/English/Racing/DisplaySectionalTime.aspx"
            sp = {"RaceDate": last.run_date.strftime("%d/%m/%Y"), "RaceNo": no}
            try:
                sh = cached(("hk_sec", sp["RaceDate"], no), 86400, lambda: get_text(surl, sp))
                data = parse_sectional(sh).get(norm_name(e.name))
                if data and data.get("last_400"):
                    e.sectional_600 = data["last_400"]
                    e.extras["sectional"] = data
            except SourceError:
                pass
        with ThreadPoolExecutor(max_workers=6) as pool:
            list(pool.map(load_sec, card.entries))
    return card
