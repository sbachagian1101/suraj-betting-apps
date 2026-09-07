"""Horse Racing Nation entries for US tracks (plain HTML).

``entries.horseracingnation.com/entries-results/<track-slug>/<date>`` lists
each race with program number, post position, horse, sire, trainer, jockey,
scratch flag and morning-line odds.  Horse pages sit behind a JavaScript check,
so US past-run detail comes from the Ladbrokes form string instead.
"""
from __future__ import annotations

import re
from datetime import date

from bs4 import BeautifulSoup

from common import Entry, RaceCard, SourceError, cached, frac_to_decimal, get_text, imperial_to_m

NAME = "Horse Racing Nation"
BASE = "https://entries.horseracingnation.com"


def track_slug(venue: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", venue.lower()).strip("-")


def parse_track_page(html: str, race_no: int, race_day: date) -> RaceCard:
    soup = BeautifulSoup(html, "lxml")
    head = soup.find(id=f"race-{race_no}")
    if head is None:
        raise SourceError(f"{NAME}: race {race_no} not on the page")
    card = RaceCard(country="USA", race_no=race_no, race_date=race_day, currency="USD", sources=[NAME])
    tm = head.find("time")
    if tm:
        card.start_time = tm.get_text(" ", strip=True)
        if tm.get("datetime"):
            card.extras["start_iso"] = tm["datetime"]
    # conditions line follows the header: "6F, Dirt, Maiden Claiming | 3 Year Olds Purse: $20,800"
    cond = head.find_next(["p", "div"])
    desc = re.sub(r"\s+", " ", cond.get_text(" ", strip=True)) if cond else ""
    card.extras["conditions"] = desc
    dm = re.match(r"^\s*(\d+(?:\s+\d/\d)?)\s*([FMY])\b", desc, re.I)
    if dm:
        v = 0.0
        for part in dm.group(1).split():
            v += float(part.split("/")[0]) / float(part.split("/")[1]) if "/" in part else float(part)
        unit = dm.group(2).upper()
        card.dist_m = imperial_to_m(f"{v}f" if unit == "F" else (f"{v}m" if unit == "M" else f"{int(v)}y"))
    sm = re.search(r"\b(Dirt|Turf|Synthetic|Tapeta|All Weather)\b", desc, re.I)
    if sm:
        s = sm.group(1).upper()
        card.surface = "DIRT" if s == "DIRT" else ("TURF" if s == "TURF" else "AW")
    pm = re.search(r"Purse:?\s*\$([\d,]+)", desc, re.I)
    if pm:
        card.prize = float(pm.group(1).replace(",", ""))
    parts = [p.strip() for p in desc.split(",")]
    if len(parts) >= 3:
        card.race_class = parts[2].split("Purse")[0].strip()[:60]
    table = head.find_next("table")
    if table is None:
        raise SourceError(f"{NAME}: race {race_no} has no entries table")
    for tr in table.find_all("tr"):
        tds = tr.find_all("td")
        if len(tds) < 5:
            continue
        num_img = tds[0].find("img")
        num = None
        if num_img and (num_img.get("alt") or "").strip().isdigit():
            num = int(num_img["alt"].strip())
        else:
            lab = tds[0].get("data-label") or ""
            m = re.search(r"(\d+)", lab)
            num = int(m.group(1)) if m else None
        pp = tds[1].get_text(" ", strip=True)
        h4 = tds[2].find("h4")
        name = (h4.get_text(" ", strip=True) if h4 else tds[2].get_text(" ", strip=True)).strip()
        # "Coco No Loco (92)": the bracketed number is HRN's power rating
        rating = None
        rm = re.match(r"^(.*?)\s*\((\d{2,3})\)\s*$", name)
        if rm:
            name, rating = rm.group(1).strip(), float(rm.group(2))
        sire = tds[2].find("p")
        ps = tds[3].find_all("p")
        trainer = ps[0].get_text(" ", strip=True) if ps else ""
        jockey = ps[1].get_text(" ", strip=True) if len(ps) > 1 else ""
        scratch_cell = tds[4].get_text(" ", strip=True).lower()
        ml = tds[5].find("p").get_text(" ", strip=True) if len(tds) > 5 and tds[5].find("p") else (
            tds[5].get_text(" ", strip=True) if len(tds) > 5 else "")
        e = Entry(number=num, name=name, barrier=int(pp) if pp.isdigit() else None,
                  trainer=trainer, jockey=jockey, sources=[NAME],
                  scratched="scr" in scratch_cell or "scratched" in (tr.get("class") or []))
        if sire:
            e.extras["sire"] = sire.get_text(" ", strip=True)
        if rating is not None:
            e.official_rating = rating
            e.extras["hrn_power_rating"] = rating
        e.extras["morning_line"] = frac_to_decimal(ml)
        card.entries.append(e)
    if not card.entries:
        raise SourceError(f"{NAME}: no entries parsed for race {race_no}")
    return card


def fetch(day: date, venue: str, race_no: int, country: str = "USA") -> RaceCard:
    url = f"{BASE}/entries-results/{track_slug(venue)}/{day.isoformat()}"
    html = cached(("hrn", url), 600, lambda: get_text(url))
    card = parse_track_page(html, race_no, day)
    card.venue = venue
    return card
