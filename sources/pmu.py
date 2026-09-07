"""PMU open programme feed for French racing.

``programme/DDMMYYYY`` lists every reunion and course; ``.../participants``
carries the field (musique, career counts, earnings, jockey, trainer, draw,
weight, blinkers, shoes); ``.../performances-detaillees/pretty`` gives every past
run with the full finishing order, so beaten lengths can be rebuilt; the
``rapports`` endpoints give tote win odds on race day, with an opening reference.
"""
from __future__ import annotations

import re
from datetime import date, datetime, timezone
from typing import Optional

from common import (Entry, PastRun, RaceCard, SourceError, cached, get_json,
                    venue_matches)

NAME = "PMU"
BASE = "https://online.turfinfo.api.pmu.fr/rest/client/1"

_GAP_WORDS = {
    "NEZ": 0.05, "COURTE_TETE": 0.1, "TETE": 0.2, "COURTE_ENCOLURE": 0.25, "ENCOLURE": 0.3,
    "DEMI_LONGUEUR": 0.5, "TROIS_QUARTS_LONGUEUR": 0.75, "UNE_LONGUEUR": 1.0,
    "UNE_LONGUEUR_ET_DEMI": 1.5, "DEUX_LONGUEURS": 2.0, "DEUX_LONGUEURS_ET_DEMI": 2.5,
    "TROIS_LONGUEURS": 3.0, "LOIN": 15.0, "DEAD_HEAT": 0.0,
}


def gap_lengths(gap: dict | None) -> Optional[float]:
    """PMU 'distanceAvecPrecedent' -> lengths."""
    if not gap:
        return None
    kv = (gap.get("knownValue") or "").upper()
    if kv in _GAP_WORDS:
        return _GAP_WORDS[kv]
    raw = (gap.get("rawValue") or "").upper().replace("L", "").strip()
    m = re.match(r"^(\d+)?\s*(?:(\d)/(\d))?$", raw)
    if m and (m.group(1) or m.group(2)):
        return float(m.group(1) or 0) + (float(m.group(2)) / float(m.group(3)) if m.group(2) else 0)
    if "TETE" in raw:
        return 0.2
    if "ENCOL" in raw:
        return 0.3
    if "NEZ" in raw:
        return 0.05
    if "LOIN" in raw:
        return 15.0
    return None


def programme(day: date) -> dict:
    key = day.strftime("%d%m%Y")
    return cached(("pmu_prog", key), 600, lambda: get_json(f"{BASE}/programme/{key}") or {})


def find_course(day: date, venue: str, race_no: int) -> Optional[tuple[dict, dict]]:
    prog = programme(day).get("programme") or {}
    for reu in prog.get("reunions", []):
        hip = reu.get("hippodrome") or {}
        names = [hip.get("libelleCourt", ""), hip.get("libelleLong", "").replace("HIPPODROME DE ", "")
                 .replace("HIPPODROME D'", "")]
        if not any(venue_matches(n, venue) for n in names if n):
            continue
        for c in reu.get("courses", []):
            if int(c.get("numOrdre") or 0) == race_no or int(c.get("numExterne") or 0) == race_no:
                return reu, c
    return None


def parse_musique(mus: str) -> str:
    """'4a0a4a1a(25)6m' -> '4041-6': one character per run, latest first, unplaced as 0,
    D/T/A/R for disqualified, fell, stopped, retired."""
    out = []
    for tok in re.findall(r"\(\d+\)|\d+|[A-Za-z]", mus or ""):
        if tok.startswith("("):
            out.append("-")
        elif tok.isdigit():
            out.append(tok if len(tok) == 1 else "0")
        elif tok in ("D", "T", "A", "R"):
            out.append(tok)          # upper-case incidents; lower-case letters are the discipline
    return "".join(out)[:16]


def _past_runs(pp: dict, race_day: date) -> list[PastRun]:
    runs: list[PastRun] = []
    for cc in pp.get("coursesCourues", []) or []:
        try:
            d = datetime.fromtimestamp(cc["date"] / 1000, tz=timezone.utc).date()
        except (KeyError, TypeError, ValueError):
            d = None
        run = PastRun(run_date=d, days_ago=(race_day - d).days if d else None,
                      track=cc.get("hippodrome", ""), race_name=cc.get("nomPrix", ""),
                      race_class=cc.get("discipline", ""), prize=cc.get("allocation"),
                      currency="EUR", dist_m=cc.get("distance"),
                      field_size=cc.get("nbParticipants"), going=cc.get("etatTerrain") or "",
                      surface="TURF")
        if cc.get("tempsDuPremier"):
            run.time_s = cc["tempsDuPremier"] / 1000.0
        cum = 0.0
        found = False
        for p in cc.get("participants", []) or []:
            pl = (p.get("place") or {})
            pos = pl.get("place")
            if pos and pos > 1:
                g = gap_lengths(p.get("distanceAvecPrecedent"))
                cum += g if g is not None else 1.0
            if p.get("itsHim"):
                found = True
                run.pos = pos
                run.jockey = p.get("nomJockey") or ""
                run.weight = p.get("poidsJockey")
                run.barrier = p.get("corde")
                status = (pl.get("statusArrivee") or "").upper()
                if pos:
                    run.margin_l = -0.5 if pos == 1 else cum
                else:
                    run.non_finish = True
                    run.comment = pl.get("rawValue") or status
                if p.get("reductionKilometrique"):
                    run.extras_rk = p["reductionKilometrique"]
                break
        if found:
            runs.append(run)
    return runs


def fetch(day: date, venue: str, race_no: int, country: str = "FR") -> RaceCard:
    hit = find_course(day, venue, race_no)
    if not hit:
        raise SourceError(f"{NAME}: no {venue} course {race_no} on {day}")
    reu, c = hit
    key = day.strftime("%d%m%Y")
    base = f"{BASE}/programme/{key}/R{reu['numOfficiel']}/C{c['numOrdre']}"
    hip = reu.get("hippodrome") or {}
    card = RaceCard(country=country, venue=hip.get("libelleCourt", venue).title(), race_no=race_no,
                    race_date=day, name=c.get("libelle", ""), dist_m=c.get("distance"),
                    race_class=" ".join(x for x in (c.get("discipline"), c.get("categorieParticularite"))
                                        if x and x != "INCONNU"),
                    prize=c.get("montantPrix"), currency="EUR", surface="TURF", sources=[NAME])
    if c.get("heureDepart"):
        card.extras["start_utc"] = datetime.fromtimestamp(c["heureDepart"] / 1000, tz=timezone.utc)
    if c.get("corde"):
        card.rail = c["corde"].replace("_", " ").title()
    meteo = reu.get("meteo") or {}
    if meteo:
        card.weather = f"{meteo.get('nebulositeLibelleCourt', '')} {meteo.get('temperature', '')}°C".strip()
    if c.get("conditions"):
        card.extras["conditions"] = c["conditions"]
    part = cached(("pmu_part", base), 120, lambda: get_json(base + "/participants") or {})
    perf = cached(("pmu_perf", base), 3600,
                  lambda: get_json(base + "/performances-detaillees/pretty") or {})
    perf_by_num = {p.get("numPmu"): p for p in perf.get("participants", []) or []}
    rapports: dict[int, dict] = {}
    for typ in ("E_SIMPLE_GAGNANT", "SIMPLE_GAGNANT"):
        try:
            rp = get_json(base + f"/rapports/{typ}", retries=0) or {}
        except SourceError:
            rp = {}
        for r in rp.get("rapportsParticipant", []) or []:
            if r.get("numPmu") and r.get("rapportDirect"):
                rapports.setdefault(int(r["numPmu"]), r)
        if rapports:
            break
    for p in part.get("participants", []) or []:
        num = p.get("numPmu")
        e = Entry(number=num, name=(p.get("nom") or "").title(), age=p.get("age"),
                  sex={"MALES": "C", "HONGRES": "G", "FEMELLES": "F"}.get(p.get("sexe"), p.get("sexe") or ""),
                  jockey=p.get("driver") or "", trainer=p.get("entraineur") or "",
                  scratched=(p.get("statut") or "PARTANT") != "PARTANT",
                  barrier=p.get("placeCorde"), sources=[NAME])
        if p.get("handicapPoids"):
            e.weight = p["handicapPoids"] / 10.0
        starts = int(p.get("nombreCourses") or 0)
        wins = int(p.get("nombreVictoires") or 0)
        sec = int(p.get("nombrePlacesSecond") or 0)
        thr = int(p.get("nombrePlacesTroisieme") or 0)
        e.career = (starts, wins, sec, thr)
        gains = (p.get("gainsParticipant") or {}).get("gainsCarriere")
        if gains:
            e.prize_money = gains / 100.0
        e.form_string = parse_musique(p.get("musique") or "")
        gear = []
        if (p.get("oeilleres") or "SANS_OEILLERES") != "SANS_OEILLERES":
            gear.append(p["oeilleres"].replace("_", " ").title())
        if p.get("deferre") and p["deferre"] != "SANS_DEFERRAGE":
            gear.append(p["deferre"].replace("_", " ").title())
        e.gear = ", ".join(gear)
        ref = (p.get("dernierRapportReference") or {}).get("rapport")
        direct = (p.get("dernierRapportDirect") or {}).get("rapport")
        if direct and direct > 1:
            e.odds = float(direct)
        if ref and ref > 1:
            e.odds_open = float(ref)
        rp = rapports.get(num)
        if rp and rp.get("rapportDirect") and rp["rapportDirect"] > 1:
            e.odds = float(rp["rapportDirect"])
            if e.odds_open is None and rp.get("tendance") is not None:
                e.odds_open = e.odds / (1 + rp["tendance"] / 100.0) if rp["tendance"] > -100 else None
            e.extras["pmu_favori"] = bool(rp.get("favoris"))
        if p.get("avisEntraineur"):
            e.extras["trainer_opinion"] = p["avisEntraineur"]
        if p.get("driverChange"):
            e.extras["jockey_change"] = True
        pp = perf_by_num.get(num)
        if pp:
            e.runs = _past_runs(pp, day)
            days = [r.days_ago for r in e.runs if r.days_ago is not None]
            if days:
                e.last_run_days = min(days)
        card.entries.append(e)
    if not card.entries:
        raise SourceError(f"{NAME}: no participants returned")
    # Early on race day the tote pool holds a handful of bets and only the backed
    # runners show a rapport (one at 1.1, the rest blank).  That is not a market:
    # drop PMU prices unless most of the field is priced.
    live = [e for e in card.entries if not e.scratched]
    priced = [e for e in live if e.odds]
    if live and len(priced) < 0.8 * len(live):
        for e in live:
            e.odds, e.odds_open = None, None
        card.notes.append(f"{NAME}: tote pool not formed yet ({len(priced)} of {len(live)} priced) - "
                          "PMU odds ignored; fixed odds arrive from Ladbrokes on race day.")
    return card
