"""The Universal Horse-Racing Scoring Framework (Reference V2), implemented.

Fifteen weighted categories summing to 100. Each is scored 0-10 from the form
page, multiplied by a weight that adapts to today's distance and surface, and
the field is then ranked on the result.

Four rules from the framework shape the whole implementation:

* **Missing means unknown, not poor.** A category with no evidence gets
  `available = False`, its weight leaves the denominator, and the rest rescale
  to 100. A horse never raced on today's going is not punished for it.
* **Field-relative thinking.** Ratings, jockeys, pace and draw are scored
  against today's opponents, not against an absolute standard.
* **No double counting.** Class uses the single strongest applicable piece of
  evidence rather than stacking a higher-class win on top of a same-class win.
* **The market stays out.** Odds are parsed and displayed but contribute
  nothing to the score, so the rating is genuinely independent of the price.

`FINAL = 100 x sum(contributions) / sum(available adjusted weights)`, and the
Field Index puts the best horse on 100.

**This is an implementation of a stated method, not a validated model.** The
framework itself says the weights are "a rational baseline for testing, not
universal scientific constants" and asks that they be frozen and then validated
over many races. Nothing here has been checked against results.
"""
from __future__ import annotations

import math
import re
from typing import Any

import numpy as np

# --------------------------------------------------------------------------
# master weights (section 2)
# --------------------------------------------------------------------------
CATEGORIES: list[tuple[str, str, float]] = [
    ("recent_form", "Recent Form", 15),
    ("ability", "Ability / Ratings / Speed", 14),
    ("pace", "Pace / Race Shape", 10),
    ("class_", "Class / Opposition Strength", 9),
    ("distance", "Distance / Stamina", 9),
    ("surface", "Surface / Going", 8),
    ("sectionals", "Sectionals / Efficiency", 6),
    ("weight", "Weight / Claim / Allowances", 6),
    ("h2h", "Direct H2H / Comparable Races", 6),
    ("course", "Course / Track Suitability", 4),
    ("barrier", "Barrier / Draw", 3),
    ("fitness", "Fitness / Preparation", 3),
    ("jockey", "Jockey", 3),
    ("trainer", "Trainer", 2),
    ("trip", "Trip / Gear / Stewards", 2),
]
BASE_WEIGHTS = {k: w for k, _, w in CATEGORIES}
LABELS = {k: lbl for k, lbl, _ in CATEGORIES}

# section 4 - distance bands
DISTANCE_MULT = {
    "<=1200": {"recent_form": 1.0, "ability": 1.1, "class_": 1.0, "pace": 1.25,
               "distance": 0.9, "surface": 1.05, "sectionals": 1.1, "weight": 0.9,
               "h2h": 1.0, "course": 1.05, "barrier": 1.35, "fitness": 1.0,
               "jockey": 1.05, "trainer": 1.0, "trip": 1.0},
    "1201-1600": {k: 1.0 for k in BASE_WEIGHTS},
    "1601-2000": {"recent_form": 1.0, "ability": 1.0, "class_": 1.0, "pace": 0.95,
                  "distance": 1.15, "surface": 1.05, "sectionals": 1.0,
                  "weight": 1.05, "h2h": 1.0, "course": 1.0, "barrier": 0.8,
                  "fitness": 1.05, "jockey": 1.0, "trainer": 1.0, "trip": 1.0},
    ">2000": {"recent_form": 1.0, "ability": 0.95, "class_": 1.0, "pace": 0.9,
              "distance": 1.35, "surface": 1.15, "sectionals": 0.9, "weight": 1.15,
              "h2h": 1.0, "course": 1.0, "barrier": 0.6, "fitness": 1.15,
              "jockey": 1.0, "trainer": 1.0, "trip": 1.0},
}

# section 5 - surface. Only the categories the document names are moved.
SURFACE_MULT = {
    "TURF": {"surface": 1.2, "sectionals": 1.1, "course": 1.1, "pace": 1.0},
    "SYNTHETIC": {"pace": 1.05, "barrier": 1.05, "surface": 0.5},
    "DIRT": {"pace": 1.05, "barrier": 1.05, "surface": 0.5},
}

# section 17 - race type
RACE_TYPE_MULT = {
    "handicap": {},
    "wfa": {"weight": 0.42},          # 6% -> ~2.5%
    "group": {"ability": 1.15, "class_": 1.15},
    "maiden": {"recent_form": 1.15},
}

# section 16 - confidence by career starts
CONFIDENCE_BANDS = [(0, 0.30), (1, 0.45), (3, 0.58), (5, 0.70),
                    (10, 0.80), (20, 0.885), (10 ** 6, 0.935)]

# section 19 - recommended outputs. WIN leans on peak ability, class and pace;
# PLACE on consistency, suitability and repeatability.
PROFILES = {
    "overall": {},
    "win": {"ability": 1.20, "class_": 1.20, "pace": 1.15, "recent_form": 1.05,
            "sectionals": 1.10, "fitness": 0.90, "trainer": 0.90, "trip": 0.90},
    "place": {"recent_form": 1.15, "distance": 1.15, "surface": 1.15,
              "h2h": 1.20, "course": 1.10, "ability": 0.90, "pace": 0.90,
              "sectionals": 0.90},
}

RUNNING_STYLES = {1: "Leader", 2: "On Pace", 3: "Prominent",
                  4: "Midfield", 5: "Backmarker"}


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def _f(v, d=0.0) -> float:
    try:
        if v is None:
            return d
        return float(v)
    except (TypeError, ValueError):
        return d


def _scale(v: float, lo: float, hi: float) -> float:
    """Linear 0-10 between lo and hi, clamped. hi may be below lo to invert."""
    if hi == lo:
        return 5.0
    return float(np.clip(10.0 * (v - lo) / (hi - lo), 0.0, 10.0))


class Acc:
    """Accumulates sub-parameter points against their maxima.

    Sub-parameters that have no evidence are simply not added, so they drop out
    of both numerator and denominator - the missing-data rule applied one level
    below the category.
    """

    def __init__(self) -> None:
        self.pts = 0.0
        self.max = 0.0
        self.notes: list[str] = []

    def add(self, points: float, maximum: float, note: str = "") -> None:
        self.pts += float(points)
        self.max += float(maximum)
        if note:
            self.notes.append(note)

    def skip(self, note: str) -> None:
        if note:
            self.notes.append(note)

    def out(self) -> tuple[float, bool, list[str]]:
        if self.max <= 0:
            return 0.0, False, self.notes
        return 10.0 * self.pts / self.max, True, self.notes


def distance_band(metres: float) -> str:
    if metres <= 1200:
        return "<=1200"
    if metres <= 1600:
        return "1201-1600"
    if metres <= 2000:
        return "1601-2000"
    return ">2000"


def surface_key(surface: str | None) -> str:
    s = (surface or "").upper()
    if "AW" in s or "SYNTH" in s or "POLY" in s or "TAPETA" in s:
        return "SYNTHETIC"
    if "DIRT" in s or "SAND" in s:
        return "DIRT"
    return "TURF"


def race_type_key(header: dict[str, Any]) -> str:
    t = f"{header.get('race_type') or ''} {header.get('race_name') or ''}".upper()
    if "MDN" in t or "MAIDEN" in t:
        return "maiden"
    if re.search(r"\bG[123]\b|GROUP|LISTED|\bLR\b|STAKES", t):
        return "group"
    if "WFA" in t or "SET WEIGHT" in t:
        return "wfa"
    return "handicap"


# --------------------------------------------------------------------------
# field context - things that can only be known relative to today's opponents
# --------------------------------------------------------------------------
def running_style(r: dict[str, Any]) -> int:
    """1 Leader .. 5 Backmarker, from average early position share.

    Uses the settling/800m position where the page gives it and the turn
    position otherwise, expressed as a fraction of field size so an 8th of 20 is
    not read like an 8th of 9.
    """
    fracs = []
    for run in (r.get("recent_runs") or [])[:5]:
        pos = run.get("settle_pos")
        if pos is None:
            pos = run.get("turn_pos")
        n = _f(run.get("field_size"), 0)
        if pos and n > 1:
            fracs.append((_f(pos) - 1) / (n - 1))
    if not fracs:
        return 4
    m = float(np.mean(fracs))
    for cut, style in ((0.15, 1), (0.35, 2), (0.55, 3), (0.78, 4)):
        if m <= cut:
            return style
    return 5


def field_context(active: list[dict[str, Any]], header: dict[str, Any]) -> dict[str, Any]:
    styles = {r["tab"]: running_style(r) for r in active}
    ppi = (1.0 * sum(1 for s in styles.values() if s == 1)
           + 0.7 * sum(1 for s in styles.values() if s == 2)
           + 0.35 * sum(1 for s in styles.values() if s == 3))
    ohrs = [_f(r.get("ohr")) for r in active if _f(r.get("ohr")) > 0]
    jrats = [_f(r.get("jrat")) for r in active if _f(r.get("jrat")) > 0]
    trats = [_f(r.get("trat")) for r in active if _f(r.get("trat")) > 0]
    wts = [_f(r.get("wt")) - _f(r.get("claim")) for r in active]
    return {
        "styles": styles,
        "ppi": ppi,
        "n_leaders": sum(1 for s in styles.values() if s == 1),
        "ohr_lo": min(ohrs) if ohrs else 0.0,
        "ohr_hi": max(ohrs) if ohrs else 0.0,
        "jrat_lo": min(jrats) if jrats else 0.0,
        "jrat_hi": max(jrats) if jrats else 0.0,
        "trat_lo": min(trats) if trats else 0.0,
        "trat_hi": max(trats) if trats else 0.0,
        "wt_lo": min(wts) if wts else 0.0,
        "wt_hi": max(wts) if wts else 0.0,
        "field_size": len(active),
        "names": {r["horse"].upper() for r in active},
        "distance": _f(header.get("distance_m"), 1200),
        "surface": surface_key(header.get("surface")),
        "going": (header.get("going") or "").upper(),
    }


# --------------------------------------------------------------------------
# the fifteen categories
# --------------------------------------------------------------------------
def _run_quality(run: dict[str, Any]) -> float:
    """0-1 quality of a single completed run, before recency weighting."""
    fin = _f(run.get("finish"), 99)
    n = max(_f(run.get("field_size"), 0), 1)
    margin = _f(run.get("margin"), 99)
    pos = 1.0 - (fin - 1) / max(n - 1, 1)            # 1st = 1.0, last = 0.0
    # A beaten margin of a length is a much better run than a distant second, so
    # margin carries as much weight as finishing position.
    mar = float(np.clip(1.0 - margin / 8.0, 0.0, 1.0))
    if fin == 1:
        mar = 1.0
    return float(np.clip(0.55 * pos + 0.45 * mar, 0.0, 1.0))


def cat_recent_form(r, ctx):
    a = Acc()
    runs = (r.get("recent_runs") or [])[:5]
    if not runs:
        a.skip("no completed runs on the page")
        return a.out()

    a.add(5.0 * _run_quality(runs[0]), 5.0,
          f"last start {int(_f(runs[0].get('finish'),0))} of "
          f"{int(_f(runs[0].get('field_size'),0))}, "
          f"{_f(runs[0].get('margin')):.1f}L")

    prev = runs[1:4]
    if prev:
        w = [0.5, 0.3, 0.2][:len(prev)]
        q = sum(wi * _run_quality(x) for wi, x in zip(w, prev)) / sum(w)
        a.add(4.0 * q, 4.0, f"{len(prev)} prior run(s) scored {10*q:.1f}/10")

    margins = [_f(x.get("margin"), 99) for x in runs[:4]]
    margins = [m for m in margins if m < 90]
    if margins:
        a.add(2.0 * float(np.clip(1 - np.mean(margins) / 8.0, 0, 1)), 2.0,
              f"average beaten margin {np.mean(margins):.1f}L")

    if len(runs) >= 3:
        q = [_run_quality(x) for x in runs[:4]]
        # runs[0] is the most recent, so a positive slope here means improving
        trend = float(np.polyfit(range(len(q)), q, 1)[0]) * -1
        a.add(2.0 * float(np.clip(0.5 + trend * 2.5, 0, 1)), 2.0,
              "improving" if trend > 0.02 else
              ("declining" if trend < -0.02 else "stable"))
        a.add(2.0 * float(np.clip(1 - np.std(q) * 2.2, 0, 1)), 2.0,
              f"consistency sd {np.std(q):.2f}")
    return a.out()


def cat_ability(r, ctx):
    a = Acc()
    ohr = _f(r.get("ohr"))
    if ohr > 0 and ctx["ohr_hi"] > ctx["ohr_lo"]:
        a.add(0.5 * _scale(ohr, ctx["ohr_lo"] - 2, ctx["ohr_hi"]), 5.0,
              f"OHR {ohr:.0f} in a field of {ctx['ohr_lo']:.0f}-{ctx['ohr_hi']:.0f}")
    elif ohr > 0:
        a.add(2.5, 5.0, f"OHR {ohr:.0f}, no spread in the field")
    else:
        a.skip("no official rating")

    hist = [x for x in (r.get("recent_runs") or []) if x.get("ohr")]
    if len(hist) >= 2:
        vals = [_f(x["ohr"]) for x in hist[:5]]
        peak, med = max(vals), float(np.median(vals))
        lo, hi = ctx["ohr_lo"] - 2, max(ctx["ohr_hi"], peak)
        a.add(0.2 * _scale(peak, lo, hi), 2.0, f"peak rating {peak:.0f}")
        a.add(0.3 * _scale(med, lo, hi), 3.0, f"median rating {med:.0f}")
    else:
        a.skip("too few rated runs for peak/median")

    # "Recent adjusted speed figure" - a true speed figure is not on the page.
    # API is the quality of the race, not of the horse, so it is used for class
    # instead. This sub-parameter is left unavailable rather than faked.
    a.skip("no comparable speed figure on the page")
    return a.out()


_CLASS_ORDER = [
    (r"\bG1\b|GROUP 1", 10), (r"\bG2\b|GROUP 2", 9), (r"\bG3\b|GROUP 3", 8),
    (r"\bLR\b|LISTED", 7), (r"OPEN|WFA", 6), (r"BM(\d+)", None),
    (r"\bC(\d)\b|CLASS (\d)", None), (r"MDN|MAIDEN", 1),
]


def class_rank(text: str | None) -> float | None:
    """A rough 0-10 ladder for a class string, jurisdiction-aware where it can be."""
    if not text:
        return None
    t = text.upper()
    for pat, val in _CLASS_ORDER:
        m = re.search(pat, t)
        if not m:
            continue
        if val is not None:
            return float(val)
        if "BM" in pat:
            # benchmark: BM50 ~ 2, BM100 ~ 7
            return float(np.clip(2 + (_f(m.group(1)) - 50) / 10.0, 1, 8))
        g = next((x for x in m.groups() if x), None)
        if g is None:
            return None
        # South African / Australian classes: C1 is the best, C6 the weakest
        return float(np.clip(7 - _f(g), 1, 7))
    return None


def cat_class(r, ctx, today_class):
    a = Acc()
    runs = [x for x in (r.get("recent_runs") or []) if x.get("race_class")]
    if not runs or today_class is None:
        a.skip("class of today's race or past runs not readable")
        return a.out()

    # Section 8: use the strongest APPLICABLE evidence, never stack class facts.
    best = None
    for x in runs[:6]:
        cr = class_rank(x.get("race_class"))
        if cr is None:
            continue
        fin = _f(x.get("finish"), 99)
        rel = cr - today_class                       # + = ran above today's grade
        if fin == 1:
            s = 8.75 if rel > 0.5 else (7.0 if rel > -0.5 else 4.75)
        elif fin <= 3:
            s = 8.0 if rel > 0.5 else (6.0 if rel > -0.5 else 3.75)
        elif _f(x.get("margin"), 99) <= 3.0:
            s = 6.5 if rel > 0.5 else (4.5 if rel > -0.5 else 3.0)
        else:
            s = 3.0 if rel > 0.5 else (1.8 if rel > -0.5 else 1.0)
        if best is None or s > best[0]:
            best = (s, x, rel)
    if best is None:
        a.skip("no readable class evidence")
        return a.out()
    s, x, rel = best
    where = "above" if rel > 0.5 else ("at" if rel > -0.5 else "below")
    a.add(s, 9.0, f"best evidence: {int(_f(x.get('finish'),0))} of "
                  f"{int(_f(x.get('field_size'),0))} in {x.get('race_class')} "
                  f"({where} today's grade)")

    apis = [_f(x.get("api")) for x in runs[:5] if _f(x.get("api")) > 0]
    if apis:
        a.notes.append(f"average race API {np.mean(apis):.1f}")
    return a.out()


def cat_pace(r, ctx):
    a = Acc()
    style = ctx["styles"].get(r["tab"], 4)
    ppi = ctx["ppi"]
    if not (r.get("recent_runs") or []):
        a.skip("no runs to infer a running style from")
        return a.out()

    # Section 7. Leaders are rewarded when they are alone and downgraded into a
    # speed battle; closers are the mirror image.
    if style in (1, 2):
        if ctx["n_leaders"] <= 1:
            fit = 9.0
        elif ctx["n_leaders"] == 2:
            fit = 7.0
        elif ppi >= 3.5:
            fit = 2.5
        else:
            fit = 5.0
    elif style == 3:
        fit = 7.0 if ppi < 3.0 else 6.0
    else:
        if ppi >= 3.0:
            fit = 8.5
        elif ppi >= 1.8:
            fit = 6.0
        else:
            fit = 3.5
    a.add(fit / 2.0, 5.0, f"{RUNNING_STYLES[style]}, pace-pressure index {ppi:.2f}")

    pressure = _scale(ppi, 0.0, 4.0)
    a.add(0.3 * (pressure if style >= 4 else 10 - pressure), 3.0,
          "closer helped by pressure" if style >= 4 else
          "leader prefers a soft lead")

    fr = []
    for run in (r.get("recent_runs") or [])[:5]:
        pos = run.get("settle_pos") or run.get("turn_pos")
        n = _f(run.get("field_size"), 0)
        if pos and n > 1:
            fr.append((_f(pos) - 1) / (n - 1))
    if len(fr) >= 3:
        a.add(0.2 * _scale(float(np.std(fr)), 0.05, 0.30), 2.0,
              f"tactical versatility sd {np.std(fr):.2f}")
    else:
        a.skip("not enough in-running data for versatility")
    return a.out()


def _wps(r, prefix):
    return (_f(r.get(f"{prefix}_win")), _f(r.get(f"{prefix}_plc")),
            _f(r.get(f"{prefix}_starts")))


def _record_score(win, plc, n, max_pts, neutral=0.5):
    """Score a win/place record with sample-size caution. Untested is neutral."""
    if n <= 0:
        return None
    conf = min(n / 5.0, 1.0)
    raw = float(np.clip(0.55 * win * 3.2 + 0.45 * plc * 1.6, 0, 1))
    return max_pts * (neutral + conf * (raw - neutral))


def cat_distance(r, ctx):
    a = Acc()
    w, p, n = _wps(r, "Dist")
    s = _record_score(w, p, n, 5.0)
    if s is None:
        a.skip("never raced at today's trip")
    else:
        a.add(s, 5.0, f"at the distance {int(_f(r.get('Dist_wins')))}-"
                      f"{int(_f(r.get('Dist_places')))}-{int(n)}")

    runs = r.get("recent_runs") or []
    today = ctx["distance"]
    near = [x for x in runs[:6]
            if x.get("distance") and abs(_f(x["distance"]) - today) <= 0.15 * today]
    if near:
        q = float(np.mean([_run_quality(x) for x in near]))
        a.add(4.0 * q, 4.0, f"{len(near)} recent run(s) within 15% of the trip")
    else:
        a.skip("no recent runs near today's trip")

    lo, hi = _f(r.get("dist_min")), _f(r.get("dist_max"))
    if lo > 0 and hi > 0:
        inside = lo - 100 <= today <= hi + 100
        a.add(2.0 if inside else 0.7, 2.0,
              f"proven range {int(lo)}-{int(hi)}m")
    else:
        a.skip("no proven distance range")
    return a.out()


_GOING_KEY = {"FIRM": "Firm", "GOOD": "Good", "SOFT": "Soft", "HEAVY": "Heavy",
              "YIELDING": "Soft", "SLOW": "Soft", "FAST": "Firm", "STANDARD": "Good"}


def cat_surface(r, ctx):
    a = Acc()
    skey = ctx["surface"]
    pre = {"TURF": "Turf", "SYNTHETIC": "AW", "DIRT": "AW"}[skey]
    w, p, n = _wps(r, pre)
    s = _record_score(w, p, n, 4.0)
    if s is None:
        a.skip(f"untested on {pre}")
    else:
        a.add(s, 4.0, f"on {pre}: {int(_f(r.get(pre+'_wins')))}-"
                      f"{int(_f(r.get(pre+'_places')))}-{int(n)}")

    gkey = None
    for k, v in _GOING_KEY.items():
        if ctx["going"].startswith(k):
            gkey = v
            break
    if skey != "TURF":
        # Section 5: turf going terminology is much less relevant on synthetics.
        a.skip("going categories not meaningful on this surface")
    elif gkey is None:
        a.skip("today's going not readable")
    else:
        gw, gp, gn = _wps(r, gkey)
        gs = _record_score(gw, gp, gn, 4.0)
        if gs is None:
            a.add(2.0, 4.0, f"untested on {gkey} - neutral, not a penalty")
        else:
            a.add(gs, 4.0, f"on {gkey}: {int(_f(r.get(gkey+'_wins')))}-"
                           f"{int(_f(r.get(gkey+'_places')))}-{int(gn)}")
    return a.out()


def cat_sectionals(r, ctx):
    a = Acc()
    runs = [x for x in (r.get("recent_runs") or [])[:5]
            if (x.get("settle_pos") or x.get("turn_pos")) and x.get("finish")]
    if not runs:
        a.skip("no in-running positions on the page")
        return a.out()
    gains = []
    for x in runs:
        pos = _f(x.get("turn_pos") or x.get("settle_pos"))
        n = max(_f(x.get("field_size"), 0), 1)
        gains.append((pos - _f(x.get("finish"))) / max(n - 1, 1))
    g = float(np.mean(gains))
    a.add(0.3 * _scale(g, -0.25, 0.45), 3.0,
          f"positions gained from the turn: {g:+.2f} of the field on average")
    a.add(0.2 * _scale(float(np.mean([_run_quality(x) for x in runs])), 0.2, 0.85),
          2.0, "efficiency proxy from finishing quality")
    # True L600/L400/L200 sectionals are not published on this page.
    a.skip("no true sectional times available")
    return a.out()


def cat_weight(r, ctx):
    a = Acc()
    today = _f(r.get("wt")) - _f(r.get("claim"))
    runs = r.get("recent_runs") or []
    comp = [x for x in runs[:5] if _f(x.get("weight")) > 0]
    if comp and today > 0:
        prev = float(np.mean([_f(x["weight"]) for x in comp[:3]]))
        d = prev - today                     # positive = carrying less today
        for lim, pts in ((4, 3.0), (3, 2.5), (2, 2.0), (1, 1.5), (-1, 1.0),
                         (-2, 0.75)):
            if d >= lim:
                sub = pts
                break
        else:
            sub = 0.25
        a.add(sub, 3.0, f"{d:+.1f}kg against recent runs")
    else:
        a.skip("no comparable weight history")

    if ctx["wt_hi"] > ctx["wt_lo"] and today > 0:
        # Section 9: lighter is not automatically better - it usually reflects
        # lower class - so this is scored only mildly.
        a.add(0.2 * _scale(today, ctx["wt_hi"], ctx["wt_lo"]), 2.0,
              f"effective weight {today:.1f}kg in a "
              f"{ctx['wt_lo']:.1f}-{ctx['wt_hi']:.1f}kg field")
    else:
        a.skip("no weight spread in the field")

    claim = _f(r.get("claim"))
    if claim > 0:
        a.add(min(1.0, 0.35 + claim / 6.0), 1.0, f"{claim:.1f}kg apprentice claim")
    else:
        a.add(0.5, 1.0, "no claim")
    return a.out()


def cat_h2h(r, ctx):
    a = Acc()
    meets = r.get("h2h") or []
    me = r["horse"].upper()
    rel = []
    for m in meets:
        mine = next((x for x in m["runners"] if x["horse"] == me), None)
        others = [x for x in m["runners"]
                  if x["horse"] != me and x["horse"] in ctx["names"]]
        for o in others:
            rel.append((mine, o))
    rel = [(a_, b) for a_, b in rel if a_ is not None]
    if not rel:
        a.skip("no head-to-head with today's rivals")
        return a.out()

    scores = []
    for mine, other in rel[:6]:
        beat = mine["finish"] < other["finish"]
        gap = abs(_f(mine["margin"]) - _f(other["margin"]))
        if beat:
            s = 4.0 if gap >= 2 else 3.5
        else:
            s = 1.0 if gap >= 2 else 2.0
        # carrying more weight and still beating a rival is stronger evidence
        if beat and _f(mine["weight"]) > _f(other["weight"]):
            s = 4.0
        scores.append(s)
    a.add(float(np.mean(scores)), 4.0,
          f"{sum(1 for m_, o in rel if m_['finish'] < o['finish'])} win(s) from "
          f"{len(rel)} meeting(s) with today's field")

    swings = [_f(m_["wt_swing"]) for m_, _ in rel if _f(m_["wt_swing"])]
    if swings:
        a.add(0.2 * _scale(float(np.mean(swings)), -5, 5), 2.0,
              f"average weight swing {np.mean(swings):+.1f}kg")
    else:
        a.skip("no weight swings recorded")
    return a.out()


def cat_course(r, ctx):
    a = Acc()
    cw, cp, cn = _wps(r, "CrsDist")
    s = _record_score(cw, cp, cn, 2.0)
    if s is None:
        a.add(1.0, 2.0, "never raced at course and distance - neutral")
    else:
        a.add(s, 2.0, f"course & distance {int(_f(r.get('CrsDist_wins')))}-"
                      f"{int(_f(r.get('CrsDist_places')))}-{int(cn)}")
    w, p, n = _wps(r, "Crs")
    s2 = _record_score(w, p, n, 1.0)
    if s2 is None:
        a.add(0.5, 1.0, "never raced at the course - neutral")
    else:
        a.add(s2, 1.0, f"course {int(_f(r.get('Crs_wins')))}-"
                       f"{int(_f(r.get('Crs_places')))}-{int(n)}")
    a.skip("track configuration not on the page")
    return a.out()


def cat_barrier(r, ctx):
    a = Acc()
    bp = _f(r.get("bp"))
    n = max(ctx["field_size"], 1)
    if bp <= 0:
        a.skip("no barrier drawn")
        return a.out()
    frac = (bp - 1) / max(n - 1, 1)
    style = ctx["styles"].get(r["tab"], 4)
    sprint = ctx["distance"] <= 1200
    if style in (1, 2):
        # Section 10: a wide draw costs a leader far more than a backmarker.
        base = 3.0 - (2.0 if sprint else 1.2) * frac
    elif style >= 4:
        base = 2.5 - 0.6 * frac
    else:
        base = 2.8 - 1.0 * frac
    a.add(float(np.clip(base, 0.0, 3.0)) * 2.0 / 3.0, 2.0,
          f"barrier {int(bp)} of {n}, {RUNNING_STYLES[style]}")
    a.add(float(np.clip(1.0 - abs(frac - 0.3), 0.0, 1.0)), 1.0,
          "draw against running style")
    return a.out()


def cat_fitness(r, ctx):
    a = Acc()
    d = _f(r.get("dslr"), -1)
    if d < 0:
        a.skip("days since last run unknown")
    else:
        for lim, pts in ((30, 1.0), (45, 0.75), (75, 0.5), (120, 0.25)):
            if d <= lim:
                sub = pts
                break
        else:
            sub = 0.15
        if d < 7:
            sub = 0.7
        a.add(sub, 1.0, f"{int(d)} days since its last run")

    prep = _f(r.get("runup"), 0)
    key = "FU" if prep <= 1 else ("U2" if prep == 2 else "U3")
    w, p, n = _wps(r, key)
    s = _record_score(w, p, n, 1.0)
    label = {"FU": "first-up", "U2": "second-up", "U3": "third-up"}[key]
    if s is None:
        a.add(0.5, 1.0, f"no {label} record - neutral")
    else:
        a.add(s, 1.0, f"{label} record {int(_f(r.get(key+'_wins')))}-"
                      f"{int(_f(r.get(key+'_places')))}-{int(n)}")
    a.skip("trials and workouts not on the page")
    return a.out()


def cat_jockey(r, ctx):
    a = Acc()
    jr = _f(r.get("jrat"))
    if jr > 0 and ctx["jrat_hi"] > ctx["jrat_lo"]:
        a.add(0.1 * _scale(jr, ctx["jrat_lo"], ctx["jrat_hi"]), 1.0,
              f"jockey rating {jr:.1f} in a "
              f"{ctx['jrat_lo']:.1f}-{ctx['jrat_hi']:.1f} field")
    elif _f(r.get("jky_n")) > 0:
        a.add(_f(r.get("jky_win")) * 3.0, 1.0,
              f"jockey last-50 {100*_f(r.get('jky_win')):.0f}% wins")
    else:
        a.skip("no jockey form")

    jh_n = _f(r.get("jh_n"))
    if jh_n > 0:
        conf = min(jh_n / 6.0, 1.0)
        raw = float(np.clip(_f(r.get("jh_win")) * 3.0, 0, 1))
        a.add(0.75 * (0.5 + conf * (raw - 0.5)), 0.75,
              f"jockey on this horse {int(jh_n)} time(s), "
              f"{100*_f(r.get('jh_win')):.0f}% wins")
    else:
        a.skip("jockey has not ridden this horse")

    jt_n = _f(r.get("jt_n"))
    if jt_n > 0:
        conf = min(jt_n / 40.0, 1.0)
        raw = float(np.clip(_f(r.get("jt_win")) * 3.5, 0, 1))
        a.add(0.75 * (0.5 + conf * (raw - 0.5)), 0.75,
              f"jockey/trainer {int(jt_n)} runs, "
              f"{100*_f(r.get('jt_win')):.0f}% wins")
    else:
        a.skip("no jockey/trainer history")
    a.skip("last-5-rides detail not on the page")
    return a.out()


def cat_trainer(r, ctx):
    a = Acc()
    tr = _f(r.get("trat"))
    if tr > 0 and ctx["trat_hi"] > ctx["trat_lo"]:
        a.add(0.075 * _scale(tr, ctx["trat_lo"], ctx["trat_hi"]), 0.75,
              f"trainer rating {tr:.1f} in a "
              f"{ctx['trat_lo']:.1f}-{ctx['trat_hi']:.1f} field")
    elif _f(r.get("trn_n")) > 0:
        a.add(_f(r.get("trn_win")) * 2.5, 0.75,
              f"stable last-50 {100*_f(r.get('trn_win')):.0f}% wins")
    else:
        a.skip("no trainer form")
    a.skip("course and race-type specialisation not on the page")
    a.skip("placement/targeting not machine-readable")
    return a.out()


_TRIP_PATTERNS = [
    (r"wide|without cover|three wide|caught wide", 0.5, "wide without cover"),
    (r"held up|blocked|checked|no room|denied|hampered|interfere", 0.5,
     "blocked or checked"),
    (r"slow(?:ly)? (?:away|to begin)|lost \d+ length at the start|missed the start",
     0.4, "slow away"),
    (r"never (?:a )?factor|no lu?ck", 0.3, "no luck"),
]


def cat_trip(r, ctx):
    a = Acc()
    runs = (r.get("recent_runs") or [])[:3]
    txt = " ".join(str(x.get("stewards") or "") for x in runs).lower()
    if not runs:
        a.skip("no runs to read stewards notes from")
        return a.out()
    if not txt.strip():
        a.add(1.0, 2.0, "no stewards notes - neutral")
        return a.out()
    bonus = 0.0
    hits = []
    for pat, pts, label in _TRIP_PATTERNS:
        if re.search(pat, txt):
            bonus += pts
            hits.append(label)
    if re.search(r"vet(erinary)?", txt) and not re.search(
            r"nothing obvious|no abnormalit", txt):
        bonus -= 1.0
        hits.append("unresolved veterinary note")
    a.add(float(np.clip(1.0 + bonus, 0.0, 2.0)), 2.0,
          ", ".join(hits) if hits else "clean recent runs")
    return a.out()


SCORERS = {
    "recent_form": cat_recent_form, "ability": cat_ability, "pace": cat_pace,
    "distance": cat_distance, "surface": cat_surface, "sectionals": cat_sectionals,
    "weight": cat_weight, "h2h": cat_h2h, "course": cat_course,
    "barrier": cat_barrier, "fitness": cat_fitness, "jockey": cat_jockey,
    "trainer": cat_trainer, "trip": cat_trip,
}


# --------------------------------------------------------------------------
# assembly
# --------------------------------------------------------------------------
def adjusted_weights(header: dict[str, Any], profile: str = "overall",
                     race_type: str | None = None) -> dict[str, float]:
    """Base weights x distance x surface x race type, renormalised to 100."""
    band = distance_band(_f(header.get("distance_m"), 1200))
    skey = surface_key(header.get("surface"))
    rt = race_type or race_type_key(header)
    dm = DISTANCE_MULT[band]
    sm = SURFACE_MULT.get(skey, {})
    rm = RACE_TYPE_MULT.get(rt, {})
    pm = PROFILES.get(profile, {})
    w = {}
    for k in BASE_WEIGHTS:
        w[k] = (BASE_WEIGHTS[k] * dm.get(k, 1.0) * sm.get(k, 1.0)
                * rm.get(k, 1.0) * pm.get(k, 1.0))
    tot = sum(w.values())
    return {k: 100.0 * v / tot for k, v in w.items()}


def confidence(r: dict[str, Any], available: dict[str, bool]) -> float:
    starts = _f(r.get("Car_starts"))
    base = CONFIDENCE_BANDS[-1][1]
    for lim, v in CONFIDENCE_BANDS:
        if starts <= lim:
            base = v
            break
    covered = sum(BASE_WEIGHTS[k] for k, ok in available.items() if ok)
    completeness = covered / sum(BASE_WEIGHTS.values())
    d = _f(r.get("dslr"), 999)
    recency = 1.0 if d <= 60 else (0.9 if d <= 120 else 0.8)
    return float(np.clip(100.0 * base * (0.55 + 0.45 * completeness) * recency,
                         0.0, 97.0))


def score_runner(r, ctx, weights, today_class):
    cats: dict[str, dict[str, Any]] = {}
    for key in BASE_WEIGHTS:
        fn = SCORERS.get(key)
        if key == "class_":
            s, ok, notes = cat_class(r, ctx, today_class)
        else:
            s, ok, notes = fn(r, ctx)
        cats[key] = {"score": s, "available": ok, "notes": notes,
                     "weight": weights[key],
                     "contribution": (s / 10.0) * weights[key] if ok else 0.0}
    denom = sum(c["weight"] for c in cats.values() if c["available"])
    num = sum(c["contribution"] for c in cats.values())
    final = 100.0 * num / denom if denom > 0 else 0.0
    avail = {k: c["available"] for k, c in cats.items()}
    return {"tab": r["tab"], "horse": r["horse"], "categories": cats,
            "final": final, "available_weight": denom,
            "confidence": confidence(r, avail),
            "style": RUNNING_STYLES[ctx["styles"].get(r["tab"], 4)]}


def score_race(runners: list[dict[str, Any]], header: dict[str, Any], *,
               profile: str = "overall", race_type: str | None = None
               ) -> dict[str, Any]:
    """Score every active runner. Returns rows plus the field context used."""
    active = [r for r in runners if not r.get("scratched")]
    if len(active) < 2:
        return {"rows": [], "context": {}, "weights": {}, "n": len(active)}
    ctx = field_context(active, header)
    weights = adjusted_weights(header, profile, race_type)
    today_class = class_rank(header.get("race_type") or header.get("race_name"))

    rows = [score_runner(r, ctx, weights, today_class) for r in active]
    best = max(row["final"] for row in rows) or 1.0
    for row in rows:
        row["field_index"] = 100.0 * row["final"] / best
    rows.sort(key=lambda x: -x["final"])
    for i, row in enumerate(rows, 1):
        row["rank"] = i
    return {"rows": rows, "context": ctx, "weights": weights,
            "today_class": today_class,
            "race_type": race_type or race_type_key(header),
            "band": distance_band(_f(header.get("distance_m"), 1200)),
            "surface_key": surface_key(header.get("surface")), "n": len(active)}


DEFAULT_SPREAD = 14.0


def win_probabilities(rows: list[dict[str, Any]],
                      spread: float = DEFAULT_SPREAD) -> np.ndarray:
    """Turn scores into probabilities with a softmax over the field.

    `spread` is the score difference that makes a horse roughly e times more
    likely than another. It is a **presentation choice, not a fitted
    parameter**: nothing here has been calibrated against results, so these are
    a ranking expressed as percentages, not a probability forecast. A small
    spread makes the favourite look far more certain than the framework can
    justify - at 7 the top horse in the sample race reads 39% and the bottom
    0.8%, which no 9-runner field deserves off an uncalibrated score. The
    default of 14 keeps the ordering while staying honest about its resolution.
    """
    s = np.array([r["final"] for r in rows], dtype=float)
    if len(s) == 0:
        return np.array([])
    e = np.exp((s - s.max()) / max(spread, 1e-6))
    return e / e.sum()
