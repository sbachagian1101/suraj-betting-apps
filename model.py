"""Rating, probabilities and race simulation.

The engine is the one graded across this repo's racing apps, generalised to a
runner record that any country's feed can fill partially:

1. **Form** - each usable past run becomes a beaten margin in lengths at
   today's distance, adjusted for weight carried, field size and class
   (prizemoney ratio in the same currency), weighted by recency and distance
   relevance.  Feeds without beaten lengths (Sporting Life) get a margin
   estimated from the finishing position and a lower evidence weight.
2. **Record terms** - career strike and place conversion; distance, course,
   surface and going records, each shrunk toward the runner's own career rate
   with a prior worth ``prior_starts`` starts.
3. **Connections** - jockey and trainer recent strike rates (shrunk toward 10%).
4. **Conditions** - barrier scaled by distance, weight versus the field, days
   since the last run, first-up record, published rating (official / HK / WF
   merit) versus the field, early speed and the race's pace shape, HK last-400m
   sectional versus the field.
5. **Market** - prices are de-vigged by the power method; the model's
   probability is shrunk toward the market where evidence is thin (never toward
   uniform, which would inflate every longshot) and blended in log space.
   The price move since opening enters as a small sentiment term.
6. **Simulation** - 20,000 Plackett-Luce draws give every runner's chance of
   each finishing position, plus exacta / trifecta combinations.

Known weakness, stated plainly: there is no joint fit of ability across many
races (no opposition-strength adjustment beyond the prizemoney proxy), and the
weights are hand-set.  Ranking is the product; probabilities are calibrated
only through the market blend.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Optional

import numpy as np

from common import Entry, PastRun, RaceCard

BUILD = "2026-09-07c"      # bumped with every change; app.py refuses a stale copy


@dataclass
class Params:
    tau_days: float = 180.0
    sigma_dist: float = 250.0
    w_offsurface: float = 0.70
    w_estimated_margin: float = 0.55     # evidence weight of a position-derived margin
    spread: float = 3.20                 # lengths of performance SD (softmax temperature)
    prior_starts: float = 15.0
    k_form: float = 1.00
    k_class: float = 1.80
    k_conv: float = 1.20
    k_dist: float = 1.20
    k_course: float = 0.90
    k_surface: float = 1.00
    k_going: float = 0.80
    k_firstup: float = 0.90
    k_rating: float = 0.12               # lengths per rating point vs field mean
    k_jockey: float = 4.0                # lengths per log strike-rate ratio vs 10%
    k_trainer: float = 3.0
    k_combo: float = 1.5
    k_barrier: float = 0.90
    k_weight: float = 0.35               # lengths per kg above the field mean
    k_speed: float = 0.50
    k_pace: float = 0.60
    k_sectional: float = 0.90            # lengths per second faster last-400m than field median
    k_late: float = 0.45                 # lengths per SD of R&S AFS (finishing speed) vs field
    k_jr: float = 0.30                   # lengths per point of R&S jockey rating vs field mean
    k_move: float = 0.80                 # lengths per log(open/current)
    k_tip: float = 0.35
    k_class_adj: float = 1.50
    field_ref: float = 12.0
    k_field: float = 0.15
    impute_field_size: int = 12
    lengths_per_kg: float = 0.50
    market_weight: float = 0.35          # weight on the FORM model in the log blend
    sims: int = 20000
    seed: int = 7


#: per-country starting points.  Trot margins are wide; HK ratings are tight and
#: informative; the US feed has no margins at all so the model leans on the market.
COUNTRY_DEFAULTS: dict[str, dict[str, float]] = {
    "AUS": {"market_weight": 0.35},
    "FR": {"spread": 3.6, "market_weight": 0.30},
    "UK": {"spread": 3.4, "market_weight": 0.30, "k_rating": 0.16},
    "IRE": {"spread": 3.4, "market_weight": 0.30, "k_rating": 0.16},
    "HK": {"spread": 2.8, "market_weight": 0.40, "k_rating": 0.14},
    "SA": {"spread": 3.2, "market_weight": 0.35, "k_rating": 0.10},
    "USA": {"spread": 3.4, "market_weight": 0.25},
}


def defaults_for(country: str) -> Params:
    return replace(Params(), **COUNTRY_DEFAULTS.get(country, {}))


@dataclass
class Rated:
    entry: Entry
    number: int
    name: str
    terms: dict[str, float] = field(default_factory=dict)
    evidence: float = 0.0
    used_runs: int = 0
    avg_margin: Optional[float] = None
    rating_raw: float = 0.0
    rating: float = 0.0
    odds: Optional[float] = None
    price_source: str = ""
    p_market: Optional[float] = None
    p_model: float = 0.0
    p_final: float = 0.0
    p_top3: float = 0.0
    p_top2: float = 0.0
    move_pct: Optional[float] = None
    move_label: str = ""
    speed_score: Optional[float] = None

    @property
    def fair(self) -> Optional[float]:
        return 1.0 / self.p_final if self.p_final > 0 else None

    @property
    def ev(self) -> Optional[float]:
        return None if not self.odds else self.p_final * self.odds - 1.0


@dataclass
class SimResult:
    names: list[str]
    numbers: list[int]
    position_matrix: np.ndarray          # runners x positions, probabilities
    exactas: list[tuple[str, float]]
    trifectas: list[tuple[str, float]]
    expected_pos: list[float]


# --- helpers ------------------------------------------------------------------

def _softmax(vals: list[float], temp: float) -> list[float]:
    hi = max(vals)
    ex = [math.exp((v - hi) / temp) for v in vals]
    z = sum(ex)
    return [e / z for e in ex]


def _shrunk_log_ratio(sub_starts: int, sub_wins: int, p0: float, prior: float) -> float:
    if p0 <= 0 or sub_starts <= 0:
        return 0.0
    rate = (sub_wins + p0 * prior) / (sub_starts + prior)
    return math.log(max(rate, 1e-6) / p0)


def _strike_term(stats: Optional[tuple[int, int, int]], k: float, base: float = 0.10,
                 prior: float = 40.0) -> float:
    """log( shrunk strike rate / base ) scaled by k.  (starts, wins, places)."""
    if not stats or stats[0] <= 0:
        return 0.0
    n, w = stats[0], stats[1]
    rate = (w + base * prior) / (n + prior)
    return k * math.log(rate / base)


def devig(odds: list[float]) -> list[float]:
    """Power-method de-vig: p_i = (1/o_i)^k with k chosen so the p sum to 1.
    Longshots are shrunk more than favourites, which matches observed bias."""
    inv = [1.0 / o for o in odds]
    lo, hi = 0.5, 3.0
    for _ in range(60):
        k = 0.5 * (lo + hi)
        s = sum(v ** k for v in inv)
        if s > 1.0:
            lo = k
        else:
            hi = k
    k = 0.5 * (lo + hi)
    p = [v ** k for v in inv]
    z = sum(p)
    return [v / z for v in p]


def book_percentage(odds: list[float]) -> float:
    return sum(1.0 / o for o in odds if o and o > 1.0) * 100.0


SPEED_SCORE = {"leader": 1.0, "lead": 1.0, "on pace": 0.7, "on-pace": 0.7, "handy": 0.6,
               "off pace": 0.35, "midfield": 0.4, "mid field": 0.4, "back": 0.15,
               "backmarker": 0.05, "rear": 0.05}


def speed_score(e: Entry, aes_range: Optional[tuple[float, float]] = None) -> Optional[float]:
    """0 = backmarker .. 1 = leader, from the R&S AES pace value (ranked within the
    field), else settling lengths, else the speed-map label."""
    aes = e.extras.get("aes")
    if aes is not None and aes_range and aes_range[1] > aes_range[0]:
        return max(0.0, min(1.0, (aes - aes_range[0]) / (aes_range[1] - aes_range[0])))
    if e.extras.get("settle_frac") is not None:        # R&S Enhanced Form settling positions
        return max(0.0, min(1.0, 1.0 - float(e.extras["settle_frac"])))
    if e.settling is not None:
        return max(0.0, min(1.0, 1.0 - e.settling / 10.0))
    lab = (e.speed_label or "").strip().lower()
    for k in sorted(SPEED_SCORE, key=len, reverse=True):   # 'backmarker' before 'back'
        if k in lab:
            return SPEED_SCORE[k]
    # HK running positions from the latest run: position at the first call / field size
    for r in e.runs[:3]:
        rp = [int(x) for x in r.running_pos.split() if x.isdigit()]
        if rp and r.field_size:
            return max(0.0, min(1.0, 1.0 - (rp[0] - 1) / max(r.field_size - 1, 1)))
    return None


def _class_number(text: str) -> Optional[int]:
    import re
    m = re.search(r"(?:class|cl)\s*(\d)", (text or "").lower())
    return int(m.group(1)) if m else None


def _weighted_margin(e: Entry, card: RaceCard, p: Params) -> tuple[Optional[float], float, int]:
    target = card.dist_m or 1400
    num = den = evid = 0.0
    used = 0
    today_prize = card.prize
    today_class = _class_number(card.race_class)
    for run in e.runs:
        if not run.usable:
            continue
        rel = math.exp(-((run.dist_m - target) / p.sigma_dist) ** 2)
        if run.surface and card.surface and run.surface != card.surface:
            rel *= p.w_offsurface
        if rel <= 1e-6:
            continue
        estimated = bool(run.comment) and run.field_size is not None and run.margin_l is not None \
            and abs(run.margin_l - round(1.4 * ((run.pos or 1) - 1) ** 0.9, 2)) < 1e-6 and (run.pos or 0) > 1
        w = rel * math.exp(-max(run.days_ago or 0, 0) / p.tau_days)
        if estimated:
            w *= p.w_estimated_margin
        fs = run.field_size or p.impute_field_size
        m = run.margin_l * (target / run.dist_m) + p.k_field * (p.field_ref - fs)
        if run.weight and e.weight:
            m += (e.weight - run.weight) * p.lengths_per_kg
        if today_prize and run.prize and run.currency == card.currency and run.prize > 0:
            ratio = math.log(today_prize / run.prize)
            m += p.k_class_adj * max(-1.5, min(1.5, ratio))
        elif today_class is not None:
            rc = _class_number(run.race_class) if not run.race_class.isdigit() else int(run.race_class)
            if rc is not None:
                # lower class number = better race; dropping in class helps
                m += 0.8 * (today_class - rc) * -1.0
        num += w * m
        den += w
        evid += rel * (p.w_estimated_margin if estimated else 1.0)
        used += 1
    if den <= 1e-9:
        return None, 0.0, 0
    return num / den, evid, used


def _form_string_score(fs: str) -> Optional[float]:
    """Fallback when there are no past runs: average of the last six finishing digits
    (0 = 10th+), as a beaten-lengths proxy.  x / - (spells) are skipped."""
    digs = [c for c in (fs or "")[:8] if c.isdigit()]
    if not digs:
        return None
    vals = [10 if c == "0" else int(c) for c in digs[:6]]
    avg = sum(vals) / len(vals)
    return 1.6 * (avg - 1)


# --- rating -------------------------------------------------------------------

def rate(card: RaceCard, p: Params | None = None) -> tuple[list[Rated], list[str]]:
    p = p or defaults_for(card.country)
    notes: list[str] = []
    field_ = card.field_
    if len(field_) < 2:
        return [], ["Need at least two non-scratched runners."]

    dist = card.dist_m or 1400
    weights = [e.weight for e in field_ if e.weight]
    mean_wt = sum(weights) / len(weights) if weights else None
    ratings = [e.official_rating for e in field_ if e.official_rating is not None]
    mean_rt = sum(ratings) / len(ratings) if ratings else None
    barriers = [e.barrier for e in field_ if e.barrier]
    max_bar = max(barriers) if barriers else 0
    secs = sorted(e.sectional_600 for e in field_ if e.sectional_600)
    med_sec = secs[len(secs) // 2] if secs else None
    aes_vals = [e.extras["aes"] for e in field_ if e.extras.get("aes") is not None]
    aes_range = (min(aes_vals), max(aes_vals)) if len(aes_vals) >= 2 else None
    speeds = {e.number: speed_score(e, aes_range) for e in field_}
    afs_vals = [e.extras["afs"] for e in field_ if e.extras.get("afs") is not None]
    afs_mean = sum(afs_vals) / len(afs_vals) if afs_vals else None
    afs_sd = (sum((v - afs_mean) ** 2 for v in afs_vals) / len(afs_vals)) ** 0.5 if afs_vals else 0.0
    jr_vals = [e.extras["jr"] for e in field_ if e.extras.get("jr") is not None]
    jr_mean = sum(jr_vals) / len(jr_vals) if jr_vals else None
    known_speed = [v for v in speeds.values() if v is not None]
    n_leaders = sum(1 for v in known_speed if v >= 0.7)
    pace = "even"
    if known_speed:
        if n_leaders >= 3:
            pace = "hot"
        elif n_leaders == 0:
            pace = "slow"

    rows: list[Rated] = []
    for e in field_:
        t: dict[str, float] = {}
        avg, ev, used = _weighted_margin(e, card, p)
        if avg is None:
            proxy = _form_string_score(e.form_string)
            if proxy is not None:
                avg, ev = proxy, 0.35
        t["form"] = -p.k_form * avg if avg is not None else 0.0

        starts, wins = e.career[0], e.career[1]
        top3 = e.places
        p0 = wins / starts if starts else 0.0
        t["class"] = p.k_class * math.log(((wins + 1.0) / (starts + 8.0)) / 0.125) if starts else 0.0
        t["consistency"] = p.k_conv * math.log(((top3 + 1.0) / (starts + 3.0)) / 0.33) if starts else 0.0

        dn, dw = e.record("Dist")[0], e.record("Dist")[1]
        t["distance"] = p.k_dist * _shrunk_log_ratio(dn, dw, p0, p.prior_starts)
        cn, cw = e.record("Course")[0], e.record("Course")[1]
        t["course"] = p.k_course * _shrunk_log_ratio(cn, cw, p0, p.prior_starts)
        sl = {"AW": "AW", "TURF": "Turf", "DIRT": "Dirt"}.get(card.surface, "")
        sn, sw = e.record(sl)[0], e.record(sl)[1]
        t["surface"] = p.k_surface * _shrunk_log_ratio(sn, sw, p0, p.prior_starts)
        gl = (card.going or "").title().split()[0] if card.going else ""
        gn, gw = e.record(gl)[0], e.record(gl)[1]
        if not gn and gl in ("Soft", "Heavy") and e.record("Wet")[0]:
            gn, gw = e.record("Wet")[0], e.record("Wet")[1]
        t["going"] = p.k_going * _shrunk_log_ratio(gn, gw, p0, p.prior_starts)

        days = e.last_run_days
        if days is None and e.runs:
            ds = [r.days_ago for r in e.runs if r.days_ago is not None]
            days = min(ds) if ds else None
        lay = 0.0
        if days and days > 60:
            lay = -min(2.5, 0.80 * math.log(days / 30.0))
            fn, fw = e.record("First Up")[0], e.record("First Up")[1]
            if fn:
                lay += p.k_firstup * _shrunk_log_ratio(fn, fw, max(p0, 0.05), p.prior_starts)
        elif days is not None and days < 7:
            lay = -0.3
        t["fitness"] = lay

        t["rating"] = (p.k_rating * (e.official_rating - mean_rt)
                       if e.official_rating is not None and mean_rt is not None else 0.0)
        t["jockey"] = _strike_term(e.jockey_stats, p.k_jockey)
        t["trainer"] = _strike_term(e.trainer_stats, p.k_trainer)
        t["combo"] = _strike_term(e.combo_stats, p.k_combo, prior=20.0)
        if e.barrier and max_bar > 1:
            scale = max(0.5, min(1.4, 1200.0 / dist))
            t["barrier"] = -p.k_barrier * scale * (e.barrier - 1) / (max_bar - 1)
        else:
            t["barrier"] = 0.0
        t["weight"] = -p.k_weight * (e.weight - mean_wt) if e.weight and mean_wt else 0.0
        sp = speeds.get(e.number)
        if sp is not None:
            sprint = max(0.4, min(1.2, 1200.0 / dist))
            base = p.k_speed * sprint * (sp - 0.5)
            if pace == "hot":
                base -= p.k_pace * (sp - 0.5)            # leaders burn each other; closers gain
            elif pace == "slow":
                base += p.k_pace * 0.6 * (sp - 0.5)      # lone leader gets a soft lead
            t["speed"] = base
        else:
            t["speed"] = 0.0
        # R&S L600m splits come from different tracks and distances, so they carry half
        # the weight of HKJC's same-course sectionals and are capped at +-1.5 L.
        if e.sectional_600 and med_sec:
            k = p.k_sectional * (0.5 if e.extras.get("sectional_source") else 1.0)
            t["sectional"] = max(-1.5, min(1.5, k * (med_sec - e.sectional_600)))
        else:
            t["sectional"] = 0.0
        afs = e.extras.get("afs")
        t["late speed"] = (p.k_late * (afs - afs_mean) / afs_sd
                           if afs is not None and afs_mean is not None and afs_sd > 1e-6 else 0.0)
        jr = e.extras.get("jr")
        if jr is not None and jr_mean is not None and not e.jockey_stats:
            t["jockey"] = p.k_jr * (jr - jr_mean)      # R&S jockey rating stands in for a record
        mv = e.move_pct
        t["market move"] = -p.k_move * math.log(1.0 + mv) if mv is not None and mv > -0.95 else 0.0
        t["tips"] = p.k_tip * (3 - e.tip_rank) / 2.0 if e.tip_rank and e.tip_rank <= 3 else 0.0

        r = Rated(entry=e, number=e.number or 0, name=e.name, terms=t, evidence=ev, used_runs=used,
                  avg_margin=avg, rating_raw=sum(t.values()), move_pct=mv, speed_score=sp)
        r.rating = r.rating_raw
        if mv is not None:
            r.move_label = "steamer" if mv <= -0.15 else ("drifter" if mv >= 0.20 else "steady")
        # market price: fixed-odds first, then exchange, then a morning line / paste price
        if e.odds and e.odds > 1:
            r.odds, r.price_source = e.odds, "bookmaker"
        elif e.exchange_odds and e.exchange_odds > 1:
            r.odds, r.price_source = e.exchange_odds, "exchange"
        elif e.extras.get("morning_line"):
            r.odds, r.price_source = float(e.extras["morning_line"]), "morning line"
        rows.append(r)

    # Centre every term on its field mean.  A softmax is shift-invariant, so this changes
    # no probability; it makes each term read as "vs the field" in the breakdown chart.
    for key in rows[0].terms:
        mu = sum(r.terms[key] for r in rows) / len(rows)
        for r in rows:
            r.terms[key] -= mu
    for r in rows:
        r.rating_raw = sum(r.terms.values())
        r.rating = r.rating_raw

    priced = all(r.odds for r in rows)
    mean_raw = sum(r.rating_raw for r in rows) / len(rows)
    if priced:
        pm = devig([r.odds for r in rows])
        for r, v in zip(rows, pm):
            r.p_market = v
        implied = [p.spread * math.log(r.p_market) for r in rows]
        mean_imp = sum(implied) / len(implied)
        anchors = [mean_raw + (i - mean_imp) for i in implied]
    else:
        anchors = [mean_raw] * len(rows)
    for r, anchor in zip(rows, anchors):
        conf = r.evidence / (r.evidence + 1.2)
        r.rating = anchor + conf * (r.rating_raw - anchor)
    for r, pm_ in zip(rows, _softmax([r.rating for r in rows], p.spread)):
        r.p_model = pm_

    thin = [r.name for r in rows if r.used_runs == 0]
    if thin and len(thin) == len(rows):
        notes.append("No runner has a usable past run, so the form model is anchored on the "
                     "market and shows little independent opinion.")
    elif thin:
        notes.append("No usable past run for " + ", ".join(thin)
                     + "; those are anchored on the market rather than rated on form.")
    if priced:
        w = p.market_weight
        blend = [math.exp(w * math.log(max(r.p_model, 1e-9)) + (1 - w) * math.log(r.p_market))
                 for r in rows]
        z = sum(blend)
        for r, b in zip(rows, blend):
            r.p_final = b / z
    else:
        missing = [r.name for r in rows if not r.odds]
        notes.append("No price for " + ", ".join(missing[:8])
                     + ("…" if len(missing) > 8 else "")
                     + " - showing the form model alone, with no market blend.")
        for r in rows:
            r.p_final = r.p_model
    if pace != "even":
        notes.append(f"Pace shape looks **{pace}**: {n_leaders} runner(s) map as leaders/on-pace "
                     f"out of {len(known_speed)} with a known run style.")
    _top_probs(rows)
    rows.sort(key=lambda r: -r.p_final)
    return rows, notes


def _top_probs(rows: list[Rated]) -> None:
    """Exact Plackett-Luce top-2 / top-3 by enumeration."""
    n = len(rows)
    p = [r.p_final for r in rows]
    z = sum(p) or 1.0
    p = [v / z for v in p]
    top2, top3 = [0.0] * n, [0.0] * n
    for i in range(n):
        top2[i] += p[i]
        top3[i] += p[i]
        r1 = 1.0 - p[i]
        if r1 <= 0:
            continue
        for j in range(n):
            if j == i:
                continue
            pj = p[i] * p[j] / r1
            top2[j] += pj
            top3[j] += pj
            r2 = r1 - p[j]
            if r2 <= 0:
                continue
            for k in range(n):
                if k not in (i, j):
                    top3[k] += pj * p[k] / r2
    for r, a, b in zip(rows, top2, top3):
        r.p_top2, r.p_top3 = min(a, 1.0), min(b, 1.0)


# --- simulation ---------------------------------------------------------------

def simulate(rows: list[Rated], p: Params | None = None) -> SimResult:
    p = p or Params()
    n = len(rows)
    rng = np.random.default_rng(p.seed)
    strength = np.array([math.log(max(r.p_final, 1e-9)) for r in rows])
    noise = rng.gumbel(size=(p.sims, n))
    perf = strength[None, :] + noise
    order = np.argsort(-perf, axis=1)                     # sims x positions -> runner idx
    matrix = np.zeros((n, n))
    for pos in range(n):
        counts = np.bincount(order[:, pos], minlength=n)
        matrix[:, pos] = counts / p.sims
    pair = {}
    trip = {}
    for row in order[:, :3]:
        a, b, c = int(row[0]), int(row[1]), int(row[2])
        pair[(a, b)] = pair.get((a, b), 0) + 1
        trip[(a, b, c)] = trip.get((a, b, c), 0) + 1
    lab = lambda i: f"{rows[i].number} {rows[i].name}"  # noqa: E731
    exactas = sorted(((f"{lab(a)} → {lab(b)}", v / p.sims) for (a, b), v in pair.items()),
                     key=lambda x: -x[1])[:8]
    trifectas = sorted(((f"{lab(a)} → {lab(b)} → {lab(c)}", v / p.sims) for (a, b, c), v in trip.items()),
                       key=lambda x: -x[1])[:8]
    expected = [float((matrix[i] * np.arange(1, n + 1)).sum()) for i in range(n)]
    return SimResult(names=[r.name for r in rows], numbers=[r.number for r in rows],
                     position_matrix=matrix, exactas=exactas, trifectas=trifectas,
                     expected_pos=expected)


# --- insights -----------------------------------------------------------------

def insights(card: RaceCard, rows: list[Rated], notes: list[str]) -> list[str]:
    out: list[str] = []
    if not rows:
        return out
    top = rows[0]
    strong = sorted(((k, v) for k, v in top.terms.items() if abs(v) > 0.15), key=lambda kv: -kv[1])
    plus = [k for k, v in strong if v > 0][:3]
    minus = [k for k, v in strong if v < 0][:2]
    line = f"**{top.number} {top.name}** rates top at {top.p_final:.0%}"
    if top.p_market:
        line += f" (market {top.p_market:.0%})"
    if plus:
        line += "; strongest on " + ", ".join(plus)
    if minus:
        line += "; marked down for " + ", ".join(minus)
    out.append(line + ".")
    if len(rows) > 1:
        gap = rows[0].p_final - rows[1].p_final
        if gap < 0.04:
            out.append(f"It is close: **{rows[1].number} {rows[1].name}** is within "
                       f"{gap:.1%} of the top pick.")
    odds = [r.odds for r in rows if r.odds]
    if len(odds) == len(rows):
        bp = book_percentage(odds)
        coherent = 102 <= bp <= 180
        if coherent:
            val = [r for r in rows if r.p_market and r.p_model / r.p_market >= 1.35 and r.p_final >= 0.05]
            for r in val[:3]:
                out.append(f"**{r.number} {r.name}**: form model {r.p_model:.0%} vs market "
                           f"{r.p_market:.0%} - the form reads better than the price "
                           f"({r.odds:.1f}). Treat as a prompt to look closer, not an edge.")
        else:
            out.append(f"Book percentage is {bp:.0f}%, so the prices are not a coherent market "
                       "(mixed sources or stale). Ranking shown, no value flags.")
    steam = [r for r in rows if r.move_label == "steamer"]
    drift = [r for r in rows if r.move_label == "drifter"]
    if steam:
        out.append("Steamers (firmed 15%+ since opening): " + ", ".join(
            f"**{r.number} {r.name}** {r.entry.odds_open:g}→{r.entry.odds:g}" for r in steam[:4]))
    if drift:
        out.append("Drifters (eased 20%+): " + ", ".join(
            f"**{r.number} {r.name}** {r.entry.odds_open:g}→{r.entry.odds:g}" for r in drift[:4]))
    hot = [r for r in rows if r.terms.get("jockey", 0) + r.terms.get("trainer", 0) > 0.9]
    if hot:
        out.append("Strong connections: " + ", ".join(f"{r.number} {r.name}" for r in hot[:4]))
    fresh = [r for r in rows if r.entry.last_run_days and r.entry.last_run_days > 90]
    if fresh:
        out.append("Resuming from 90+ days: " + ", ".join(
            f"{r.number} {r.name} ({r.entry.last_run_days}d)" for r in fresh[:5]))
    out.extend(notes)
    return out


def sensitivity(card: RaceCard, base: Params, draws: int = 200) -> dict[str, float]:
    """How often each runner rates top when the constants are jittered."""
    import random
    rng = random.Random(base.seed)
    tally: dict[str, int] = {}
    for _ in range(draws):
        q = replace(base,
                    tau_days=rng.uniform(100, 300), sigma_dist=base.sigma_dist * rng.uniform(0.7, 1.4),
                    spread=base.spread * rng.uniform(0.8, 1.3), k_form=rng.uniform(0.6, 1.4),
                    k_class=rng.uniform(1.0, 2.8), k_conv=rng.uniform(0.6, 2.0),
                    k_dist=rng.uniform(0.5, 2.0), k_course=rng.uniform(0.4, 1.6),
                    k_going=rng.uniform(0.3, 1.5), k_rating=base.k_rating * rng.uniform(0.5, 1.6),
                    k_jockey=rng.uniform(2.0, 6.0), k_trainer=rng.uniform(1.5, 5.0),
                    k_barrier=rng.uniform(0.3, 1.5), k_weight=rng.uniform(0.15, 0.6),
                    k_speed=rng.uniform(0.2, 0.9), k_pace=rng.uniform(0.2, 1.0),
                    k_late=rng.uniform(0.15, 0.8), k_jr=rng.uniform(0.1, 0.5),
                    k_move=rng.uniform(0.2, 1.4), market_weight=rng.uniform(0.2, 0.55),
                    k_class_adj=base.k_class_adj * rng.uniform(0.3, 1.6))
        rows, _ = rate(card, q)
        if rows:
            tally[rows[0].name] = tally.get(rows[0].name, 0) + 1
    return {k: v / draws for k, v in sorted(tally.items(), key=lambda kv: -kv[1])}
