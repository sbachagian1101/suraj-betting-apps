"""Conditional-logit rating for greyhound races parsed from R&S Full Fields.

Each runner's last-10 runs become a weighted average beaten margin expressed in
lengths at the race distance.  That margin, plus a handful of shrunk record
terms, becomes a rating in lengths; ratings go through a softmax to give
race-conditional win probabilities, which are then blended in log space with the
de-vigged market.

Every record term is shrunk toward the runner's own career strike rate with a
prior worth `PRIOR_STARTS` starts.  That is not decoration: an earlier version
used a prior worth 7 starts, which let a *two-start* distance record move a dog
1.9 lengths, and the two runners it penalised hardest in a live test finished
first and second.

Known weakness, stated plainly: there is **no opposition-strength adjustment**.
A dog beaten 1L in a weak race outrates one beaten 5L in a strong one.  Fixing
that properly means fitting abilities jointly across many races rather than
rating one race in isolation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from rs_parser import Race, Runner

# --- tunables -----------------------------------------------------------------


@dataclass
class Params:
    tau_days: float = 200.0        # recency decay scale
    sigma_dist: float = 90.0       # distance-relevance gaussian width (m)
    w_straight: float = 0.35       # weight on straight-track form in a circle race
    w_foreign: float = 0.75        # weight on overseas form
    w_offsurface: float = 0.70     # weight on turf form in an AW race, and vice versa
    spread: float = 2.40           # lengths of performance SD (softmax temperature)
    prior_starts: float = 15.0     # shrinkage prior for distance/course records
    k_class: float = 2.20
    k_conv: float = 1.90
    k_dist: float = 1.60
    k_course: float = 1.40
    k_surface: float = 1.40
    k_sectional: float = 0.45
    k_gap: float = 0.00            # vacant-box term: UNVALIDATED, off by default
    market_weight: float = 0.38    # weight on the model in the log-space blend
    devig_power: float = 1.06      # >1 damps favourite-longshot bias
    impute_field_size: int = 7     # for runs with an impossible "X of Y"


@dataclass
class Rated:
    runner: Runner
    box: int
    eff_box: int
    gap_inside: int
    odds: float | None
    avg_margin: float
    evidence: float
    terms: dict[str, float] = field(default_factory=dict)
    rating: float = 0.0
    p_model: float = 0.0
    p_market: float | None = None
    p_final: float = 0.0
    p_top2: float = 0.0
    p_top3: float = 0.0

    @property
    def name(self) -> str:
        return self.runner.name

    @property
    def fair(self) -> float:
        return 1.0 / self.p_final if self.p_final > 0 else float("inf")

    @property
    def ev(self) -> float | None:
        if not self.odds:
            return None
        return self.p_final * self.odds - 1.0


# base box advantage in lengths by *effective* position from the rail
_BOX_PROFILE = [0.50, 0.30, 0.10, -0.05, -0.20, -0.30, -0.40, -0.45]


def _box_adj(eff: int, dist_m: int | None) -> float:
    base = _BOX_PROFILE[min(eff, len(_BOX_PROFILE)) - 1]
    d = dist_m or 420
    return base * max(0.75, min(1.25, 420.0 / d))


def _shrunk_log_ratio(sub_starts: int, sub_wins: int, p0: float, prior: float) -> float:
    """log( shrunk sub-population strike rate / career strike rate )."""
    if p0 <= 0 or sub_starts <= 0:
        return 0.0
    rate = (sub_wins + p0 * prior) / (sub_starts + prior)
    return math.log(max(rate, 1e-6) / p0)


def _weighted_margin(r: Runner, race: Race, p: Params) -> tuple[float, float]:
    """(weighted average beaten margin in lengths at race distance, evidence)."""
    target = race.dist_m or 400
    num = den = 0.0
    for run in r.runs:
        if run.margin is None or not run.dist_m or run.days_ago is None:
            continue
        w = math.exp(-max(run.days_ago, 0) / p.tau_days)
        w *= math.exp(-((run.dist_m - target) / p.sigma_dist) ** 2)
        kind = run.track_kind
        if kind == "straight" and not race.is_straight:
            w *= p.w_straight
        elif kind == "circle" and race.is_straight:
            w *= p.w_straight
        elif kind == "foreign":
            w *= p.w_foreign
        # surface is a SEPARATE axis from track shape: a straight turf run is
        # discounted on both counts for an all-weather circle race.
        if run.surface and race.surface and run.surface != race.surface:
            w *= p.w_offsurface
        if w <= 1e-6:
            continue
        fs = run.field_size or p.impute_field_size
        m = run.margin * (target / run.dist_m) + 0.45 * (4.5 - fs)
        num += w * m
        den += w
    if den <= 0:
        return 8.0, 0.0
    return num / den, den


def _sectional_ref(r: Runner, race: Race) -> tuple[float | None, int]:
    """Own sectionals at this track AND this distance only - R&S measure the
    split to a different marker per track/distance, so they are not comparable
    across either."""
    code = race.track_code()
    vals = [run.sectional for run in r.runs
            if run.sectional and run.dist_m == race.dist_m
            and (not code or run.track == code)]
    if not vals:
        return None, 0
    return 0.5 * min(vals) + 0.5 * (sum(vals) / len(vals)), len(vals)


def _softmax(vals: list[float], temp: float) -> list[float]:
    hi = max(vals)
    ex = [math.exp((v - hi) / temp) for v in vals]
    z = sum(ex)
    return [e / z for e in ex]


def rate(race: Race, p: Params | None = None) -> tuple[list[Rated], list[str]]:
    """Rate a parsed race.  Returns (rated runners sorted by p_final, notes)."""
    p = p or Params()
    notes: list[str] = []
    field_ = race.field_
    if len(field_) < 2:
        return [], ["Need at least two non-scratched runners to rate a race."]

    boxes = [r.box for r in field_ if r.box is not None]
    rows: list[Rated] = []

    # sectional reference = median of whoever has one at this track+distance
    refs = [_sectional_ref(r, race) for r in field_]
    have = sorted(v for v, n in refs if v is not None)
    field_ref = have[len(have) // 2] if have else None
    if have and len(have) < len(field_):
        notes.append(
            f"Only {len(have)} of {len(field_)} runners have a sectional at "
            f"{race.dist_m}m {race.track_code() or 'this track'}; the rest are "
            "treated as field-average on early speed.")

    for r in field_:
        box = r.box or 0
        eff = sum(1 for b in boxes if b < box) + 1
        inner = [b for b in boxes if b < box]
        gap = box - (max(inner) if inner else 0) - 1

        avg, ev = _weighted_margin(r, race, p)
        starts, wins = r.career_starts, r.career_wins
        p0 = wins / starts if starts else 0.0
        seconds, thirds = r.career_places
        top3 = wins + seconds + thirds

        terms: dict[str, float] = {}
        terms["form"] = -avg
        terms["class"] = p.k_class * math.log(
            ((wins + 1.0) / (starts + 8.0)) / 0.125) if starts else 0.0
        terms["conversion"] = p.k_conv * math.log(
            ((wins + 0.5) / (top3 + 1.5)) / 0.33) if top3 else 0.0

        dn, dw = r.record("Dist")[0], r.record("Dist")[1]
        terms["distance"] = p.k_dist * _shrunk_log_ratio(dn, dw, p0, p.prior_starts)
        cn, cw = r.record("Course")[0], r.record("Course")[1]
        terms["course"] = p.k_course * _shrunk_log_ratio(cn, cw, p0, p.prior_starts)

        # surface suitability from the runner's own AW / Turf panel record
        surf_label = "AW" if race.surface == "AW" else "Turf"
        sn, sw = r.record(surf_label)[0], r.record(surf_label)[1]
        terms["surface"] = p.k_surface * _shrunk_log_ratio(sn, sw, p0, p.prior_starts)

        ref, nsec = _sectional_ref(r, race)
        if ref is not None and field_ref is not None:
            terms["early speed"] = (p.k_sectional * ((field_ref - ref) / 0.10)
                                    * (nsec / (nsec + 2.0)))
        else:
            terms["early speed"] = 0.0

        days = min((run.days_ago for run in r.runs if run.days_ago is not None),
                   default=None)
        terms["layoff"] = (-min(2.5, 0.80 * math.log(days / 30.0))
                           if days and days > 60 else 0.0)
        terms["box"] = _box_adj(eff, race.dist_m)
        terms["vacant box"] = p.k_gap * gap

        rows.append(Rated(runner=r, box=box, eff_box=eff, gap_inside=gap,
                          odds=r.odds, avg_margin=avg, evidence=ev, terms=terms,
                          rating=sum(terms.values())))

    # shrink thin-evidence runners toward the field mean rating
    mean_raw = sum(x.rating for x in rows) / len(rows)
    for x in rows:
        conf = x.evidence / (x.evidence + 0.60)
        x.rating = mean_raw + conf * (x.rating - mean_raw)

    p_model = _softmax([x.rating for x in rows], p.spread)
    for x, pm in zip(rows, p_model):
        x.p_model = pm

    priced = [x for x in rows if x.odds and x.odds > 1.0]
    if len(priced) == len(rows):
        inv = [(1.0 / x.odds) ** p.devig_power for x in rows]
        z = sum(inv)
        for x, v in zip(rows, inv):
            x.p_market = v / z
        w = p.market_weight
        blend = [math.exp(w * math.log(x.p_model) + (1 - w) * math.log(x.p_market))
                 for x in rows]
        bz = sum(blend)
        for x, b in zip(rows, blend):
            x.p_final = b / bz
    else:
        missing = [x.name for x in rows if not (x.odds and x.odds > 1.0)]
        notes.append("No usable price for " + ", ".join(missing)
                     + " - showing the form model alone, with no market blend.")
        for x in rows:
            x.p_final = x.p_model

    _plackett_luce(rows)
    rows.sort(key=lambda x: -x.p_final)
    return rows, notes


def _plackett_luce(rows: list[Rated]) -> None:
    """Exact top-2 / top-3 probabilities by enumeration (fields are small)."""
    n = len(rows)
    p = [x.p_final for x in rows]
    total = sum(p)
    if total <= 0:
        return
    p = [v / total for v in p]
    top2 = [0.0] * n
    top3 = [0.0] * n
    for i in range(n):
        top2[i] += p[i]
        top3[i] += p[i]
        for j in range(n):
            if j == i:
                continue
            r1 = 1.0 - p[i]
            if r1 <= 0:
                continue
            pj = p[i] * p[j] / r1
            top2[j] += pj
            top3[j] += pj
            for k in range(n):
                if k in (i, j):
                    continue
                r2 = 1.0 - p[i] - p[j]
                if r2 <= 0:
                    continue
                top3[k] += pj * p[k] / r2
    for x, a, b in zip(rows, top2, top3):
        x.p_top2 = min(a, 1.0)
        x.p_top3 = min(b, 1.0)


def quinellas(rows: list[Rated], limit: int = 5) -> list[tuple[str, float]]:
    """Most likely unordered first-two combinations."""
    out: list[tuple[str, float]] = []
    n = len(rows)
    for i in range(n):
        for j in range(i + 1, n):
            a, b = rows[i], rows[j]
            pa, pb = a.p_final, b.p_final
            prob = 0.0
            if 1 - pa > 0:
                prob += pa * pb / (1 - pa)
            if 1 - pb > 0:
                prob += pb * pa / (1 - pb)
            out.append((f"{a.box}-{b.box}  {a.name} / {b.name}", prob))
    out.sort(key=lambda t: -t[1])
    return out[:limit]


def sensitivity(race: Race, base: Params, draws: int = 600, seed: int = 7
                ) -> dict[str, float]:
    """How often each runner rates top when the parameters are jittered.

    A selection that survives this is a property of the form, not of one set of
    hand-picked constants.
    """
    import random
    rng = random.Random(seed)
    tally: dict[str, int] = {}
    for _ in range(draws):
        q = Params(
            tau_days=rng.uniform(120, 320),
            sigma_dist=rng.uniform(65, 130),
            w_straight=rng.uniform(0.15, 0.60),
            w_foreign=rng.uniform(0.55, 0.95),
            spread=rng.uniform(1.9, 3.2),
            prior_starts=rng.uniform(10, 22),
            k_class=rng.uniform(1.2, 3.2),
            k_conv=rng.uniform(0.8, 3.0),
            k_dist=rng.uniform(0.7, 2.6),
            k_course=rng.uniform(0.6, 2.4),
            k_surface=rng.uniform(0.6, 2.4),
            w_offsurface=rng.uniform(0.5, 0.9),
            k_sectional=rng.uniform(0.2, 0.8),
            k_gap=base.k_gap,
            market_weight=rng.uniform(0.20, 0.60),
            devig_power=rng.uniform(1.0, 1.12),
            impute_field_size=rng.choice([6, 7, 8]),
        )
        rated, _ = rate(race, q)
        if rated:
            tally[rated[0].name] = tally.get(rated[0].name, 0) + 1
    return {k: v / draws for k, v in sorted(tally.items(), key=lambda t: -t[1])}
