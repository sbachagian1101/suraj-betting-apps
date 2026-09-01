"""Conditional-logit rating for races parsed from R&S Full Fields pages.

Each runner's last-10 runs become a weighted average beaten margin expressed in
lengths at the race distance.  That margin, plus shrunk record terms, becomes a
rating in lengths; ratings go through a softmax to give race-conditional win
probabilities, which are blended in log space with the de-vigged market.

The core is identical for all three codes.  What differs:

* **thoroughbred** - past margins are adjusted for the **weight** carried, and a
  **barrier** term is applied.  Without the weight adjustment a handicapper's
  margins are not comparable across its own runs.
* **harness** - a **DQG** (broke gait) is not a result.  Those runs are excluded
  from the margin average and scored separately as a reliability term, because
  the *rate* of it is the most predictive thing on a French trot page.  Past
  runs off a distance handicap are credited for the extra ground.
* **greyhound** - early-speed sectionals and an effective box position that
  counts vacant boxes.

Every record term is shrunk toward the runner's own career strike rate with a
prior worth `prior_starts` starts.  That is not decoration: an earlier version
used a prior worth 7 starts, which let a *two-start* distance record move a dog
1.9 lengths, and the two runners it penalised hardest in a live race finished
first and second.

Known weakness, stated plainly: there is **no opposition-strength adjustment**.
A runner beaten 1L in a weak race outrates one beaten 5L in a strong one.
Fixing that properly means fitting abilities jointly across many races.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace  # noqa: F401  (re-exported)

from rs_parser import (GREYHOUND, HARNESS, METRES_PER_LENGTH, THOROUGHBRED,
                       Race, Runner)

# --- tunables -----------------------------------------------------------------


@dataclass
class Params:
    tau_days: float = 200.0        # recency decay scale
    sigma_dist: float = 90.0       # distance-relevance gaussian width (m)
    w_straight: float = 0.35       # straight-track form in a circle race
    w_foreign: float = 0.75        # overseas form
    w_offsurface: float = 0.70     # turf form in an AW race, and vice versa
    spread: float = 2.40           # lengths of performance SD (softmax temp)
    prior_starts: float = 15.0     # shrinkage prior for record terms
    k_class: float = 2.20
    k_conv: float = 1.90
    k_dist: float = 1.60
    k_course: float = 1.40
    k_surface: float = 1.40
    k_going: float = 1.00
    k_sectional: float = 0.45      # greyhound only
    k_gap: float = 0.00            # greyhound vacant-box term: UNVALIDATED
    k_barrier: float = 0.80        # thoroughbred only
    lengths_per_kg: float = 0.70   # thoroughbred weight scale
    k_dq: float = 3.50             # harness reliability term
    k_class_adj: float = 1.20      # lengths per log-unit of prizemoney ratio
    field_ref: float = 4.5         # "neutral" field size for this code
    k_field: float = 0.45          # lengths of credit per runner above field_ref
    handicap_credit: float = 0.50  # harness: share of a distance handicap credited
    market_weight: float = 0.38
    devig_power: float = 1.06
    impute_field_size: int = 7


#: sensible starting points per code.  Margins live on very different scales:
#: a greyhound sprint is decided inside 3 lengths, a French trot can be 70.
CODE_DEFAULTS: dict[str, dict[str, float]] = {
    # k_class_adj is OFF for greyhound on the evidence: across the fixtures a
    # single grade (GR 5) spans $1.4k-$9.0k, a 6.4x spread, so Australian
    # greyhound prizemoney tracks the VENUE, not the class, and using it as a
    # class proxy injected noise - it took the backtest from 2/3 to 1/3 and put
    # the blend behind the market.  In France prizemoney IS the class ladder
    # (CL1 ~E88k > CL2 ~E51k > CL3 ~E29k; trot D > E > F > G > H), each band
    # tight to <=1.3x, so the term is kept there.
    GREYHOUND:    {"spread": 2.40, "sigma_dist": 90.0, "k_class_adj": 0.00,
                   "field_ref": 4.5, "k_field": 0.45, "impute_field_size": 7},
    THOROUGHBRED: {"spread": 4.00, "sigma_dist": 220.0, "k_class_adj": 2.00,
                   "field_ref": 12.0, "k_field": 0.15, "impute_field_size": 14},
    HARNESS:      {"spread": 9.00, "sigma_dist": 320.0, "k_class_adj": 4.50,
                   "field_ref": 13.0, "k_field": 0.15, "impute_field_size": 13},
}


def defaults_for(code: str) -> Params:
    return replace(Params(), **CODE_DEFAULTS.get(code, {}))


@dataclass
class Rated:
    runner: Runner
    tab: int
    eff_box: int
    gap_inside: int
    odds: float | None
    avg_margin: float
    evidence: float
    used_runs: int
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
        return None if not self.odds else self.p_final * self.odds - 1.0


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


def _weighted_margin(r: Runner, race: Race, p: Params) -> tuple[float, float, int]:
    """(weighted average beaten margin in lengths, evidence, runs used)."""
    target = race.dist_m or 400
    today_prize = race.prize_value
    num = den = rel_total = 0.0
    used = 0
    for run in r.runs:
        if not run.counts_as_form:
            continue          # DQG / no result: excluded, scored separately
        # RELEVANCE: how much this run tells us about today's race at all.
        rel = math.exp(-((run.dist_m - target) / p.sigma_dist) ** 2)
        kind = run.track_kind
        if kind == "straight" and not race.is_straight:
            rel *= p.w_straight
        elif kind == "circle" and race.is_straight:
            rel *= p.w_straight
        elif kind == "foreign":
            rel *= p.w_foreign
        if run.surface and race.surface_code and run.surface != race.surface_code:
            rel *= p.w_offsurface
        if rel <= 1e-6:
            continue
        # RECENCY weights the average, but must NOT also drive confidence.
        # Staleness is already paid for here and again by the layoff term;
        # charging it a third time as low confidence anchored a long-spelled
        # runner on the market instead of on its own form.  Blazing Ace had ten
        # usable runs and a confidence of 0.36 purely because they were old.
        w = rel * math.exp(-max(run.days_ago, 0) / p.tau_days)

        fs = run.field_size or p.impute_field_size
        # Beating more rivals is worth more, but the credit must be on the
        # CODE's scale: a flat greyhound coefficient applied to a 20-runner
        # thoroughbred field subtracts seven lengths and swamps the margin.
        m = run.margin * (target / run.dist_m) + p.k_field * (p.field_ref - fs)

        if race.code == THOROUGHBRED and run.weight and r.weight:
            # worse off at the weights today -> its past margin flatters it
            m += (r.weight - run.weight) * p.lengths_per_kg
        if race.code == HARNESS and run.handicap_m:
            # started behind scratch: credit the extra ground it had to make up
            m -= p.handicap_credit * (run.handicap_m / METRES_PER_LENGTH)

        # Opposition strength, via prizemoney as a class proxy.  Without this a
        # runner beaten 1L in a weak race outrates one beaten 5L in a strong
        # one - the model's oldest and worst weakness.  Beating a cheaper field
        # is discounted; running well in a richer one is credited.  Currencies
        # are never mixed: a cross-currency ratio would need a rate.
        rp = run.prize_value
        if today_prize and rp and rp[0] == today_prize[0] and rp[1] > 0:
            ratio = math.log(today_prize[1] / rp[1])
            m += p.k_class_adj * max(-1.5, min(1.5, ratio))

        num += w * m
        den += w
        rel_total += rel
        used += 1
    if den <= 1e-9:
        return 8.0, 0.0, 0
    return num / den, rel_total, used


def _sectional_ref(r: Runner, race: Race) -> tuple[float | None, int]:
    """Own sectionals at this track AND distance only - R&S measure the split to
    a different marker per track and per distance."""
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
    p = p or defaults_for(race.code)
    notes: list[str] = []
    field_ = race.field_
    if len(field_) < 2:
        return [], ["Need at least two non-scratched runners to rate a race."]

    tabs = [r.tab for r in field_ if r.tab is not None]
    barriers = [r.barrier for r in field_ if r.barrier]
    max_bar = max(barriers) if barriers else 0

    if race.code == GREYHOUND:
        refs = [_sectional_ref(r, race) for r in field_]
        have = sorted(v for v, _ in refs if v is not None)
        field_ref = have[len(have) // 2] if have else None
        if have and len(have) < len(field_):
            notes.append(
                f"Only {len(have)} of {len(field_)} runners have a sectional at "
                f"{race.dist_m}m {race.track_code() or 'this track'}; the rest are "
                "treated as field-average on early speed.")
    else:
        field_ref = None

    rows: list[Rated] = []
    for r in field_:
        tab = r.tab or 0
        eff = sum(1 for t in tabs if t < tab) + 1
        inner = [t for t in tabs if t < tab]
        gap = tab - (max(inner) if inner else 0) - 1

        avg, ev, used = _weighted_margin(r, race, p)
        starts, wins = r.career_starts, r.career_wins
        p0 = wins / starts if starts else 0.0
        seconds, thirds = r.career_places
        top3 = wins + seconds + thirds

        t: dict[str, float] = {}
        t["form"] = -avg
        t["class"] = p.k_class * math.log(
            ((wins + 1.0) / (starts + 8.0)) / 0.125) if starts else 0.0
        t["conversion"] = p.k_conv * math.log(
            ((wins + 0.5) / (top3 + 1.5)) / 0.33) if top3 else 0.0

        dn, dw = r.record("Dist")[0], r.record("Dist")[1]
        t["distance"] = p.k_dist * _shrunk_log_ratio(dn, dw, p0, p.prior_starts)
        cn, cw = r.record("Course")[0], r.record("Course")[1]
        t["course"] = p.k_course * _shrunk_log_ratio(cn, cw, p0, p.prior_starts)
        sl = race.surface_record_label
        sn, sw = r.record(sl)[0], r.record(sl)[1]
        t["surface"] = p.k_surface * _shrunk_log_ratio(sn, sw, p0, p.prior_starts)
        gl = race.going_record_label
        gn, gw = r.record(gl)[0], r.record(gl)[1]
        t["going"] = p.k_going * _shrunk_log_ratio(gn, gw, p0, p.prior_starts)

        if race.code == GREYHOUND:
            ref, nsec = _sectional_ref(r, race)
            t["early speed"] = ((p.k_sectional * ((field_ref - ref) / 0.10)
                                 * (nsec / (nsec + 2.0)))
                                if ref is not None and field_ref is not None else 0.0)
            t["box"] = _box_adj(eff, race.dist_m)
            t["vacant box"] = p.k_gap * gap
        elif race.code == THOROUGHBRED:
            t["barrier"] = (-p.k_barrier * (r.barrier - 1) / (max_bar - 1)
                            if r.barrier and max_bar > 1 else 0.0)
        elif race.code == HARNESS:
            n = len(r.runs)
            t["reliability"] = -p.k_dq * ((r.dq_count + 0.5) / (n + 2.0)) if n else 0.0

        days = min((run.days_ago for run in r.runs if run.days_ago is not None),
                   default=None)
        t["layoff"] = (-min(2.5, 0.80 * math.log(days / 30.0))
                       if days and days > 60 else 0.0)

        rows.append(Rated(runner=r, tab=tab, eff_box=eff, gap_inside=gap,
                          odds=r.odds, avg_margin=avg, evidence=ev, used_runs=used,
                          terms=t, rating=sum(t.values())))

    priced = all(x.odds and x.odds > 1.0 for x in rows)
    mean_raw = sum(x.rating for x in rows) / len(rows)

    # Shrink thin evidence toward the MARKET, not toward the field mean.
    #
    # Shrinking toward the mean drives an evidence-free model to uniform, and a
    # uniform model is NOT a neutral input to the blend: with p_model constant,
    # p_final reduces to p_market^(1-w), which flattens the market and inflates
    # every longshot.  At Cabourg (where a parse bug had left every runner with
    # zero usable runs) that manufactured "+88% EV" on a 101/1 shot out of
    # nothing.  Anchoring on the market instead means a runner the model knows
    # nothing about simply gets the market's opinion and shows no edge.
    if priced:
        inv = [(1.0 / x.odds) ** p.devig_power for x in rows]
        z = sum(inv)
        for x, v in zip(rows, inv):
            x.p_market = v / z
        implied = [p.spread * math.log(x.p_market) for x in rows]
        mean_imp = sum(implied) / len(implied)
        anchors = [mean_raw + (i - mean_imp) for i in implied]
    else:
        anchors = [mean_raw] * len(rows)

    for x, anchor in zip(rows, anchors):
        conf = x.evidence / (x.evidence + 0.60)
        x.rating = anchor + conf * (x.rating - anchor)

    for x, pm in zip(rows, _softmax([x.rating for x in rows], p.spread)):
        x.p_model = pm

    thin = [x.name for x in rows if x.used_runs == 0]
    if thin and len(thin) == len(rows):
        notes.append(
            "**No runner has a usable past run for this race**, so the form "
            "model has no independent opinion and is showing the market. Check "
            "the Diagnostics tab — this usually means the distance or surface "
            "line did not parse.")
    elif thin:
        notes.append("No usable past run for " + ", ".join(thin)
                     + " - those are anchored on the market rather than rated.")

    if priced:
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
    """Exact top-2 / top-3 probabilities by enumeration."""
    n = len(rows)
    p = [x.p_final for x in rows]
    total = sum(p)
    if total <= 0:
        return
    p = [v / total for v in p]
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
    for x, a, b in zip(rows, top2, top3):
        x.p_top2, x.p_top3 = min(a, 1.0), min(b, 1.0)


def quinellas(rows: list[Rated], limit: int = 5) -> list[tuple[str, float]]:
    """Most likely unordered first-two combinations."""
    out: list[tuple[str, float]] = []
    for i in range(len(rows)):
        for j in range(i + 1, len(rows)):
            a, b = rows[i], rows[j]
            prob = 0.0
            if 1 - a.p_final > 0:
                prob += a.p_final * b.p_final / (1 - a.p_final)
            if 1 - b.p_final > 0:
                prob += b.p_final * a.p_final / (1 - b.p_final)
            out.append((f"{a.tab}-{b.tab}  {a.name} / {b.name}", prob))
    out.sort(key=lambda t: -t[1])
    return out[:limit]


def sensitivity(race: Race, base: Params, draws: int = 400, seed: int = 7
                ) -> dict[str, float]:
    """How often each runner rates top when the constants are jittered.  A
    selection that survives this is a property of the form, not of one set of
    hand-picked numbers."""
    import random
    rng = random.Random(seed)
    tally: dict[str, int] = {}
    for _ in range(draws):
        q = replace(
            base,
            tau_days=rng.uniform(120, 320),
            sigma_dist=base.sigma_dist * rng.uniform(0.7, 1.4),
            w_straight=rng.uniform(0.15, 0.60),
            w_offsurface=rng.uniform(0.5, 0.9),
            w_foreign=rng.uniform(0.55, 0.95),
            spread=base.spread * rng.uniform(0.8, 1.35),
            prior_starts=rng.uniform(10, 22),
            k_class=rng.uniform(1.2, 3.2),
            k_conv=rng.uniform(0.8, 3.0),
            k_dist=rng.uniform(0.7, 2.6),
            k_course=rng.uniform(0.6, 2.4),
            k_surface=rng.uniform(0.6, 2.4),
            k_going=rng.uniform(0.4, 1.8),
            k_sectional=rng.uniform(0.2, 0.8),
            k_barrier=rng.uniform(0.2, 1.6),
            lengths_per_kg=rng.uniform(0.4, 1.1),
            k_dq=rng.uniform(1.5, 5.5),
            handicap_credit=rng.uniform(0.2, 0.9),
            market_weight=rng.uniform(0.20, 0.60),
            devig_power=rng.uniform(1.0, 1.12),
            k_class_adj=base.k_class_adj * rng.uniform(0.0, 1.8),
            k_field=base.k_field * rng.uniform(0.5, 1.6),
            impute_field_size=max(4, base.impute_field_size + rng.choice([-1, 0, 1])),
        )
        rated, _ = rate(race, q)
        if rated:
            tally[rated[0].name] = tally.get(rated[0].name, 0) + 1
    return {k: v / draws for k, v in sorted(tally.items(), key=lambda t: -t[1])}
