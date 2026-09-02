"""Fit the calibration constants and prove them out of sample.

R&S rank well and are badly overconfident. A single temperature applied to
the PER column fixes the log-loss without touching the order, but the amount
of overconfidence is jurisdiction-specific: Australia is if anything UNDER
confident while the UK is out by 2.4x, so one global constant gets both wrong.

Everything is validated leave-one-meeting-out: the temperature is chosen on
seven meetings and graded on the eighth, never on the meeting it scores. A
sweep of shrinkage strengths showed that pulling a region toward the global
constant only ever hurt, so regions with two or more meetings behind them
trust their own number. Regions with a single meeting are a different case --
when that meeting is held out there is no region constant left, so the fold
silently grades the global one and the region constant is never tested at
all. Those are pulled halfway to global and reported as PROVISIONAL.

Run:  python fit_calibration.py        (writes calibration.json)
"""
from __future__ import annotations

import json
import math
import random
from pathlib import Path

from ws_parser import parse, meeting_name_from_filename, region_for

SAMPLES = Path("samples")
RESULTS = json.loads(Path("samples/results.json").read_text(encoding="utf-8"))
GRID = [0.6 + 0.05 * i for i in range(0, 289)]      # 0.60 .. 15.00
# Shrinkage is by MEETINGS, not races. A holdout sweep showed that pulling a
# region toward the global constant only ever made things worse for AUS and
# UK, which have two and four meetings behind them -- so those trust
# themselves. But a region with a single meeting was never actually validated:
# when that meeting is held out no region constant exists for it and the fold
# silently falls back to global. Those get pulled halfway as a hedge and are
# reported as provisional.


def load():
    """[(meeting, region, race_index, live_runners, first_four)]"""
    out = []
    for f in sorted(SAMPLES.glob("*.csv")):
        name = meeting_name_from_filename(f.name)
        mt = parse(f.read_text(encoding="utf-8-sig"), name)
        res = RESULTS.get(f.stem, [])
        for rc in mt.races:
            if rc.index - 1 >= len(res) or len(rc.live) < 2:
                continue
            out.append((name, mt.region, rc.index, rc.live,
                        res[rc.index - 1]))
    return out


def probs(live, t):
    v = [max(r.per, 0.01) ** (1.0 / t) for r in live]
    s = sum(v)
    return [x / s for x in v] if s > 0 else [1 / len(v)] * len(v)


def logloss(races, tfn):
    ll, n = 0.0, 0
    for _, reg, _, live, res in races:
        tabs = [r.tab for r in live]
        if res[0] not in tabs:
            continue
        ll -= math.log(max(probs(live, tfn(reg))[tabs.index(res[0])], 1e-9))
        n += 1
    return (ll / n if n else 0.0), n


def toppick(races, t):
    """(mean claimed probability, actual strike rate) for the top pick."""
    claim = hit = 0.0
    for _, _, _, live, res in races:
        p = probs(live, t)
        i = max(range(len(live)), key=lambda k: p[k])
        claim += p[i]
        hit += live[i].tab == res[0]
    n = max(len(races), 1)
    return claim / n, hit / n


def best_t(races, grid=GRID):
    """Choose t so the top pick CLAIMS what it actually WINS.

    Minimising log-loss instead is dominated by the 69% of races the top pick
    does not win, so it over-flattens the favourite: fitted that way the
    calibrated top pick landed 12.8 points below its real strike rate out of
    sample, and 14.9 points below in Australia. Moment matching scored better
    on BOTH measures leave-one-meeting-out -- log-loss 2.0293 against 2.0352,
    and calibration error -7.1 points against -12.8 -- so it wins on the
    proper scoring rule as well as on the one it targets.
    """
    def err(t):
        c, h = toppick(races, t)
        return abs(c - h)
    return min(grid, key=err)


def toppick_place(races, t, tp):
    """(mean claimed top-3 probability, actual top-3 rate) for the top pick."""
    from ws_model import top_n_probs
    claim = hit = 0.0
    for _, _, _, live, res in races:
        p = probs(live, t)
        i = max(range(len(live)), key=lambda k: p[k])
        pl = top_n_probs(probs(live, t * tp), min(3, max(1, len(live) - 1)))
        claim += pl[i]
        hit += live[i].tab in res[:3]
    n = max(len(races), 1)
    return claim / n, hit / n


def best_place_t(races, t, grid=None):
    """Harville reads too high off sharpened win probabilities.

    It is computed from a FLATTER copy of the same probabilities, with the
    extra flattening chosen so the top pick claims the top-3 rate it actually
    achieved -- the same moment matching used for the win column.
    """
    grid = grid or [0.8 + 0.05 * i for i in range(0, 89)]     # 0.80 .. 5.20

    def err(tp):
        c, h = toppick_place(races, t, tp)
        return abs(c - h)
    return min(grid, key=err)


def fit(races):
    """Global temperature, then one per region.

    Returns (global_t, {region: t}, {region: meetings_backing}).
    """
    g = best_t(races)
    gp = best_place_t(races, g)
    per, place, backing = {}, {}, {}
    for reg in sorted({r[1] for r in races}):
        sub = [r for r in races if r[1] == reg]
        meets = len({r[0] for r in sub})
        w = 1.0 if meets >= 2 else 0.5
        t = round(w * best_t(sub) + (1 - w) * g, 3)
        per[reg] = t
        place[reg] = round(w * best_place_t(sub, t) + (1 - w) * gp, 3)
        backing[reg] = meets
    return round(g, 3), per, backing, round(gp, 3), place


def temp_fn(g, per):
    return lambda reg: per.get(reg, g)


def main():
    races = load()
    random.seed(5)
    print(f"{len(races)} races, {sum(len(r[3]) for r in races)} live runners")

    meetings = sorted({r[0] for r in races})
    tot_c = tot_r = tot_u = tot_n = 0.0
    print(f"\n{'held-out meeting':18s} {'reg':5s} {'t used':>7} "
          f"{'calibrated':>11} {'published':>10} {'uniform':>9}")
    print("-" * 66)
    for hold in meetings:
        train = [r for r in races if r[0] != hold]
        test = [r for r in races if r[0] == hold]
        g, per, _, _, _ = fit(train)
        fn = temp_fn(g, per)
        c, n = logloss(test, fn)
        p, _ = logloss(test, lambda _r: 1.0)
        u = sum(math.log(len(L)) for _, _, _, L, res in test
                if res[0] in [x.tab for x in L]) / max(n, 1)
        reg = test[0][1]
        tot_c += c * n
        tot_r += p * n
        tot_u += u * n
        tot_n += n
        flag = "  <-- worse" if c > p else ""
        print(f"{hold:18s} {reg:5s} {fn(reg):7.2f} {c:11.4f} {p:10.4f} "
              f"{u:9.4f}{flag}")
    print("-" * 66)
    print(f"{'OUT OF SAMPLE':18s} {'':5s} {'':7s} {tot_c/tot_n:11.4f} "
          f"{tot_r/tot_n:10.4f} {tot_u/tot_n:9.4f}")

    g, per, backing, gp, place = fit(races)
    counts = {}
    hits = {}
    claims = {}
    for _, reg, _, live, res in races:
        counts[reg] = counts.get(reg, 0) + 1
        top = max(live, key=lambda r: r.per)
        hits[reg] = hits.get(reg, 0) + (top.tab == res[0])
        claims[reg] = claims.get(reg, 0.0) + top.per / 100.0

    print(f"\nFitted on everything -- global t = {g}")
    print(f"{'region':7s} {'meets':>6} {'races':>6} {'t':>6} "
          f"{'claims':>8} {'actual':>8} {'ratio':>7}  status")
    for reg in sorted(per):
        st = "validated" if backing[reg] >= 2 else "PROVISIONAL (1 meeting)"
        print(f"{reg:7s} {backing[reg]:6d} {counts[reg]:6d} {per[reg]:6.2f} "
              f"{claims[reg]/counts[reg]*100:7.1f}% "
              f"{hits[reg]/counts[reg]*100:7.1f}% "
              f"{claims[reg]/max(hits[reg],1):6.2f}x  {st}")

    w = sum(hits.values())
    n = sum(counts.values())
    t3 = sum(1 for _, _, _, L, res in races
             if res[0] in [x.tab for x in sorted(L, key=lambda y: -y.per)[:3]])
    rand_w = sum(1.0 / len(L) for _, _, _, L, _ in races) / n
    rand_3 = sum(min(3, len(L)) / len(L) for _, _, _, L, _ in races) / n

    out = {
        "global_t": g, "regions": per, "backing": backing,
        "global_place_t": gp, "place_regions": place,
        "fitted_on": {"races": n, "runners": sum(len(r[3]) for r in races),
                      "meetings": len(meetings), "date": "2026-09-01"},
        "record": {
            "top_pick_win": round(w / n, 4),
            "top_three": round(t3 / n, 4),
            "random_win": round(rand_w, 4),
            "random_top_three": round(rand_3, 4),
            "logloss_calibrated_oos": round(tot_c / tot_n, 4),
            "logloss_published": round(tot_r / tot_n, 4),
            "logloss_uniform": round(tot_u / tot_n, 4),
            "overall_ratio": round(sum(claims.values()) / w, 3),
            "objective": "top-pick strike rate (moment matching)",
        },
        "region_detail": {
            reg: {"races": counts[reg],
                  "claims": round(claims[reg] / counts[reg], 4),
                  "actual": round(hits[reg] / counts[reg], 4),
                  "ratio": round(claims[reg] / max(hits[reg], 1), 3),
                  "meetings": backing[reg],
                  "validated": backing[reg] >= 2}
            for reg in sorted(counts)},
    }
    Path("calibration.json").write_text(json.dumps(out, indent=2) + "\n",
                                        encoding="utf-8")
    print(f"\nTop pick wins {w}/{n} = {w/n*100:.1f}% "
          f"(random {rand_w*100:.1f}%), top three hold the winner "
          f"{t3}/{n} = {t3/n*100:.1f}% (random {rand_3*100:.1f}%)")
    print("wrote calibration.json")


if __name__ == "__main__":
    main()
