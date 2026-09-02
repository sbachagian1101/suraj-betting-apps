"""Recalibrate R&S worksheet percentages and price a bet off them.

The finding this app exists for
-------------------------------
Measured on 61 races and 551 runners from 1 September 2026:

    R&S top-rated wins            31.1%   (random 12.4%)
    R&S top three hold the winner 60.7%   (random 37.3%)

So the ORDER is good and is left completely alone. The percentages are not.
Published PER scores a log-loss of 2.6500, which is worse than calling every
runner equally likely (2.1445), because the top pick claims 43.0% and wins
31.1% -- it is 1.38x overconfident.

Two constants per region fix it, both chosen by moment matching and graded
leave-one-meeting-out:

    win temperature    so the top pick claims the strike rate it achieves
    place temperature  extra flattening before Harville, which otherwise
                       reads far too high off sharpened probabilities

Out of sample that reaches log-loss 2.0293 -- better than published PER and
better than uniform -- and in-sample calibration lands within a few points
everywhere:

    region  races   win claim / actual      top-3 claim / actual
    AUS        16      47.5% / 50.0%           81.2% / 81.2%
    FR          8      29.0% / 37.5%           52.3% / 50.0%
    IRE         8      27.0% / 25.0%           41.0% / 37.5%
    UK         29      20.7% / 20.7%           55.0% / 55.2%
    ALL        61      29.6% / 31.1%           59.7% / 59.0%

The overconfidence is nothing like uniform across jurisdictions -- Australia
is if anything UNDERconfident at 0.67x while the UK is out by 2.42x -- so one
global constant would get those two wrong in opposite directions. FR and IRE
rest on a single meeting each and are reported as provisional.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

CAL_PATH = Path("calibration.json")


def load_calibration(path: Path | None = None) -> dict:
    p = path or CAL_PATH
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    return {"global_t": 4.0, "regions": {}, "backing": {},
            "record": {}, "region_detail": {}, "fitted_on": {}}


def temperature(cal: dict, region: str) -> tuple[float, bool]:
    """(win temperature, is_validated) for a region."""
    t = cal.get("regions", {}).get(region, cal.get("global_t", 2.0))
    val = cal.get("backing", {}).get(region, 0) >= 2
    return float(t), bool(val)


def place_temperature(cal: dict, region: str) -> float:
    """Extra flattening applied before Harville, as a multiple of the win
    temperature. Harville reads high off sharpened probabilities."""
    return float(cal.get("place_regions", {}).get(
        region, cal.get("global_place_t", 1.0)))


def calibrate(pers, t: float):
    """Flatten a list of published percentages. The ORDER never changes."""
    v = [max(float(p), 0.01) ** (1.0 / t) for p in pers]
    s = sum(v)
    return [x / s for x in v] if s > 0 else [1.0 / len(v)] * len(v)


def top_n_probs(p, k: int = 3):
    """Harville: probability of finishing in the first k.

    Sequential draws without replacement from the win probabilities. It is
    known to flatter short-priced runners slightly; the bundled data says the
    top pick makes the first three 60.7% of the time, and a test pins the
    model against that.
    """
    n = len(p)
    k = min(k, n)
    out = [0.0] * n
    for i in range(n):
        out[i] = p[i]
    if k == 1:
        return out
    for i in range(n):
        # P(2nd) = sum over j!=i of p_j * p_i/(1-p_j)
        for j in range(n):
            if j == i:
                continue
            d = 1.0 - p[j]
            if d <= 1e-9:
                continue
            out[i] += p[j] * p[i] / d
            if k >= 3:
                for m in range(n):
                    if m in (i, j):
                        continue
                    d2 = 1.0 - p[j] - p[m]
                    if d2 <= 1e-9:
                        continue
                    out[i] += p[j] * (p[m] / (1.0 - p[j])) * p[i] / d2
    return [min(max(x, 0.0), 0.999) for x in out]


@dataclass
class Row:
    tab: int
    horse: str
    published: float          # PER as printed, %
    prob: float               # calibrated win probability
    place: float              # calibrated top-3 probability
    fair_win: float
    fair_place: float
    div: float | None
    fr: float
    em: float
    first_up: bool
    dls: float
    shift: float              # calibrated minus published, in points


def analyse_race(live, region: str, cal: dict, places: int = 3):
    t, validated = temperature(cal, region)
    pers = [r.per for r in live]
    p = calibrate(pers, t)
    k = min(places, max(1, len(live) - 1))
    pl = top_n_probs(calibrate(pers, t * place_temperature(cal, region)), k)
    rows = []
    for r, pw, pp in zip(live, p, pl):
        rows.append(Row(
            tab=r.tab, horse=r.horse, published=r.per, prob=pw, place=pp,
            fair_win=(1.0 / pw if pw > 0 else 0.0),
            fair_place=(1.0 / pp if pp > 0 else 0.0),
            div=r.div, fr=r.fr, em=r.em, first_up=r.first_up, dls=r.dls,
            shift=pw * 100.0 - r.per))
    rows.sort(key=lambda x: x.prob, reverse=True)
    meta = {"region": region, "temperature": t, "validated": validated,
            "runners": len(live), "places": k,
            "spread": (rows[0].prob - rows[1].prob) if len(rows) > 1 else 0.0}
    meta["confidence"], meta["confidence_parts"] = confidence(rows, meta, cal)
    return rows, meta


def confidence(rows, meta, cal):
    """0-100: how much the calibration for THIS race can be leaned on.

    It is not a hit rate. It reflects how much evidence stands behind the
    region's constant, how clearly one runner leads, and the field size.
    """
    det = cal.get("region_detail", {}).get(meta["region"], {})
    races = det.get("races", 0)
    evidence = min(1.0, races / 40.0) * (1.0 if meta["validated"] else 0.45)
    lead = min(1.0, meta["spread"] / 0.10)
    top = rows[0].prob
    strength = min(1.0, top / 0.35)
    size = max(0.0, min(1.0, (18 - meta["runners"]) / 12.0))
    parts = {"Region evidence": evidence, "Clear leader": lead,
             "Top pick strength": strength, "Field size": size}
    score = 100.0 * (0.38 * evidence + 0.24 * lead + 0.22 * strength
                     + 0.16 * size)
    return round(score, 1), parts


def band(c: float):
    if c >= 68:
        return "High", "#14804a"
    if c >= 48:
        return "Moderate", "#b7791f"
    if c >= 30:
        return "Low", "#c05621"
    return "Very low", "#9b2c2c"


def recommend(rows, meta, price=None, kelly_fraction=0.25, max_points=5.0):
    """Win, place, or no bet -- and the price it needs to be worth taking.

    Without a market price no expected value exists, so the headline number
    is the BREAK-EVEN price. Enter what you can actually get and it becomes a
    stake.
    """
    top = rows[0]
    conf = meta["confidence"]
    market = "Win"
    line = top
    p = top.prob
    fair = top.fair_win
    # A modest win chance with a strong place chance is a place bet, not a
    # win bet. The bundled data has the top pick winning 31% and making the
    # first three 61%, so this is the common case rather than the exception.
    if top.prob < 0.28 and top.place >= 0.55 and meta["places"] >= 3:
        market, p, fair = "Place", top.place, top.fair_place

    ev = pts = None
    if price and price > 1.0:
        ev = p * price - 1.0
        if ev > 0:
            k = ev / (price - 1.0)
            pts = min(k * kelly_fraction * 100.0, max_points)
            if conf < 30:
                pts *= 0.5
            pts = round(max(pts, 0.25), 2)

    if conf < 22:
        action = "No bet"
        why = ("Too little behind the calibration for this race -- either the "
               "region has almost no graded evidence, or nothing separates "
               "the top of the field.")
    elif price and price > 1.0 and (ev is None or ev <= 0):
        action = "No bet"
        why = (f"At ${price:.2f} the {market.lower()} is worth "
               f"{ev:+.1%}. You need better than ${fair:.2f} for this to be "
               f"a bet at all.")
    else:
        action = "Back"
        why = (f"Calibrated chance {p*100:.1f}%, so anything longer than "
               f"${fair:.2f} is value. R&S published "
               f"{top.published:.1f}% for this runner; the "
               f"{meta['region']} correction moves it to {top.prob*100:.1f}%.")
    return {"action": action, "market": market, "line": line, "prob": p,
            "fair": fair, "price": price, "ev": ev, "points": pts,
            "why": why, "confidence": conf}


def insights(rows, meta, cal):
    out = []
    top = rows[0]
    det = cal.get("region_detail", {}).get(meta["region"], {})
    ratio = det.get("ratio")
    if ratio:
        direction = ("overconfident" if ratio > 1.05
                     else ("underconfident" if ratio < 0.95 else "about right"))
        out.append((
            f"{meta['region']} calibration",
            f"On {det.get('races', 0)} graded races R&S were **{ratio:.2f}x "
            f"{direction}** here — they claimed "
            f"{det.get('claims', 0)*100:.1f}% for the top pick and it won "
            f"{det.get('actual', 0)*100:.1f}%. Temperature "
            f"{meta['temperature']:.2f} is applied."
            + ("" if meta["validated"] else
               " **This constant rests on a single meeting and is "
               "provisional.**")))
    out.append((
        "What changed",
        f"#{top.tab} {top.horse} published at {top.published:.1f}% and comes "
        f"out at {top.prob*100:.1f}% — a shift of {top.shift:+.1f} points. "
        f"The order is untouched; only the confidence moves."))
    if len(rows) > 1 and meta["spread"] < 0.03:
        out.append((
            "Nothing separates the top two",
            f"#{top.tab} {top.horse} and #{rows[1].tab} {rows[1].horse} are "
            f"within {meta['spread']*100:.1f} points. Treat them as a pair."))
    big = max(rows, key=lambda r: r.shift)
    if big.shift > 1.0 and big.tab != top.tab:
        out.append((
            "Most improved by the correction",
            f"#{big.tab} {big.horse} gains {big.shift:+.1f} points "
            f"({big.published:.1f}% to {big.prob*100:.1f}%). Flattening an "
            f"overconfident book always lifts the outsiders."))
    fu = [r for r in rows[:5] if r.first_up]
    if fu:
        out.append((
            "First-up runners near the top",
            ", ".join(f"#{r.tab} {r.horse}" for r in fu)
            + " are resuming. The worksheet already accounts for it, but "
              "these carry more uncertainty than the number suggests."))
    out.append((
        "What this is not",
        "The ranking is R&S's, not this app's. All that changes is how "
        "confident the percentages are, which is the part their own numbers "
        "get wrong."))
    return out
