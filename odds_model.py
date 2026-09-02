"""Turn two sets of market prices into probabilities, and size a bet.

Where the numbers come from
---------------------------
Both price sources carry a margin. The fixed-odds book on the two graded
races ran 122-127%, the tote book 118%. Summing 1/price and dividing through
removes that margin and leaves probabilities that add to 1.

The model is then deliberately almost nothing: average the two de-vigged
pools. That is not modesty for its own sake -- it is what the grading showed.

    Two market-only races graded          winner rank (lower better)
      fixed odds alone                    2, 7   mean 4.5
      tote alone                          1, 7   mean 4.0
      blend 50/50                         1, 6   mean 3.5
      random guess                               mean 6.8

Signals that were tried and dropped
-----------------------------------
* Steam money. Ipswich: the biggest mover (Cantarito, 4.20 -> 3.00) won.
  Murray Bridge: the biggest mover by far (Magic Island, 14.00 -> 7.50, +87%)
  ran nowhere and the winner had DRIFTED 15%. One for two, and adding a steam
  bonus never changed a single ranking. Shown as context, not scored.
* Drift as an elimination. Refuted twice. Sweet Pretender drifted 35% and ran
  second; Play Bouzouki and Win to Retire both drifted 15% and ran first and
  second.
* Tote priority. Won race one decisively, lost race two. Now equal-weighted.

Where an edge could actually exist
----------------------------------
Not in out-predicting the market -- the probabilities ARE the market. It is
in the disagreement: when the two pools differ, the fair estimate is near the
average while one pool is offering a longer price than that average. Backing
the longer of the two prices is then +EV against your own estimate. Every
positive expected value this app reports comes from that gap and nowhere else.
"""
from __future__ import annotations

from dataclasses import dataclass

PLACES_FOR_FIELD = [(0, 1), (5, 2), (8, 3)]     # runners -> paid places


def paid_places(n: int) -> int:
    out = 1
    for lo, p in PLACES_FOR_FIELD:
        if n > lo:
            out = p
    return out


def devig(prices):
    """1/price normalised to sum to 1. Returns [] if nothing usable."""
    inv = [(1.0 / p if p and p > 1.0 else 0.0) for p in prices]
    tot = sum(inv)
    if tot <= 0:
        return [], 0.0
    return [x / tot for x in inv], tot


@dataclass
class Line:
    number: int
    name: str
    p_fixed: float
    p_tote: float
    p_win: float
    p_place: float
    fixed_win: float
    tote_win: float
    fixed_place: float
    tote_place: float
    best_win: float
    best_win_src: str
    best_place: float
    best_place_src: str
    move: float
    ev_win: float
    ev_place: float
    kelly_win: float
    kelly_place: float
    fair_win: float
    fair_place: float


def analyse(race, tote_weight: float = 0.5, kelly_fraction: float = 0.25,
            max_points: float = 5.0):
    """Score one parsed race. Returns (lines, meta)."""
    rs = race.active
    if len(rs) < 2:
        return [], {"error": "Need at least two priced runners."}

    p_fx, book_fx = devig([r.fixed_win for r in rs])
    tote_ok = all(r.tote_win and r.tote_win > 1.0 for r in rs)
    if tote_ok:
        p_to, book_to = devig([r.tote_win for r in rs])
    else:
        p_to, book_to = list(p_fx), 0.0
    have_pl = all(r.fixed_place and r.fixed_place > 1.0 for r in rs)
    npl = paid_places(len(rs))
    # The place probability has to be blended across BOTH place books for the
    # same reason the win probability is. Estimating from the fixed book and
    # then comparing against the tote price is apples to oranges, and on
    # outsiders it manufactures huge fake edges -- an early build recommended
    # a 0.7% chance at $37.50 place on exactly that mistake.
    if have_pl:
        raw_fp, book_pl = devig([r.fixed_place for r in rs])
        tote_pl_ok = all(r.tote_place and r.tote_place > 1.0 for r in rs)
        if tote_pl_ok:
            raw_tp, _ = devig([r.tote_place for r in rs])
        else:
            raw_tp = list(raw_fp)
        wp = tote_weight if tote_pl_ok else 0.0
        p_pl = [min(((1 - wp) * a + wp * b) * npl, 0.98)
                for a, b in zip(raw_fp, raw_tp)]
    else:
        p_pl, book_pl = [0.0] * len(rs), 0.0

    w = tote_weight if tote_ok else 0.0
    lines = []
    for i, r in enumerate(rs):
        pw = (1 - w) * p_fx[i] + w * p_to[i]
        pp = p_pl[i]
        bw, bws = r.fixed_win, "fixed"
        if r.tote_win and r.tote_win > r.fixed_win:
            bw, bws = r.tote_win, "tote"
        bp, bps = (r.fixed_place or 0.0), "fixed"
        if r.tote_place and r.tote_place > (r.fixed_place or 0.0):
            bp, bps = r.tote_place, "tote"
        ev_w = pw * bw - 1.0
        ev_p = (pp * bp - 1.0) if (pp and bp) else -1.0
        k_w = (ev_w / (bw - 1.0)) if bw > 1.0 else 0.0
        k_p = (ev_p / (bp - 1.0)) if bp > 1.0 else 0.0
        move = (r.opening / r.fixed_win - 1.0) if r.opening else 0.0
        lines.append(Line(
            number=r.number, name=r.name, p_fixed=p_fx[i], p_tote=p_to[i],
            p_win=pw, p_place=pp, fixed_win=r.fixed_win,
            tote_win=r.tote_win or 0.0, fixed_place=r.fixed_place or 0.0,
            tote_place=r.tote_place or 0.0, best_win=bw, best_win_src=bws,
            best_place=bp, best_place_src=bps, move=move,
            ev_win=ev_w, ev_place=ev_p,
            kelly_win=max(k_w, 0.0) * kelly_fraction,
            kelly_place=max(k_p, 0.0) * kelly_fraction,
            fair_win=(1.0 / pw if pw > 0 else 0.0),
            fair_place=(1.0 / pp if pp > 0 else 0.0)))

    lines.sort(key=lambda x: x.p_win, reverse=True)
    meta = {
        "runners": len(rs), "places_paid": npl,
        "book_fixed": book_fx, "book_tote": book_to,
        "book_place": (book_pl / npl if npl else 0.0),
        "has_tote": tote_ok, "has_place": have_pl,
        "tote_weight": w, "kelly_fraction": kelly_fraction,
        "max_points": max_points,
    }
    meta["confidence"], meta["confidence_parts"] = confidence(lines, meta)
    meta["recommendation"] = recommend(lines, meta)
    return lines, meta


def confidence(lines, meta):
    """0-100. Describes how well-defined the market is, NOT hit rate.

    Four things make a market read trustworthy: the two pools agreeing, a
    tight book, a small field, and a clear top rating. None of them promise
    a winner, and the app says so.
    """
    top = lines[0]
    gap = abs(top.p_fixed - top.p_tote)
    agree = max(0.0, 1.0 - gap / 0.10) if meta["has_tote"] else 0.35
    tight = max(0.0, min(1.0, (1.35 - meta["book_fixed"]) / 0.20))
    size = max(0.0, min(1.0, (16 - meta["runners"]) / 10.0))
    sep = 0.0
    if len(lines) > 1 and top.p_win > 0:
        sep = max(0.0, min(1.0, (top.p_win - lines[1].p_win) / 0.12))
    parts = {"Pools agree": agree, "Tight book": tight,
             "Field size": size, "Clear top pick": sep}
    score = 100.0 * (0.40 * agree + 0.22 * tight + 0.18 * size + 0.20 * sep)
    return round(score, 1), parts


def band(c):
    if c >= 70:
        return "High", "#14804a"
    if c >= 50:
        return "Moderate", "#b7791f"
    if c >= 32:
        return "Low", "#c05621"
    return "Very low", "#9b2c2c"


def recommend(lines, meta):
    """Pick win, place or no bet, and stake it at fractional Kelly.

    A point is 1% of the betting bank. Stakes are capped because these
    probabilities come from the market itself, so a large computed edge is
    far more likely to be a stale price than a real one.
    """
    cap = meta["max_points"]
    conf = meta.get("confidence", 0.0)
    best_w = max(lines, key=lambda x: x.ev_win)
    best_p = max(lines, key=lambda x: x.ev_place) if meta["has_place"] \
        else None
    top = lines[0]

    # Expected value on a very small probability is dominated by the error in
    # the probability, not by any real edge, and pre-race tote dividends on
    # outsiders swing wildly. So thin runners cannot be recommended at all.
    MIN_WIN, MIN_PLACE = 0.06, 0.15
    cands = []
    if best_w.ev_win > 0.005 and best_w.p_win >= MIN_WIN:
        cands.append(("Win", best_w, best_w.ev_win, best_w.kelly_win,
                      best_w.best_win, best_w.best_win_src))
    else:
        alt = [x for x in lines if x.p_win >= MIN_WIN and x.ev_win > 0.005]
        if alt:
            a = max(alt, key=lambda x: x.ev_win)
            cands.append(("Win", a, a.ev_win, a.kelly_win, a.best_win,
                          a.best_win_src))
    if meta["has_place"]:
        alt = [x for x in lines if x.p_place >= MIN_PLACE
               and x.ev_place > 0.005]
        if alt:
            a = max(alt, key=lambda x: x.ev_place)
            cands.append(("Place", a, a.ev_place, a.kelly_place,
                          a.best_place, a.best_place_src))
    if not cands:
        return {
            "action": "No bet", "line": top, "market": "-", "points": 0.0,
            "ev": max(best_w.ev_win, best_p.ev_place if best_p else -1.0),
            "price": top.best_win, "source": top.best_win_src,
            "why": ("Neither pool is offering more than the blended estimate "
                    "thinks the runner is worth, so there is nothing to back. "
                    "The top rating is still shown above. Runners under "
                    "6% (win) or 15% (place) are never recommended -- the "
                    "error in the estimate swamps any apparent edge."),
        }

    cands.sort(key=lambda c: c[2], reverse=True)
    market, line, ev, kelly, price, src = cands[0]
    pts = min(kelly * 100.0, cap)
    if conf < 32:
        pts *= 0.5
    return {
        "action": "Back", "line": line, "market": market,
        "points": round(max(pts, 0.25), 2), "ev": ev, "price": price,
        "source": src,
        "why": (f"The {src} pool is offering ${price:.2f} while the blend of "
                f"both pools rates it a "
                f"${(line.fair_win if market == 'Win' else line.fair_place):.2f} "
                f"chance. The edge is the gap between the two pools, not a "
                f"view about the horse."),
    }


def insights(lines, meta, race):
    """Plain-language observations, each one checkable against the table."""
    out = []
    top, second = lines[0], (lines[1] if len(lines) > 1 else None)
    out.append(("Top rated",
                f"#{top.number} {top.name} at {top.p_win*100:.1f}%, a fair "
                f"price of ${top.fair_win:.2f} against ${top.best_win:.2f} "
                f"available."))
    if second and (top.p_win - second.p_win) < 0.03:
        out.append(("Too close to split",
                    f"#{top.number} {top.name} and #{second.number} "
                    f"{second.name} are within "
                    f"{(top.p_win-second.p_win)*100:.1f} points. The market "
                    f"cannot separate them and neither can this."))
    if meta["has_tote"]:
        d = max(lines, key=lambda x: abs(x.p_fixed - x.p_tote))
        if abs(d.p_fixed - d.p_tote) > 0.03:
            richer = "tote" if d.p_tote < d.p_fixed else "bookmaker"
            out.append(("Pools disagree",
                        f"#{d.number} {d.name}: bookmakers rate it "
                        f"{d.p_fixed*100:.1f}%, the tote {d.p_tote*100:.1f}%. "
                        f"The {richer} is the one offering the longer price, "
                        f"and that gap is where any value in this race sits."))
    movers = [x for x in lines if x.move > 0.15]
    if movers:
        m = max(movers, key=lambda x: x.move)
        out.append(("Money moved",
                    f"#{m.number} {m.name} shortened {m.move*100:.0f}% from "
                    f"its opening price. Context only -- steam won one of the "
                    f"two graded races and failed the other."))
    drift = [x for x in lines[:6] if x.move < -0.20]
    if drift:
        d = min(drift, key=lambda x: x.move)
        out.append(("A drifter is still live",
                    f"#{d.number} {d.name} eased {abs(d.move)*100:.0f}%. "
                    f"Drift is NOT an elimination -- in both graded races a "
                    f"drifting runner hit the board."))
    out.append(("Market margin",
                f"Fixed book {meta['book_fixed']*100:.1f}%"
                + (f", tote {meta['book_tote']*100:.1f}%"
                   if meta["has_tote"] else "")
                + f". Every probability above has that margin removed."))
    for n in race.notes:
        out.append(("Header note", n))
    return out
