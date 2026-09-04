"""Lengths-based rating of a thoroughbred field from an R&S Enhanced Form page.

Every component is expressed in **lengths at the finish of today's race**, so
the pieces add up transparently and the sum drives both the win probabilities
and the finishing margins the animation draws. The market is *not* one of the
fundamental components: it is blended in afterwards with a small, explicit,
user-controlled weight (default 5 %, hard cap 10 %) so odds can never dominate.

Rules of thumb behind the constants (Australian handicapping conventions):
    1 length              ~ 2.4 m ~ 0.15 s at 1400 m
    1.5 kg                ~ 1 length at 1400 m (we damp this to 0.4 L/kg because
                           handicappers already set weights to equalise)
    1 rating point (OHR)  ~ 0.5 kg
    class scale           MDN 46 < CL1 50 < CL2 54 < BM55 55 < CL3 58 < BM66 66 < OPEN 72
"""
from __future__ import annotations

import math
import re
from typing import Any

import numpy as np

LENGTH_M = 2.4
SEC_PER_LENGTH = 0.15
MARKET_WEIGHT_CAP = 0.10          # the market can never be more than 10 % of the rating
SIGMA_LENGTHS = 3.0               # race-day noise on a runner's finishing margin (generous: country fields are open)
MAX_RATING_SPREAD = 2.0           # std-dev cap on the fundamental rating (over-confidence guard)

GOING_SPEED_ADJ = {"H": 1.035, "S": 1.015, "G": 1.0, "F": 0.995, "N": 1.0, "": 1.0}


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def par_speed(distance_m: float) -> float:
    """Typical winning speed (m/s) on a Good track for a country BM55 field."""
    return 17.9 - 0.00075 * distance_m


def class_points(cls: str, prize: float | None = None) -> float:
    """Map an R&S class string (BM58, CL1, MDN, 3U CL3, 34 OPEN, TRPY) to
    benchmark rating points, nudged by prizemoney (metropolitan money = tougher)."""
    c = (cls or "").upper()
    m = re.search(r"BM\s*(\d+)", c)
    mcl = re.search(r"\bCL\s*(\d)", c)
    if m:
        base = float(m.group(1))
    elif "MDN" in c or "MAIDEN" in c:
        base = 46.0
    elif mcl:
        base = 46.0 + 4.0 * int(mcl.group(1))
    elif "G1" in c:
        base = 100.0
    elif "G2" in c:
        base = 95.0
    elif "G3" in c:
        base = 90.0
    elif "LR" in c or "LISTED" in c:
        base = 85.0
    elif "OPEN" in c:
        base = 72.0
    elif "TRPY" in c or "PIC" in c:
        base = 48.0
    elif "HCP" in c:
        base = 62.0
    else:
        base = 55.0
    if prize and prize > 0:
        base += float(np.clip(4.0 * math.log2(prize / 22000.0), -3.0, 8.0))
    return base


def _going_letter(going: str) -> str:
    g = (going or "").upper()
    return {"HEAVY": "H", "SOFT": "S", "GOOD": "G", "FIRM": "F",
            "SYNTHETIC": "N", "SLOW": "S", "FAST": "G", "WET": "S"}.get(g, g[:1])


def _shrunk_score(w: float, p: float, s: float, prior: float, k: float = 4.0) -> float:
    """(wins + half the places) per start, shrunk toward `prior` with k pseudo-starts."""
    return (w + 0.5 * p + k * prior) / (s + k)


def _zscores(x: np.ndarray) -> np.ndarray:
    sd = float(np.std(x))
    return (x - x.mean()) / sd if sd > 1e-9 else np.zeros_like(x)


# --------------------------------------------------------------------------
# per-runner feature extraction
# --------------------------------------------------------------------------
def _run_perf_points(run: dict[str, Any]) -> float | None:
    """Performance figure (rating points) of one past run."""
    margin = run.get("margin")
    if margin is None:
        return None
    pts = class_points(run.get("race_class", ""), run.get("prize"))
    pts -= 3.0 * float(margin)                       # 1 length ~ 3 points
    cd = run.get("cd") or run.get("weight")
    if cd:
        pts += 2.0 * (float(cd) - 57.0)              # carried more than standard
    return pts


def _run_speed_lengths(run: dict[str, Any], today_d: float) -> float | None:
    """Speed figure of one run converted to lengths at today's distance."""
    t = run.get("race_time_s")
    d = run.get("distance")
    if not t or not d or t <= 0 or d <= 0:
        return None
    margin = float(run.get("margin") or 0.0)
    speed = d / (t + margin * SEC_PER_LENGTH)
    going = (run.get("going") or "").upper()[:1]
    adj = GOING_SPEED_ADJ.get(going, 1.0)
    if going == "S" and (run.get("going_rating") or 6) <= 5:
        adj = 1.008
    par = par_speed(d)
    pct = (speed * adj - par) / par
    if abs(pct) > 0.08:                              # implausible time / typo guard
        return None
    return 0.5 * pct * today_d / LENGTH_M            # damped: track-to-track variance is large


def features(r: dict[str, Any], header: dict[str, Any], sm: dict[str, float] | None) -> dict[str, Any]:
    d_today = float(header.get("distance_m") or 1400)
    going = _going_letter(header.get("going", ""))
    runs = r.get("recent_runs", [])[:5]
    f: dict[str, Any] = {}

    # --- class & weight -----------------------------------------------------
    f["ohr"] = float(r.get("ohr") or 0.0)
    f["carried"] = float(r.get("wt") or 0.0) - float(r.get("claim") or 0.0)

    # --- recent form: recency-weighted performance figure -------------------
    perf, wts = [], []
    for idx, run in enumerate(runs):
        p = _run_perf_points(run)
        if p is None:
            continue
        w = math.exp(-float(run.get("days_ago") or 60) / 120.0) * (1.3 if idx == 0 else 1.0)
        perf.append(p)
        wts.append(w)
    f["form_pts"] = float(np.average(perf, weights=wts)) if perf else None
    f["n_runs"] = len(runs)

    # --- speed figures -----------------------------------------------------
    sp = [x for x in (_run_speed_lengths(run, d_today) for run in runs[:4]) if x is not None]
    f["speed_L"] = float(np.median(sorted(sp, reverse=True)[:3])) if sp else None

    # --- fitness ------------------------------------------------------------
    dslr = int(r.get("dslr") or 30)
    runup = r.get("runup")
    fu_w, fu_p, fu_s = r.get("FU_wins", 0), r.get("FU_places", 0), r.get("FU_starts", 0)
    fit = 0.0
    if dslr <= 6:
        fit -= 0.6
    elif dslr <= 35:
        fit += 0.0
    elif dslr <= 60:
        fit -= 0.3
    elif dslr <= 120:
        fit -= 0.7
    else:
        fit -= 1.0
    if dslr > 60 and fu_s >= 3 and (fu_w + 0.5 * fu_p) / fu_s >= 0.3:
        fit += 0.5                                   # proven fresh
    if runup in (2, 3):
        fit += 0.25                                  # horses peak 2nd/3rd-up
    if runup and runup >= 10:
        fit -= 0.2                                   # long preparation
    age = int(r.get("age") or 5)
    if age >= 8:
        fit -= 0.3
    f["fitness_L"] = fit
    f["dslr"] = dslr

    # --- barrier ------------------------------------------------------------
    bp = int((sm or {}).get("bp") or r.get("bp") or 0)
    f["bp"] = bp

    # --- jockey / trainer ---------------------------------------------------
    f["jrat"] = float((sm or {}).get("jr") or r.get("jrat") or 0.0)
    f["trat"] = float(r.get("trat") or 0.0)
    f["jky_win"] = float(r.get("jky_win") or 0.0)
    f["trn_win"] = float(r.get("trn_win") or 0.0)
    f["has_jockey"] = bool(r.get("jockey"))
    jt_n, jh_n = int(r.get("jt_n") or 0), int(r.get("jh_n") or 0)
    combo = 0.0
    if jt_n >= 5:
        combo += 1.5 * (float(r.get("jt_win") or 0.0) - 0.10)
    if jh_n >= 2:
        combo += 1.0 * (float(r.get("jh_win") or 0.0) - 0.15)
    f["combo_L"] = float(np.clip(combo, -0.5, 0.6))

    # --- going / surface suitability ---------------------------------------
    cw, cp, cs = r.get("Car_wins", 0), r.get("Car_places", 0), r.get("Car_starts", 0)
    car_score = (cw + 0.5 * cp) / cs if cs else 0.12
    key = {"H": "Heavy", "S": "Soft", "G": "Good", "F": "Firm"}.get(going)
    suit = 0.0
    if key:
        gw, gp, gs = r.get(f"{key}_wins", 0), r.get(f"{key}_places", 0), r.get(f"{key}_starts", 0)
        if going in ("S", "H"):                      # wet form transfers between Soft and Heavy
            ow = "Heavy" if key == "Soft" else "Soft"
            gw += 0.5 * r.get(f"{ow}_wins", 0)
            gp += 0.5 * r.get(f"{ow}_places", 0)
            gs += 0.5 * r.get(f"{ow}_starts", 0)
        if gs > 0:
            suit += 3.0 * (_shrunk_score(gw, gp, gs, car_score) - car_score)
        else:
            suit -= 0.15                             # never raced on this going
    surface = (header.get("surface") or "TURF").upper()
    turf_s, aw_s = r.get("Turf_starts", 0), r.get("AW_starts", 0)
    if surface.startswith("TURF") and turf_s == 0 and aw_s > 0:
        suit -= 0.4                                  # all the form is on synthetic
    if not surface.startswith("TURF") and aw_s == 0 and turf_s > 0:
        suit -= 0.3
    f["going_L"] = float(np.clip(suit, -1.2, 1.2))

    # --- distance suitability ----------------------------------------------
    dist = 0.0
    dmin, dmax = r.get("dist_min"), r.get("dist_max")
    if dmax and d_today > dmax:
        dist -= 0.5 * (d_today - dmax) / 100.0
    if dmin and d_today < dmin:
        dist -= 0.3 * (dmin - d_today) / 100.0
    dw, dp, ds = r.get("Dist_wins", 0), r.get("Dist_places", 0), r.get("Dist_starts", 0)
    if ds >= 2:
        dist += 2.0 * (_shrunk_score(dw, dp, ds, car_score) - car_score)
    if any(abs(wd - d_today) <= 100 for wd in r.get("win_dists", [])):
        dist += 0.2
    if r.get("CrsDist_wins", 0) > 0:
        dist += 0.2
    f["distance_L"] = float(np.clip(dist, -2.0, 1.0))

    # --- pace profile -------------------------------------------------------
    settle = [run["settle_pos"] / run["field_size"] for run in runs
              if run.get("settle_pos") and run.get("field_size")]
    f["settle_frac"] = float(np.mean(settle)) if settle else 0.5
    f["aes"] = float((sm or {}).get("aes") or 0.0) or None
    f["afs"] = float((sm or {}).get("afs") or 0.0) or None
    f["slow_begin"] = sum(1 for run in runs[:3] if run.get("slow_begin")) >= 2
    f["odds"] = float(r.get("bf_odds") or 999.0)
    return f


# --------------------------------------------------------------------------
# field rating
# --------------------------------------------------------------------------
def rate_field(header: dict[str, Any], runners: list[dict[str, Any]],
               speed_map: dict[int, dict[str, float]] | None = None,
               market_weight: float = 0.05, sigma: float = SIGMA_LENGTHS,
               n_sims: int = 20000, seed: int = 0) -> dict[str, Any]:
    """Rate the active runners. Returns {"runners": [...], "meta": {...}}."""
    active = [r for r in runners if not r.get("scratched")]
    if not active:
        return {"runners": [], "meta": {}}
    speed_map = speed_map or {}
    d_today = float(header.get("distance_m") or 1400)
    n = len(active)
    market_weight = float(np.clip(market_weight, 0.0, MARKET_WEIGHT_CAP))

    F = [features(r, header, speed_map.get(r["tab"])) for r in active]

    def col(key, default=0.0):
        return np.array([f[key] if f.get(key) is not None else default for f in F], dtype=float)

    # class (OHR) relative to field
    ohr = col("ohr")
    have_ohr = ohr > 0
    ohr_mean = ohr[have_ohr].mean() if have_ohr.any() else 0.0
    class_L = np.where(have_ohr, 0.3 * (ohr - ohr_mean), -0.3)

    # weight carried relative to field
    carried = col("carried")
    weight_L = -0.4 * (carried - carried.mean()) * (d_today / 1400.0)

    # recent form (points -> lengths, relative to field)
    form_pts = np.array([f["form_pts"] if f["form_pts"] is not None else np.nan for f in F])
    fp_mean = np.nanmean(form_pts) if np.isfinite(form_pts).any() else 0.0
    form_L = np.where(np.isfinite(form_pts), 0.75 * (form_pts - fp_mean) / 3.0, -0.5)
    form_L = np.clip(form_L, -5.0, 5.0)

    # speed figures relative to field
    speed = np.array([f["speed_L"] if f["speed_L"] is not None else np.nan for f in F])
    sp_mean = np.nanmean(speed) if np.isfinite(speed).any() else 0.0
    speed_L = np.where(np.isfinite(speed), 0.6 * (speed - sp_mean), -0.3)
    speed_L = np.clip(speed_L, -3.0, 3.0)

    # early-speed score for positioning (AES + where it usually settles)
    aes = np.array([f["aes"] if f["aes"] else np.nan for f in F])
    settle = col("settle_frac", 0.5)
    es = np.zeros(n)
    if np.isfinite(aes).any():
        aes_f = np.where(np.isfinite(aes), aes, np.nanmean(aes))
        es += 0.6 * _zscores(aes_f)
    es += (0.4 if np.isfinite(aes).any() else 1.0) * _zscores(-settle)
    es -= np.array([0.6 if f["slow_begin"] else 0.0 for f in F])

    # late-speed score (AFS)
    afs = np.array([f["afs"] if f["afs"] else np.nan for f in F])
    late = _zscores(np.where(np.isfinite(afs), afs, np.nanmean(afs))) if np.isfinite(afs).any() else np.zeros(n)
    late_L = 0.35 * late

    # barrier
    bp = col("bp")
    pen = np.where(bp <= 4, 0.0, np.where(bp <= 10, -0.12 * (bp - 4), -0.72 - 0.2 * (bp - 10)))
    pen = np.clip(pen, -2.0, 0.0)
    pen *= 1.2 if d_today <= 1200 else (0.7 if d_today >= 1600 else 1.0)
    pen = np.where(es >= 0.8, pen * 0.6, np.where(es <= -0.8, pen * 0.8, pen))
    barrier_L = pen

    # jockey / trainer
    jrat, trat = col("jrat"), col("trat")
    jw, tw = col("jky_win"), col("trn_win")
    jockey_L = 0.25 * (jrat - jrat.mean()) + 3.0 * (jw - jw.mean())
    jockey_L -= np.array([0.0 if f["has_jockey"] else 0.5 for f in F])
    trainer_L = 0.15 * (trat - trat.mean()) + 2.0 * (tw - tw.mean())
    combo_L = col("combo_L")

    fitness_L = col("fitness_L")
    going_L = col("going_L")
    distance_L = col("distance_L")

    comps = {
        "class": class_L, "weight": weight_L, "form": form_L, "speed": speed_L,
        "fitness": fitness_L, "barrier": barrier_L, "jockey": jockey_L,
        "trainer": trainer_L, "combo": combo_L, "going": going_L,
        "distance": distance_L, "late_speed": late_L,
    }
    model = sum(comps.values())
    model = model - model.mean()
    sd = float(np.std(model))
    if sd > MAX_RATING_SPREAD:                       # over-confidence guard
        model *= MAX_RATING_SPREAD / sd

    # market, blended with a capped weight
    odds = col("odds", 999.0)
    p_mkt = np.where(odds < 999, 1.0 / np.maximum(odds, 1.01), np.nan)
    if np.isfinite(p_mkt).any():
        p_mkt = np.where(np.isfinite(p_mkt), p_mkt, np.nanmin(p_mkt) / 2)
        p_mkt = p_mkt / p_mkt.sum()
        market_L = 2.0 * (np.log(p_mkt) - np.log(p_mkt).mean())
    else:
        p_mkt = np.full(n, 1.0 / n)
        market_L = np.zeros(n)
        market_weight = 0.0
    rating = (1.0 - market_weight) * model + market_weight * market_L

    # Monte Carlo win / place
    rng = np.random.default_rng(seed)
    noise = rng.normal(0.0, sigma, size=(n_sims, n))
    finish = -rating[None, :] + noise
    order = np.argsort(finish, axis=1)
    win = np.bincount(order[:, 0], minlength=n) / n_sims
    k = 3 if n >= 8 else 2
    place = np.zeros(n)
    for j in range(k):
        place += np.bincount(order[:, j], minlength=n)
    place /= n_sims

    # expected finishing margins (lengths behind the top-rated runner)
    margins = rating.max() - rating
    if margins.max() > 14:
        margins *= 14 / margins.max()

    # predicted winning time
    going = _going_letter(header.get("going", ""))
    adj = GOING_SPEED_ADJ.get(going, 1.0)
    if going == "S" and (header.get("going_rating") or 6) <= 5:
        adj = 1.008
    speed_edge = 0.0
    if np.isfinite(speed).any():
        speed_edge = float(np.nanmax(speed)) * LENGTH_M / d_today / 0.5   # undo damping
    win_speed = par_speed(d_today) * (1 + float(np.clip(speed_edge, -0.03, 0.03))) / adj
    pred_time = d_today / win_speed

    out = []
    for i, (r, f) in enumerate(zip(active, F)):
        out.append({
            "tab": r["tab"], "horse": r["horse"], "jockey": r.get("jockey", ""),
            "trainer": r.get("trainer", ""), "bp": int(bp[i]), "carried": float(carried[i]),
            "wt": float(r.get("wt") or 0.0), "claim": float(r.get("claim") or 0.0),
            "odds": float(odds[i]) if odds[i] < 999 else None,
            "ohr": int(ohr[i]) if ohr[i] > 0 else None, "form": r.get("form", ""),
            "rating": float(rating[i]), "model_rating": float(model[i]),
            "market_L": float(market_L[i]), "market_prob": float(p_mkt[i]),
            "win_prob": float(win[i]), "place_prob": float(place[i]),
            "exp_margin": float(margins[i]), "early_speed": float(es[i]),
            "late_speed": float(late[i]), "slow_begin": bool(f["slow_begin"]),
            "aes": f["aes"], "afs": f["afs"], "dslr": f["dslr"],
            "components": {k: float(v[i]) for k, v in comps.items()},
        })
    out.sort(key=lambda x: -x["rating"])
    for rank, x in enumerate(out, 1):
        x["rank"] = rank
        x["model_odds"] = round(1.0 / x["win_prob"], 1) if x["win_prob"] > 0 else None
    meta = {
        "n": n, "market_weight": market_weight, "sigma": sigma,
        "pred_time_s": float(pred_time), "distance_m": d_today,
        "has_speed_map": bool(speed_map), "components": list(comps.keys()),
    }
    return {"runners": out, "meta": meta}


def fmt_time(seconds: float) -> str:
    m = int(seconds // 60)
    s = seconds - 60 * m
    return f"{m}:{s:05.2f}"
