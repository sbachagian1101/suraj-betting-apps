"""Greyhound race probability model.

A transparent ensemble built for copied Racing & Sports Enhanced Form data.
It scores market strength, recent adjusted speed, early pace, current box
history, track/distance suitability, recent finishing form and trainer form,
then blends fundamentals with the de-vigged market and simulates finishing
orders with a Plackett-Luce model.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np

MARKET_ALPHA = 0.60
COMPONENT_WEIGHTS = {
    "speed": 0.29, "early": 0.23, "box": 0.14, "trackdist": 0.14,
    "form": 0.10, "trainer": 0.05, "freshness": 0.05,
}


def _safe(v: Any, default: float = 0.0) -> float:
    try:
        x = float(v)
        return x if math.isfinite(x) else default
    except Exception:
        return default


def _z(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    finite = np.isfinite(values)
    if not finite.any():
        return np.zeros_like(values)
    med = float(np.nanmedian(values[finite]))
    values = np.where(finite, values, med)
    sd = float(values.std())
    return (values - values.mean()) / (sd if sd > 1e-9 else 1.0)


def _softmax(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    x = x - np.max(x)
    e = np.exp(np.clip(x, -30, 30))
    return e / max(e.sum(), 1e-12)


def _market_probs(runners: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    def probs(key: str) -> np.ndarray:
        inv = np.array([1.0 / max(_safe(r.get(key), 999.0), 1.01) for r in runners])
        return inv / max(inv.sum(), 1e-12)
    p_tab = probs("tab_odds")
    p_bf = probs("bf_odds")
    valid_bf = np.array([1.0 if 1.01 < _safe(r.get("bf_odds"), 999.0) < 500 else 0.0 for r in runners])
    if valid_bf.mean() < 0.5:
        p_mkt = p_tab
    else:
        p_mkt = 0.55 * p_tab + 0.45 * p_bf
        p_mkt /= p_mkt.sum()
    return p_mkt, p_tab, p_bf


def _weighted_recent(runs: list[dict[str, Any]], key: str, default: float, *, lower_better: bool = False,
                     target_dist: int = 0, target_track: str = "") -> float:
    vals, ws = [], []
    for k, run in enumerate(runs[:6]):
        if key not in run:
            continue
        v = _safe(run.get(key), default)
        w = math.exp(-0.33 * k)
        if target_dist and abs(int(run.get("distance", 0)) - target_dist) <= 40:
            w *= 1.30
        if target_track and str(run.get("track", "")).upper() == target_track.upper():
            w *= 1.18
        vals.append(v); ws.append(w)
    if not vals:
        return default
    avg = float(np.average(vals, weights=ws))
    return -avg if lower_better else avg


def _record_rate(r: dict[str, Any], prefix: str, place_weight: float = 0.35) -> float:
    starts = max(int(r.get(f"{prefix}_starts", 0)), 0)
    wins = int(r.get(f"{prefix}_wins", 0)); p23 = int(r.get(f"{prefix}_places23", 0))
    win = (wins + 1.0) / (starts + 8.0)
    top3 = (wins + p23 + 3.0) / (starts + 8.0)
    return win + place_weight * top3


def _box_score(r: dict[str, Any]) -> tuple[float, float, float, int]:
    box = int(r.get("box", 0))
    stats = r.get("box_stats", {}).get(box, {}) if box else {}
    starts = int(stats.get("starts", 0)); wins = int(stats.get("wins", 0)); p23 = int(stats.get("places23", 0))
    national = {1: .187, 2: .141, 3: .117, 4: .110, 5: .108, 6: .100, 7: .107, 8: .130}.get(box, .125)
    win_rate = (wins + national * 6.0) / (starts + 6.0)
    place_rate = (wins + p23 + .375 * 6.0) / (starts + 6.0)
    return win_rate + .32 * place_rate, win_rate, place_rate, starts


def _form_finish_score(runs: list[dict[str, Any]]) -> float:
    vals, ws = [], []
    for k, run in enumerate(runs[:6]):
        if "finish" not in run:
            continue
        field = max(int(run.get("field", 8)), 2); finish = max(int(run.get("finish", field)), 1)
        pct = 1.0 - (finish - 1) / (field - 1)
        margin = _safe(run.get("margin"), 5.0)
        vals.append(pct - min(margin, 15.0) / 60.0); ws.append(math.exp(-.33 * k))
    return float(np.average(vals, weights=ws)) if vals else .35


def _early_score(r: dict[str, Any], target_dist: int, target_track: str) -> tuple[float, float, float]:
    runs = r.get("recent_runs", []); vals, ws = [], []; quick, slow = 0, 0
    for k, run in enumerate(runs[:6]):
        field = max(int(run.get("field", 8)), 2)
        if "settle_pos" in run:
            pos = int(run["settle_pos"]); pace = 1.0 - (pos - 1) / (field - 1)
            w = math.exp(-.32 * k)
            if target_dist and abs(int(run.get("distance", 0)) - target_dist) <= 40: w *= 1.25
            if target_track and str(run.get("track", "")).upper() == target_track.upper(): w *= 1.12
            vals.append(pace); ws.append(w)
        note = str(run.get("stewards", "")).lower()
        if "quick to begin" in note: quick += 1
        if "slow to begin" in note or "slowly away" in note: slow += 1
    base = float(np.average(vals, weights=ws)) if vals else .45
    return base + 0.035 * quick - 0.035 * slow, float(quick), float(slow)


def _trackdist_score(r: dict[str, Any]) -> float:
    return .48 * _record_rate(r, "course_distance") + .31 * _record_rate(r, "distance") + .21 * _record_rate(r, "course")


def _freshness_score(days: float) -> float:
    d = max(days, 0.0)
    if d <= 2: return .45
    if d <= 12: return 1.0 - abs(d - 7.0) / 25.0
    if d <= 28: return .75 - (d - 12.0) / 80.0
    if d <= 56: return .55 - (d - 28.0) / 100.0
    return .25


def _components(runners: list[dict[str, Any]], header: dict[str, Any]) -> dict[str, np.ndarray]:
    target_dist = int(header.get("distance_m", 0) or 0); target_track = str(header.get("track", "")).upper()
    speed_raw, early_raw, box_raw, td_raw, form_raw, trn_raw, fresh_raw = [], [], [], [], [], [], []
    box_meta, early_meta = [], []
    for r in runners:
        runs = r.get("recent_runs", [])
        bom = _weighted_recent(runs, "bom_adj", .65, lower_better=True, target_dist=target_dist, target_track=target_track)
        mrk = _weighted_recent(runs, "mrk_delta", .75, lower_better=True, target_dist=target_dist, target_track=target_track)
        speed_raw.append(.58 * bom + .42 * mrk)
        early, q, s = _early_score(r, target_dist, target_track); early_raw.append(early); early_meta.append((q, s))
        bscore, bwr, bpr, bstarts = _box_score(r); box_raw.append(bscore); box_meta.append((bwr, bpr, bstarts))
        td_raw.append(_trackdist_score(r)); form_raw.append(_form_finish_score(runs))
        trn_raw.append(.72 * _safe(r.get("trainer_win"), .10) + .28 * _safe(r.get("trainer_place"), .35))
        fresh_raw.append(_freshness_score(_safe(r.get("dls"), 14)))
    comp = {
        "speed": _z(np.array(speed_raw)), "early": _z(np.array(early_raw)), "box": _z(np.array(box_raw)),
        "trackdist": _z(np.array(td_raw)), "form": _z(np.array(form_raw)), "trainer": _z(np.array(trn_raw)),
        "freshness": _z(np.array(fresh_raw)), "speed_raw": np.array(speed_raw), "early_raw": np.array(early_raw),
        "box_raw": np.array(box_raw), "trackdist_raw": np.array(td_raw), "form_raw": np.array(form_raw),
        "trainer_raw": np.array(trn_raw), "freshness_raw": np.array(fresh_raw), "box_meta": box_meta, "early_meta": early_meta,
    }
    td_best = np.array([_safe(r.get("tra_dist_best"), np.nan) or np.nan for r in runners], dtype=float)
    if np.isfinite(td_best).sum() >= max(3, len(runners)//2):
        comp["trackdist"] = .70 * comp["trackdist"] + .30 * (-_z(td_best))
    pace = .72 * comp["early"] + .28 * comp["box"]
    pace_adj = np.zeros(len(runners)); order = np.argsort(-pace)
    if len(order) >= 2:
        gap = pace[order[0]] - pace[order[1]]
        if gap > .55:
            pace_adj[order[0]] += .22
        else:
            b0, b1 = int(runners[order[0]].get("box", 0)), int(runners[order[1]].get("box", 0))
            if b0 and b1 and abs(b0-b1) <= 2:
                pace_adj[order[0]] -= .07; pace_adj[order[1]] -= .07
    comp["pace"], comp["pace_adjust"], comp["early_order"] = pace, pace_adj, order
    return comp


def _simulate(p: np.ndarray, sims: int, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = len(p); rng = np.random.default_rng(seed); counts = np.zeros((n, n), dtype=int)
    for _ in range(sims):
        remaining = list(range(n))
        for pos in range(n):
            w = p[remaining]; w = w / w.sum(); j = int(rng.choice(len(remaining), p=w)); idx = remaining.pop(j); counts[idx, pos] += 1
    pos_prob = counts / sims
    top2 = pos_prob[:, :min(2,n)].sum(axis=1); top3 = pos_prob[:, :min(3,n)].sum(axis=1)
    exp_pos = (pos_prob * np.arange(1, n+1)).sum(axis=1)
    return top2, top3, exp_pos


def predict(all_runners: list[dict[str, Any]], header: dict[str, Any], *, alpha: float = MARKET_ALPHA,
            sims: int = 20000, seed: int = 42) -> dict[str, Any]:
    active = [r for r in all_runners if not r.get("scratched")]
    if len(active) < 2:
        raise ValueError("At least two active greyhounds are required.")
    p_mkt, p_tab, p_bf = _market_probs(active); comp = _components(active, header)
    fund_score = np.zeros(len(active))
    for name, weight in COMPONENT_WEIGHTS.items(): fund_score += weight * comp[name]
    fund_score += comp["pace_adjust"]; p_fund = _softmax(1.10 * fund_score)
    a = float(np.clip(alpha, 0.0, 1.0))
    p_win = _softmax(a*np.log(np.clip(p_mkt,1e-9,1)) + (1-a)*np.log(np.clip(p_fund,1e-9,1)))
    top2, top3, exp_pos = _simulate(p_win, max(int(sims),1000), int(seed)); order = list(np.argsort(-p_win))
    early_order = list(comp["early_order"]); early_rank = np.empty(len(active), dtype=int)
    for rank, idx in enumerate(early_order, start=1): early_rank[idx] = rank
    bf = np.array([_safe(r.get("bf_odds"),999.0) for r in active]); ev = p_win*bf - 1.0; fair = 1.0/np.clip(p_win,1e-9,None)
    conf = []
    for i, r in enumerate(active):
        completeness = min(len(r.get("recent_runs",[]))/5.0,1.0) + (.20 if r.get("course_distance_starts",0) else 0) + (.15 if r.get("box_stats",{}).get(int(r.get("box",0)),{}).get("starts",0) else 0)
        completeness = min(completeness/1.35,1.0); comp_vals = np.array([comp[k][i] for k in COMPONENT_WEIGHTS]); agreement = 1.0/(1.0+float(comp_vals.std()))
        rank = order.index(i); sep = p_win[i]-p_win[order[rank+1]] if rank+1 < len(order) else p_win[i]
        conf.append(int(np.clip(round(2.0+3.0*completeness+2.0*agreement+4.0*max(sep,0)),1,9)))
    recs, why = [], []
    for i, r in enumerate(active):
        rank = order.index(i)+1
        rec = "TOP PICK" if rank==1 else "DANGER" if rank<=3 else "VALUE" if ev[i]>=.12 and p_win[i]>=.07 else "EXOTICS / PLACE" if top3[i]>=.35 else "WATCH"
        recs.append(rec); bwr,bpr,bstarts = comp["box_meta"][i]; q,s = comp["early_meta"][i]
        reasons = [f"Projected early position {early_rank[i]} of {len(active)}.",
                   f"Current box {r.get('box','?')} history: {bwr*100:.1f}% win, {bpr*100:.1f}% top-3 equivalent over {bstarts} starts.",
                   f"Market {p_mkt[i]*100:.1f}% vs fundamental {p_fund[i]*100:.1f}%; blended win {p_win[i]*100:.1f}%."]
        if q or s: reasons.append(f"Recent steward starts: {int(q)} quick / {int(s)} slow to begin.")
        reasons.append(f"Model fair odds ${fair[i]:.2f}; listed exchange odds ${bf[i]:.2f}; EV {ev[i]:+.2f}.")
        why.append(" ".join(reasons))
    top_gap = p_win[order[0]]-p_win[order[1]] if len(order)>1 else p_win[order[0]]
    overall_conf = int(np.clip(round(4+15*top_gap+np.mean(conf)/3),1,9))
    return {"runners":active,"p_mkt":p_mkt,"p_tab":p_tab,"p_bf":p_bf,"p_fund":p_fund,"p_win":p_win,
            "top2":top2,"top3":top3,"exp_pos":exp_pos,"fair":fair,"ev_win":ev,"order":order,
            "early_rank":early_rank,"early_order":early_order,"components":comp,"conf":conf,"recs":recs,"why":why,
            "overall_conf":overall_conf}
