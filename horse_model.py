"""
horse_model.py - thoroughbred prediction engine
===============================================
Ported unchanged (bar two small guards for single-runner fields) from the
validated model in horse-race-predictor, so the calibration below still holds.

Peer-reviewed pipeline:
  Shin (1993, Economic Journal)            - bookmaker de-vig
  Smith, Paton & Vaughan Williams (2006)   - exchange-price weighting
  Bolton & Chapman (1986, Mgmt Science)    - conditional logit fundamentals
  Benter (1994, Efficiency of Racetrack
      Betting Markets)                     - log-odds market/fundamental blend
  Harville (1973, JASA) +
  Lo, Bacon-Shone & Busche (1995)          - discounted rank probabilities
  Plackett (1975) / Luce (1959)            - sequential-choice Monte Carlo
"""

import numpy as np

MARKET_ALPHA = 0.85       # validated optimum 0.80-0.95 (user backtests, ~1,700 races)
BF_WEIGHT = 0.70          # Betfair vs Shin-corrected TAB within the market term
SIMS = 30_000
LAMBDAS = [1.00, 0.81, 0.65]   # Lo-Bacon-Shone discounts; 0.60 thereafter

FEATURES = [
    ("ohr",        0.90, "official rating"),
    ("neg_wt",     0.25, "weight carried"),
    ("car_win",    0.35, "career strike rate"),
    ("dist_plc",   0.30, "distance placing record"),
    ("going_win",  0.25, "record on today's going"),
    ("jky_win",    0.20, "jockey L50 strike rate"),
    ("trn_win",    0.20, "trainer L50 strike rate"),
    ("neg_lastfin",0.35, "last-start finish"),
    ("freshness",  0.15, "fitness/freshness curve"),
    ("jt_rat",     0.25, "R&S jockey+trainer rating"),
]


def _z(x):
    x = np.asarray(x, float)
    s = x.std()
    return (x - x.mean()) / (s + 1e-9)


def _shin_sum(z, q, beta):
    """Sum of Shin (1993) implied probabilities for a given insider fraction z."""
    if z <= 0:
        return float((q / np.sqrt(beta)).sum())
    return float(((np.sqrt(z * z + 4 * (1 - z) * q * q / beta) - z) / (2 * (1 - z))).sum())


def shin_probs(odds):
    """Shin (1993) de-vig: strip the bookmaker margin and the favourite-longshot bias.

    Solved by bisection on z. The sum of implied probabilities is monotonically
    decreasing in z, so bisection is guaranteed to converge; the fixed-point
    iteration this replaced oscillated between two values and never converged,
    leaving the result at an arbitrary point in that cycle.
    """
    q = np.asarray(1.0 / np.asarray(odds, float), float)
    beta = float(q.sum())
    if beta <= 1.0:                       # no margin to strip
        return q / max(beta, 1e-12), 0.0, beta

    lo, hi = 0.0, 0.5
    for _ in range(60):                   # widen until the sum drops below 1
        if _shin_sum(hi, q, beta) <= 1.0:
            break
        lo, hi = hi, min(hi * 2, 0.999999)
        if hi >= 0.999999:
            break
    for _ in range(200):                  # bisect to the root
        mid = 0.5 * (lo + hi)
        if _shin_sum(mid, q, beta) > 1.0:
            lo = mid
        else:
            hi = mid
    z = 0.5 * (lo + hi)
    p = (np.sqrt(z * z + 4 * (1 - z) * q * q / beta) - z) / (2 * (1 - z))
    return p / p.sum(), z, beta


def build_features(runners, header):
    going = (header.get("going", "GOOD") or "GOOD").upper()
    going_lbl = "Soft" if going.startswith(("S", "H")) else "Good"
    ohr = np.array([r["ohr"] for r in runners], float)
    if ohr.max() > 0:
        ohr[ohr == 0] = np.median(ohr[ohr > 0])  # impute missing
    feats = {
        "ohr": ohr,
        "neg_wt": -np.array([r["wt"] for r in runners]),
        "car_win": np.array([r["Car_win"] for r in runners]),
        "dist_plc": np.array([r["Dist_plc"] for r in runners]),
        "going_win": np.array([r[f"{going_lbl}_win"] for r in runners]),
        "jky_win": np.array([r["jky_win"] for r in runners]),
        "trn_win": np.array([r["trn_win"] for r in runners]),
        "neg_lastfin": -np.array([min(r["last_fin"], 8) for r in runners], float),
        "freshness": -np.abs(np.array([r["dslr"] for r in runners], float) - 21),
        "jt_rat": np.array([r["jrat"] + r["trat"] for r in runners]),
    }
    return feats


def predict(runners, header, alpha=MARKET_ALPHA, sims=SIMS, seed=42,
            bf_weight=BF_WEIGHT):
    rng = np.random.default_rng(seed)
    n = len(runners)
    tab_odds = np.array([r["tab_odds"] for r in runners])
    bf_odds = np.array([r["bf_odds"] for r in runners])

    # -------- market --------
    p_tab, z_shin, overround = shin_probs(tab_odds)
    q_bf = 1 / bf_odds
    p_bf = q_bf / q_bf.sum()
    p_mkt = bf_weight * p_bf + (1 - bf_weight) * p_tab
    p_mkt /= p_mkt.sum()

    # -------- fundamental (Bolton-Chapman conditional logit) --------
    feats = build_features(runners, header)
    V = np.zeros(n)
    contribs = {}                       # per-horse feature contributions
    for name, beta, label in FEATURES:
        zc = _z(feats[name]) * beta
        V += zc
        contribs[name] = zc
    p_fund = np.exp(V - V.max())
    p_fund /= p_fund.sum()

    # -------- Benter blend --------
    lg = alpha * np.log(np.clip(p_mkt, 1e-9, 1)) + \
         (1 - alpha) * np.log(np.clip(p_fund, 1e-9, 1))
    p_win = np.exp(lg - lg.max())
    p_win /= p_win.sum()

    # -------- finishing-order Monte Carlo (discounted Plackett-Luce) ----
    lam = LAMBDAS + [0.60] * max(0, n - len(LAMBDAS))
    pos_counts = np.zeros((n, n), dtype=np.int64)
    idx = np.arange(n)
    for _ in range(sims):
        remaining = idx.tolist()
        for k in range(n):
            s = p_win[remaining] ** lam[k]
            cs = np.cumsum(s)
            j = int(np.searchsorted(cs, rng.random() * cs[-1]))
            pick = remaining.pop(j)
            pos_counts[pick, k] += 1
    pos_prob = pos_counts / sims
    exp_pos = pos_prob @ np.arange(1, n + 1)
    top2 = pos_prob[:, :min(2, n)].sum(axis=1)
    top3 = pos_prob[:, :min(3, n)].sum(axis=1)

    # -------- EV & recommendations --------
    ev_win = p_win * bf_odds
    fund_mkt_ratio = p_fund / np.clip(p_mkt, 1e-9, 1)

    # -------- confidence 0-9 per horse --------
    # sharpness: how concentrated is this horse's position distribution
    with np.errstate(divide="ignore", invalid="ignore"):
        ent = -np.nansum(np.where(pos_prob > 0,
                                  pos_prob * np.log(pos_prob), 0), axis=1)
    max_ent = np.log(n) if n > 1 else 1.0
    sharp = 1 - ent / max_ent                          # 0..1
    agree = 1 - np.tanh(np.abs(np.log(fund_mkt_ratio)) / 1.5)   # 0..1
    starts = np.array([r["Car_starts"] for r in runners], float)
    depth = np.clip(starts / 15, 0.2, 1.0)             # data depth 0.2..1
    conf = np.clip(np.rint(9 * (0.45 * sharp + 0.35 * agree + 0.20 * depth)),
                   0, 9).astype(int)

    # overall model confidence 0-9
    sorted_p = np.sort(p_win)
    fav_gap = float(sorted_p[-1] - sorted_p[-2]) if n > 1 else 1.0
    book_ok = 1.0 if 1.0 <= (1 / bf_odds).sum() <= 1.10 else 0.6
    overall_conf = int(np.clip(np.rint(
        9 * (0.4 * float(np.mean(sharp)) + 0.3 * float(np.mean(agree)) +
             0.3 * min(fav_gap / 0.15, 1.0)) * book_ok), 0, 9))

    recs, why = [], []
    order = np.argsort(exp_pos)
    for i in range(n):
        r = runners[i]
        rec = "NO BET"
        if p_win[i] < 0.02:
            rec = "AVOID"
        elif ev_win[i] >= 1.10 and p_win[i] >= 0.15:
            rec = "WIN BET (overlay)"
        elif ev_win[i] >= 1.10 and p_win[i] >= 0.05:
            rec = "SMALL WIN / EACH-WAY (overlay)"
        elif fund_mkt_ratio[i] >= 2.5 and p_win[i] >= 0.03:
            rec = "EXOTICS ANCHOR (form-model roughie)"
        elif 0.95 <= ev_win[i] < 1.10 and exp_pos[i] == exp_pos.min():
            rec = "FAIR PRICE - saver only"
        recs.append(rec)
        why.append(_explain(r, i, contribs, p_mkt, p_fund, p_win, ev_win,
                            fund_mkt_ratio, conf, sharp, agree, depth, rec,
                            exp_pos, top3))

    return {
        "p_mkt": p_mkt, "p_fund": p_fund, "p_win": p_win,
        "pos_prob": pos_prob, "exp_pos": exp_pos, "top2": top2, "top3": top3,
        "ev_win": ev_win, "conf": conf, "overall_conf": overall_conf,
        "recs": recs, "why": why, "order": order,
        "overround_tab": overround, "shin_z": z_shin,
        "book_bf": float((1 / bf_odds).sum()),
        "alpha": alpha, "fund_mkt_ratio": fund_mkt_ratio, "sims": sims,
    }


def _explain(r, i, contribs, p_mkt, p_fund, p_win, ev, ratio, conf,
             sharp, agree, depth, rec, exp_pos, top3):
    """Template explanation citing the actual feature contributions."""
    labels = {name: lbl for name, _, lbl in FEATURES}
    cvals = sorted(((contribs[k][i], labels[k]) for k in contribs),
                   key=lambda t: -abs(t[0]))
    pos = [f"{lbl} (+{v:.2f})" for v, lbl in cvals if v > 0.08][:3]
    neg = [f"{lbl} ({v:.2f})" for v, lbl in cvals if v < -0.08][:3]

    lines = []
    lines.append(f"Market {p_mkt[i]*100:.1f}% | Fundamental {p_fund[i]*100:.1f}% "
                 f"| Blended win {p_win[i]*100:.1f}% | Top-3 {top3[i]*100:.1f}% "
                 f"| E[pos] {exp_pos[i]:.2f} | EV {ev[i]:.2f} @ ${r['bf_odds']}")
    if pos:
        lines.append("Form model likes: " + "; ".join(pos) + ".")
    if neg:
        lines.append("Form model dislikes: " + "; ".join(neg) + ".")
    if ratio[i] >= 2.0:
        lines.append(f"Fundamental model rates this runner {ratio[i]:.1f}x the "
                     f"market - a divergence flag (exotics angle, per the "
                     f">=2.5x screen).")
    elif ratio[i] <= 0.5:
        lines.append(f"Market is {1/max(ratio[i],1e-9):.1f}x keener than the "
                     f"form model - price likely reflects stable/market "
                     f"intelligence not visible in the form.")
    lines.append(f"Confidence {conf[i]}/9 = position-distribution sharpness "
                 f"{sharp[i]:.2f} (45%), market/form agreement {agree[i]:.2f} "
                 f"(35%), data depth {depth[i]:.2f} on {r['Car_starts']} "
                 f"starts (20%).")
    lines.append(f"Recommendation: {rec}. "
                 + {"WIN BET (overlay)": "Blended probability exceeds the "
                        "Betfair price by >=10% on a genuine winning chance.",
                    "SMALL WIN / EACH-WAY (overlay)": "Priced over fair odds "
                        "but win probability is modest - stake accordingly.",
                    "EXOTICS ANCHOR (form-model roughie)": "Large form-vs-"
                        "market divergence; best used in trifecta/first4 "
                        "legs rather than win bets.",
                    "FAIR PRICE - saver only": "Top pick but no overlay; the "
                        "market has it about right.",
                    "AVOID": "Negligible winning probability.",
                    "NO BET": "No value edge at current prices."}[rec])
    return "\n".join(lines)
