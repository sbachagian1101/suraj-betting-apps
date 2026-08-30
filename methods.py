"""Twelve prediction methods, and an honest way of combining them.

Each method returns 1X2, and most also return a full score matrix from which
BTTS, over/under and the correct-score grid are read off.

**What an ensemble of twelve can and cannot do here.** Ten of these methods are
fed by the same two numbers - a home scoring rate and an away scoring rate -
and differ only in the count distribution wrapped around them. They are not
twelve independent opinions; they are one opinion under twelve parameterisations
plus two (Bradley-Terry, Elo) that look at results instead of rates. So their
agreement is close to meaningless as evidence, and the spread between them
measures model form, not uncertainty about the match. The genuinely independent
uncertainty is in the rates themselves, and with five matches a side that is
large. The app shows every method separately so this is visible rather than
hidden behind one confident-looking number.
"""
from __future__ import annotations

import math

import numpy as np
from scipy.stats import nbinom, poisson, skellam

MAX_GOALS = 8
GRID = np.arange(MAX_GOALS + 1)


# --------------------------------------------------------------- helpers
def _norm(m: np.ndarray) -> np.ndarray:
    s = m.sum()
    return m / s if s > 0 else np.full_like(m, 1.0 / m.size)


def outcomes(m: np.ndarray) -> dict:
    """1X2, BTTS and totals read off a score matrix."""
    m = _norm(np.asarray(m, dtype=float))
    idx = np.arange(m.shape[0])
    hw = float(m[np.tril_indices_from(m, -1)].sum())
    aw = float(m[np.triu_indices_from(m, 1)].sum())
    dr = float(np.trace(m))
    btts = float(m[1:, 1:].sum())
    tot = idx[:, None] + idx[None, :]
    return {"home": hw, "draw": dr, "away": aw, "btts": btts,
            "over25": float(m[tot > 2].sum()),
            "over15": float(m[tot > 1].sum()),
            "over35": float(m[tot > 3].sum())}


def poisson_matrix(lh: float, la: float) -> np.ndarray:
    return _norm(np.outer(poisson.pmf(GRID, lh), poisson.pmf(GRID, la)))


def _tau(lh: float, la: float, rho: float) -> np.ndarray:
    t = np.ones((MAX_GOALS + 1, MAX_GOALS + 1))
    t[0, 0] = 1 - lh * la * rho
    t[0, 1] = 1 + lh * rho
    t[1, 0] = 1 + la * rho
    t[1, 1] = 1 - rho
    return np.clip(t, 1e-6, None)


# ---------------------------------------------------------------- methods
def m_poisson_goals(ctx) -> dict:
    lh, la = ctx["lh_goals"], ctx["la_goals"]
    return {"matrix": poisson_matrix(lh, la), "lh": lh, "la": la}


def m_poisson_xg(ctx) -> dict:
    lh, la = ctx["lh_xg"], ctx["la_xg"]
    return {"matrix": poisson_matrix(lh, la), "lh": lh, "la": la}


def m_dixon_coles(ctx) -> dict:
    """Poisson with the low-score dependence correction (Dixon & Coles 1997)."""
    lh, la, rho = ctx["lh"], ctx["la"], ctx["rho"]
    return {"matrix": _norm(poisson_matrix(lh, la) * _tau(lh, la, rho)),
            "lh": lh, "la": la, "note": f"rho={rho:+.2f}"}


def m_bivariate_poisson(ctx) -> dict:
    """Karlis & Ntzoufras: a shared component makes the scores correlated."""
    lh, la, c = ctx["lh"], ctx["la"], ctx["cov"]
    l3 = c * min(lh, la)
    l1, l2 = max(lh - l3, 1e-6), max(la - l3, 1e-6)
    m = np.zeros((MAX_GOALS + 1, MAX_GOALS + 1))
    for x in GRID:
        for y in GRID:
            s = 0.0
            for k in range(min(x, y) + 1):
                s += (math.comb(x, k) * math.comb(y, k) * math.factorial(k)
                      * (l3 / (l1 * l2)) ** k)
            m[x, y] = (math.exp(-(l1 + l2 + l3)) * l1 ** x / math.factorial(x)
                       * l2 ** y / math.factorial(y) * s)
    return {"matrix": _norm(m), "lh": lh, "la": la, "note": f"lambda3={l3:.2f}"}


def m_negative_binomial(ctx) -> dict:
    """Overdispersed counts: football scores are a little wilder than Poisson."""
    lh, la, r = ctx["lh"], ctx["la"], ctx["nb_r"]
    ph = nbinom.pmf(GRID, r, r / (r + lh))
    pa = nbinom.pmf(GRID, r, r / (r + la))
    return {"matrix": _norm(np.outer(ph, pa)), "lh": lh, "la": la,
            "note": f"r={r:.0f}"}


def m_skellam(ctx) -> dict:
    """The exact distribution of the goal difference of two Poissons."""
    lh, la = ctx["lh"], ctx["la"]
    ks = np.arange(-MAX_GOALS, MAX_GOALS + 1)
    p = skellam.pmf(ks, lh, la)
    p = p / p.sum()
    home = float(p[ks > 0].sum())
    draw = float(p[ks == 0].sum())
    away = float(p[ks < 0].sum())
    base = outcomes(poisson_matrix(lh, la))
    return {"home": home, "draw": draw, "away": away,
            "btts": base["btts"], "over25": base["over25"],
            "over15": base["over15"], "over35": base["over35"],
            "lh": lh, "la": la}


def m_monte_carlo(ctx) -> dict:
    """Simulation with the rate itself uncertain, which is the real risk."""
    lh, la, sd = ctx["lh"], ctx["la"], ctx["rate_sd"]
    rng = np.random.default_rng(ctx.get("seed", 7))
    n = int(ctx.get("sims", 40000))
    rh = np.clip(lh * np.exp(rng.normal(0, sd, n)), 0.05, 8)
    ra = np.clip(la * np.exp(rng.normal(0, sd, n)), 0.05, 8)
    gh, ga = rng.poisson(rh), rng.poisson(ra)
    m = np.zeros((MAX_GOALS + 1, MAX_GOALS + 1))
    np.add.at(m, (np.clip(gh, 0, MAX_GOALS), np.clip(ga, 0, MAX_GOALS)), 1)
    return {"matrix": _norm(m), "lh": lh, "la": la,
            "note": f"rate sd={sd:.2f}"}


def m_shots_conversion(ctx) -> dict:
    """Goals as shots on target times a shrunk conversion rate."""
    h, a, base = ctx["home"], ctx["away"], ctx["base"]
    def rate(att, dfn):
        sot = np.nanmean([att.get("sot_for", np.nan),
                          dfn.get("sot_against", np.nan)])
        if not np.isfinite(sot) or sot <= 0:
            return np.nan
        conv = att.get("gf", np.nan) / att.get("sot_for", np.nan) \
            if att.get("sot_for", 0) else np.nan
        conv = 0.5 * (conv if np.isfinite(conv) else 0.30) + 0.5 * 0.30
        return float(np.clip(sot * conv, 0.15, 5.0))
    lh, la = rate(h, a), rate(a, h)
    if not (np.isfinite(lh) and np.isfinite(la)):
        lh, la = ctx["lh"], ctx["la"]
    lh *= base["home_adv"]
    la /= base["home_adv"]
    return {"matrix": poisson_matrix(lh, la), "lh": lh, "la": la}


def m_empirical(ctx) -> dict:
    """Resample the scorelines the two teams actually produced."""
    h, a = ctx["home"], ctx["away"]
    rng = np.random.default_rng(ctx.get("seed", 11) + 1)
    hm, am = h.get("matches"), a.get("matches")
    if hm is None or am is None or hm.empty or am.empty:
        return m_poisson_xg(ctx)
    hw = hm["weight"].to_numpy(float)
    aw = am["weight"].to_numpy(float)
    hg = hm["gf"].to_numpy(float)
    ag = am["gf"].to_numpy(float)
    n = int(ctx.get("sims", 40000))
    ih = rng.choice(len(hg), n, p=hw / hw.sum())
    ia = rng.choice(len(ag), n, p=aw / aw.sum())
    m = np.zeros((MAX_GOALS + 1, MAX_GOALS + 1))
    np.add.at(m, (np.clip(hg[ih].astype(int), 0, MAX_GOALS),
                  np.clip(ag[ia].astype(int), 0, MAX_GOALS)), 1)
    m = _norm(m + 0.004)                 # a little smoothing: n is five
    return {"matrix": m, "lh": float(np.average(hg, weights=hw)),
            "la": float(np.average(ag, weights=aw)),
            "note": "resampled, smoothed"}


def m_bradley_terry(ctx) -> dict:
    """Strengths fitted to who beat whom, with Davidson's draw term."""
    df, home, away = ctx["df"], ctx["home"]["team"], ctx["away"]["team"]
    names = sorted(set(df.home) | set(df.away))
    idx = {t: i for i, t in enumerate(names)}
    r = np.zeros(len(names))
    nu = 1.0
    for _ in range(400):
        num = np.zeros(len(names))
        den = np.zeros(len(names))
        for _, mrow in df.iterrows():
            i, j = idx[mrow.home], idx[mrow.away]
            pi, pj = math.exp(r[i]), math.exp(r[j])
            d = pi + pj + nu * math.sqrt(pi * pj)
            si = 1.0 if mrow.hg > mrow.ag else (0.5 if mrow.hg == mrow.ag else 0.0)
            num[i] += si
            num[j] += 1 - si
            den[i] += (pi + 0.5 * nu * math.sqrt(pi * pj)) / d
            den[j] += (pj + 0.5 * nu * math.sqrt(pi * pj)) / d
        new = np.log(np.clip(num, 1e-6, None) / np.clip(den, 1e-6, None))
        new -= new.mean()
        new = 0.5 * new + 0.5 * r            # damped, the sample is tiny
        if np.max(np.abs(new - r)) < 1e-9:
            r = new
            break
        r = new
    pi, pj = math.exp(r[idx[home]]), math.exp(r[idx[away]])
    pi *= ctx["base"]["home_adv"]
    d = pi + pj + nu * math.sqrt(pi * pj)
    base = outcomes(poisson_matrix(ctx["lh"], ctx["la"]))
    return {"home": pi / d, "draw": nu * math.sqrt(pi * pj) / d, "away": pj / d,
            "btts": base["btts"], "over25": base["over25"],
            "over15": base["over15"], "over35": base["over35"],
            "note": "Davidson ties"}


def m_elo(ctx) -> dict:
    """Sequential ratings over the parsed matches, oldest first."""
    df, home, away = ctx["df"], ctx["home"]["team"], ctx["away"]["team"]
    R: dict = {}
    K, HA = 20.0, 40.0
    for _, mrow in df.sort_values("date").iterrows():
        h, a = mrow.home, mrow.away
        R.setdefault(h, 1500.0)
        R.setdefault(a, 1500.0)
        e = 1 / (1 + 10 ** ((R[a] - (R[h] + HA)) / 400))
        s = 1.0 if mrow.hg > mrow.ag else (0.5 if mrow.hg == mrow.ag else 0.0)
        g = 1 + 0.5 * min(abs(mrow.hg - mrow.ag), 4)
        R[h] += K * g * (s - e)
        R[a] -= K * g * (s - e)
    dh = (R.get(home, 1500.0) + HA) - R.get(away, 1500.0)
    p_home_or_half = 1 / (1 + 10 ** (-dh / 400))
    draw = float(np.clip(0.28 - abs(dh) / 4000, 0.10, 0.32))
    home_p = float(np.clip(p_home_or_half - draw / 2, 0.02, 0.96))
    away_p = float(np.clip(1 - home_p - draw, 0.02, 0.96))
    base = outcomes(poisson_matrix(ctx["lh"], ctx["la"]))
    return {"home": home_p, "draw": draw, "away": away_p,
            "btts": base["btts"], "over25": base["over25"],
            "over15": base["over15"], "over35": base["over35"],
            "note": f"Elo gap {dh:+.0f}"}


def m_form_logistic(ctx) -> dict:
    """Ordered logit on the weighted points-per-game gap."""
    h, a = ctx["home"], ctx["away"]
    gap = (h.get("ppg", 1.2) - a.get("ppg", 1.2)) + 0.25 * math.log(
        max(ctx["base"]["home_adv"], 1e-6)) * 4
    b = 1.05
    z = b * gap
    c1, c2 = -0.62, 0.62
    p_away = 1 / (1 + math.exp(z - c1))
    p_not_home = 1 / (1 + math.exp(z - c2))
    draw = max(p_not_home - p_away, 1e-4)
    home_p = max(1 - p_not_home, 1e-4)
    s = home_p + draw + p_away
    base = outcomes(poisson_matrix(ctx["lh"], ctx["la"]))
    return {"home": home_p / s, "draw": draw / s, "away": p_away / s,
            "btts": base["btts"], "over25": base["over25"],
            "over15": base["over15"], "over35": base["over35"],
            "note": f"PPG gap {h.get('ppg', 0) - a.get('ppg', 0):+.2f}"}


def m_weakness_adjusted(ctx) -> dict:
    """Poisson on rates nudged by the finishing and keeping gaps.

    Attack weakness is a team scoring below its chances; defence weakness is
    conceding above them. Both are half-believed - over five matches most of
    that gap is luck, not a property of the team.
    """
    h, a, base = ctx["home"], ctx["away"], ctx["base"]
    lh, la = ctx["lh"], ctx["la"]
    lh *= float(np.clip(1 - 0.5 * h.get("atk_weakness", 0.0), 0.6, 1.5))
    lh *= float(np.clip(1 + 0.5 * a.get("def_weakness", 0.0), 0.6, 1.5))
    la *= float(np.clip(1 - 0.5 * a.get("atk_weakness", 0.0), 0.6, 1.5))
    la *= float(np.clip(1 + 0.5 * h.get("def_weakness", 0.0), 0.6, 1.5))
    lh, la = float(np.clip(lh, 0.15, 6)), float(np.clip(la, 0.15, 6))
    return {"matrix": poisson_matrix(lh, la), "lh": lh, "la": la,
            "note": "half-weighted finishing gaps"}


METHODS = [
    ("Poisson (goals)", m_poisson_goals, "Independent Poisson on goal rates."),
    ("Poisson (xG)", m_poisson_xg, "The same, driven by xG instead of goals."),
    ("Dixon–Coles", m_dixon_coles, "Poisson with the low-score correction."),
    ("Bivariate Poisson", m_bivariate_poisson, "Correlated scores via a shared term."),
    ("Negative binomial", m_negative_binomial, "Allows scores wilder than Poisson."),
    ("Skellam", m_skellam, "Exact goal-difference distribution."),
    ("Monte Carlo", m_monte_carlo, "Simulation with the rates themselves uncertain."),
    ("Shots × conversion", m_shots_conversion, "Shots on target times conversion."),
    ("Empirical resample", m_empirical, "Resamples the scorelines actually produced."),
    ("Bradley–Terry", m_bradley_terry, "Strength fitted to results, Davidson ties."),
    ("Elo", m_elo, "Sequential ratings over the parsed matches."),
    ("Form logistic", m_form_logistic, "Ordered logit on the points-per-game gap."),
    ("Weakness-adjusted", m_weakness_adjusted, "Rates nudged by finishing gaps."),
]


def build_context(df, home, away, base, *, rho=-0.05, cov=0.10, nb_r=8.0,
                  rate_sd=0.28, sims=40000, seed=7) -> dict:
    import metrics as M
    lh_xg, la_xg = M.expected_goals(home, away, base, use_xg=True)
    lh_g, la_g = M.expected_goals(home, away, base, use_xg=False)
    lh = 0.6 * lh_xg + 0.4 * lh_g      # xG is the steadier of the two
    la = 0.6 * la_xg + 0.4 * la_g
    return {"df": df, "home": home, "away": away, "base": base,
            "lh": lh, "la": la, "lh_xg": lh_xg, "la_xg": la_xg,
            "lh_goals": lh_g, "la_goals": la_g,
            "rho": rho, "cov": cov, "nb_r": nb_r, "rate_sd": rate_sd,
            "sims": sims, "seed": seed}


def run_all(ctx) -> list[dict]:
    out = []
    for name, fn, blurb in METHODS:
        try:
            res = fn(ctx)
        except Exception as exc:                              # noqa: BLE001
            out.append({"method": name, "error": f"{type(exc).__name__}: {exc}",
                        "blurb": blurb})
            continue
        row = {"method": name, "blurb": blurb, "note": res.get("note", "")}
        if "matrix" in res:
            row["matrix"] = res["matrix"]
            row.update(outcomes(res["matrix"]))
        else:
            row.update({k: res[k] for k in
                        ("home", "draw", "away", "btts", "over25",
                         "over15", "over35") if k in res})
        row["lh"], row["la"] = res.get("lh"), res.get("la")
        out.append(row)
    return out


def ensemble(results: list[dict], weights: dict | None = None) -> dict:
    """Weighted average of the methods, with the score grid from those that
    produce one. Averaging probabilities (not log-odds) keeps a single wild
    method from dominating."""
    ok = [r for r in results if "home" in r and "error" not in r]
    if not ok:
        raise ValueError("no method produced a prediction")
    w = np.array([float((weights or {}).get(r["method"], 1.0)) for r in ok])
    w = w / w.sum()
    keys = ("home", "draw", "away", "btts", "over25", "over15", "over35")
    agg = {k: float(np.sum([wi * r.get(k, np.nan) for wi, r in zip(w, ok)
                            if np.isfinite(r.get(k, np.nan))])) for k in keys}
    s = agg["home"] + agg["draw"] + agg["away"]
    for k in ("home", "draw", "away"):
        agg[k] /= s

    mats = [(wi, r["matrix"]) for wi, r in zip(w, ok) if "matrix" in r]
    if mats:
        tot = sum(wi for wi, _ in mats)
        agg["matrix"] = _norm(sum(wi * m for wi, m in mats) / tot)
    agg["n_methods"] = len(ok)
    agg["spread"] = float(np.std([r["home"] for r in ok]))
    return agg


def corners_markets(ch: float, ca: float, line: float = 8.5,
                    nb_r: float = 12.0) -> dict:
    """Total corners over/under, by Poisson and by negative binomial."""
    tot = ch + ca
    ks = np.arange(0, 41)
    p_pois = poisson.pmf(ks, tot)
    p_nb = nbinom.pmf(ks, nb_r, nb_r / (nb_r + tot))
    over_p = float(p_pois[ks > line].sum())
    over_nb = float(p_nb[ks > line].sum())
    over = 0.5 * (over_p + over_nb)
    return {"expected": tot, "home": ch, "away": ca,
            "over": over, "under": 1 - over,
            "over_poisson": over_p, "over_negbin": over_nb,
            "line": line, "dist": 0.5 * (p_pois + p_nb), "ks": ks}


def top_scores(matrix: np.ndarray, k: int = 8) -> list[tuple[str, float]]:
    m = np.asarray(matrix, dtype=float)
    flat = [(f"{i}-{j}", float(m[i, j])) for i in range(m.shape[0])
            for j in range(m.shape[1])]
    return sorted(flat, key=lambda x: -x[1])[:k]
