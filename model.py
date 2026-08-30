"""Full-time score matrix from two FootyStats team panels.

The two scoring rates come from pairing each side's attack with the other's
defence, **at the venue each will actually be playing**::

    lambda_home = mean(home.scored_at_home,  away.conceded_away)
    lambda_away = mean(away.scored_away,     home.conceded_at_home)

and the same pairing again on xG/xGA, the two blended.

**There is deliberately no home-advantage multiplier.** The venue columns
already contain it - a home scoring rate is a home scoring rate - so applying a
further home factor on top would count the same effect twice. That is the one
specification error this shape of input invites.

The venue figures are averaged over half a season, so they are shrunk toward
the team's overall figure rather than trusted outright: eight or nine matches
is enough to be worth using and not enough to be taken at face value.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import nbinom, poisson

import panel as PN

MAX_GOALS = 10
GRID = np.arange(MAX_GOALS + 1)


def _blend_venue(p: dict, key: str, venue: str, w: float) -> float:
    v = PN.value(p, key, venue)
    o = PN.value(p, key, "overall")
    if not np.isfinite(v):
        return o
    if not np.isfinite(o):
        return v
    return w * v + (1 - w) * o


def rates(home: dict, away: dict, *, venue_weight: float = 0.65,
          xg_weight: float = 0.5) -> dict:
    """The two scoring rates, and the parts they were built from."""
    h_scored = _blend_venue(home, "scored", "home", venue_weight)
    h_conc = _blend_venue(home, "conceded", "home", venue_weight)
    a_scored = _blend_venue(away, "scored", "away", venue_weight)
    a_conc = _blend_venue(away, "conceded", "away", venue_weight)
    h_xg = _blend_venue(home, "xg", "home", venue_weight)
    h_xga = _blend_venue(home, "xga", "home", venue_weight)
    a_xg = _blend_venue(away, "xg", "away", venue_weight)
    a_xga = _blend_venue(away, "xga", "away", venue_weight)

    def pair(att, dfn):
        vals = [v for v in (att, dfn) if np.isfinite(v)]
        return float(np.mean(vals)) if vals else np.nan

    lh_g, la_g = pair(h_scored, a_conc), pair(a_scored, h_conc)
    lh_x, la_x = pair(h_xg, a_xga), pair(a_xg, h_xga)

    def mix(g, x):
        if not np.isfinite(x):
            return g
        if not np.isfinite(g):
            return x
        return xg_weight * x + (1 - xg_weight) * g

    lh, la = mix(lh_g, lh_x), mix(la_g, la_x)
    return {
        "lh": float(np.clip(lh, 0.15, 6.0)), "la": float(np.clip(la, 0.15, 6.0)),
        "lh_goals": lh_g, "la_goals": la_g, "lh_xg": lh_x, "la_xg": la_x,
        "parts": {"home scored": h_scored, "away conceded": a_conc,
                  "away scored": a_scored, "home conceded": h_conc,
                  "home xG": h_xg, "away xGA": a_xga,
                  "away xG": a_xg, "home xGA": h_xga},
    }


def _tau(lh: float, la: float, rho: float, n: int) -> np.ndarray:
    t = np.ones((n, n))
    t[0, 0] = 1 - lh * la * rho
    t[0, 1] = 1 + lh * rho
    t[1, 0] = 1 + la * rho
    t[1, 1] = 1 - rho
    return np.clip(t, 1e-6, None)


def score_matrix(lh: float, la: float, *, rho: float = -0.05,
                 dispersion: float | None = None) -> np.ndarray:
    """Dixon-Coles corrected score grid.

    `dispersion` switches the count distribution from Poisson to negative
    binomial with that `r`. It fattens **both** tails, not just the top: at
    football scoring rates the extra mass at 0-0 outweighs the extra at the
    high end, so the over-lines drift slightly *down* while 0-0 rises sharply
    (at mu=1.5 a side, r=4 moves 0-0 from 5.5% to 8.7% and over 3.5 from 35.3%
    to 35.1%). Use it to spread the grid, not to chase goals.
    """
    if dispersion and dispersion > 0:
        ph = nbinom.pmf(GRID, dispersion, dispersion / (dispersion + lh))
        pa = nbinom.pmf(GRID, dispersion, dispersion / (dispersion + la))
    else:
        ph, pa = poisson.pmf(GRID, lh), poisson.pmf(GRID, la)
    m = np.outer(ph, pa) * _tau(lh, la, rho, MAX_GOALS + 1)
    return m / m.sum()


def markets(m: np.ndarray) -> dict:
    m = np.asarray(m, dtype=float)
    idx = np.arange(m.shape[0])
    tot = idx[:, None] + idx[None, :]
    return {
        "home": float(m[np.tril_indices_from(m, -1)].sum()),
        "draw": float(np.trace(m)),
        "away": float(m[np.triu_indices_from(m, 1)].sum()),
        "btts": float(m[1:, 1:].sum()),
        "over15": float(m[tot > 1].sum()),
        "over25": float(m[tot > 2].sum()),
        "over35": float(m[tot > 3].sum()),
        "exp_goals": float((m * tot).sum()),
    }


def top_scores(m: np.ndarray, k: int = 10):
    m = np.asarray(m, dtype=float)
    flat = [((i, j), float(m[i, j])) for i in range(m.shape[0])
            for j in range(m.shape[1])]
    return sorted(flat, key=lambda x: -x[1])[:k]


def score_prob(m: np.ndarray, hg: int, ag: int) -> float:
    if hg > MAX_GOALS or ag > MAX_GOALS:
        return 0.0
    return float(m[hg, ag])


def predict(home: dict, away: dict, **kw) -> dict:
    vk = {k: kw.pop(k) for k in ("venue_weight", "xg_weight") if k in kw}
    r = rates(home, away, **vk)
    m = score_matrix(r["lh"], r["la"], **kw)
    out = {**r, "matrix": m, **markets(m)}
    (hg, ag), p = top_scores(m, 1)[0]
    out["pick"] = (hg, ag)
    out["pick_prob"] = p
    return out
