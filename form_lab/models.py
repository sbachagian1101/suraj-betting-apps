"""Race-level probability models and held-out combination/calibration."""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize, minimize_scalar
from scipy.special import logsumexp


def groups(ids):
    ids = np.asarray(ids)
    boundaries = np.flatnonzero(ids[1:] != ids[:-1]) + 1
    starts = np.r_[0, boundaries]
    ends = np.r_[boundaries, len(ids)]
    if len(set(ids)) != len(starts):
        raise ValueError("Race rows must be contiguous")
    return [slice(int(a), int(b)) for a, b in zip(starts, ends) if b > a]


def race_softmax(scores, slices):
    scores = np.asarray(scores, dtype=float)
    if not np.isfinite(scores).all():
        raise ValueError("Non-finite model score")
    out = np.empty_like(scores)
    for sl in slices:
        out[sl] = np.exp(scores[sl] - logsumexp(scores[sl]))
    return out


def logloss(p, y, slices):
    return float(np.mean([-np.dot(y[s], np.log(np.clip(p[s], 1e-15, 1))) for s in slices]))


class ConditionalLogit:
    """Regularised baseline; this is not the later dynamic Bayesian model."""
    def __init__(self, l2=.03):
        self.l2 = l2

    def fit(self, x, y, slices):
        x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
        if any(not np.isclose(y[s].sum(), 1) for s in slices):
            raise ValueError("Each race must have unit winner mass")
        def objective(beta):
            score = x @ beta
            p = race_softmax(score, slices)
            loss = np.mean([logsumexp(score[s]) - np.dot(y[s], score[s]) for s in slices])
            grad = x.T @ (p - y) / len(slices) + self.l2 * beta
            return loss + .5 * self.l2 * np.dot(beta, beta), grad
        fit = minimize(objective, np.zeros(x.shape[1]), jac=True, method="L-BFGS-B",
                       options={"maxiter": 400, "ftol": 1e-10})
        if not fit.success:
            raise RuntimeError(f"Conditional logit failed to converge: {fit.message}")
        self.beta = fit.x
        return self

    def predict(self, x, slices):
        return race_softmax(np.asarray(x) @ self.beta, slices)


def blend(probabilities, weights, slices, temperature=1):
    weights = np.asarray(weights, dtype=float)
    if np.any(weights < 0) or not np.isclose(weights.sum(), 1) or temperature <= 0:
        raise ValueError("Invalid simplex weights or temperature")
    score = np.log(np.clip(np.asarray(probabilities), 1e-15, 1)).T @ weights
    return race_softmax(score / temperature, slices)


def fit_blend(probabilities, y, slices):
    # Two-component simplex. Endpoints allow either model to receive zero weight.
    objective = lambda w: logloss(blend(probabilities, [w, 1-w], slices), y, slices)
    fit = minimize_scalar(objective, bounds=(0, 1), method="bounded")
    if not fit.success:
        raise RuntimeError("Blend optimisation failed")
    w = min([0., float(fit.x), 1.], key=objective)
    return [w, 1-w]


def fit_temperature(p, y, slices):
    objective = lambda log_t: logloss(race_softmax(np.log(np.clip(p, 1e-15, 1)) / np.exp(log_t), slices), y, slices)
    fit = minimize_scalar(objective, bounds=(np.log(.25), np.log(4)), method="bounded")
    if not fit.success:
        raise RuntimeError("Temperature optimisation failed")
    return float(np.exp(fit.x))


def pl_topk(p, k=3):
    """Exact PL top-k marginals, k <= 3; no Monte Carlo sampling error.

Assumes a complete strict finishing order. DNF and tied-place corrections are
not modelled here. Top-three is not bookmaker place probability.
"""
    p = np.asarray(p, dtype=float)
    if not np.isfinite(p).all() or np.any(p < 0) or not np.isclose(p.sum(), 1):
        raise ValueError("Invalid winner probabilities")
    k = min(k, len(p))
    if k not in (1, 2, 3):
        raise ValueError("Only top-1, top-2 and top-3 are supported")
    # Tiny floor avoids zero denominators in degenerate deterministic fields.
    p = np.clip(p, 1e-12, None)
    p /= p.sum()
    hits = np.zeros(len(p))
    def visit(chosen, mass):
        if len(chosen) == k:
            hits[list(chosen)] += mass
            return
        remaining = [i for i in range(len(p)) if i not in chosen]
        denom = p[remaining].sum()
        for i in remaining:
            visit(chosen + (i,), mass * p[i] / denom)
    visit((), 1.)
    return np.clip(hits, 0, 1)


def metrics(p, frame):
    slices = groups(frame.race_id)
    y = frame.y.to_numpy(float)
    rows = []
    for s in slices:
        g, v, target = frame.iloc[s], p[s], y[s]
        order = np.argsort(-v, kind="stable")
        rows.append({"race_id": g.race_id.iloc[0], "meeting_id": g.meeting_id.iloc[0],
                     "log_loss": -float(target @ np.log(np.clip(v, 1e-15, 1))),
                     "brier": float(np.square(v-target).sum()),
                     "top_pick_won": float(target[order[0]] > 0),
                     "winner_in_top3": float(target[order[:3]].sum()),
                     "top_pick_probability": float(v[order[0]])})
    summary = {k: float(np.mean([r[k] for r in rows])) for k in
               ("log_loss", "brier", "top_pick_won", "winner_in_top3")}
    summary["races"] = len(rows)
    # Per-runner reliability, explicitly distinct from top-pick calibration.
    summary["reliability"] = []
    for low, high in zip(np.linspace(0, 1, 11)[:-1], np.linspace(0, 1, 11)[1:]):
        mask = (p >= low) & (p < high if high < 1 else p <= high)
        if mask.any():
            summary["reliability"].append({"lower": float(low), "upper": float(high),
                "runners": int(mask.sum()), "predicted": float(p[mask].mean()), "observed": float(y[mask].mean())})
    return summary, rows
