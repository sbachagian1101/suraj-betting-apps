"""The models.

Four different mathematical objects, combined. They are not four flavours of the
same thing - each optimises a genuinely different criterion, which is why
blending them is worth anything at all.

**Conditional logit** (Bolton & Chapman, 1986). The likelihood that actually
matches the question: exactly one horse wins each race, so the probability of a
runner is a softmax over its own field. Fitted by maximising the grouped
log-likelihood directly, with an L2 penalty. It is linear, so it cannot overfit
the way a tree can, and it produces calibrated probabilities by construction.

**Gradient-boosted classifier.** Binary win/lose per runner, then renormalised
within the race. Non-linear and catches interactions the logit cannot, at the
cost of needing the renormalisation step to become a proper race distribution.

**LambdaRank.** Optimises the *ordering* within a race rather than a
per-runner probability. It does not care about absolute calibration, which makes
its mistakes different from the other two - the useful property in an ensemble.

**Plackett-Luce.** Not a fitted model: given win probabilities it produces
place and top-N probabilities by sampling finishing orders, which is the correct
way to go from "who wins" to "who runs top three".

The blend is a weighted geometric mean in log space - the standard way to
combine probability forecasts, and the form Benter (1994) uses for the
market/fundamental combination.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy.optimize import minimize

warnings.filterwarnings("ignore")

try:
    import lightgbm as lgb
    HAVE_LGB = True
except ImportError:                                    # pragma: no cover
    HAVE_LGB = False


# --------------------------------------------------------------------------
# grouping helpers
# --------------------------------------------------------------------------
def race_slices(race_ids: np.ndarray) -> list[slice]:
    """Contiguous slices, one per race. Requires rows grouped by race."""
    out, start = [], 0
    for i in range(1, len(race_ids) + 1):
        if i == len(race_ids) or race_ids[i] != race_ids[start]:
            out.append(slice(start, i))
            start = i
    return out


def softmax_by_race(scores: np.ndarray, slices: list[slice]) -> np.ndarray:
    p = np.empty_like(scores, dtype=float)
    for s in slices:
        v = scores[s]
        e = np.exp(v - v.max())
        p[s] = e / e.sum()
    return p


def normalise_by_race(p: np.ndarray, slices: list[slice]) -> np.ndarray:
    out = np.empty_like(p, dtype=float)
    for s in slices:
        v = np.clip(p[s], 1e-12, None)
        out[s] = v / v.sum()
    return out


# --------------------------------------------------------------------------
# 1. conditional logit
# --------------------------------------------------------------------------
class ConditionalLogit:
    """Race-conditional multinomial logit, fitted by L-BFGS on the exact
    grouped log-likelihood.

    For race r with winner w:  log P = x_w·b - log sum_j exp(x_j·b)
    """

    def __init__(self, l2: float = 1.0, max_iter: int = 400):
        self.l2 = float(l2)
        self.max_iter = int(max_iter)
        self.beta_: np.ndarray | None = None

    def fit(self, X, y_win, race_ids):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y_win, dtype=float)
        sl = race_slices(np.asarray(race_ids))
        n_races = len(sl)

        def obj(b):
            s = X @ b
            nll = 0.0
            grad = np.zeros_like(b)
            for sc in sl:
                v = s[sc]
                mx = v.max()
                e = np.exp(v - mx)
                Z = e.sum()
                p = e / Z
                yi = y[sc]
                nll -= (v[yi == 1][0] - (mx + np.log(Z)))
                grad += X[sc].T @ (p - yi)
            nll = nll / n_races + 0.5 * self.l2 * b @ b
            grad = grad / n_races + self.l2 * b
            return nll, grad

        b0 = np.zeros(X.shape[1])
        res = minimize(obj, b0, jac=True, method="L-BFGS-B",
                       options={"maxiter": self.max_iter})
        self.beta_ = res.x
        self.converged_ = bool(res.success)
        return self

    def decision(self, X):
        return np.asarray(X, dtype=float) @ self.beta_

    def predict_proba(self, X, race_ids):
        return softmax_by_race(self.decision(X), race_slices(np.asarray(race_ids)))


# --------------------------------------------------------------------------
# 2 & 3. boosted models
# --------------------------------------------------------------------------
def fit_gbm_classifier(X, y_win, seed=0, **kw):
    params = dict(n_estimators=700, learning_rate=0.03, num_leaves=31,
                  min_child_samples=60, subsample=0.8, subsample_freq=1,
                  colsample_bytree=0.7, reg_lambda=5.0, random_state=seed,
                  verbose=-1)
    params.update(kw)
    m = lgb.LGBMClassifier(**params)
    m.fit(X, y_win)
    return m


def fit_ranker(X, finish, race_ids, seed=0, **kw):
    """LambdaRank. Relevance is a small ladder off the finishing position -
    winning is worth more than placing, placing more than the rest."""
    sl = race_slices(np.asarray(race_ids))
    group = [s.stop - s.start for s in sl]
    fin = np.asarray(finish)
    rel = np.where(fin == 1, 4, np.where(fin == 2, 3,
                   np.where(fin == 3, 2, np.where(fin <= 5, 1, 0))))
    params = dict(objective="lambdarank", n_estimators=700, learning_rate=0.03,
                  num_leaves=31, min_child_samples=60, colsample_bytree=0.7,
                  reg_lambda=5.0, random_state=seed, verbose=-1,
                  label_gain=list(range(32)))
    params.update(kw)
    m = lgb.LGBMRanker(**params)
    m.fit(X, rel, group=group)
    return m


# --------------------------------------------------------------------------
# blending
# --------------------------------------------------------------------------
def blend_log(probs: list[np.ndarray], weights: np.ndarray,
              slices: list[slice]) -> np.ndarray:
    """Weighted geometric mean, renormalised within each race."""
    w = np.asarray(weights, dtype=float)
    w = w / w.sum()
    lp = sum(wi * np.log(np.clip(p, 1e-12, None)) for wi, p in zip(w, probs))
    return softmax_by_race(lp, slices)


def fit_blend_weights(probs: list[np.ndarray], y_win: np.ndarray,
                      slices: list[slice], grid: int = 9) -> np.ndarray:
    """Choose blend weights by grid search on held-out log-loss.

    Coarse on purpose: with a few thousand races a fine search would be fitting
    the validation fold rather than choosing a weight.
    """
    k = len(probs)
    best, best_w = np.inf, np.ones(k) / k
    for w in _simplex_grid(k, grid):
        ll = race_logloss(blend_log(probs, w, slices), y_win, slices)
        if ll < best:
            best, best_w = ll, w
    return best_w


def _simplex_grid(k: int, n: int):
    """Every weight vector on the k-simplex at resolution 1/n.

    Enumerated as integer compositions of n into k parts, which covers the
    simplex evenly - a recursive rescaling does not, and quietly samples the
    corners more densely than the middle.
    """
    def comps(k, n):
        if k == 1:
            yield (n,)
            return
        for i in range(n + 1):
            for rest in comps(k - 1, n - i):
                yield (i,) + rest

    for c in comps(k, n):
        yield np.asarray(c, dtype=float) / n


# --------------------------------------------------------------------------
# 4. Plackett-Luce
# --------------------------------------------------------------------------
def place_probabilities(p_win: np.ndarray, slices: list[slice],
                        places: np.ndarray, sims: int = 20000,
                        seed: int = 0) -> np.ndarray:
    """P(finishing inside the paying places), by Gumbel top-k sampling.

    Sampling a full finishing order respects the fact that exactly one horse can
    be first - which is precisely what independent per-horse draws get wrong.
    """
    rng = np.random.default_rng(seed)
    out = np.zeros_like(p_win, dtype=float)
    for s in slices:
        p = np.clip(p_win[s], 1e-12, None)
        p = p / p.sum()
        n = len(p)
        k = int(np.clip(places[s][0], 1, n))
        if n == 1:
            out[s] = 1.0
            continue
        g = rng.gumbel(size=(sims, n))
        order = np.argsort(-(np.log(p) + g), axis=1)[:, :k]
        hits = np.zeros(n)
        np.add.at(hits, order.ravel(), 1.0)
        out[s] = hits / sims
    return out


# --------------------------------------------------------------------------
# metrics
# --------------------------------------------------------------------------
def race_logloss(p, y_win, slices):
    tot = 0.0
    for s in slices:
        pi = np.clip(p[s], 1e-12, None)
        pi = pi / pi.sum()
        tot -= np.log(pi[y_win[s] == 1][0])
    return tot / len(slices)


def rps(p, finish, slices):
    """Ranked probability score on the win distribution, ordered by finish."""
    tot = 0.0
    for s in slices:
        pi = np.clip(p[s], 1e-12, None)
        pi = pi / pi.sum()
        order = np.argsort(finish[s])
        cp = np.cumsum(pi[order])
        obs = np.cumsum((finish[s][order] == 1).astype(float))
        tot += np.mean((cp - obs) ** 2)
    return tot / len(slices)


def top_k_hit(p, y_win, slices, k=1):
    hit = 0
    for s in slices:
        idx = np.argsort(-p[s])[:k]
        hit += int(y_win[s][idx].max() == 1)
    return hit / len(slices)
