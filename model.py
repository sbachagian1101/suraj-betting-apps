"""Rank-ordered conditional logit for partial fields.

Each past race in these PDFs is seen only through the horses that happen to be
entered at this weekend's meetings - typically 2 to 5 runners out of a field of
11. That is exactly the situation the conditional logit is built for: under
independence of irrelevant alternatives the likelihood of an ordering **within
any subset** of the alternatives is valid on its own, so a partial field costs
information but does not bias the estimates.

The Plackett-Luce form is used rather than "who won", because the full ordering
inside the subset carries more signal than the top place alone - and in most of
these groups the actual race winner isn't among the horses we can see, so a
plain win-indicator likelihood would have nothing to condition on::

    LL = sum_k [ v_(k) - log sum_{j >= k} exp(v_(j)) ]

with runners indexed in finishing order. Fitted by L-BFGS on the exact
gradient, with an L2 penalty chosen by the caller.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize


def _groups(race_keys: np.ndarray) -> list[np.ndarray]:
    order = np.argsort(race_keys, kind="stable")
    keys = race_keys[order]
    bounds = np.flatnonzero(np.r_[True, keys[1:] != keys[:-1], True])
    return [order[bounds[i]:bounds[i + 1]] for i in range(len(bounds) - 1)]


class RankLogit:
    """Plackett-Luce conditional logit fitted on partial fields."""

    def __init__(self, l2: float = 1.0):
        self.l2 = l2
        self.beta_: np.ndarray | None = None
        self.mean_: np.ndarray | None = None
        self.scale_: np.ndarray | None = None

    # ---------------------------------------------------------------- fitting
    def _standardise(self, X, fit=False):
        if fit:
            self.mean_ = X.mean(axis=0)
            s = X.std(axis=0)
            self.scale_ = np.where(s > 1e-9, s, 1.0)
        return (X - self.mean_) / self.scale_

    def fit(self, X, finish, race_keys):
        X = np.asarray(X, dtype=float)
        Z = self._standardise(X, fit=True)
        finish = np.asarray(finish, dtype=float)

        # keep the groups worth fitting, each ordered best-finisher first
        blocks = []
        for idx in _groups(np.asarray(race_keys)):
            if len(idx) < 2:
                continue
            blocks.append(idx[np.argsort(finish[idx], kind="stable")])
        if not blocks:
            raise ValueError("no group has two or more runners to rank")
        self.n_groups_ = len(blocks)
        self.n_runners_ = sum(len(b) for b in blocks)

        def objective(beta):
            v = Z @ beta
            ll = 0.0
            grad = np.zeros_like(beta)
            for b in blocks:
                vb, Zb = v[b], Z[b]
                n = len(b)
                m = vb.max()
                e = np.exp(vb - m)
                tail = np.cumsum(e[::-1])[::-1]      # tail[k] = sum_{j>=k} e_j
                # the last stage contributes v - log(exp(v)) = 0, so stop at n-1
                ll += float(np.sum(vb[:-1] - m - np.log(tail[:-1])))
                grad += Zb[:-1].sum(axis=0)
                for k in range(n - 1):
                    grad -= (e[k:] / tail[k]) @ Zb[k:]
            ll -= 0.5 * self.l2 * float(beta @ beta)
            grad -= self.l2 * beta
            return -ll, -grad

        beta0 = np.zeros(Z.shape[1])
        res = minimize(objective, beta0, jac=True, method="L-BFGS-B",
                       options={"maxiter": 800, "ftol": 1e-12, "gtol": 1e-9})
        self.beta_ = res.x
        self.result_ = res
        return self

    # --------------------------------------------------------------- scoring
    def utility(self, X) -> np.ndarray:
        Z = (np.asarray(X, dtype=float) - self.mean_) / self.scale_
        return Z @ self.beta_

    def probabilities(self, X, race_keys=None) -> np.ndarray:
        v = self.utility(X)
        if race_keys is None:
            return _softmax(v)
        p = np.empty_like(v)
        for idx in _groups(np.asarray(race_keys)):
            p[idx] = _softmax(v[idx])
        return p


def _softmax(v):
    e = np.exp(v - np.max(v))
    return e / e.sum()


def fit_temperature(v, finish, race_keys, lo=0.05, hi=8.0):
    """Scale utilities so the probabilities are calibrated, not just ordered.

    Strong L2 is what keeps the ranking honest on 1,300-odd small groups, but it
    also shrinks the utilities toward zero, which pushes every probability
    toward uniform and leaves the log-loss barely better than chance even when
    the *order* is good. Rank and scale are separate problems, so they get
    separated: beta is fitted under whatever penalty ranks best, then a single
    temperature is fitted on held-out groups by golden-section search on the
    same grouped log-loss the model is judged by.
    """
    v = np.asarray(v, dtype=float)
    finish = np.asarray(finish, dtype=float)
    blocks = [idx for idx in _groups(np.asarray(race_keys)) if len(idx) >= 2]
    if not blocks:
        return 1.0

    def loss(t):
        tot = 0.0
        for b in blocks:
            u = t * v[b]
            u -= u.max()
            e = np.exp(u)
            best = int(np.argmin(finish[b]))
            tot -= u[best] - np.log(e.sum())
        return tot / len(blocks)

    phi = (np.sqrt(5.0) - 1.0) / 2.0
    a, b = lo, hi
    c, d = b - phi * (b - a), a + phi * (b - a)
    fc, fd = loss(c), loss(d)
    for _ in range(60):
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - phi * (b - a)
            fc = loss(c)
        else:
            a, c, fc = c, d, fd
            d = a + phi * (b - a)
            fd = loss(d)
        if abs(b - a) < 1e-6:
            break
    return float((a + b) / 2.0)


def normalise_within(values, race_keys):
    """Turn positive scores into within-race probabilities."""
    values = np.asarray(values, dtype=float)
    out = np.empty_like(values)
    for idx in _groups(np.asarray(race_keys)):
        s = values[idx].sum()
        out[idx] = values[idx] / s if s > 0 else 1.0 / len(idx)
    return out


# ----------------------------------------------------------------- metrics
def top1_logloss(p, finish, race_keys):
    """Log-loss of the best-finisher-in-subset event, and top-1 hit rate."""
    p = np.asarray(p, dtype=float)
    finish = np.asarray(finish, dtype=float)
    lls, hits, n = [], [], 0
    for idx in _groups(np.asarray(race_keys)):
        if len(idx) < 2:
            continue
        best = idx[np.argmin(finish[idx])]
        q = np.clip(p[idx], 1e-12, 1.0)
        q = q / q.sum()
        lls.append(-np.log(q[list(idx).index(best)]))
        hits.append(int(idx[np.argmax(p[idx])] == best))
        n += 1
    return float(np.mean(lls)), float(np.mean(hits)), n


def uniform_logloss(finish, race_keys):
    lls = []
    for idx in _groups(np.asarray(race_keys)):
        if len(idx) < 2:
            continue
        lls.append(np.log(len(idx)))
    return float(np.mean(lls)), len(lls)
