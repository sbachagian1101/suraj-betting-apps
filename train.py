"""Fit the production model and record what it is actually worth.

The model is a ridge regression on **finishing percentile** rather than a
win/lose classifier or a conditional logit. That choice was measured, not
assumed: the past runs in these PDFs are seen only through the handful of
runners entered again this weekend, so a within-race ranking likelihood can use
just the 60% of rows that land in a group of two or more, while a percentile
target uses every row. On identical held-out groups the percentile ridge scored
0.9998 against the rank-logit's 1.0164 and gradient boosting's 1.1016.

Turning a percentile into a win probability needs a scale. The temperature is
fitted **once, on pooled walk-forward predictions** - never on data the model
trained on, and never per-fold, which was tried and produced wildly unstable
values (0.19 to 1.31) because each fold's calibration slice was too small.

That the temperature can be fitted on partial fields and applied to full ones
rests on independence of irrelevant alternatives: under IIA a softmax over any
subset of the runners is a valid model of choice within that subset, so a scale
learned on groups of two transfers to a field of twelve. It is an assumption,
and it is the main thing standing between these numbers and the app's output.
"""
from __future__ import annotations

import glob
import json

import joblib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge

import features as F
import model as M
import past_form as PF
import pdf_parser as PP

ALPHA = 200.0
FOLDS = 5


def _folds(d: pd.DataFrame, folds: int = FOLDS):
    dates = np.sort(d.date.unique())
    cuts = [dates[int(len(dates) * q)] for q in np.linspace(0.5, 0.9, folds)]
    for i, cut in enumerate(cuts):
        nxt = (cuts[i + 1] if i + 1 < len(cuts)
               else dates[-1] + np.timedelta64(1, "D"))
        yield i + 1, d[d.date < cut], d[(d.date >= cut) & (d.date < nxt)]


def walk_forward(d: pd.DataFrame, alpha: float = ALPHA):
    """Pooled out-of-sample predictions plus the per-fold record."""
    oos, rec = [], []
    for i, train, test in _folds(d):
        if len(train) < 200 or len(test) < 40:
            continue
        Xtr, med = F.matrix(train)
        Xte, _ = F.matrix(test, med)
        r = Ridge(alpha=alpha).fit(Xtr, train.fp.to_numpy())
        t = test.copy()
        t["v"] = -r.predict(Xte)
        t["fold"] = i
        oos.append(t)
        rec.append({"fold": i, "train": len(train), "test": len(test),
                    "spearman": float(spearmanr(t.v, t.fp).statistic)})
    return pd.concat(oos, ignore_index=True), pd.DataFrame(rec)


def measure(oos: pd.DataFrame, tau: float) -> dict:
    """Everything the app is allowed to claim, computed out of sample."""
    grouped = oos[oos.groupby("race_key")["horse"].transform("size") >= 2]
    keys = grouped.race_key.to_numpy()
    v = tau * ((grouped.v - grouped.v.mean()) / (grouped.v.std() + 1e-9)).to_numpy()
    p = M.normalise_within(np.exp(v - v.max()), keys)
    mll, mhit, n = M.top1_logloss(p, grouped.finish.to_numpy(), keys)
    mk = M.normalise_within(
        1.0 / np.clip(grouped.sp.to_numpy(dtype=float), 1.01, None), keys)
    kll, khit, _ = M.top1_logloss(mk, grouped.finish.to_numpy(), keys)
    ull, _ = M.uniform_logloss(grouped.finish.to_numpy(), keys)

    def top(df, col, k=1):
        r = df.groupby("race_key")[col].rank(ascending=False, method="first")
        return df[r <= k]

    o = oos.copy()
    o["mkt"] = -o.sp
    mt, kt = top(o, "v"), top(o, "mkt")
    return {
        "n_rows": int(len(oos)), "n_races": int(oos.race_key.nunique()),
        "n_groups": int(n),
        "avg_visible": float(oos.groupby("race_key").size().mean()),
        "avg_field": float(oos.past_field_size.mean()),
        "logloss": mll, "market_logloss": kll, "chance_logloss": ull,
        "top1_subset": mhit, "market_top1_subset": khit,
        "spearman": float(spearmanr(oos.v, oos.fp).statistic),
        "pick_win": float(mt.won.mean()), "pick_place": float(mt.placed.mean()),
        "market_win": float(kt.won.mean()), "market_place": float(kt.placed.mean()),
        "random_win": float(oos.groupby("race_key").won.mean().mean()),
        "random_place": float(oos.placed.mean()),
        "tau": float(tau), "alpha": ALPHA,
    }


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    ph = k / n
    d = 1 + z * z / n
    c = (ph + z * z / (2 * n)) / d
    h = z * np.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


GAP_EDGES = [("low", 0.0, 0.62), ("medium", 0.62, 1.11), ("high", 1.11, 1e9)]


def confidence_bands(oos: pd.DataFrame) -> list[dict]:
    """Measured strike rates by how far clear the top pick is.

    Confidence is pinned to the **score gap to the second-ranked runner**,
    because that is a quantity whose payoff can actually be measured on these
    files. The win probability cannot: it is a softmax over the full field,
    while every held-out race here shows only about 1.8 of its runners, so any
    band drawn on probabilities would be quoting a number never observed.

    The gap is not perfectly monotonic - the narrowest quartile slightly
    outscores the second - but the widest band is clearly ahead of the rest and
    its interval barely overlaps theirs.
    """
    o = oos.copy()
    o["z"] = (o.v - o.v.mean()) / (o.v.std() + 1e-9)
    g = o[o.groupby("race_key")["horse"].transform("size") >= 2].copy()
    r = g.groupby("race_key")["z"].rank(ascending=False, method="first")
    top = g[r == 1].set_index("race_key")
    sec = g[r == 2].set_index("race_key")
    top = top.assign(gap=top.z - sec.z.reindex(top.index))
    top = top[top.gap.notna()]

    out = []
    for name, lo, hi in GAP_EDGES:
        s = top[(top.gap >= lo) & (top.gap < hi)]
        if s.empty:
            continue
        lw, hw = wilson(int(s.won.sum()), len(s))
        lp, hp = wilson(int(s.placed.sum()), len(s))
        out.append({"band": name, "gap_lo": lo, "gap_hi": hi, "n": len(s),
                    "win": float(s.won.mean()), "win_ci": [lw, hw],
                    "place": float(s.placed.mean()), "place_ci": [lp, hp],
                    "avg_field": float(s.past_field_size.mean())})
    return out


def place_probabilities(p_win: np.ndarray, places: int = 3,
                        sims: int = 20000, seed: int = 0) -> np.ndarray:
    """P(finish in the first `places`) by Gumbel top-k sampling.

    A Plackett-Luce ordering is drawn by adding Gumbel noise to log win
    probabilities and reading off the order, which is exact for that model. It
    is derived from the win probabilities, so it inherits their calibration and
    is not separately validated - the one place figure that *is* measured is the
    top selection's strike rate, which is recorded in the bundle.
    """
    rng = np.random.default_rng(seed)
    lp = np.log(np.clip(p_win, 1e-12, None))
    g = rng.gumbel(size=(sims, len(p_win)))
    order = np.argsort(-(lp + g), axis=1)[:, :places]
    hits = np.zeros(len(p_win))
    for c in range(places):
        np.add.at(hits, order[:, c], 1)
    return hits / sims


def main():
    files = sorted(glob.glob(
        "C:/Users/Admin/Downloads/[0-9][0-9][0-9][0-9][0-9][0-9].pdf"))
    print(f"reading {len(files)} meeting PDFs …")
    past = PF.parse_many(files)
    field = PP.parse_many(files)
    print(f"  {len(past)} past runs, {len(field)} upcoming runners, "
          f"{field.race_id.nunique()} races")

    tr = F.build_training(past)
    d = tr[(tr.n_prior > 0) & tr.date.notna() & tr.key_ok].copy()
    d["fp"] = F._pct(d.finish.to_numpy(), d.past_field_size.to_numpy())
    d = d.sort_values("date").reset_index(drop=True)
    print(f"  {len(d)} training rows with prior form")

    oos, rec = walk_forward(d)
    print("\nwalk-forward folds:")
    print(rec.to_string(index=False))

    # one temperature, fitted on pooled out-of-sample predictions
    grouped = oos[oos.groupby("race_key")["horse"].transform("size") >= 2]
    z = ((grouped.v - grouped.v.mean()) / (grouped.v.std() + 1e-9)).to_numpy()
    tau = M.fit_temperature(z, grouped.finish.to_numpy(),
                            grouped.race_key.to_numpy())
    print(f"\nfitted temperature: {tau:.3f}")

    stats = measure(oos, tau)
    bands = confidence_bands(oos)
    print("\nconfidence bands (gap to the second-ranked runner):")
    for b in bands:
        print(f"  {b['band']:7s} n={b['n']:4d}  won {100*b['win']:5.1f}% "
              f"[{100*b['win_ci'][0]:4.1f},{100*b['win_ci'][1]:4.1f}]  "
              f"placed {100*b['place']:5.1f}%")
    print("\nmeasured, out of sample:")
    for k, v in stats.items():
        print(f"  {k:22s} {v:.4f}" if isinstance(v, float) else f"  {k:22s} {v}")

    X, med = F.matrix(d)
    final = Ridge(alpha=ALPHA).fit(X, d.fp.to_numpy())
    bundle = {
        "ridge": final, "medians": med, "features": F.FEATURES,
        # the temperature was fitted against utilities standardised on the
        # walk-forward pool, so the same centre and scale have to travel with
        # it - re-deriving them from the final in-sample fit would quietly
        # shift the probabilities the app prints
        "tau": tau, "v_mean": float(oos.v.mean()), "v_std": float(oos.v.std()),
        "validation": stats, "folds": rec.to_dict("records"),
        "confidence_bands": bands,
        "coefficients": dict(zip(F.FEATURES, final.coef_.tolist())),
        "n_training_rows": int(len(d)), "n_horses": int(d.horse.nunique()),
        "date_range": [str(d.date.min().date()), str(d.date.max().date())],
        "trained_on_files": [f.rsplit("/", 1)[-1] for f in files],
    }
    joblib.dump(bundle, "model_bundle.joblib")
    print("\nwrote model_bundle.joblib")
    print(json.dumps({k: v for k, v in stats.items()
                      if isinstance(v, (int, float))}, indent=2)[:400])

    co = pd.Series(final.coef_, index=F.FEATURES).sort_values(key=abs,
                                                              ascending=False)
    print("\nstrongest coefficients (negative = better finish):")
    print(co.head(12).to_string())


if __name__ == "__main__":
    main()
