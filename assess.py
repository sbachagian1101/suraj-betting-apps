"""Tuning, calibration, and the disagreement flag.

Three jobs the fitting core does not do.

**Tuning per league.** The shipped defaults were tuned on a ten-team Latvian
league. On a twenty-team Dutch one they rank 24th of 27 settings tried - the
shrinkage in particular wants to be ten times stronger. Every league gets its
own search, on its own data, with an inner time split so the tuning never sees
the matches it is judged on.

**Calibration.** A model can rank well and still lie about magnitudes. Saying
60% when it happens 45% of the time is a different failure from picking the
wrong team, and only a reliability table shows it.

**The disagreement flag, which is inverted from what anyone expects.** The
obvious idea is that where the model departs from the market it has found
something. Measured on 723 Dutch matches, the opposite holds: in the
75th-90th percentile of disagreement the model is **+0.0730 log-loss worse**
than the market (95% CI +0.0275 to +0.1164, excluding zero), and where the two
name different favourites the market is right 41.5% against the model's 36.8%.

So a large disagreement is evidence that **the model** is wrong. The flag is a
confidence warning, not a tip, and this module computes the evidence for that
claim from whatever data is loaded rather than asserting it.
"""
from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

import soccer_model as sm

OUTCOME = {"H": 0, "D": 1, "A": 2}

# Searched per league. Deliberately coarse: with a few hundred matches a fine
# grid would be fitting the validation split rather than choosing a setting.
GRID_XI = (0.0005, 0.002, 0.005)
GRID_REG = (0.005, 0.02, 0.05)
GRID_RESPONSE = ((0.25, 0.25), (0.0, 0.0), (0.35, 0.15))


def _fit_score(fit_df, eval_df, **kw) -> tuple[float, int]:
    """Mean 1X2 log-loss of a single fit over a later block of matches."""
    try:
        model = sm.fit(fit_df, **kw)
    except (ValueError, KeyError):
        return float("inf"), 0
    tot, n = 0.0, 0
    for r in eval_df.itertuples():
        try:
            p = sm.predict(model, r.home_team_name, r.away_team_name)
        except KeyError:
            continue                      # team unseen before this point
        v = np.clip([p["home_win"], p["draw"], p["away_win"]], 1e-12, None)
        v = v / v.sum()
        res = "H" if r.hg > r.ag else ("A" if r.hg < r.ag else "D")
        tot -= np.log(v[OUTCOME[res]])
        n += 1
    return (tot / n if n else float("inf")), n


def tune(df: pd.DataFrame, frac: float = 0.75,
         progress=None) -> tuple[dict[str, float], pd.DataFrame]:
    """Search the grid on an inner time split. Returns the winner and the table.

    The split is by date, so the settings are chosen on matches that come after
    the ones they were fitted on - the same relationship they will have in use.
    """
    dates = sorted(df["date"].unique())
    if len(dates) < 20:
        return dict(xi=sm.XI, reg=sm.REG, w_xg=sm.W_XG, w_sot=sm.W_SOT), pd.DataFrame()
    cut = dates[int(len(dates) * frac)]
    tr, va = df[df["date"] < cut], df[df["date"] >= cut]

    rows, total = [], len(GRID_XI) * len(GRID_REG) * len(GRID_RESPONSE)
    k = 0
    for xi in GRID_XI:
        for reg in GRID_REG:
            for w_xg, w_sot in GRID_RESPONSE:
                ll, n = _fit_score(tr, va, xi=xi, reg=reg, w_xg=w_xg, w_sot=w_sot)
                rows.append({"xi": xi, "reg": reg, "w_xg": w_xg, "w_sot": w_sot,
                             "log_loss": ll, "n": n})
                k += 1
                if progress is not None:
                    progress(k / total)
    table = pd.DataFrame(rows).sort_values("log_loss").reset_index(drop=True)
    best = table.iloc[0]
    return (dict(xi=float(best.xi), reg=float(best.reg),
                 w_xg=float(best.w_xg), w_sot=float(best.w_sot)), table)


def default_params() -> dict[str, float]:
    return dict(xi=sm.XI, reg=sm.REG, w_xg=sm.W_XG, w_sot=sm.W_SOT)


# --------------------------------------------------------------------------
# market comparison
# --------------------------------------------------------------------------
def with_market(bt: pd.DataFrame) -> pd.DataFrame:
    """Attach de-vigged market probabilities and per-match log-losses.

    Matches without a complete, sane book are dropped rather than filled in: a
    missing price is not a price.
    """
    d = bt.dropna(subset=["odds_H", "odds_D", "odds_A"]).copy()
    d = d[(d.odds_H > 1) & (d.odds_D > 1) & (d.odds_A > 1)].reset_index(drop=True)
    if d.empty:
        return d
    P = d[["p_H", "p_D", "p_A"]].to_numpy(float)
    Mk = np.array([sm.devig_1x2(r.odds_H, r.odds_D, r.odds_A)
                   for r in d.itertuples()])
    y = np.array([OUTCOME[r] for r in d["result"]])
    idx = np.arange(len(d))
    d["m_H"], d["m_D"], d["m_A"] = Mk[:, 0], Mk[:, 1], Mk[:, 2]
    d["ll_model"] = -np.log(np.clip(P[idx, y], 1e-12, None))
    d["ll_market"] = -np.log(np.clip(Mk[idx, y], 1e-12, None))
    d["disagreement"] = np.abs(P - Mk).max(axis=1)
    d["model_pick"] = P.argmax(axis=1)
    d["market_pick"] = Mk.argmax(axis=1)
    d["actual"] = y
    return d


def disagreement_table(d: pd.DataFrame, boots: int = 20000,
                       seed: int = 0) -> pd.DataFrame:
    """Model vs market, split by how far apart they are.

    The column that matters is `model_minus_market`: **positive means the model
    is worse**. If it grows with disagreement, a large gap is a warning about
    the model rather than a signal about the match.
    """
    if d.empty:
        return pd.DataFrame()
    rng = np.random.default_rng(seed)
    g = d["disagreement"].to_numpy()
    qs = np.quantile(g, [0.0, 0.5, 0.75, 0.9, 1.0])
    labels = ["smallest half", "50th-75th pct", "75th-90th pct", "top 10%"]
    rows = []
    for (lo, hi), lab in zip(zip(qs[:-1], qs[1:]), labels):
        s = (g >= lo) & (g <= hi)
        if s.sum() < 10:
            continue
        diff = (d["ll_model"] - d["ll_market"]).to_numpy()[s]
        bs = diff[rng.integers(0, len(diff), size=(boots, len(diff)))].mean(axis=1)
        lo_ci, hi_ci = np.percentile(bs, [2.5, 97.5])
        rows.append({
            "disagreement": lab, "matches": int(s.sum()),
            "model": float(d["ll_model"].to_numpy()[s].mean()),
            "market": float(d["ll_market"].to_numpy()[s].mean()),
            "model_minus_market": float(diff.mean()),
            "ci_low": float(lo_ci), "ci_high": float(hi_ci),
            "model_worse": bool(lo_ci > 0),
        })
    return pd.DataFrame(rows)


def pick_conflict(d: pd.DataFrame) -> dict[str, Any]:
    """How often the two name a different favourite, and who is right."""
    if d.empty:
        return {}
    dis = d["model_pick"].to_numpy() != d["market_pick"].to_numpy()
    if dis.sum() == 0:
        return {"conflicts": 0, "share": 0.0}
    y = d["actual"].to_numpy()
    return {
        "conflicts": int(dis.sum()),
        "share": float(dis.mean()),
        "model_right": float((d["model_pick"].to_numpy()[dis] == y[dis]).mean()),
        "market_right": float((d["market_pick"].to_numpy()[dis] == y[dis]).mean()),
        "neither": float(((d["model_pick"].to_numpy()[dis] != y[dis])
                          & (d["market_pick"].to_numpy()[dis] != y[dis])).mean()),
    }


def overall_gap(d: pd.DataFrame, boots: int = 20000, seed: int = 1) -> dict[str, Any]:
    if d.empty:
        return {}
    diff = (d["ll_model"] - d["ll_market"]).to_numpy()
    rng = np.random.default_rng(seed)
    bs = diff[rng.integers(0, len(diff), size=(boots, len(diff)))].mean(axis=1)
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return {"n": int(len(d)), "model": float(d["ll_model"].mean()),
            "market": float(d["ll_market"].mean()),
            "gap": float(diff.mean()), "ci_low": float(lo), "ci_high": float(hi),
            "market_ahead": bool(lo > 0)}


# --------------------------------------------------------------------------
# calibration
# --------------------------------------------------------------------------
def calibration_table(bt: pd.DataFrame, bins=(0, .1, .2, .3, .4, .55, 1.0)
                      ) -> pd.DataFrame:
    """Reliability over all three outcomes pooled.

    Every match contributes three (probability, happened) pairs, which is the
    only way to see the whole distribution rather than just the favourite.
    """
    if bt.empty:
        return pd.DataFrame()
    p, hit = [], []
    for r in bt.itertuples():
        y = OUTCOME[r.result]
        for i, v in enumerate((r.p_H, r.p_D, r.p_A)):
            p.append(v)
            hit.append(int(i == y))
    d = pd.DataFrame({"p": p, "hit": hit})
    d["band"] = pd.cut(d["p"], list(bins))
    g = d.groupby("band", observed=True).agg(
        forecasts=("hit", "size"), predicted=("p", "mean"),
        actual=("hit", "mean")).reset_index()
    g["band"] = g["band"].astype(str)
    g["error"] = g["actual"] - g["predicted"]
    return g


def calibration_error(bt: pd.DataFrame) -> float:
    """Forecast-weighted mean absolute gap between predicted and actual."""
    t = calibration_table(bt)
    if t.empty:
        return float("nan")
    return float(np.average(np.abs(t["error"]), weights=t["forecasts"]))


# --------------------------------------------------------------------------
# the flag, for a single upcoming fixture
# --------------------------------------------------------------------------
def flag(pred: dict[str, Any], odds: tuple[float, float, float] | None,
         table: pd.DataFrame | None = None) -> dict[str, Any] | None:
    """Compare one prediction with the market and rate the disagreement.

    Returns None when there is no usable book. `level` is 'aligned', 'watch' or
    'caution' - never anything that sounds like a recommendation, because the
    measured evidence says a large gap means the model is the unreliable one.
    """
    if not odds or any(o is None or not np.isfinite(o) or o <= 1 for o in odds):
        return None
    mk = sm.devig_1x2(*odds)
    p = np.array([pred["home_win"], pred["draw"], pred["away_win"]], dtype=float)
    gap = float(np.abs(p - mk).max())
    names = [pred["home"], "Draw", pred["away"]]

    cut_watch, cut_caution = 0.05, 0.10
    if table is not None and not table.empty and "disagreement" in table.columns:
        pass                                   # thresholds stay fixed and simple
    level = ("aligned" if gap < cut_watch else
             ("watch" if gap < cut_caution else "caution"))
    i = int(np.argmax(np.abs(p - mk)))
    return {
        "gap": gap, "level": level,
        "outcome": names[i],
        "model_p": float(p[i]), "market_p": float(mk[i]),
        "direction": "higher" if p[i] > mk[i] else "lower",
        "model": {n: float(v) for n, v in zip(names, p)},
        "market": {n: float(v) for n, v in zip(names, mk)},
        "same_favourite": bool(np.argmax(p) == np.argmax(mk)),
    }


FLAG_TEXT = {
    "aligned": "The model and the market broadly agree here. That is the "
               "situation the model is most reliable in.",
    "watch": "A moderate disagreement. Worth understanding before you act on "
             "either number.",
    "caution": "A large disagreement — and on this league's history that is a "
               "warning about **the model**, not a signal about the match. "
               "Where the two are furthest apart the model has been clearly "
               "worse than the market. Read it as: the market probably knows "
               "something this model cannot see, usually team news.",
}
