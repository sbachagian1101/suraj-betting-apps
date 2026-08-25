"""Dixon-Coles style team-strength model for football match prediction.

Method
------
Each team gets an **attack** and a **defence** parameter. Expected goals for a
fixture are

    lambda_home = exp(mu + attack_home + defence_away + gamma)
    lambda_away = exp(mu + attack_away + defence_home)

with `gamma` the home advantage, fitted from the data rather than assumed.
Parameters are estimated by weighted maximum likelihood, and three choices
matter more than the structure itself:

1. **What counts as "attacking output".** Goals alone are noisy. The response
   fitted is a blend of goals, xG and shots-on-target-implied goals. The weights
   were chosen by walk-forward backtest, not by taste - see README.
2. **Time decay.** Matches are weighted `exp(-xi * days_ago)`. The backtest
   preferred a *very* light decay: in a ten-team league, older matches carry
   more information than recency-chasing gives them credit for.
3. **Low-score dependence.** Independent Poissons misprice 0-0, 1-0, 0-1 and
   1-1. The Dixon & Coles (1997) `tau` correction is fitted on the *actual*
   scorelines and applied to the score matrix.

Markets are then read off the joint score matrix, so 1X2, BTTS and Over/Under
are guaranteed mutually consistent - they come from one distribution.

References
----------
Dixon, M. & Coles, S. (1997) *Modelling Association Football Scores and
Inefficiencies in the Football Betting Market*, JRSS-C 46(2).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize

# Defaults chosen by walk-forward backtest on 309 Latvian Virsliga matches.
XI = 0.0005          # time-decay per day (very light)
REG = 0.005          # L2 shrink of attack/defence toward league average
W_XG = 0.25          # weight on xG in the fitted response
W_SOT = 0.25         # weight on shots-on-target-implied goals
MAX_GOALS = 11       # score-matrix truncation


@dataclass
class FittedModel:
    teams: list[str]
    index: dict[str, int]
    attack: np.ndarray
    defence: np.ndarray
    mu: float
    gamma: float
    rho: float
    n_matches: int
    conversion: float
    weights: dict[str, float] = field(default_factory=dict)

    @property
    def home_advantage_goals(self) -> float:
        """Home advantage expressed as extra goals for an average fixture."""
        base = math.exp(self.mu)
        return base * math.exp(self.gamma) - base


def _sot_conversion(df: pd.DataFrame) -> float:
    """League goals per shot on target, used to put SoT on a goals scale."""
    if not {"home_team_shots_on_target", "away_team_shots_on_target"} <= set(df.columns):
        return 0.0
    sot = pd.concat([df["home_team_shots_on_target"], df["away_team_shots_on_target"]])
    goals = pd.concat([df["hg"], df["ag"]])
    m = sot.notna() & (sot > 0)
    if m.sum() < 20:
        return 0.0
    return float(goals[m.values].sum() / sot[m].sum())


def _response(df: pd.DataFrame, w_xg: float, w_sot: float, conv: float):
    """Blended attacking output per team-match.

    Any component missing for a given match falls back to that match's goals, so
    the weights always sum to one and a team is never penalised for a data gap.
    """
    have_xg = {"team_a_xg", "team_b_xg"} <= set(df.columns)
    have_sot = ({"home_team_shots_on_target", "away_team_shots_on_target"} <= set(df.columns)
                and conv > 0)
    w_xg = w_xg if have_xg else 0.0
    w_sot = w_sot if have_sot else 0.0
    w_g = 1.0 - w_xg - w_sot
    hg, ag = df["hg"].astype(float), df["ag"].astype(float)
    yh, ya = w_g * hg, w_g * ag
    if w_xg:
        yh = yh + w_xg * df["team_a_xg"].fillna(hg)
        ya = ya + w_xg * df["team_b_xg"].fillna(ag)
    if w_sot:
        yh = yh + w_sot * (df["home_team_shots_on_target"] * conv).fillna(hg)
        ya = ya + w_sot * (df["away_team_shots_on_target"] * conv).fillna(ag)
    return yh.values, ya.values, {"goals": w_g, "xg": w_xg, "sot": w_sot}


def _tau(x, y, lh, la, rho):
    """Dixon-Coles correction for dependence between low scorelines."""
    t = np.ones(np.shape(x), dtype=float)
    t = np.where((x == 0) & (y == 0), 1 - lh * la * rho, t)
    t = np.where((x == 0) & (y == 1), 1 + lh * rho, t)
    t = np.where((x == 1) & (y == 0), 1 + la * rho, t)
    t = np.where((x == 1) & (y == 1), 1 - rho, t)
    return t


def fit(df: pd.DataFrame, asof: pd.Timestamp | None = None, xi: float = XI,
        reg: float = REG, w_xg: float = W_XG, w_sot: float = W_SOT) -> FittedModel:
    """Fit team strengths on every match strictly before `asof`."""
    d = df if asof is None else df[df["date"] < asof]
    if len(d) < 20:
        raise ValueError(f"Need at least 20 completed matches to fit a model; got {len(d)}.")
    teams = sorted(set(d["home_team_name"]) | set(d["away_team_name"]))
    index = {t: i for i, t in enumerate(teams)}
    n = len(teams)
    hi = d["home_team_name"].map(index).values
    ai = d["away_team_name"].map(index).values
    conv = _sot_conversion(d)
    yh, ya, weights = _response(d, w_xg, w_sot, conv)

    ref = d["date"].max() if asof is None else asof
    days = (ref - d["date"]).dt.total_seconds().values / 86400.0
    w = np.exp(-xi * np.clip(days, 0, None))

    def unpack(p):
        att = np.concatenate([p[:n - 1], [-p[:n - 1].sum()]])
        dfn = np.concatenate([p[n - 1:2 * n - 2], [-p[n - 1:2 * n - 2].sum()]])
        return att, dfn, p[-2], p[-1]

    def nll(p):
        att, dfn, mu, gamma = unpack(p)
        lh = np.exp(np.clip(mu + att[hi] + dfn[ai] + gamma, -8, 4))
        la = np.exp(np.clip(mu + att[ai] + dfn[hi], -8, 4))
        ll = w * (yh * np.log(lh) - lh + ya * np.log(la) - la)
        return -ll.sum() + reg * len(d) * (np.sum(att ** 2) + np.sum(dfn ** 2))

    p0 = np.zeros(2 * n)
    p0[-2] = math.log(max(float(np.mean(np.r_[yh, ya])), 0.1))
    res = minimize(nll, p0, method="L-BFGS-B")
    att, dfn, mu, gamma = unpack(res.x)

    model = FittedModel(teams=teams, index=index, attack=att, defence=dfn,
                        mu=float(mu), gamma=float(gamma), rho=0.0,
                        n_matches=len(d), conversion=conv, weights=weights)
    model.rho = _fit_rho(d, model, w)
    return model


def _fit_rho(d: pd.DataFrame, model: FittedModel, w: np.ndarray) -> float:
    lh = np.empty(len(d)); la = np.empty(len(d))
    for k, (h, a) in enumerate(zip(d["home_team_name"], d["away_team_name"])):
        lh[k], la[k] = expected_goals(model, h, a)
    x, y = d["hg"].values, d["ag"].values

    def nll(r):
        return -np.sum(w * np.log(np.clip(_tau(x, y, lh, la, r[0]), 1e-9, None)))

    r = minimize(nll, [-0.05], method="L-BFGS-B", bounds=[(-0.35, 0.35)])
    return float(r.x[0])


def expected_goals(model: FittedModel, home: str, away: str) -> tuple[float, float]:
    i, j = model.index.get(home), model.index.get(away)
    if i is None:
        raise KeyError(f"{home!r} does not appear in the loaded data.")
    if j is None:
        raise KeyError(f"{away!r} does not appear in the loaded data.")
    lh = math.exp(model.mu + model.attack[i] + model.defence[j] + model.gamma)
    la = math.exp(model.mu + model.attack[j] + model.defence[i])
    return lh, la


def score_matrix(lh: float, la: float, rho: float, max_goals: int = MAX_GOALS) -> np.ndarray:
    k = np.arange(max_goals)
    fact = np.array([math.factorial(int(v)) for v in k], dtype=float)
    ph = np.exp(-lh) * lh ** k / fact
    pa = np.exp(-la) * la ** k / fact
    m = np.outer(ph, pa)
    x, y = np.meshgrid(k, k, indexing="ij")
    m = m * _tau(x, y, lh, la, rho)
    m = np.clip(m, 0, None)
    return m / m.sum()


def markets(m: np.ndarray) -> dict[str, float]:
    k = m.shape[0]
    x, y = np.meshgrid(np.arange(k), np.arange(k), indexing="ij")
    tot = x + y
    return {
        "home_win": float(m[x > y].sum()),
        "draw": float(m[x == y].sum()),
        "away_win": float(m[x < y].sum()),
        "btts_yes": float(m[1:, 1:].sum()),
        "btts_no": float(1 - m[1:, 1:].sum()),
        "over_25": float(m[tot >= 3].sum()),
        "under_25": float(m[tot <= 2].sum()),
        "over_15": float(m[tot >= 2].sum()),
        "over_35": float(m[tot >= 4].sum()),
        "home_cs": float(m[:, 0].sum()),       # home clean sheet
        "away_cs": float(m[0, :].sum()),
    }


def top_scorelines(m: np.ndarray, n: int = 8) -> list[tuple[int, int, float]]:
    flat = [(i, j, float(m[i, j])) for i in range(m.shape[0]) for j in range(m.shape[1])]
    flat.sort(key=lambda t: -t[2])
    return flat[:n]


def predict(model: FittedModel, home: str, away: str,
            max_goals: int = MAX_GOALS) -> dict[str, Any]:
    lh, la = expected_goals(model, home, away)
    m = score_matrix(lh, la, model.rho, max_goals)
    out = markets(m)
    out.update({"lambda_home": lh, "lambda_away": la, "matrix": m,
                "top_scorelines": top_scorelines(m), "home": home, "away": away})
    return out


# --------------------------------------------------------------------------
# optional market blending
# --------------------------------------------------------------------------
def devig_1x2(odds_home: float, odds_draw: float, odds_away: float) -> np.ndarray:
    q = np.array([1.0 / odds_home, 1.0 / odds_draw, 1.0 / odds_away], dtype=float)
    return q / q.sum()


def devig_pair(odds_yes: float, odds_no: float) -> float:
    qy, qn = 1.0 / odds_yes, 1.0 / odds_no
    return float(qy / (qy + qn))


def blend(model_p, market_p, w_market: float) -> np.ndarray:
    """Log-opinion pool of model and market probabilities."""
    mp = np.asarray(model_p, dtype=float)
    kp = np.asarray(market_p, dtype=float)
    lg = w_market * np.log(np.clip(kp, 1e-9, 1)) + (1 - w_market) * np.log(np.clip(mp, 1e-9, 1))
    q = np.exp(lg - lg.max())
    return q / q.sum()


# --------------------------------------------------------------------------
# walk-forward validation
# --------------------------------------------------------------------------
def walk_forward(df: pd.DataFrame, xi: float = XI, reg: float = REG,
                 w_xg: float = W_XG, w_sot: float = W_SOT,
                 min_train: int = 60, progress=None) -> pd.DataFrame:
    """Re-fit before every matchday and predict it. No match sees its own result.

    This is the only honest way to quote accuracy: fitting and scoring on the
    same matches would flatter the model badly.
    """
    rows = []
    dates = sorted(df["date"].unique())
    for k, day in enumerate(dates):
        past = df[df["date"] < day]
        if len(past) < min_train:
            continue
        try:
            model = fit(df, asof=pd.Timestamp(day), xi=xi, reg=reg, w_xg=w_xg, w_sot=w_sot)
        except ValueError:
            continue
        for _, m in df[df["date"] == day].iterrows():
            try:
                p = predict(model, m["home_team_name"], m["away_team_name"])
            except KeyError:
                continue                      # team never seen before this date
            rows.append({
                "date": m["date"], "home": m["home_team_name"], "away": m["away_team_name"],
                "hg": m["hg"], "ag": m["ag"],
                "result": "H" if m.hg > m.ag else ("A" if m.hg < m.ag else "D"),
                "p_H": p["home_win"], "p_D": p["draw"], "p_A": p["away_win"],
                "p_btts": p["btts_yes"], "btts": int(m.hg > 0 and m.ag > 0),
                "p_o25": p["over_25"], "o25": int(m.hg + m.ag > 2.5),
                "odds_H": m.get("odds_ft_home_team_win", np.nan),
                "odds_D": m.get("odds_ft_draw", np.nan),
                "odds_A": m.get("odds_ft_away_team_win", np.nan),
            })
        if progress is not None:
            progress((k + 1) / max(len(dates), 1))
    return pd.DataFrame(rows)


def _logloss_multi(P, res):
    col = {"H": 0, "D": 1, "A": 2}
    p = np.clip([P[i, col[r]] for i, r in enumerate(res)], 1e-12, 1)
    return float(-np.mean(np.log(p)))


def _rps(P, res):
    """Ranked probability score - the standard ordered metric for 1X2."""
    col = {"H": 0, "D": 1, "A": 2}
    tot = 0.0
    for i, r in enumerate(res):
        o = np.zeros(3); o[col[r]] = 1
        tot += float(np.sum((np.cumsum(P[i]) - np.cumsum(o)) ** 2) / 2)
    return tot / len(res)


def _logloss_binary(p, y):
    p = np.asarray(p, dtype=float); y = np.asarray(y)
    return float(-np.mean(np.log(np.clip(np.where(y == 1, p, 1 - p), 1e-12, 1))))


def evaluate(bt: pd.DataFrame) -> dict[str, Any]:
    """Metrics for a walk-forward frame, with the bookmaker as the benchmark."""
    if bt.empty:
        return {}
    P = bt[["p_H", "p_D", "p_A"]].values
    res = bt["result"].tolist()
    pick = np.array(["H", "D", "A"])[P.argmax(1)]
    out = {
        "n": len(bt),
        "logloss_1x2": _logloss_multi(P, res),
        "rps_1x2": _rps(P, res),
        "acc_1x2": float(np.mean(np.array(res) == pick)),
        "logloss_btts": _logloss_binary(bt["p_btts"], bt["btts"]),
        "acc_btts": float(np.mean((bt["p_btts"] > 0.5) == (bt["btts"] == 1))),
        "logloss_o25": _logloss_binary(bt["p_o25"], bt["o25"]),
        "acc_o25": float(np.mean((bt["p_o25"] > 0.5) == (bt["o25"] == 1))),
    }
    # Naive baseline: always predict the league's own base rates.
    base = np.tile([[np.mean(np.array(res) == "H"), np.mean(np.array(res) == "D"),
                     np.mean(np.array(res) == "A")]], (len(bt), 1))
    out["logloss_baserate"] = _logloss_multi(base, res)
    out["rps_baserate"] = _rps(base, res)

    ok = bt[["odds_H", "odds_D", "odds_A"]].notna().all(axis=1) & (bt[["odds_H", "odds_D", "odds_A"]] > 1).all(axis=1)
    if ok.sum() >= 20:
        sub = bt[ok]
        B = np.array([devig_1x2(r.odds_H, r.odds_D, r.odds_A) for r in sub.itertuples()])
        r2 = sub["result"].tolist()
        out["market_n"] = int(ok.sum())
        out["market_logloss"] = _logloss_multi(B, r2)
        out["market_rps"] = _rps(B, r2)
        out["market_acc"] = float(np.mean(np.array(r2) == np.array(["H", "D", "A"])[B.argmax(1)]))
        # Same subset for the model, so the comparison is like-for-like.
        Pm = sub[["p_H", "p_D", "p_A"]].values
        out["model_logloss_on_market_subset"] = _logloss_multi(Pm, r2)
        out["model_rps_on_market_subset"] = _rps(Pm, r2)
    return out
