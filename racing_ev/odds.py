"""Market de-vigging and expected-value calculations."""
from __future__ import annotations

import numpy as np
import pandas as pd


def effective_decimal_odds(odds: np.ndarray, commission: float = 0.0) -> np.ndarray:
    odds = np.asarray(odds, dtype=float)
    c = float(np.clip(commission, 0.0, 0.25))
    return 1.0 + np.maximum(odds - 1.0, 0.0) * (1.0 - c)


def _power_devig(implied: np.ndarray) -> np.ndarray:
    """Power-method de-vig: choose k so sum(q_i ** k) = 1."""
    q = np.clip(np.asarray(implied, dtype=float), 1e-12, 1.0)
    if q.sum() <= 1.0:
        return q / q.sum()
    lo, hi = 0.01, 20.0
    for _ in range(100):
        mid = (lo + hi) / 2.0
        value = np.power(q, mid).sum()
        if value > 1.0:
            lo = mid
        else:
            hi = mid
    p = np.power(q, (lo + hi) / 2.0)
    return p / p.sum()


def devig_probabilities(odds: np.ndarray, method: str = "power") -> tuple[np.ndarray, float]:
    odds = np.asarray(odds, dtype=float)
    valid = np.isfinite(odds) & (odds > 1.0)
    if not valid.all():
        raise ValueError("All active runners need decimal odds greater than 1.00.")
    implied = 1.0 / odds
    overround = float(implied.sum() - 1.0)
    if method.lower() == "proportional":
        p = implied / implied.sum()
    else:
        p = _power_devig(implied)
    return p, overround


def add_value_columns(
    probability_table: pd.DataFrame,
    odds_col: str = "market_odds",
    method: str = "power",
    commission: float = 0.0,
) -> tuple[pd.DataFrame, float]:
    df = probability_table.copy()
    odds = pd.to_numeric(df[odds_col], errors="coerce").to_numpy(float)
    net_odds = effective_decimal_odds(odds, commission)
    market_p, overround = devig_probabilities(odds, method)
    model_p = pd.to_numeric(df["win_probability"], errors="coerce").to_numpy(float)
    df["net_odds"] = net_odds
    df["market_probability_raw"] = 1.0 / odds
    df["market_probability_fair"] = market_p
    df["probability_edge"] = model_p - market_p
    df["ev_per_unit"] = model_p * net_odds - 1.0
    df["ev_pct"] = 100.0 * df["ev_per_unit"]
    df["full_kelly"] = np.maximum(0.0, (model_p * net_odds - 1.0) / np.maximum(net_odds - 1.0, 1e-9))
    df["quarter_kelly"] = 0.25 * df["full_kelly"]
    df["value_grade"] = [
        _grade(ev, edge, q)
        for ev, edge, q in zip(df["ev_per_unit"], df["probability_edge"], df["data_quality"])
    ]
    return df, overround


def _grade(ev: float, edge: float, quality: float) -> str:
    if ev >= 0.15 and edge >= 0.04 and quality >= 0.65:
        return "Strong positive EV"
    if ev >= 0.05 and edge >= 0.02:
        return "Positive EV"
    if ev > 0:
        return "Marginal / uncertain"
    return "No model value"
