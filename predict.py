"""Score a single race from the saved bundle.

The one subtlety: every feature is a *within-race* figure, so a race is scored
as a whole and can never be scored a runner at a time. Feeding one horse in
isolation would give it a z-score of zero on everything.
"""
from __future__ import annotations

import warnings

import joblib
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

import data as D
import models as M

BUNDLE = "model_bundle.joblib"


def load_bundle(path: str = BUNDLE):
    return joblib.load(path)


def score_race(race_df: pd.DataFrame, bundle: dict, *,
               use_market: bool = True, sims: int = 20000) -> pd.DataFrame:
    df = D.prepare(race_df, require_result=False)
    n = len(df)
    if n < 2:
        raise ValueError("need at least two runners")

    X, _ = D.build_features(df, cols=bundle["base_columns"], with_market=False)
    F = X.reindex(columns=bundle["feature_names"], fill_value=0.0).to_numpy(float)
    sl = M.race_slices(df["race_id"].to_numpy())

    p_logit = M.softmax_by_race(F @ bundle["logit_beta"], sl)
    p_gbm = M.normalise_by_race(bundle["gbm"].predict_proba(F)[:, 1], sl)
    p_rank = M.softmax_by_race(bundle["ranker"].predict(F), sl)

    odds = pd.to_numeric(df.get(D.ODDS), errors="coerce").to_numpy(float)
    has_book = np.isfinite(odds).all() and (odds > 1).all()

    if use_market and has_book:
        p_mkt = D.shin_devig(odds)
        p = M.blend_log([p_logit, p_gbm, p_rank, p_mkt], bundle["weights"], sl)
        mode = "blend + market"
    else:
        p_mkt = np.full(n, np.nan)
        p = M.blend_log([p_logit, p_gbm, p_rank],
                        bundle["weights_fundamentals"], sl)
        mode = "fundamentals only" + ("" if has_book else " (no complete book)")

    places = np.where(n >= 8, 3, np.where(n >= 5, 2, 1))
    p_place = M.place_probabilities(p, sl, np.full(n, places), sims=sims)

    out = pd.DataFrame({
        "Num": pd.to_numeric(df.get("Num"), errors="coerce"),
        "Horse": df.get("Horse Name"),
        "Odds": odds,
        "Win %": 100 * p,
        "Place %": 100 * p_place,
        "Market %": 100 * p_mkt,
        "Fair $": 1.0 / np.clip(p, 1e-9, None),
        "Edge": p * odds - 1.0,
        "_logit": 100 * p_logit,
        "_gbm": 100 * p_gbm,
        "_rank": 100 * p_rank,
    })
    out = out.sort_values("Win %", ascending=False).reset_index(drop=True)
    out.insert(0, "Rank", np.arange(1, n + 1))
    out.attrs["mode"] = mode
    out.attrs["places_paid"] = int(places)
    out.attrs["overround"] = float((1 / odds).sum()) if has_book else float("nan")
    return out


if __name__ == "__main__":
    b = load_bundle()
    race = pd.read_excel("D:/01_PREDICTION MODELS/20260818-lingfield-r06.xlsx")
    t = score_race(race, b)
    print(f"mode: {t.attrs['mode']} | places paid {t.attrs['places_paid']} | "
          f"book {t.attrs['overround']:.3f}\n")
    show = t.drop(columns=[c for c in t.columns if c.startswith("_")])
    print(show.to_string(index=False, float_format=lambda v: f"{v:.2f}"))
    print("\nthe three components, win % each:")
    print(t[["Horse", "_logit", "_gbm", "_rank", "Win %", "Market %"]].to_string(
        index=False, float_format=lambda v: f"{v:.1f}"))
    val = t[t.Edge > 0]
    print(f"\nrunners the model prices above the market: {len(val)}")
    if len(val):
        print(val[["Horse", "Odds", "Win %", "Market %", "Edge"]].to_string(
            index=False, float_format=lambda v: f"{v:.2f}"))
