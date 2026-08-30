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
               use_market: bool = True, sims: int = 20000,
               feature_set: str = "all") -> pd.DataFrame:
    """Score one race.

    `feature_set` is "all" or "jockey". The jockey model sees only the rider's
    own record - earnings, starts, wins, places, strike rate and ROI over four
    horizons, plus the apprentice claim - and nothing about the horse. It is a
    genuinely weaker model, kept because it answers a different question.
    """
    if feature_set not in ("all", "jockey"):
        raise ValueError("feature_set must be 'all' or 'jockey'")
    jockey = feature_set == "jockey"

    df = D.prepare(race_df, require_result=False)
    n = len(df)
    if n < 2:
        raise ValueError("need at least two runners")

    cols = bundle["jockey_columns"] if jockey else bundle["base_columns"]
    names = bundle["jockey_feature_names"] if jockey else bundle["feature_names"]
    X, _ = D.build_features(df, cols=cols, with_market=False)
    F = X.reindex(columns=names, fill_value=0.0).to_numpy(float)
    sl = M.race_slices(df["race_id"].to_numpy())

    beta = bundle["jockey_logit_beta"] if jockey else bundle["logit_beta"]
    gbm = bundle["jockey_gbm"] if jockey else bundle["gbm"]
    rank = bundle["jockey_ranker"] if jockey else bundle["ranker"]
    p_logit = M.softmax_by_race(F @ beta, sl)
    p_gbm = M.normalise_by_race(gbm.predict_proba(F)[:, 1], sl)
    p_rank = M.softmax_by_race(rank.predict(F), sl)

    odds = pd.to_numeric(df.get(D.ODDS), errors="coerce").to_numpy(float)
    has_book = np.isfinite(odds).all() and (odds > 1).all()

    w_market = bundle["jockey_weights_market"] if jockey else bundle["weights"]
    w_alone = bundle["jockey_weights"] if jockey else bundle["weights_fundamentals"]
    label = "jockey" if jockey else "fundamentals"
    if use_market and has_book:
        p_mkt = D.shin_devig(odds)
        p = M.blend_log([p_logit, p_gbm, p_rank, p_mkt], w_market, sl)
        mode = f"{label} + market"
    else:
        p_mkt = np.full(n, np.nan)
        p = M.blend_log([p_logit, p_gbm, p_rank], w_alone, sl)
        mode = f"{label} only" + ("" if has_book else " (no complete book)")

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

    # `DataFrame.attrs` is written into the Arrow schema metadata as JSON when
    # Streamlit ships the frame to the browser, and **JavaScript's JSON.parse
    # rejects NaN** even though Python's json.loads accepts it. A NaN here does
    # not fail in Python at all - it fails client-side, as an unreadable
    # SyntaxError in the middle of the rendered page. So attrs carries None,
    # never NaN.
    out.attrs["mode"] = mode
    out.attrs["places_paid"] = int(places)
    out.attrs["overround"] = float((1 / odds).sum()) if has_book else None
    out.attrs["has_book"] = bool(has_book)
    out.attrs["priced_runners"] = int(np.isfinite(odds).sum())
    out.attrs["validated_config"] = bool(has_book and use_market and not jockey)
    out.attrs["feature_set"] = feature_set
    out.attrs["market_weight"] = (float(w_market[3])
                                  if (use_market and has_book) else None)
    assert not any(isinstance(v, float) and np.isnan(v)
                   for v in out.attrs.values()), "NaN in attrs would break the UI"
    return out


if __name__ == "__main__":
    b = load_bundle()
    race = D.read_race_file("D:/01_PREDICTION MODELS/20260818-lingfield-r06.xlsx")
    t = score_race(race, b)
    ov = t.attrs["overround"]
    print(f"mode: {t.attrs['mode']} | places paid {t.attrs['places_paid']} | "
          f"book {ov:.3f}" if ov else
          f"mode: {t.attrs['mode']} | places paid {t.attrs['places_paid']} | "
          f"no complete book")
    print()
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
