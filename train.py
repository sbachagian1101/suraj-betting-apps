"""Fit the production model and save it as a single bundle.

The method was validated by walk-forward on held-out periods. Having established
*that* it works, the shipped model is fitted on everything - holding data back
from the final fit buys nothing once the method itself is settled, and costs
accuracy.

The blend weights are the one thing that must not be fitted on everything: they
are chosen on the most recent validation period only, before the final refit, so
they are never tuned on races the models also learned from.
"""
from __future__ import annotations

import json
import time
import warnings

import joblib
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

import data as D
import models as M

OUT = "model_bundle.joblib"


def main(out: str = OUT):
    t0 = time.time()
    df = D.load()
    t = D.targets(df)
    y, fin = t["win"], t["finish"]
    X, names = D.build_features(df, with_market=False)
    F = X.to_numpy(float)
    rid = df["race_id"].to_numpy()
    p_mkt = D.market_probability(df).groupby(df["race_id"]).transform(
        lambda s: s.fillna(1.0 / len(s))).to_numpy()

    races = list(dict.fromkeys(rid))
    cut = int(len(races) * 0.85)
    tr = np.isin(rid, races[:cut])
    va = np.isin(rid, races[cut:])
    sl_va = M.race_slices(rid[va])

    print(f"{len(races)} races | {len(df)} runners | {len(names)} features")
    print("choosing blend weights on the last 15% of races …")
    prov = M.ConditionalLogit(l2=0.05).fit(F[tr], y[tr], rid[tr])
    prov_gb = M.fit_gbm_classifier(F[tr], y[tr])
    prov_rk = M.fit_ranker(F[tr], fin[tr], rid[tr])
    pv = (M.softmax_by_race(prov.decision(F[va]), sl_va),
          M.normalise_by_race(prov_gb.predict_proba(F[va])[:, 1], sl_va),
          M.softmax_by_race(prov_rk.predict(F[va]), sl_va),
          M.normalise_by_race(p_mkt[va], sl_va))
    w = M.fit_blend_weights(list(pv), y[va], sl_va)
    print(f"  logit {w[0]:.2f} | gbm {w[1]:.2f} | rank {w[2]:.2f} | "
          f"market {w[3]:.2f}")
    w_fund = M.fit_blend_weights(list(pv[:3]), y[va], sl_va)
    print(f"  fundamentals-only: logit {w_fund[0]:.2f} | gbm {w_fund[1]:.2f} | "
          f"rank {w_fund[2]:.2f}")

    print("refitting on every race …")
    cl = M.ConditionalLogit(l2=0.05).fit(F, y, rid)
    gb = M.fit_gbm_classifier(F, y)
    rk = M.fit_ranker(F, fin, rid)

    # A second, deliberately blinkered model: the jockey columns and nothing
    # else. Much weaker than the full model - see JOCKEY_VALIDATION - but it
    # answers a different question, "what do the riders alone say", and it is
    # the only way to look at that without the horse's own form drowning it out.
    print("fitting the jockey-only model …")
    jcols = D.jockey_columns(df)
    Xj, jnames = D.build_features(df, cols=jcols, with_market=False)
    Fj = Xj.to_numpy(float)
    jl_prov = M.ConditionalLogit(l2=0.05).fit(Fj[tr], y[tr], rid[tr])
    jg_prov = M.fit_gbm_classifier(Fj[tr], y[tr])
    jr_prov = M.fit_ranker(Fj[tr], fin[tr], rid[tr])
    pvj = (M.softmax_by_race(jl_prov.decision(Fj[va]), sl_va),
           M.normalise_by_race(jg_prov.predict_proba(Fj[va])[:, 1], sl_va),
           M.softmax_by_race(jr_prov.predict(Fj[va]), sl_va),
           M.normalise_by_race(p_mkt[va], sl_va))
    wj = M.fit_blend_weights(list(pvj[:3]), y[va], sl_va)
    wj_m = M.fit_blend_weights(list(pvj), y[va], sl_va)
    print(f"  jockey-only blend  logit {wj[0]:.2f} | gbm {wj[1]:.2f} | "
          f"rank {wj[2]:.2f}")
    print(f"  with the market    market weight {wj_m[3]:.2f}")
    jcl = M.ConditionalLogit(l2=0.05).fit(Fj, y, rid)
    jgb = M.fit_gbm_classifier(Fj, y)
    jrk = M.fit_ranker(Fj, fin, rid)

    bundle = {
        "feature_names": names,
        "base_columns": D.base_columns(df),
        "logit_beta": cl.beta_,
        "gbm": gb,
        "ranker": rk,
        "weights": w,
        "weights_fundamentals": w_fund,
        "jockey_feature_names": jnames,
        "jockey_columns": jcols,
        "jockey_logit_beta": jcl.beta_,
        "jockey_gbm": jgb,
        "jockey_ranker": jrk,
        "jockey_weights": wj,
        "jockey_weights_market": wj_m,
        "jockey_validation": {
            "test_races": 465,
            "logloss": 2.0888,
            "top1": 0.224,
            "top3": 0.546,
            "market_logloss": 1.8275,
            "market_top1": 0.381,
            "market_top3": 0.665,
            "full_logloss": 1.8111,
            "full_top1": 0.355,
            "dart_throw_top1": 0.113,
            "market_weight_when_blended": 1.00,
            "n_columns": len(jcols),
            "top_drivers": ["Jockey This Season ROI", "Jockey 12 Months ROI",
                            "Jockey Last Season ROI", "Jockey Last 100 ROI",
                            "Jockey This Season Strike Rate",
                            "Jockey This Season Avg Horse Earnings"],
        },
        "n_races": len(races),
        "n_runners": int(len(df)),
        "date_range": [str(df.date.min()), str(df.date.max())],
        "validation": {
            "test_races": 465,
            "market_logloss": 1.8275,
            "blend_logloss": 1.8111,
            "logloss_edge": 0.0164,
            "edge_ci": [0.0023, 0.0307],
            "folds_positive_logloss": "5/5",
            "value_roi": 0.226,
            "value_roi_ci": [0.058, 0.405],
            "value_bets": 3841,
            "value_strike": 0.114,
            "folds_positive_roi": "5/5",
            "roi_at_5pct_worse": 0.165,
            "roi_at_10pct_worse": 0.104,
            "roi_at_20pct_worse": -0.019,
            "market_top1": 0.381,
            "blend_top1": 0.355,
            "overround": 1.129,
        },
    }
    joblib.dump(bundle, out, compress=3)
    import os
    print(f"saved {out} ({os.path.getsize(out)/1e6:.1f} MB) "
          f"in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
