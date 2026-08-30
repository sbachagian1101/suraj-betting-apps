"""Score one upcoming race and say why.

The explanation is not a story told after the fact. The model is a ridge
regression, so a runner's score is a sum of `coefficient x feature`, and the
part of that sum which actually separates one runner from another is its
deviation from the **race mean** - a field where every horse is dropping in
class learns nothing from the fact that this one is too. So each reason is a
real term of the fitted linear predictor, measured against its own race, and
the reasons are ranked by how much they moved the score.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import features as F
import train as T

# how a feature reads to a person, and which direction is good
LABELS = {
    "barrier_pct": ("barrier", "inside draw", "wide draw"),
    "class_change": ("class move", "dropping in class", "stepping up in class"),
    "avg_log_sp": ("market history", "short prices in the past",
                   "long prices in the past"),
    "last_log_sp": ("last start price", "was well backed last start",
                    "was unfancied last start"),
    "win_rate": ("win strike rate", "wins often", "rarely wins"),
    "place_rate": ("place strike rate", "places often", "rarely places"),
    "going_fin_pct": ("record on this going", "goes well on this going",
                      "struggles on this going"),
    "track_fin_pct": ("record at this track", "goes well here", "struggles here"),
    "dist_fin_pct": ("record at this trip", "suited by this trip",
                     "unsuited by this trip"),
    "last_fin_pct": ("last start", "ran well last start", "ran poorly last start"),
    "last3_fin_pct": ("recent form", "in form", "out of form"),
    "avg_fin_pct": ("career form", "consistent", "inconsistent"),
    "best_fin_pct": ("best run", "has a strong best run", "no strong run"),
    "avg_settled_pct": ("running style", "races on speed", "races back in the field"),
    "best_speed": ("best time", "has run fast time", "has not run fast time"),
    "avg_speed": ("average time", "quick on the clock", "slow on the clock"),
    "log_days_since": ("freshness", "well spaced", "quick back-up or long spell"),
    "weight": ("weight", "well weighted", "carrying weight"),
    "weight_change": ("weight change", "down in weight", "up in weight"),
    "dist_change": ("trip change", "suited by the trip change",
                    "facing a trip change"),
    "avg_margin": ("beaten margins", "runs close up", "beaten a long way"),
    "last_margin": ("last margin", "close up last start", "well beaten last start"),
    "fav_rate": ("favouritism", "often favourite", "rarely favourite"),
    "n_prior": ("experience", "experienced", "lightly raced"),
    "prior_span_days": ("time in work", "long campaign", "short campaign"),
    "avg_api": ("class raced in", "has raced in good class", "modest class"),
    "best_api": ("best class", "has met good horses", "modest company"),
    "best_log_sp": ("shortest price", "has been well fancied",
                    "has never been fancied"),
    "going_fin_pct_": ("going", "", ""),
}


def confidence_for(gap: float, bands: list[dict]) -> dict:
    for b in sorted(bands, key=lambda x: -x["gap_lo"]):
        if gap >= b["gap_lo"]:
            return b
    return bands[0]


def score_race(field: pd.DataFrame, past: pd.DataFrame, bundle: dict,
               places: int = 3, sims: int = 20000) -> pd.DataFrame:
    """Rank one race. `field` is the runners, `past` every parsed past run."""
    feats = F.build_upcoming(field, past)
    X, _ = F.matrix(feats, bundle["medians"])
    v = -bundle["ridge"].predict(X)
    z = (v - bundle["v_mean"]) / (bundle["v_std"] + 1e-12)

    u = bundle["tau"] * z
    e = np.exp(u - u.max())
    p_win = e / e.sum()
    p_place = T.place_probabilities(p_win, places=min(places, max(len(p_win) - 1, 1)),
                                    sims=sims)

    out = feats[["tab", "horse", "jockey", "trainer", "n_history",
                 "comment", "gear"]].copy()
    out["score"] = z
    out["Win %"] = 100 * p_win
    out["Place %"] = 100 * p_place
    out = out.sort_values("score", ascending=False).reset_index(drop=True)
    out.insert(0, "Rank", np.arange(1, len(out) + 1))

    gap = float(out.score.iloc[0] - out.score.iloc[1]) if len(out) > 1 else 0.0
    band = confidence_for(gap, bundle["confidence_bands"])
    out.attrs["gap"] = gap
    out.attrs["confidence"] = band["band"]
    out.attrs["band"] = band
    out.attrs["field_size"] = int(len(out))
    out.attrs["no_history"] = int((feats.n_history == 0).sum())
    out.attrs["reasons"] = explain(feats, X, bundle,
                                   int(out.index[out.Rank == 1][0]),
                                   order=out.index.to_numpy())
    return out


def explain(feats: pd.DataFrame, X: np.ndarray, bundle: dict,
            row_in_sorted: int, order: np.ndarray, top_n: int = 5) -> list[str]:
    """Why the top-ranked runner scored where it did, against this field."""
    names = bundle["features"]
    coef = np.asarray(bundle["ridge"].coef_, dtype=float)
    centred = X - X.mean(axis=0)
    # the score is -prediction, so a negative coef*dev raises the score
    contrib = -(coef * centred)
    idx = int(np.argmax(-(bundle["ridge"].predict(X))))
    row = contrib[idx]
    rank = np.argsort(-np.abs(row))[:top_n]

    out = []
    for j in rank:
        name = names[j]
        if abs(row[j]) < 1e-9:
            continue
        label, good, bad = LABELS.get(name, (name, "above the field",
                                             "below the field"))
        phrase = good if row[j] > 0 else bad
        sign = "+" if row[j] > 0 else "-"
        out.append(f"{label}: {phrase} ({sign}{abs(row[j]):.3f})")
    return out


def summary_line(t: pd.DataFrame) -> str:
    b = t.attrs["band"]
    top = t.iloc[0]
    return (
        f"**{top.horse}** (tab {int(top.tab)}) — {t.attrs['confidence']} "
        f"confidence. In this band the top pick won "
        f"{100*b['win']:.1f}% and placed {100*b['place']:.1f}% "
        f"across {b['n']} held-out races."
    )
