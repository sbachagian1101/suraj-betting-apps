"""Feature construction for the Australian PDF form guides.

Every feature for a run on date *t* is built from that horse's runs strictly
**before** *t*. The history is walked in date order and each row sees only the
accumulator as it stood before that row was folded in, so there is no way for
the target run to inform its own features - the usual quiet killer in racing
models, where a "career win rate" silently includes the race being predicted.

Features split into two kinds:

* **history** - what the horse has done (form, class, speed, the market's past
  opinion of it). Available identically for a past run and for an upcoming one.
* **conditions** - the race itself: weight, barrier, distance, field size,
  prize money, days since the last run. For a past run these come off the run's
  own line; for an upcoming one they come off the field table.

The market's historical prices are used as *history* (how short has this horse
been in the past) and never as the price of the race being predicted, which is
not in these files at all.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

HISTORY = [
    "n_prior", "win_rate", "place_rate", "avg_fin_pct", "last_fin_pct",
    "last3_fin_pct", "best_fin_pct", "avg_margin", "last_margin",
    "avg_log_sp", "best_log_sp", "last_log_sp", "fav_rate",
    "avg_api", "best_api", "avg_speed", "best_speed",
    "avg_settled_pct", "dist_fin_pct", "track_fin_pct", "going_fin_pct",
    "prior_span_days",
]
# Race-constant columns - field size, distance, prize - are deliberately NOT
# in CONDITIONS. A conditional logit compares runners *within* a race, so a
# column with one value per race cannot be identified and contributes nothing
# but noise. Leaving field size in was how a phantom coefficient of -0.39, the
# largest in the model, appeared: the race key was merging divided races, so
# field size varied inside a "race" and the model learned to tell the two
# divisions apart. They are kept on the frame for display and for the
# field-size-dependent parts of the app, just not fitted.
CONDITIONS = [
    "weight", "barrier_pct", "log_days_since", "dist_change", "class_change",
    "weight_change",
]
FEATURES = HISTORY + CONDITIONS
RACE_CONSTANT = ["log_distance", "field_size", "log_prize"]


def _pct(finish, field):
    """Finishing position as a fraction of the field: 0 = won, 1 = last."""
    field = np.where(field > 1, field, 2)
    return (finish - 1) / (field - 1)


def _safe_log(x, floor=1e-6):
    return np.log(np.maximum(np.asarray(x, dtype=float), floor))


def _blank():
    return {k: np.nan for k in HISTORY}


def _summarise(prior: pd.DataFrame, row) -> dict:
    """History features from `prior` — runs strictly before `row`."""
    if prior.empty:
        d = _blank()
        d["n_prior"] = 0.0
        return d

    fp = _pct(prior["finish"].to_numpy(), prior["past_field_size"].to_numpy())
    sp = prior["sp"].to_numpy(dtype=float)
    api = prior["api"].to_numpy(dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        speed = prior["distance"].to_numpy(float) / prior["race_time"].to_numpy(float)
    speed = np.where(np.isfinite(speed), speed, np.nan)

    def near(mask):
        return float(np.nanmean(fp[mask])) if mask.any() else np.nan

    dist = prior["distance"].to_numpy(float)
    same_dist = np.abs(dist - float(row["distance"])) <= 200
    same_track = (prior["track"].to_numpy() == row["track"])
    same_going = (prior["going"].to_numpy() == row.get("going", ""))

    d = {
        "n_prior": float(len(prior)),
        "win_rate": float(prior["won"].mean()),
        "place_rate": float(prior["placed"].mean()),
        "avg_fin_pct": float(np.nanmean(fp)),
        "last_fin_pct": float(fp[-1]),
        "last3_fin_pct": float(np.nanmean(fp[-3:])),
        "best_fin_pct": float(np.nanmin(fp)),
        "avg_margin": float(np.nanmean(prior["margin"])),
        "last_margin": float(prior["margin"].to_numpy()[-1]),
        "avg_log_sp": float(np.nanmean(_safe_log(sp))),
        "best_log_sp": float(np.nanmin(_safe_log(sp))),
        "last_log_sp": float(_safe_log(sp[-1:])[0]),
        "fav_rate": float(prior["was_favourite"].mean()),
        "avg_api": float(np.nanmean(api)),
        "best_api": float(np.nanmax(api)),
        "avg_speed": float(np.nanmean(speed)) if np.isfinite(speed).any() else np.nan,
        "best_speed": float(np.nanmax(speed)) if np.isfinite(speed).any() else np.nan,
        "avg_settled_pct": float(np.nanmean(
            prior["settled"].to_numpy(float)
            / np.maximum(prior["past_field_size"].to_numpy(float), 1))),
        "dist_fin_pct": near(same_dist),
        "track_fin_pct": near(same_track),
        "going_fin_pct": near(same_going),
        "prior_span_days": float(
            (row["date"] - prior["date"].min()).days) if pd.notna(row["date"]) else np.nan,
    }
    return d


def build_training(past: pd.DataFrame) -> pd.DataFrame:
    """One row per past run, with history features from earlier runs only."""
    past = past.sort_values(["horse", "date"]).reset_index(drop=True)
    out = []
    for horse, g in past.groupby("horse", sort=False):
        g = g.reset_index(drop=True)
        for i in range(len(g)):
            row = g.loc[i]
            prior = g.iloc[:i]
            feat = _summarise(prior, row)
            prev = prior.iloc[-1] if len(prior) else None
            days = ((row["date"] - prev["date"]).days
                    if prev is not None and pd.notna(row["date"]) else np.nan)
            feat.update({
                "weight": row["weight"],
                "barrier_pct": (row["barrier"] / max(row["past_field_size"], 2)
                                if pd.notna(row["barrier"]) else np.nan),
                "log_distance": np.log(max(row["distance"], 1)),
                "field_size": row["past_field_size"],
                "log_prize": _safe_log([row.get("race_prize", np.nan)])[0],
                "race_prize": row.get("race_prize", np.nan),
                "log_days_since": np.log1p(days) if days == days else np.nan,
                "dist_change": (row["distance"] - prev["distance"]
                                if prev is not None else np.nan),
                "class_change": (_safe_log([row.get("race_prize", np.nan)])[0]
                                 - feat["avg_api"] * 0
                                 - _safe_log([prev.get("race_prize", np.nan)])[0]
                                 if prev is not None else np.nan),
                "weight_change": (row["weight"] - prev["weight"]
                                  if prev is not None else np.nan),
                "horse": horse, "date": row["date"], "track": row["track"],
                "distance": row["distance"], "finish": row["finish"],
                "past_field_size": row["past_field_size"], "sp": row["sp"],
                "won": row["won"], "placed": row["placed"],
            })
            out.append(feat)
    df = pd.DataFrame(out)
    # Date + track + distance alone merges divided races and heats: 676 of
    # 3,904 such groups held two different field sizes, and 90 held two
    # winners. Field size and prize money separate them and leave 25.
    df["race_key"] = (df.date.dt.strftime("%Y%m%d") + "|" + df.track + "|"
                      + df.distance.astype("Int64").astype(str) + "|"
                      + df.past_field_size.astype("Int64").astype(str) + "|"
                      + df.race_prize.fillna(-1).astype(int).astype(str))
    winners = df.groupby("race_key")["won"].transform("sum")
    df["key_ok"] = winners <= 1
    return df


def build_upcoming(field: pd.DataFrame, past: pd.DataFrame) -> pd.DataFrame:
    """Features for an upcoming race: the whole history is prior by definition."""
    hist = past.sort_values(["horse", "date"])
    rows = []
    for _, r in field.iterrows():
        prior = hist[hist["horse"].str.lower() == str(r["horse"]).lower()]
        row = {"distance": r["distance"], "track": r["track"],
               "date": pd.Timestamp.now().normalize(), "going": ""}
        feat = _summarise(prior, row)
        prev = prior.iloc[-1] if len(prior) else None
        days = r.get("days_since_run", np.nan)
        feat.update({
            "weight": r["weight"],
            "barrier_pct": (r["barrier"] / max(r["field_size"], 2)
                            if pd.notna(r["barrier"]) else np.nan),
            "log_distance": np.log(max(r["distance"], 1)),
            "field_size": r["field_size"],
            "log_prize": _safe_log([r.get("prize", np.nan)])[0],
            "log_days_since": np.log1p(days) if days == days else np.nan,
            "dist_change": (r["distance"] - prev["distance"]
                            if prev is not None else np.nan),
            "class_change": (_safe_log([r.get("prize", np.nan)])[0]
                             - _safe_log([prev.get("race_prize", np.nan)])[0]
                             if prev is not None else np.nan),
            "weight_change": (r["weight"] - prev["weight"]
                              if prev is not None else np.nan),
            "horse": r["horse"], "tab": r["tab"], "jockey": r.get("jockey", ""),
            "trainer": r.get("trainer", ""), "race_id": r.get("race_id", ""),
            "comment": r.get("comment", ""), "gear": r.get("gear", ""),
            "n_history": len(prior),
        })
        rows.append(feat)
    return pd.DataFrame(rows)


def matrix(df: pd.DataFrame, medians: pd.Series | None = None):
    """Feature matrix, median-imputed, plus the medians used."""
    X = df.reindex(columns=FEATURES).astype(float)
    if medians is None:
        medians = X.median(numeric_only=True)
    X = X.fillna(medians).fillna(0.0)
    return X.to_numpy(dtype=float), medians


if __name__ == "__main__":
    past = pd.read_parquet("past_runs.parquet")
    tr = build_training(past)
    tr.to_parquet("training.parquet")
    print(f"{len(tr)} rows | {tr.horse.nunique()} horses")
    print(f"with prior form: {(tr.n_prior > 0).sum()}")
    grp = tr[tr.n_prior > 0].groupby("race_key").size()
    print(f"groups with >=2 runners: {(grp >= 2).sum()} "
          f"({grp[grp >= 2].sum()} runners)")
    print("\nfeature coverage (rows with >=1 prior run):")
    sub = tr[tr.n_prior > 0]
    for c in FEATURES:
        print(f"  {c:18s} {100*sub[c].notna().mean():5.1f}%")
