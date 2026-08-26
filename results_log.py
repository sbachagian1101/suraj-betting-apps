"""Results ledger and threshold tuning for the Place Finder.

What this can and cannot do
---------------------------
It **cannot** retrain the underlying model. HorsePredictor's feature weights were
calibrated on roughly 1,700 races; logging finishing positions gives no way to
move them, and re-fitting them on a few dozen races would be worse than leaving
them alone.

It **can** tune the five Place Finder thresholds, because those are the part
currently resting on sixteen placegetters. How much data that needs is not a
matter of taste:

===========================================  =============
Question you want answered                   Races needed
===========================================  =============
Has a rule quietly broken (47% -> 20%)?      about 17
Does it beat a dart throw?                   about 45
Does consensus really beat model top-3?      about 71
Is 47% genuinely better than 40%?            about 273
===========================================  =============

So the ledger is worth keeping from the first race for **monitoring**, and the
tuner stays locked until there is enough data to say anything. Tuning is scored
by **leave-one-race-out** cross-validation, never in-sample: picking the best
thresholds on the same races you then score them against would manufacture an
edge that does not exist.

Storage note: Streamlit Community Cloud has an ephemeral filesystem, so the
download button is the real persistence. Keep the CSV and re-upload it.
"""
from __future__ import annotations

import itertools
from typing import Any

import numpy as np
import pandas as pd

import horse_model as hm

# Enough per runner to re-derive any threshold combination offline.
CORE_COLUMNS = [
    "race_id", "date", "track", "race_no", "going", "surface", "distance_m",
    "field_size", "places_paid",
    "tab", "horse", "model_rank", "win_pct", "place_pct_raw", "place_pct_adj",
    "fm", "mkt_rank", "bf_odds", "tab_odds", "conf", "status_at_prediction",
    "finish_pos", "placed",
]

# The raw parsed fields. Stored so that features nobody has thought of yet can
# still be derived from races logged today - the whole point of keeping them.
_RECORD_KEYS = ["Car", "M12", "Crs", "Dist", "CrsDist", "Firm", "Good", "Soft",
                "Heavy", "AW", "Turf", "FU", "U2", "U3"]
RAW_COLUMNS = (
    ["wt", "bp", "claim", "jrat", "trat", "age", "sex", "maiden", "form",
     "ohr", "dslr", "runup", "runup_tag",
     "jky_win", "jky_place", "jky_n", "trn_win", "trn_place", "trn_n",
     "jh_win", "jh_place", "jh_n", "jt_win", "jt_place", "jt_n",
     "dist_min", "dist_max", "win_dist",
     "last_fin", "last_margin", "last_sp", "api_avg", "n_recent_runs"]
    + [f"{k}_{s}" for k in _RECORD_KEYS for s in ("wins", "places", "starts")]
)

# The model's own feature values, so weights can be re-fitted directly once
# enough races have been settled.
FEATURE_COLUMNS = ["feat_" + n for n, _, _ in hm.FEATURES + hm.EXTRA_FEATURES]

LEDGER_COLUMNS = CORE_COLUMNS + RAW_COLUMNS + FEATURE_COLUMNS

MIN_RACES_TO_REPORT = 5     # below this, summary stats are noise
MIN_RACES_TO_TUNE = 40      # below this, tuning fits noise
CONFIDENT_RACES = 70        # roughly where consensus vs model-top-3 separates


def race_id(header: dict[str, Any]) -> str:
    track = str(header.get("track", "race")).strip().replace(" ", "-")
    return f"{track}_R{header.get('race_no', '?')}_{header.get('date', '')}".strip("_")


def snapshot(header: dict[str, Any], active: list[dict[str, Any]],
             result: dict[str, Any], table: pd.DataFrame,
             meta: dict[str, Any]) -> pd.DataFrame:
    """Freeze one race's prediction into ledger rows, results not yet known."""
    rid = race_id(header)
    by_tab = {int(r.get("tab") or -1): r for r in active}
    idx_by_tab = {int(r.get("tab") or -1): i for i, r in enumerate(active)}
    try:
        feats = hm.build_features(active, header,
                                  extended=bool(result.get("extended", True)))
    except Exception:                                        # noqa: BLE001
        feats = {}
    rows = []
    for _, t in table.iterrows():
        tab = int(t["Tab"])
        r = by_tab.get(tab, {})
        i = idx_by_tab.get(tab)
        extra = {c: r.get(c) for c in RAW_COLUMNS}
        extra["n_recent_runs"] = len(r.get("recent_runs", []) or [])
        for name in (n for n, _, _ in hm.FEATURES + hm.EXTRA_FEATURES):
            v = feats.get(name)
            extra["feat_" + name] = (float(v[i]) if v is not None and i is not None
                                     and i < len(v) else np.nan)
        rows.append({
            **extra,
            "race_id": rid,
            "date": header.get("date", ""),
            "track": header.get("track", ""),
            "race_no": header.get("race_no", ""),
            "going": header.get("going", ""),
            "surface": header.get("surface", ""),
            "distance_m": header.get("distance_m", ""),
            "field_size": meta["runners"],
            "places_paid": meta["places"],
            "tab": tab,
            "horse": t["Horse"],
            "model_rank": int(t["Rank"]),
            "win_pct": float(t["Win%"]),
            "place_pct_raw": float(t["Place% (raw)"]),
            "place_pct_adj": float(t["Place% (adj)"]),
            "fm": float(t["F/M"]),
            "mkt_rank": int(t["Mkt rank"]),
            "bf_odds": float(r.get("bf_odds") or np.nan),
            "tab_odds": float(r.get("tab_odds") or np.nan),
            "conf": int(t["Conf"]),
            "status_at_prediction": t["Status"],
            "finish_pos": np.nan,
            "placed": np.nan,
        })
    return pd.DataFrame(rows, columns=LEDGER_COLUMNS)


def empty_ledger() -> pd.DataFrame:
    return pd.DataFrame(columns=LEDGER_COLUMNS)


def conform(df: pd.DataFrame) -> pd.DataFrame:
    """Bring an older ledger up to the current schema without losing rows."""
    out = df.copy()
    missing = [c for c in LEDGER_COLUMNS if c not in out.columns]
    if missing:
        # Added in one concat rather than column by column, which pandas warns
        # about as a highly fragmented frame.
        out = pd.concat(
            [out, pd.DataFrame(np.nan, index=out.index, columns=missing)], axis=1)
    return out[LEDGER_COLUMNS]


def merge(ledger: pd.DataFrame, new: pd.DataFrame) -> pd.DataFrame:
    """Add a race, replacing any earlier snapshot of the same race."""
    if ledger is None or ledger.empty:
        return new.copy()
    keep = ledger[~ledger["race_id"].isin(set(new["race_id"]))]
    return pd.concat([keep, new], ignore_index=True)


def record_result(ledger: pd.DataFrame, rid: str, finishers: list[int]) -> pd.DataFrame:
    """Attach a finishing order (tab numbers, winner first) to a logged race."""
    out = ledger.copy()
    m = out["race_id"] == rid
    if not m.any():
        raise KeyError(f"No logged prediction for race {rid!r}.")
    places = int(out.loc[m, "places_paid"].iloc[0])
    pos = {int(t): i + 1 for i, t in enumerate(finishers)}
    out.loc[m, "finish_pos"] = out.loc[m, "tab"].map(pos)
    placed = out.loc[m, "finish_pos"].apply(
        lambda v: 0 if pd.isna(v) else int(v <= places))
    out.loc[m, "placed"] = placed
    return out


def settled(ledger: pd.DataFrame) -> pd.DataFrame:
    if ledger is None or ledger.empty:
        return empty_ledger()
    return ledger[ledger["placed"].notna()].copy()


# --------------------------------------------------------------------------
# selection logic, re-derivable from the ledger alone
# --------------------------------------------------------------------------
def selections_for_race(d: pd.DataFrame, top_n: int, market_top: int,
                        fm_max: float, max_picks: int) -> list[int]:
    """Re-apply the Place Finder gates to one logged race."""
    d = d.sort_values("model_rank")
    picks: list[int] = []
    for _, r in d.iterrows():
        if r["model_rank"] > top_n:
            break
        if len(picks) >= max_picks:
            break
        if r["fm"] >= fm_max:
            continue
        if r["mkt_rank"] > market_top:
            continue
        picks.append(int(r["tab"]))
    return picks


def score_params(df: pd.DataFrame, top_n: int, market_top: int,
                 fm_max: float, max_picks: int) -> dict[str, float]:
    """Precision and volume for one threshold set across settled races."""
    hits = sel = races = 0
    for _, d in df.groupby("race_id"):
        picks = selections_for_race(d, top_n, market_top, fm_max, max_picks)
        if not picks:
            races += 1
            continue
        placed = set(d[d["placed"] == 1]["tab"].astype(int))
        hits += len(set(picks) & placed)
        sel += len(picks)
        races += 1
    return {
        "precision": (hits / sel) if sel else 0.0,
        "hits": hits, "selections": sel, "races": races,
        "picks_per_race": (sel / races) if races else 0.0,
    }


DEFAULT_GRID = {
    "top_n": [3, 4, 5, 6],
    "market_top": [2, 3, 4],
    "fm_max": [1.5, 2.0, 3.0, 99.0],
    "max_picks": [2, 3],
}


def _per_race_stats(df: pd.DataFrame, combos: list[dict]) -> tuple[list[str], list[dict]]:
    """(hits, selections) for every combo on every race, computed once.

    Cross-validation then becomes arithmetic on these totals instead of
    re-scoring the whole ledger inside every fold, which turns an O(combos x
    races^2) search into O(combos x races).
    """
    groups = {rid: d.sort_values("model_rank") for rid, d in df.groupby("race_id")}
    placed = {rid: set(d[d["placed"] == 1]["tab"].astype(int))
              for rid, d in groups.items()}
    races = sorted(groups)
    stats = []
    for c in combos:
        per = {}
        for rid in races:
            picks = selections_for_race(groups[rid], **c)
            per[rid] = (len(set(picks) & placed[rid]), len(picks))
        stats.append(per)
    return races, stats


def tune(df: pd.DataFrame, grid: dict[str, list] | None = None,
         min_picks_per_race: float = 1.0) -> dict[str, Any]:
    """Pick thresholds by leave-one-race-out cross-validation.

    For every held-out race the thresholds are chosen on the *other* races only,
    then applied to the held-out one. That is the only estimate worth quoting:
    choosing and scoring on the same races invents an edge that is not there.
    """
    grid = grid or DEFAULT_GRID
    combos = [dict(zip(grid.keys(), v)) for v in itertools.product(*grid.values())]
    if df.empty or df["race_id"].nunique() < 3:
        return {"ok": False, "races": int(df["race_id"].nunique()) if not df.empty else 0,
                "reason": "Need at least three settled races to tune."}

    races, stats = _per_race_stats(df, combos)
    n_races = len(races)
    tot_hits = [sum(h for h, _ in per.values()) for per in stats]
    tot_sel = [sum(s for _, s in per.values()) for per in stats]

    def pick(hits: list[int], sels: list[int], races_in: int) -> int:
        best, best_p = None, -1.0
        for k in range(len(combos)):
            if races_in and sels[k] / races_in < min_picks_per_race:
                continue
            p = (hits[k] / sels[k]) if sels[k] else 0.0
            if p > best_p:
                best, best_p = k, p
        return best if best is not None else 0

    cv_hits = cv_sel = 0
    for rid in races:
        h = [tot_hits[k] - stats[k][rid][0] for k in range(len(combos))]
        sl = [tot_sel[k] - stats[k][rid][1] for k in range(len(combos))]
        k = pick(h, sl, n_races - 1)
        hh, ss = stats[k][rid]
        cv_hits += hh
        cv_sel += ss

    best_k = pick(tot_hits, tot_sel, n_races)
    chosen = combos[best_k]
    return {
        "ok": True,
        "races": n_races,
        "params": chosen,
        "in_sample": score_params(df, **chosen),
        "cv_precision": (cv_hits / cv_sel) if cv_sel else 0.0,
        "cv_selections": cv_sel,
        "current": score_params(df, top_n=5, market_top=3, fm_max=2.0, max_picks=3),
        "trustworthy": n_races >= CONFIDENT_RACES,
    }


# --------------------------------------------------------------------------
# performance reporting
# --------------------------------------------------------------------------
def performance(ledger: pd.DataFrame) -> dict[str, Any]:
    df = settled(ledger)
    if df.empty:
        return {"races": 0}
    races = df["race_id"].nunique()
    sel = df[df["status_at_prediction"].str.contains("SELECTION", na=False)]
    out: dict[str, Any] = {
        "races": races,
        "runners": len(df),
        "placegetters": int(df["placed"].sum()),
        "selections": len(sel),
        "selection_hits": int(sel["placed"].sum()) if len(sel) else 0,
    }
    if len(sel):
        p = out["selection_hits"] / len(sel)
        se = float(np.sqrt(max(p * (1 - p), 1e-9) / len(sel)))
        out["precision"] = p
        out["ci_low"] = max(0.0, p - 1.96 * se)
        out["ci_high"] = min(1.0, p + 1.96 * se)
    base = float((df["places_paid"] / df["field_size"]).mean())
    out["base_rate"] = base
    # Calibration of the adjusted place probability.
    p = (df["place_pct_adj"] / 100).values
    y = df["placed"].values.astype(float)
    out["logloss"] = float(-np.mean(np.log(np.clip(np.where(y == 1, p, 1 - p), 1e-12, 1))))
    out["brier"] = float(np.mean((p - y) ** 2))
    out["mean_predicted"] = float(p.mean())
    out["actual_rate"] = float(y.mean())
    bands = []
    for lo, hi in [(0, .15), (.15, .30), (.30, .45), (.45, .60), (.60, 1.01)]:
        m = (p >= lo) & (p < hi)
        if m.sum() >= 5:
            bands.append({"band": f"{lo:.0%}-{hi:.0%}", "n": int(m.sum()),
                          "predicted": float(p[m].mean()), "actual": float(y[m].mean())})
    out["calibration"] = bands
    return out


def readiness(races: int) -> tuple[str, str]:
    """Plain-English statement of what this much data can and cannot support."""
    if races == 0:
        return ("empty", "No settled races yet. Log a prediction, then enter the "
                         "finishing order once the race is run.")
    if races < MIN_RACES_TO_REPORT:
        return ("thin", f"{races} settled race(s). Far too few to read anything into — "
                        "the ledger is just recording for now.")
    if races < MIN_RACES_TO_TUNE:
        return ("monitor", f"{races} settled races. Enough to spot a **badly broken** "
                           f"rule, not enough to tune. Tuning unlocks at "
                           f"{MIN_RACES_TO_TUNE}.")
    if races < CONFIDENT_RACES:
        return ("tune", f"{races} settled races. Tuning is available, but only coarse "
                        f"differences are real at this size. Treat suggestions as "
                        f"provisional until about {CONFIDENT_RACES}.")
    return ("confident", f"{races} settled races. Enough to separate genuinely "
                         "different rules. Fine distinctions (a few points of "
                         "precision) still need several hundred races.")
