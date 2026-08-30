"""Data preparation for the horse-race model.

Three decisions here shape everything downstream.

**Races, not rows.** Every operation groups by race. A finishing position is a
rank inside a field, so a runner only means something next to its opponents.

**Relative, not absolute.** Each feature is turned into a within-race z-score
*and* a within-race rank. A handicap rating of 82 says nothing on its own; being
the top-rated of nine says a lot. Ranks are kept alongside z-scores because most
of these columns are heavily skewed (prize money, earnings) and a rank is immune
to that.

**Race-level columns stay raw.** Distance, prize and field size are identical for
every runner in a race, so normalising them within the race would produce a
column of zeros. They are detected automatically and passed through.

The market is handled separately and can be switched off entirely, because a
model that is fed the price cannot then be said to have found value in it.
"""
from __future__ import annotations

import re
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

SRC = "D:/01_PREDICTION MODELS/June2026_Updated Training data.xlsx"
TARGET = "Finish Result (Updates after race)"
ODDS = "Best Fixed Odds"

# Identifiers and anything that is not evidence about the horse.
EXCLUDE = {
    TARGET, ODDS, "BetEasy Odds", "Num",
    "Horse Name", "Jockey", "Trainer", "Gender", "Apprentice",
    "Form Guide Url", "Horse Profile Url", "Jockey Profile Url",
    "Trainer Profile Url", "race_id", "date", "track", "raceno",
}


# --------------------------------------------------------------------------
# de-vigging
# --------------------------------------------------------------------------
def shin_devig(odds: np.ndarray, tol: float = 1e-10, iters: int = 200) -> np.ndarray:
    """Shin (1993) de-vig, solved by bisection.

    The bookmaker's implied probabilities sum to more than one. Shin's model
    attributes the excess to a proportion `z` of insider money and recovers the
    underlying probabilities. Bisection is used rather than the usual fixed-point
    iteration because that iteration oscillates instead of converging on books
    with a large favourite.
    """
    q = 1.0 / np.asarray(odds, dtype=float)
    q = q / q.sum() if q.sum() <= 0 else q
    s = q.sum()
    if not np.isfinite(s) or s <= 1.0 + 1e-12:
        return q / q.sum()

    def p_of(z):
        r = np.sqrt(z * z + 4.0 * (1.0 - z) * q * q / s)
        p = (r - z) / (2.0 * (1.0 - z))
        return p / p.sum()

    lo, hi = 0.0, 0.99
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        r = np.sqrt(mid * mid + 4.0 * (1.0 - mid) * q * q / s)
        tot = ((r - mid) / (2.0 * (1.0 - mid))).sum()
        if abs(tot - 1.0) < tol:
            break
        if tot > 1.0:
            lo = mid
        else:
            hi = mid
    return p_of(0.5 * (lo + hi))


# --------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------
def read_race_file(src, filename: str | None = None) -> pd.DataFrame:
    """Read one race sheet, CSV or Excel, without letting the columns shift.

    The CSV export ends every data row with a trailing comma, so the rows carry
    **129 fields against a 128-name header**. Pandas resolves that by silently
    promoting the first column to the index, which shifts every column one to
    the left: `Horse Name` comes back holding ages and `Best Fixed Odds` holding
    carried weights. Nothing raises - the frame looks fine and every prediction
    made from it is nonsense.

    `index_col=False` refuses that inference. It is harmless when the counts
    already match, so it is applied unconditionally, and the result is then
    checked: horse names must not be numbers.
    """
    name = (filename or getattr(src, "name", "") or str(src)).lower()
    if name.endswith((".xlsx", ".xlsm", ".xls")):
        df = pd.read_excel(src)
    else:
        df = pd.read_csv(src, index_col=False)

    horse = df.get("Horse Name")
    if horse is not None and len(horse):
        numeric = pd.to_numeric(horse, errors="coerce").notna().mean()
        if numeric > 0.5:
            raise ValueError(
                "The columns in this file are misaligned - 'Horse Name' holds "
                "numbers. That usually means the header and the data rows have "
                "different field counts. Re-export the race sheet."
            )
    return df


def load(src: str = SRC, require_result: bool = True) -> pd.DataFrame:
    df = pd.read_excel(src)
    return prepare(df, require_result=require_result)


def prepare(df: pd.DataFrame, require_result: bool = True) -> pd.DataFrame:
    df = df.copy()
    if "Num" in df.columns:
        df = df[df["Num"].astype(str).str.lower().ne("num")]

    u = df.get("Form Guide Url", pd.Series("", index=df.index)).astype(str)
    m = u.str.extract(r"/form-guide/horses/(.+?)-(\d{8})(?:-\d+)?/(.+?)-race-(\d+)/")
    df["track"], df["date"], df["raceno"] = m[0], m[1], m[3]
    df["race_id"] = df["track"] + "_" + df["date"] + "_R" + df["raceno"]
    if df["race_id"].isna().all():
        # a single-race file with no usable urls still forms one race
        df["race_id"] = "RACE"
        df["date"] = df.get("date", "99999999")
        df["track"] = df.get("track", "UNKNOWN")

    df = df[df["race_id"].notna()]
    df = df.drop_duplicates(subset=["race_id", "Horse Name"], keep="first")

    if TARGET in df.columns:
        df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")
    if require_result:
        g = df.groupby("race_id")
        ok = g.apply(lambda d: (d[TARGET].notna().all()
                                and sorted(d[TARGET].tolist())
                                == list(range(1, len(d) + 1))))
        df = df[df["race_id"].isin(ok[ok].index)]

    df[ODDS] = pd.to_numeric(df.get(ODDS), errors="coerce")
    df = df.sort_values(["date", "race_id"]).reset_index(drop=True)
    df["field_size"] = df.groupby("race_id")["race_id"].transform("size")
    return df


def market_probability(df: pd.DataFrame) -> pd.Series:
    """Shin-de-vigged win probability per runner, NaN where a race has no book."""
    out = pd.Series(np.nan, index=df.index, dtype=float)
    for rid, g in df.groupby("race_id", sort=False):
        o = g[ODDS]
        if o.notna().all() and (o > 1).all() and len(g) >= 2:
            out.loc[g.index] = shin_devig(o.to_numpy(dtype=float))
    return out


# --------------------------------------------------------------------------
# features
# --------------------------------------------------------------------------
def base_columns(df: pd.DataFrame, min_fill: float = 0.90) -> list[str]:
    num = df.select_dtypes(include=[np.number]).columns
    cols = [c for c in num if c not in EXCLUDE and c != "field_size"]
    fill = df[cols].notna().mean()
    return [c for c in cols if fill[c] >= min_fill]


def jockey_columns(df: pd.DataFrame, min_fill: float = 0.90) -> list[str]:
    """Only what is known about the rider.

    Thirty-two statistics - earnings, starts, wins, places, strike rate and ROI
    over the last 100 rides, twelve months, this season and last season - plus
    the apprentice claim, which is a property of the jockey and shows up as a
    real weight advantage.

    Nothing about the horse, trainer, distance, going or draw.
    """
    cols = [c for c in base_columns(df, min_fill) if c.lower().startswith("jockey")]
    claim = "Jockey Weight Claim"
    if claim in df.columns and claim not in cols:
        cols.append(claim)
    return cols


def build_features(df: pd.DataFrame, cols: list[str] | None = None,
                   with_market: bool = True
                   ) -> tuple[pd.DataFrame, list[str]]:
    """Within-race z-scores and ranks, plus race-level columns passed through."""
    cols = cols or base_columns(df)
    X = df[cols].astype(float)
    X = X.fillna(X.median())

    grp = df["race_id"]
    mean = X.groupby(grp).transform("mean")
    std = X.groupby(grp).transform("std")

    # A column with no within-race variation is a property of the race, not the
    # runner. Z-scoring it yields zeros and throws the information away.
    varies = (std.fillna(0) > 1e-9).mean() > 0.5
    dyn = [c for c in cols if varies[c]]
    static = [c for c in cols if not varies[c]]

    Z = ((X[dyn] - mean[dyn]) / std[dyn].replace(0, np.nan)).fillna(0.0)
    Z.columns = [f"z_{c}" for c in dyn]

    R = X[dyn].groupby(grp).rank(pct=True, method="average").fillna(0.5)
    R.columns = [f"r_{c}" for c in dyn]

    parts = [Z, R]
    names = list(Z.columns) + list(R.columns)

    if static:
        S = X[static].copy()
        S.columns = [f"race_{c}" for c in static]
        parts.append(S)
        names += list(S.columns)

    F = pd.DataFrame({"field_size": df["field_size"].astype(float).values},
                     index=df.index)
    parts.append(F)
    names.append("field_size")

    if with_market:
        p = market_probability(df)
        med = p.groupby(grp).transform(lambda s: s.fillna(1.0 / len(s)))
        lp = np.log(med.clip(1e-6))
        M = pd.DataFrame({
            "mkt_logp": lp.values,
            "z_mkt": ((lp - lp.groupby(grp).transform("mean"))
                      / lp.groupby(grp).transform("std").replace(0, np.nan)
                      ).fillna(0.0).values,
            "r_mkt": med.groupby(grp).rank(pct=True, ascending=False).values,
        }, index=df.index)
        parts.append(M)
        names += list(M.columns)

    out = pd.concat(parts, axis=1)
    out.columns = names
    return out, names


def targets(df: pd.DataFrame) -> dict[str, np.ndarray]:
    fin = df[TARGET].to_numpy(dtype=float)
    n = df["field_size"].to_numpy(dtype=float)
    places = np.where(n >= 8, 3, np.where(n >= 5, 2, 1))
    return {
        "win": (fin == 1).astype(int),
        "place": (fin <= places).astype(int),
        "finish": fin,
        "places_paid": places,
    }


if __name__ == "__main__":
    d = load()
    X, names = build_features(d)
    t = targets(d)
    print(f"races {d.race_id.nunique()} | runners {len(d)} | features {len(names)}")
    print(f"date range {d.date.min()}–{d.date.max()}")
    print(f"win rate {t['win'].mean():.3f} | place rate {t['place'].mean():.3f}")
    p = market_probability(d)
    print(f"market probability available for {p.notna().mean():.1%} of runners")
    fav = d.assign(_p=p).dropna(subset=["_p"])
    idx = fav.groupby("race_id")["_p"].idxmax()
    print(f"favourite wins {(d.loc[idx, TARGET] == 1).mean():.3f} of "
          f"{len(idx)} races")
    book = d.assign(_q=1 / d[ODDS]).groupby("race_id")["_q"].sum()
    print(f"median overround {book.median():.3f}")
