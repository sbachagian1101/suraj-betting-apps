"""Loading and cleaning of FootyStats-style season match CSVs.

The loader is deliberately defensive about two things found in the real files:

* **`-1` sentinels.** Shot, foul and possession columns use `-1` to mean "not
  recorded", not a real value. Averaging them in would silently drag a team's
  attacking rating down.
* **Both-zero xG.** Three matches in the Latvian sample carry `team_a_xg == 0`
  *and* `team_b_xg == 0`, one of them a 6-1 result. That is missing data, not a
  genuine goalless-chance game, so it is dropped rather than believed.

Only rows with `status == "complete"` are ever used to fit a model.
"""
from __future__ import annotations

import io
from typing import Any

import numpy as np
import pandas as pd

# Columns the model and the ratings table rely on.
REQUIRED = ["home_team_name", "away_team_name",
            "home_team_goal_count", "away_team_goal_count"]
NUMERIC_SENTINEL = [
    "home_team_shots", "away_team_shots",
    "home_team_shots_on_target", "away_team_shots_on_target",
    "home_team_shots_off_target", "away_team_shots_off_target",
    "home_team_fouls", "away_team_fouls",
    "home_team_possession", "away_team_possession",
    "home_team_corner_count", "away_team_corner_count",
    "team_a_xg", "team_b_xg",
]
ODDS_COLS = ["odds_ft_home_team_win", "odds_ft_draw", "odds_ft_away_team_win",
             "odds_btts_yes", "odds_btts_no", "odds_ft_over25"]

_DATE_FORMATS = ("%b %d %Y %I:%M%p", "%b %d %Y %H:%M", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d")


def _parse_dates(s: pd.Series) -> pd.Series:
    raw = s.astype(str).str.replace(" - ", " ", regex=False).str.strip()
    out = pd.Series(pd.NaT, index=s.index, dtype="datetime64[ns]")
    for fmt in _DATE_FORMATS:
        miss = out.isna()
        if not miss.any():
            break
        out.loc[miss] = pd.to_datetime(raw[miss], format=fmt, errors="coerce")
    miss = out.isna()
    if miss.any():
        out.loc[miss] = pd.to_datetime(raw[miss], errors="coerce")
    return out


def load_frames(files: list[Any], names: list[str] | None = None) -> tuple[pd.DataFrame, list[str]]:
    """Read one or more CSVs (paths or file-like) into a single tidy frame."""
    notes: list[str] = []
    frames = []
    for i, f in enumerate(files):
        label = (names[i] if names else None) or getattr(f, "name", f"file {i+1}")
        try:
            if hasattr(f, "read"):
                data = f.read()
                d = pd.read_csv(io.BytesIO(data) if isinstance(data, bytes) else io.StringIO(data))
            else:
                d = pd.read_csv(f)
        except Exception as exc:                              # noqa: BLE001
            notes.append(f"❌ {label}: could not be read ({exc}).")
            continue
        missing = [c for c in REQUIRED if c not in d.columns]
        if missing:
            notes.append(f"❌ {label}: skipped, missing required column(s) {', '.join(missing)}.")
            continue
        d["source_file"] = str(label)
        frames.append(d)
    if not frames:
        return pd.DataFrame(), notes + ["No usable files were loaded."]
    a = pd.concat(frames, ignore_index=True)
    return clean(a, notes)


def clean(a: pd.DataFrame, notes: list[str] | None = None) -> tuple[pd.DataFrame, list[str]]:
    notes = list(notes or [])
    n_all = len(a)

    if "status" in a.columns:
        incomplete = int((a["status"] != "complete").sum())
        a = a[a["status"] == "complete"].copy()
        if incomplete:
            notes.append(
                f"Excluded {incomplete} match(es) not marked complete "
                "(fixtures not yet played, abandoned or suspended).")
    else:
        a = a.copy()
        notes.append("No `status` column found - assuming every row is a completed match.")

    for c in NUMERIC_SENTINEL + ODDS_COLS:
        if c in a.columns:
            a[c] = pd.to_numeric(a[c], errors="coerce")

    sent = 0
    for c in NUMERIC_SENTINEL:
        if c in a.columns:
            hit = (a[c] == -1)
            sent += int(hit.sum())
            a.loc[hit, c] = np.nan
    if sent:
        notes.append(f"Treated {sent} `-1` entries as missing rather than as real values.")

    if {"team_a_xg", "team_b_xg"} <= set(a.columns):
        both0 = (a["team_a_xg"] == 0) & (a["team_b_xg"] == 0)
        if both0.any():
            notes.append(
                f"Dropped xG for {int(both0.sum())} match(es) where both teams recorded "
                "exactly 0.00 xG - that is missing data, not a real scoreline.")
            a.loc[both0, ["team_a_xg", "team_b_xg"]] = np.nan

    a["home_team_name"] = a["home_team_name"].astype(str).str.strip()
    a["away_team_name"] = a["away_team_name"].astype(str).str.strip()
    a["hg"] = pd.to_numeric(a["home_team_goal_count"], errors="coerce")
    a["ag"] = pd.to_numeric(a["away_team_goal_count"], errors="coerce")
    bad = a["hg"].isna() | a["ag"].isna()
    if bad.any():
        notes.append(f"Dropped {int(bad.sum())} match(es) with an unreadable score.")
        a = a[~bad]
    a["hg"] = a["hg"].astype(int)
    a["ag"] = a["ag"].astype(int)

    if "date_GMT" in a.columns:
        a["date"] = _parse_dates(a["date_GMT"])
    else:
        a["date"] = pd.NaT
    if a["date"].isna().all():
        notes.append("No usable dates found - matches are kept in file order and "
                     "time-decay weighting is disabled.")
        a["date"] = pd.Timestamp("2000-01-01") + pd.to_timedelta(np.arange(len(a)), unit="D")
    elif a["date"].isna().any():
        n = int(a["date"].isna().sum())
        a = a[~a["date"].isna()]
        notes.append(f"Dropped {n} match(es) with an unreadable date.")

    a = a.sort_values("date").reset_index(drop=True)
    a["season"] = a["date"].dt.year
    a = a.drop_duplicates(subset=["date", "home_team_name", "away_team_name"], keep="first")

    dropped = n_all - len(a)
    notes.insert(0, f"Loaded **{len(a)}** completed matches "
                    f"({dropped} row(s) excluded) across "
                    f"{a['season'].nunique()} season(s), {teams_of(a).__len__()} teams.")
    return a.reset_index(drop=True), notes


def teams_of(a: pd.DataFrame) -> list[str]:
    if a.empty:
        return []
    return sorted(set(a["home_team_name"]) | set(a["away_team_name"]))


def team_table(a: pd.DataFrame, model=None) -> pd.DataFrame:
    """Per-team descriptive record plus, if given, the fitted strength indexes."""
    rows = []
    for t in teams_of(a):
        h = a[a["home_team_name"] == t]
        v = a[a["away_team_name"] == t]
        n = len(h) + len(v)
        gf = h["hg"].sum() + v["ag"].sum()
        ga = h["ag"].sum() + v["hg"].sum()
        w = int((h["hg"] > h["ag"]).sum() + (v["ag"] > v["hg"]).sum())
        d = int((h["hg"] == h["ag"]).sum() + (v["ag"] == v["hg"]).sum())
        row = {
            "Team": t, "P": n, "W": w, "D": d, "L": n - w - d,
            "GF": int(gf), "GA": int(ga), "GD": int(gf - ga),
            "PPG": round((3 * w + d) / n, 2) if n else 0.0,
            "GF/g": round(gf / n, 2) if n else 0.0,
            "GA/g": round(ga / n, 2) if n else 0.0,
        }
        for label, hc, vc in (("xGF/g", "team_a_xg", "team_b_xg"),
                              ("xGA/g", "team_b_xg", "team_a_xg"),
                              ("SoT/g", "home_team_shots_on_target", "away_team_shots_on_target"),
                              ("SoTA/g", "away_team_shots_on_target", "home_team_shots_on_target")):
            if hc in a.columns and vc in a.columns:
                vals = pd.concat([h[hc], v[vc]]).dropna()
                row[label] = round(float(vals.mean()), 2) if len(vals) else np.nan
        if "team_a_xg" in a.columns and row.get("xGF/g") == row.get("xGF/g"):
            row["xG diff/g"] = round(row["GF/g"] - row["xGF/g"], 2)
        if model is not None and t in model.index:
            i = model.index[t]
            row["Attack idx"] = round(float(np.exp(model.attack[i])), 3)
            row["Defence idx"] = round(float(np.exp(model.defence[i])), 3)
            row["Exp GF (neutral)"] = round(float(np.exp(model.mu + model.attack[i])), 2)
            row["Exp GA (neutral)"] = round(float(np.exp(model.mu + model.defence[i])), 2)
        rows.append(row)
    df = pd.DataFrame(rows)
    sort_col = "Attack idx" if "Attack idx" in df.columns else "PPG"
    return df.sort_values(sort_col, ascending=False).reset_index(drop=True)
