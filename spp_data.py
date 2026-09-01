"""Loading for SoccerPredictorPro.

Accepts **either** FootyStats export shape and works out which is which from the
columns present:

* **Match files** (`...-matches-...csv`) - one row per fixture. Richer: they
  carry individual scorelines, so a Dixon-Coles low-score correction can be
  *fitted* rather than assumed, and matches can be time-decayed.
* **Team files** (`...-teams-...csv`) - one row per team per season, holding
  season-to-date scoring and conceding rates split by venue, at full time and
  at half time. Everything the model needs is already aggregated.

Both routes end at the same place: per-team home/away attack and defence rates
on a goals-per-match scale, which `spp_model` turns into a score distribution.

The team-file route is the simpler one and maps directly onto the two-season
recommendation: last season's file is a completed table, this season's file is a
to-date snapshot, and the loader weights them by matches played.
"""
from __future__ import annotations

import hashlib
import io
from typing import Any

import numpy as np
import pandas as pd

# --- how each shape is recognised -------------------------------------------
MATCH_MARKERS = ("home_team_name", "away_team_name", "home_team_goal_count")
TEAM_MARKERS = ("common_name", "goals_scored_per_match_home", "matches_played")

TEAM_RATE_COLS = [
    "goals_scored_per_match_home", "goals_scored_per_match_away",
    "goals_conceded_per_match_home", "goals_conceded_per_match_away",
]
TEAM_HT_COLS = [
    "goals_scored_per_match_half_time_home", "goals_scored_per_match_half_time_away",
    "goals_conceded_per_match_half_time_home", "goals_conceded_per_match_half_time_away",
]

KIND_MATCH = "match"
KIND_TEAM = "team"


def sniff(df: pd.DataFrame) -> str | None:
    cols = set(df.columns)
    if all(c in cols for c in MATCH_MARKERS):
        return KIND_MATCH
    if all(c in cols for c in TEAM_MARKERS):
        return KIND_TEAM
    return None


def _read(f: Any) -> pd.DataFrame:
    if hasattr(f, "read"):
        data = f.read()
        return pd.read_csv(io.BytesIO(data) if isinstance(data, bytes) else io.StringIO(data))
    return pd.read_csv(f)


def load(files: list[Any], names: list[str] | None = None):
    """Read uploads and return ``(kind, frame, notes)``.

    Mixed uploads are not merged - whichever shape supplies more files wins and
    the rest are reported as skipped, because silently blending a season of
    match rows with a season of team rows would double-count that season.
    """
    notes: list[str] = []
    buckets: dict[str, list[pd.DataFrame]] = {KIND_MATCH: [], KIND_TEAM: []}
    seen: set[str] = set()

    for i, f in enumerate(files):
        label = (names[i] if names else None) or getattr(f, "name", f"file {i + 1}")
        try:
            d = _read(f)
        except Exception as exc:                                   # noqa: BLE001
            notes.append(f"Could not read **{label}** ({exc}).")
            continue
        # A duplicated download ("... (1).csv") is common in a Downloads folder
        # and would double every rate it touches. The signature must be the
        # file's *contents*: an earlier version keyed on row count plus column
        # names, and since every season of a 16-team league exports 16 rows with
        # identical headers, it threw away a whole season as a false duplicate.
        sig = hashlib.sha1(
            pd.util.hash_pandas_object(d, index=False).values.tobytes()).hexdigest()
        if sig in seen:
            notes.append(f"Skipped **{label}** - it is a duplicate of a file already loaded.")
            continue
        kind = sniff(d)
        if kind is None:
            notes.append(f"Skipped **{label}** - not a recognised FootyStats match or team export.")
            continue
        seen.add(sig)
        d["source_file"] = str(label)
        buckets[kind].append(d)

    if not buckets[KIND_MATCH] and not buckets[KIND_TEAM]:
        return None, pd.DataFrame(), notes + ["No usable files were loaded."]

    kind = KIND_MATCH if len(buckets[KIND_MATCH]) >= len(buckets[KIND_TEAM]) else KIND_TEAM
    other = KIND_TEAM if kind == KIND_MATCH else KIND_MATCH
    if buckets[other]:
        notes.append(
            f"Ignored {len(buckets[other])} {other} file(s) - this run is using the "
            f"{len(buckets[kind])} {kind} file(s). Upload one shape at a time so a "
            "season is not counted twice.")

    frame = pd.concat(buckets[kind], ignore_index=True)
    if kind == KIND_TEAM:
        return KIND_TEAM, *_clean_team(frame, notes)
    return KIND_MATCH, *_clean_match(frame, notes)


# --- team-file route ---------------------------------------------------------

def _clean_team(a: pd.DataFrame, notes: list[str]):
    a = a.copy()
    for c in TEAM_RATE_COLS + TEAM_HT_COLS + ["matches_played", "matches_played_home",
                                              "matches_played_away"]:
        if c in a.columns:
            a[c] = pd.to_numeric(a[c], errors="coerce")

    a["team"] = a["common_name"].astype(str).str.strip()
    if "season" in a.columns:
        a["season_label"] = a["season"].astype(str)
        a["season_start"] = pd.to_numeric(
            a["season_label"].str.slice(0, 4), errors="coerce")
    else:
        a["season_label"] = "unknown"
        a["season_start"] = 0

    missing = [c for c in TEAM_RATE_COLS if c not in a.columns]
    if missing:
        notes.append(f"Team files are missing {', '.join(missing)} - cannot build ratings.")
        return pd.DataFrame(), notes

    a = a[a["matches_played"].fillna(0) > 0]
    if a.empty:
        notes.append("Every team row shows zero matches played - nothing to fit.")
        return pd.DataFrame(), notes

    a = a.drop_duplicates(subset=["team", "season_label"], keep="last")
    seasons = sorted(a["season_label"].unique())
    notes.insert(0, f"Loaded **{a['team'].nunique()} teams** across "
                    f"{len(seasons)} season(s): {', '.join(seasons)}.")
    thin = a[a["matches_played"] < 5]
    if not thin.empty:
        notes.append(
            f"{len(thin)} team-season row(s) have under 5 matches played; those rates are "
            "shrunk hard toward the league average rather than trusted.")
    return a.reset_index(drop=True), notes


# --- match-file route --------------------------------------------------------

def _clean_match(a: pd.DataFrame, notes: list[str]):
    import soccer_data as sd
    a = a.copy()
    # Reuse the proven cleaner (status filter, -1 sentinels, dates, dedupe).
    frame, more = sd.clean(a, [])
    notes.extend(more)
    if frame.empty:
        return frame, notes
    for c in ("home_team_goal_count_half_time", "away_team_goal_count_half_time"):
        if c in frame.columns:
            frame[c] = pd.to_numeric(frame[c], errors="coerce")
    if "home_team_goal_count_half_time" in frame.columns:
        ok = frame["home_team_goal_count_half_time"].notna()
        notes.append(f"Half-time scores present for {int(ok.sum())} of {len(frame)} matches.")
    else:
        notes.append("No half-time columns found - half-time and HT/FT markets will be "
                     "unavailable for this upload.")
    return frame, notes


def teams(kind: str, frame: pd.DataFrame) -> list[str]:
    if frame.empty:
        return []
    if kind == KIND_TEAM:
        return sorted(frame["team"].unique())
    return sorted(set(frame["home_team_name"]) | set(frame["away_team_name"]))
