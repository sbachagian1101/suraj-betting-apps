"""Parser for the Racing & Sports `<date>-<TRACK>-T.xlsx` meeting export.

A different input path to `horse_parser`, which reads a pasted Enhanced Form
page. This one reads a whole meeting from a spreadsheet, laid out as stacked
race blocks on a single sheet:

    <race no> |   | <race name>
              |   | Type : ...
    <time>    |   | <distance>m, <surface> <going>
              |   |
    Tab | Horse | Form L5 | BP | 12m% | ...      <- header, repeated per race
    <runner rows, until a blank line>

Two rules keep it honest:

1. **Blocks are found by their header row**, never by counting lines. The number
   of runners and the number of blank rows between blocks both vary, and Ffos Las
   and Navan disagree about both.
2. **A race ends at the first blank tab cell.** Not at a fixed offset, and not at
   the next header - a runner row is only a runner row while it has a tab number.

The percentage triplets (`12-28-40`) are **win% - place% - starts**, and are
split into three real columns. Read as text they are useless; read as only their
first number they throw away both the place rate and the sample size behind it,
and "0-0-0" from a first-starter would be indistinguishable from a genuine 0%
off forty runs.

**This export carries no market prices.** There is no odds column in it at all,
which is why `form_model` is a fundamentals-only model and why `race_quality`
cannot be applied to these files.
"""
from __future__ import annotations

import os
import re
from typing import Any

import numpy as np
import pandas as pd

HEADER = ["Tab", "Horse", "Form L5", "BP", "12m%", "Car%", "Dist%", "DLR",
          "Crs%", "JRat", "TRat", "PM 12m", "GD%", "Turf%", "AW%", "SH%",
          "PM Car", "LS Det", "CD%", "JH%", "JT%"]

# Columns holding a win%-place%-starts triplet.
TRIPLETS = ["12m%", "Car%", "Dist%", "Crs%", "GD%", "Turf%", "AW%", "SH%",
            "CD%", "JH%", "JT%"]

_DIST_RE = re.compile(r"^\s*(\d+)\s*m\s*,\s*(\S+)\s*(\S*)")
_TRIPLET_RE = re.compile(r"^\s*(-?\d+)\s*-\s*(-?\d+)\s*-\s*(-?\d+)\s*$")


def _num(v: Any) -> float:
    """Numeric value of a cell, tolerating '$1,234' and blanks."""
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return np.nan
    s = str(v).replace("$", "").replace(",", "").strip()
    if not s:
        return np.nan
    try:
        return float(s)
    except ValueError:
        return np.nan


def split_triplet(v: Any) -> tuple[float, float, float]:
    """'12-28-40' -> (win%, place%, starts). Anything else -> NaNs."""
    if not isinstance(v, str):
        return (np.nan, np.nan, np.nan)
    m = _TRIPLET_RE.match(v)
    if not m:
        return (np.nan, np.nan, np.nan)
    return tuple(float(x) for x in m.groups())


def form_figures(v: Any) -> tuple[float, int]:
    """Mean recent finishing position from a form string like 'x604x'.

    Digits are placings and `0` means "out of the placings", counted as 10 so it
    is treated as a bad run rather than the best possible one. `x` marks a spell
    and is skipped. Also returns how many runs were actually read, so a horse
    with one figure is not mistaken for one with five.
    """
    if not isinstance(v, str):
        return (np.nan, 0)
    ds = [10 if ch == "0" else int(ch) for ch in v if ch.isdigit()]
    return (float(np.mean(ds)) if ds else np.nan, len(ds))


def track_name(path: str) -> str:
    """'2026-08-27-FFOS-LAS-T.xlsx' -> 'FFOS-LAS'."""
    base = os.path.basename(path)
    base = re.sub(r"\.xlsx?$", "", base, flags=re.I)
    base = re.sub(r"^\d{4}-\d{2}-\d{2}-", "", base)
    return re.sub(r"-T$", "", base)


def meeting_date(path: str) -> str | None:
    m = re.match(r"^(\d{4}-\d{2}-\d{2})-", os.path.basename(path))
    return m.group(1) if m else None


def parse_grid(raw: pd.DataFrame, track: str, date: str | None = None) -> list[dict]:
    """Rows of runners from an already-loaded sheet."""
    hdr_rows = [i for i in range(len(raw))
                if isinstance(raw.iat[i, 0], str) and raw.iat[i, 0].strip() == "Tab"]
    out: list[dict] = []
    for block, h in enumerate(hdr_rows):
        rno, dist, surface, going, rname = None, np.nan, None, None, None
        # Metadata sits in the few rows above the header.
        for j in range(h - 1, max(-1, h - 8), -1):
            a, c = raw.iat[j, 0], raw.iat[j, 2]
            if isinstance(c, str):
                m = _DIST_RE.match(c)
                if m:
                    dist = float(m.group(1))
                    surface = m.group(2)
                    going = m.group(3) or None
            if (rno is None and isinstance(a, (int, float)) and not pd.isna(a)
                    and float(a) == int(a) and 1 <= int(a) <= 20
                    and isinstance(c, str) and not c.startswith("Type")):
                rno, rname = int(a), c
        if rno is None:
            rno = block + 1

        r = h + 1
        while r < len(raw):
            tab = raw.iat[r, 0]
            if pd.isna(tab) or not str(tab).strip():
                break
            if isinstance(tab, str) and tab.strip() == "Tab":
                break
            row = {"track": track, "date": date, "race": rno, "race_name": rname,
                   "dist": dist, "surface": surface, "going": going}
            for ci, name in enumerate(HEADER):
                row[name] = raw.iat[r, ci] if ci < raw.shape[1] else np.nan
            out.append(row)
            r += 1
    return out


def parse_file(path: str) -> list[dict]:
    raw = pd.read_excel(path, header=None)
    return parse_grid(raw, track_name(path), meeting_date(path))


def to_frame(rows: list[dict]) -> pd.DataFrame:
    """Runner rows -> a typed frame with the derived columns the model needs."""
    df = pd.DataFrame(rows)
    if df.empty:
        return df

    df["tab"] = df["Tab"].apply(_num)
    df["horse"] = df["Horse"].astype(str).str.strip()
    df["race_id"] = df["track"] + "_R" + df["race"].astype(str)
    for src, dst in [("BP", "bp"), ("DLR", "dlr"), ("JRat", "jrat"),
                     ("TRat", "trat"), ("PM 12m", "pm12"), ("PM Car", "pmcar")]:
        df[dst] = df[src].apply(_num)

    for c in TRIPLETS:
        w, p, n = zip(*df[c].apply(split_triplet))
        key = c.replace("%", "").lower()
        df[f"{key}_win"], df[f"{key}_plc"], df[f"{key}_runs"] = w, p, n

    ff = df["Form L5"].apply(form_figures)
    df["form_mean"] = [a for a, _ in ff]
    df["form_n"] = [b for _, b in ff]

    # Last start, e.g. "4-5.3L-BVY5-2011m-4U" or "1st-1.8L-GBHM5-1709m"
    ls = df["LS Det"].astype(str)
    df["ls_pos"] = ls.str.extract(r"^(\d+|1st)")[0].replace({"1st": "1"}).apply(_num)
    df["ls_mgn"] = ls.str.extract(r"-([\d.]+)L")[0].apply(_num)

    df["pm12_log"] = np.log1p(df["pm12"].clip(lower=0))
    df["pmcar_log"] = np.log1p(df["pmcar"].clip(lower=0))

    # Pick the going/surface record that matches the race actually being run,
    # instead of offering all four and hoping the weights sort it out.
    surf = df["surface"].astype(str).str.upper()
    synth = (surf.str.startswith("AW") | surf.str.startswith("DIRT")
             | surf.str.contains("SAND") | surf.str.contains("POLY"))
    turf = surf.str.startswith("TURF")
    df["surf_win"] = np.where(turf, df["turf_win"],
                              np.where(synth, df["aw_win"], df["gd_win"]))
    df["surf_plc"] = np.where(turf, df["turf_plc"],
                              np.where(synth, df["aw_plc"], df["gd_plc"]))

    df["n_runners"] = df.groupby("race_id")["tab"].transform("size")
    return df


def load(paths: list[str]) -> pd.DataFrame:
    rows: list[dict] = []
    for p in paths:
        rows += parse_file(p)
    return to_frame(rows)


def races(df: pd.DataFrame) -> list[str]:
    """Race ids in card order: each meeting's races together, in race order.

    Sorting on the race number alone would interleave the meetings - Carlisle R1,
    Ffos Las R1, Ipswich R1 - which is not how anyone reads a card.
    """
    if df.empty:
        return []
    key = df.groupby("race_id").agg(track=("track", "first"), race=("race", "first"))
    return list(key.sort_values(["track", "race"]).index)


def warnings_for(df: pd.DataFrame) -> list[str]:
    """Anything that should make the reader distrust a race."""
    out = []
    for rid, g in df.groupby("race_id", sort=False):
        tabs = sorted(int(t) for t in g["tab"].dropna())
        if not tabs:
            out.append(f"{rid}: no runners read")
            continue
        if tabs != list(range(1, len(tabs) + 1)):
            missing = sorted(set(range(1, max(tabs) + 1)) - set(tabs))
            out.append(f"{rid}: tab numbers are not 1..{len(tabs)} "
                       f"(missing {missing}) — scratchings, or a parse problem")
        if len(g) < 2:
            out.append(f"{rid}: only {len(g)} runner read")
    return out
