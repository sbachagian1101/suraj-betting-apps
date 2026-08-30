"""Parser for Racing & Sports meeting form-guide PDFs (Australia).

The text layer emits one token per line, so the field table can be read as a
fixed run of fields per runner rather than by hunting whitespace. Two layout
quirks make a naive scan wrong, and both cost real runners:

* The tab number is **sometimes on its own line and sometimes glued** to the
  form figures::

      1.                        4. x8929Bold Starlet
      33xGolden Cloud           4m

  Matching only the glued form recovered 4 of 9 runners at Strathalbyn.

* The form figures are **glued to the horse name** with no separator, so the
  split has to be made on case. Matching the figures case-insensitively ate the
  leading letter of every horse whose name began with one of them - giving
  `x7036D` + `ance Dance Dance` instead of `x7036` + `Dance Dance Dance`.

Two fields here are genuinely absent from the CSV export and are the reason
these files are worth parsing at all: **gear** (`[Blinkers]`, `[Tongue Tie]`,
`[Ear Muffs]`) and the **comment line** ("First up; Recent trial; Suited by
better draw"), which encode trials, gear moves and class changes that no
numeric column carries.
"""
from __future__ import annotations

import re
from typing import Any

import numpy as np
import pandas as pd

try:
    import pymupdf
except ImportError:                                   # pragma: no cover
    import fitz as pymupdf

_TAB = re.compile(r"^\s*(\d{1,2})\.\s*(.*)$")
# case-sensitive: the figures are lowercase, the horse name is Title Case
_FF_HORSE = re.compile(r"^([0-9xfdsopu/-]*?)([A-Z].*)$")
_AGE_SEX = re.compile(r"^\s*(\d{1,2})\s*([a-zA-Z])\s*$")
_WEIGHT = re.compile(r"^\s*([\d.]+)\s*kg\s*$", re.I)
_BP_JOCKEY = re.compile(r"^\s*(\d{1,2})?\s*(.*?)(?:\s*\(a([\d.]+)\))?\s*$")
_CAREER = re.compile(r"^\s*(\d+)\s*-\s*(\d+)\s*-\s*(\d+)\s*$")
_MONEY = re.compile(r"^\s*\$([\d,]+)\s*$")
_RACE_HDR = re.compile(
    r"(\d{1,2} \w{3} \d{2})\s+(\d{1,2}:\d{2}[ap]m)\s+([A-Z][A-Z' ]+?)\s+(\d+)m")
_PRIZE = re.compile(r"Prizemoney:\s*\$([\d,]+)")
_GEAR = re.compile(r"\[([^\]]+)\]")
_CLASS = re.compile(r"^\s*(.*?)\s*Prizemoney:")


def _num(s, default=np.nan) -> float:
    try:
        return float(str(s).strip().replace(",", ""))
    except (TypeError, ValueError):
        return default


def _money(s: str) -> float:
    m = _MONEY.match(s or "")
    return float(m.group(1).replace(",", "")) if m else np.nan


def _race_header(text: str, lines: list[str], path: str):
    hdr = _RACE_HDR.search(text)
    if not hdr:
        return None
    race_no, name = None, ""
    for i, ln in enumerate(lines):
        if ln.strip() == "Race No":
            for j in range(i + 1, min(i + 4, len(lines))):
                if lines[j].strip().isdigit():
                    race_no = int(lines[j].strip())
                    break
            break
    for i, ln in enumerate(lines):
        if _RACE_HDR.search(ln):
            if i + 1 < len(lines):
                name = lines[i + 1].strip()
            break
    pm = _PRIZE.search(text)
    cls = ""
    for ln in lines:
        c = _CLASS.match(ln)
        if c and c.group(1):
            cls = c.group(1).strip()
            break
    return {
        "date": hdr.group(1), "time": hdr.group(2),
        "track": hdr.group(3).strip(), "distance": int(hdr.group(4)),
        "race_no": race_no, "race_name": name, "race_class": cls,
        "prize": float(pm.group(1).replace(",", "")) if pm else np.nan,
        "source": path.replace("\\", "/").rsplit("/", 1)[-1],
    }


def _read_runner(lines: list[str], i: int):
    """One field-table row starting at line `i`, or None if this isn't one."""
    m = _TAB.match(lines[i])
    if not m:
        return None
    tab, rest = int(m.group(1)), m.group(2).strip()
    j = i + 1
    if not rest:                                  # tab alone on its own line
        if j >= len(lines):
            return None
        rest = lines[j].strip()
        j += 1
    fh = _FF_HORSE.match(rest)
    if not fh or j + 8 >= len(lines):
        return None

    a = _AGE_SEX.match(lines[j])
    w = _WEIGHT.match(lines[j + 1])
    car = _CAREER.match(lines[j + 4])
    # all three must hold: a detail page can throw up a plausible-looking tab
    # line, and without the career triple those rows land in the field table
    if not (a and w and car):
        return None

    bp_j = _BP_JOCKEY.match(lines[j + 2])
    rtc = lines[j + 6].strip()
    dlw = lines[j + 8].strip()
    return {
        "tab": tab,
        "form_figures": fh.group(1),
        "horse": fh.group(2).strip(),
        "age": _num(a.group(1)),
        "sex": a.group(2).lower(),
        "weight": float(w.group(1)),
        "barrier": _num(bp_j.group(1)) if bp_j and bp_j.group(1) else np.nan,
        "jockey": bp_j.group(2).strip() if bp_j else "",
        "claim": float(bp_j.group(3)) if bp_j and bp_j.group(3) else 0.0,
        "trainer": lines[j + 3].strip(),
        "career_wins": _num(car.group(1)),
        "career_places": _num(car.group(2)),
        "career_starts": _num(car.group(3)),
        "career_prize": _money(lines[j + 5]),
        "first_up": rtc.upper() == "FU",
        "second_up": rtc.upper() == "SU",
        "runs_this_prep": _num(rtc),
        "days_since_run": _num(lines[j + 7]),
        "days_since_win": np.nan if dlw.lower().startswith("mdn") else _num(dlw),
        "never_won": dlw.lower().startswith("mdn"),
    }, j + 9


def parse_meeting(path: str) -> pd.DataFrame:
    """Every runner in every race of one meeting PDF."""
    doc = pymupdf.open(path)
    rows: list[dict[str, Any]] = []
    race = None

    for pno in range(doc.page_count):
        text = doc[pno].get_text()
        lines = text.splitlines()

        hdr = _race_header(text, lines, path)
        if hdr:
            race = hdr
        if race is None:
            continue

        comment = ""
        for ln in lines[:4]:
            t = ln.strip()
            if ";" in t and len(t) > 20:
                comment = t
                break
        gear = "; ".join(_GEAR.findall(comment)) if comment else ""

        i = 0
        while i < len(lines):
            got = _read_runner(lines, i)
            if got is None:
                i += 1
                continue
            row, i = got
            rows.append({**race, **row, "page": pno + 1,
                         "comment": comment, "gear": gear})

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df = df.drop_duplicates(subset=["track", "date", "race_no", "tab"],
                            keep="first")
    df["race_id"] = (df["track"].str.replace(" ", "_") + "_"
                     + df["date"].str.replace(" ", "") + "_R"
                     + df["race_no"].astype("Int64").astype(str))
    df["field_size"] = df.groupby("race_id")["tab"].transform("size")
    return df.reset_index(drop=True)


def parse_many(paths: list[str]) -> pd.DataFrame:
    frames = [parse_meeting(p) for p in paths]
    frames = [f for f in frames if not f.empty]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def warnings_for(df: pd.DataFrame) -> list[str]:
    if df.empty:
        return ["No runners were read from those files."]
    out = []
    for rid, g in df.groupby("race_id", sort=False):
        tabs = sorted(int(t) for t in g["tab"].dropna())
        if tabs != list(range(1, len(tabs) + 1)):
            missing = sorted(set(range(1, max(tabs) + 1)) - set(tabs))
            out.append(f"{rid}: tabs run to {max(tabs)} but {len(tabs)} read "
                       f"(gaps at {missing}) - scratchings, or a parse problem")
        if len(g) < 4:
            out.append(f"{rid}: only {len(g)} runner(s) read")
    return out


if __name__ == "__main__":
    import glob
    fs = sorted(glob.glob(
        "C:/Users/Admin/Downloads/[0-9][0-9][0-9][0-9][0-9][0-9].pdf"))
    d = parse_many(fs)
    print(f"{len(d)} runners | {d.race_id.nunique()} races | "
          f"{d.track.nunique()} tracks | {len(fs)} files")
    sz = d.groupby("race_id").size()
    print(f"field sizes {sz.min()}-{sz.max()}, mean {sz.mean():.1f}")
    w = warnings_for(d)
    print(f"warnings: {len(w)}")
    for x in w[:10]:
        print("  ", x)
    print("\nblank horses:", int((d.horse.str.strip() == "").sum()),
          "| blank jockeys:", int((d.jockey.str.strip() == "").sum()))
    print("\nStrathalbyn R1:")
    print(d[d.race_id == "STRATHALBYN_30Aug26_R1"][
        ["tab", "form_figures", "horse", "age", "sex", "weight", "barrier",
         "jockey", "claim", "career_wins", "career_starts"]
    ].to_string(index=False))
