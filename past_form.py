"""Past-performance extraction from Racing & Sports meeting PDFs.

Each runner's detail section carries its full race history, one block per past
run, wrapped across text lines::

    6th of 8 18/7/2026 MORPHETTVILLE PARKS Margin 7.3 Lengths Distance 1000m
    SOT S RST MDN Race Ule Hoof Oil Maiden Plate Prize $50,000 API 2.71
    Race Time 0:59.89 Sec Time 35.17 (600) Sec Time Adj 1.17 Jockey Connor
    Murtagh Weight 57.5 CD 57.5 BP 3 Odds 17 Prize Won $1,000 Trainer Roslyn
    Day ... Winner Stand Alone 53.5 (2) Second Sensational Secret 56 (8)
    Third Street Legal 57.5 (1) Settled 4th 800m 4th Turn 5th

That is a labelled observation - finishing position out of a known field size,
**with the starting price** - and there are ~7,900 of them across these files.
They are the reason a model can be fitted at all: the current races in the
field tables carry no prices and no results.

Segmenting on the runner anchors matters. Both the field table and the detail
section start a runner with a bare ``N.`` line; they are told apart by what
follows - Title Case (``33xGolden Cloud``) in the table, ALL CAPS
(``GOLDEN CLOUD``) in the detail. History also runs across page boundaries, so
the document is flattened before it is split.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

try:
    import pymupdf
except ImportError:                                   # pragma: no cover
    import fitz as pymupdf

# a past run starts here; a trial has no finishing position of this shape
_RUN = re.compile(r"(\d+)(?:st|nd|rd|th)\s+of\s+(\d+)\s+(\d{1,2}/\d{1,2}/\d{4})")
_ANCHOR = re.compile(r"^\s*(\d{1,2})\.\s*$")
_CAPS = re.compile(r"^[A-Z][A-Z0-9 '&./()-]{2,}$")

_F = {
    "margin":    re.compile(r"Margin\s+([\d.]+)\s+Lengths"),
    "distance":  re.compile(r"Distance\s+(\d+)m"),
    "api":       re.compile(r"\bAPI\s+([\d.]+)"),
    "weight":    re.compile(r"\bWeight\s+([\d.]+)"),
    "cd":        re.compile(r"\bCD\s+([\d.]+)"),
    "barrier":   re.compile(r"\bBP\s+(\d+)"),
    "prize_won": re.compile(r"Prize Won\s+\$([\d,]+)"),
    "race_prize": re.compile(r"\bPrize\s+\$([\d,]+)"),
}
# "Odds 0.3F" is decimal odds MINUS ONE, with F/EF/JF marking favouritism.
# Read raw it turns a $1.30 winner into a 0.3 shot and every implied
# probability with it; the suffix is the only place favouritism is recorded.
_ODDS = re.compile(r"Odds\s+([\d.]+)\s*([A-Z]{0,2})")
_TIME = re.compile(r"Race Time\s+(\d+):([\d.]+)")
_SEC = re.compile(r"Sec Time\s+([\d.]+)\s*\((\d+)\)")
_JOCKEY = re.compile(r"\bJockey\s+([A-Z][A-Za-z'.-]*(?:\s+[A-Z][A-Za-z'.-]*){0,3})")
_TRACK = re.compile(r"\d{1,2}/\d{1,2}/\d{4}\s+([A-Z][A-Z' ]+?)(?=\s+Margin|\s+Distance|\s+Race\b)")
_COND = re.compile(r"\b(SOT|AWT|TRF|SND)\s+([A-Za-z0-9]+)")
_RUNPOS = re.compile(r"Settled\s+(\d+)\w*\s+800m\s+(\d+)\w*\s+Turn\s+(\d+)")
_WINNER = re.compile(r"\bWinner\s+(.+?)\s+[\d.]+\s*\(\d+\)")
_GEAR_CH = re.compile(r"Gear Change\s+(.+?)(?=\s+Winner\b|\s+Settled\b|$)")
_TRIAL = re.compile(r"(\d+)(?:st|nd|rd|th)\s+(\d{1,2}/\d{1,2}/\d{4})\s+([A-Z]+)\s+(\d+)m\s+Race Time")


def _n(m, g=1, cast=float):
    if not m:
        return np.nan
    try:
        return cast(m.group(g).replace(",", ""))
    except (ValueError, AttributeError):
        return np.nan


def _sections(path: str) -> list[tuple[str, str]]:
    """(horse name, flattened history text) for every runner in the file."""
    doc = pymupdf.open(path)
    lines: list[str] = []
    for p in range(doc.page_count):
        lines.extend(doc[p].get_text().splitlines())

    anchors = []
    for i in range(len(lines) - 1):
        if _ANCHOR.match(lines[i]) and _CAPS.match(lines[i + 1].strip()):
            anchors.append((i, lines[i + 1].strip()))
    out = []
    for k, (i, name) in enumerate(anchors):
        end = anchors[k + 1][0] if k + 1 < len(anchors) else len(lines)
        out.append((name, " ".join(x.strip() for x in lines[i:end])))
    return out


def parse_past_runs(path: str) -> pd.DataFrame:
    rows = []
    src = path.replace("\\", "/").rsplit("/", 1)[-1]
    for name, blob in _sections(path):
        starts = [m.start() for m in _RUN.finditer(blob)]
        for k, s in enumerate(starts):
            end = starts[k + 1] if k + 1 < len(starts) else len(blob)
            seg = blob[s:end]
            m = _RUN.match(seg)
            if not m:
                continue
            finish, field, date = int(m.group(1)), int(m.group(2)), m.group(3)
            t, sec, cond = _TIME.search(seg), _SEC.search(seg), _COND.search(seg)
            rp = _RUNPOS.search(seg)
            row = {
                "source": src, "horse": name.title(), "run_index": k,
                "finish": finish, "past_field_size": field,
                "date": pd.to_datetime(date, dayfirst=True, errors="coerce"),
                "track": (_TRACK.search(seg).group(1).strip()
                          if _TRACK.search(seg) else ""),
                "surface": cond.group(1) if cond else "",
                "going": cond.group(2) if cond else "",
                "race_time": (_n(t, 1) * 60 + _n(t, 2)) if t else np.nan,
                "sec_time": _n(sec, 1),
                "sec_at": _n(sec, 2),
                "jockey": (_JOCKEY.search(seg).group(1).strip()
                           if _JOCKEY.search(seg) else ""),
                "winner": (_WINNER.search(seg).group(1).strip()
                           if _WINNER.search(seg) else ""),
                "gear_change": (_GEAR_CH.search(seg).group(1).strip()[:80]
                                if _GEAR_CH.search(seg) else ""),
                "settled": _n(rp, 1), "at_800m": _n(rp, 2), "at_turn": _n(rp, 3),
            }
            for key, pat in _F.items():
                row[key] = _n(pat.search(seg))
            od = _ODDS.search(seg)
            raw = _n(od, 1)
            row["sp"] = raw + 1.0 if raw == raw else np.nan
            row["was_favourite"] = bool(od and od.group(2).endswith("F"))
            row["won"] = int(finish == 1)
            row["placed"] = int(finish <= 3)
            rows.append(row)
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    # a horse can appear at two meetings on the same weekend; its history is
    # identical, so keep one copy per (horse, date, distance)
    return df.drop_duplicates(subset=["horse", "date", "distance", "finish"])


def parse_trials(path: str) -> pd.DataFrame:
    rows = []
    for name, blob in _sections(path):
        for m in _TRIAL.finditer(blob):
            rows.append({"horse": name.title(), "place": int(m.group(1)),
                         "date": pd.to_datetime(m.group(2), dayfirst=True,
                                                errors="coerce"),
                         "track": m.group(3), "distance": int(m.group(4))})
    return pd.DataFrame(rows)


def parse_many(paths: list[str]) -> pd.DataFrame:
    fs = [parse_past_runs(p) for p in paths]
    fs = [f for f in fs if not f.empty]
    return pd.concat(fs, ignore_index=True) if fs else pd.DataFrame()


if __name__ == "__main__":
    import glob
    fs = sorted(glob.glob(
        "C:/Users/Admin/Downloads/[0-9][0-9][0-9][0-9][0-9][0-9].pdf"))
    d = parse_many(fs)
    d.to_parquet("past_runs.parquet")
    print(f"{len(d)} past runs | {d.horse.nunique()} horses | "
          f"{d.date.min():%Y-%m-%d} to {d.date.max():%Y-%m-%d}")
    print("\nfill rates:")
    for c in ["finish", "past_field_size", "sp", "margin", "distance",
              "weight", "barrier", "race_time", "sec_time", "api", "going",
              "settled", "jockey", "prize_won"]:
        v = d[c]
        ok = v.notna().mean() if v.dtype != object else (v != "").mean()
        print(f"  {c:16s} {100*ok:5.1f}%")
    print("\nSP sanity — win rate by starting price:")
    b = pd.cut(d.sp, [1, 2, 3, 4, 6, 9, 16, 31, 1000])
    print(d.groupby(b, observed=True).agg(
        n=("won", "size"), win=("won", "mean"), place=("placed", "mean"),
        implied=("sp", lambda s: (1 / s).mean())).to_string())
