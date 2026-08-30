"""Split a Racing & Sports meeting export into the race blocks the engine reads.

The `-T.xlsx` meeting file is laid out exactly like the text people paste in -
race header, `Type :` line, time and distance, blank, the `Tab Horse Form L5`
header, then one row per runner - just distributed across spreadsheet cells
instead of tab characters. So the conversion is a join, not a translation, and
the engine's parser is used unchanged.

The meeting file carries two columns the pasted format does not, `JH%` and
`JT%`. They are kept in the emitted text: the parser reads columns by header
name, so trailing extras are ignored rather than shifting anything.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

import pandas as pd


@dataclass
class RaceBlock:
    number: int
    name: str
    text: str
    runners: int


def _cells(row) -> list[str]:
    out = []
    for x in row:
        if x is None or (isinstance(x, float) and pd.isna(x)):
            out.append("")
        elif isinstance(x, float) and x.is_integer():
            out.append(str(int(x)))
        else:
            out.append(str(x).strip())
    return out


def read_meeting(src, filename: str | None = None) -> pd.DataFrame:
    """Read a meeting export as raw cells.

    ``filename`` has to be passed for an in-memory stream. A ``BytesIO`` has no
    ``.name``, so sniffing the extension off the object falls through to the
    CSV reader and an .xlsx dies on ``UnicodeDecodeError`` at the first binary
    byte — which is exactly what happened to the bundled sample.

    ``header=None`` is deliberate: the sheet's first row is the first race's
    header, not column names, so letting pandas promote it would swallow a race.
    """
    name = str(filename or getattr(src, "name", "") or src or "").lower()
    if name.endswith((".xlsx", ".xlsm", ".xls")):
        return pd.read_excel(src, header=None)
    return pd.read_csv(src, header=None, index_col=False)


def split_races(df: pd.DataFrame) -> list[RaceBlock]:
    """Every race in the meeting, as text the engine can parse."""
    rows = [_cells(df.iloc[i]) for i in range(len(df))]

    # A race starts where column 0 is a bare number and column 2 names the
    # race. Column 1 must be EMPTY: a runner row also carries a number in
    # column 0 and text in column 2 (its form figures), so without that test
    # every runner is mistaken for the start of a new race.
    #
    # Nothing else is tested against column 2. Excluding names beginning
    # "tab" to skip the header row looks harmless and is not: it silently ate
    # the TABTOUCH CARNARVON CUP, merging it into the previous race. TABtouch
    # sponsors races all over Australia. The header row is already excluded by
    # the numeric test, since it carries "Tab" in column 0.
    starts = []
    for i, r in enumerate(rows):
        if (re.fullmatch(r"\d{1,2}", r[0] or "")
                and len(r) > 2 and r[2] and not r[1]
                and not re.fullmatch(r"[\d.,]+", r[2])):
            starts.append(i)

    blocks: list[RaceBlock] = []
    for k, s in enumerate(starts):
        end = starts[k + 1] if k + 1 < len(starts) else len(rows)
        chunk = rows[s:end]
        # drop the run of blank rows that separates meetings
        while chunk and not any(c for c in chunk[-1]):
            chunk.pop()
        if not chunk:
            continue

        n_runners = 0
        seen_header = False
        for r in chunk:
            if r[0] == "Tab":
                seen_header = True
            elif seen_header and re.fullmatch(r"\d{1,2}", r[0] or ""):
                n_runners += 1
        if not seen_header or n_runners < 2:
            continue

        text = "\n".join("\t".join(r).rstrip("\t") for r in chunk)
        blocks.append(RaceBlock(number=int(chunk[0][0]), name=chunk[0][2],
                                text=text, runners=n_runners))
    return blocks


def meeting_label(df: pd.DataFrame) -> str:
    for i in range(min(len(df), 40)):
        for cell in _cells(df.iloc[i]):
            m = re.search(r"\b([A-Z][A-Z' ]{3,})\b", cell)
            if m and "TYPE" not in m.group(1):
                return m.group(1).strip()
    return "meeting"


if __name__ == "__main__":
    import sys
    path = (sys.argv[1] if len(sys.argv) > 1
            else "C:/Users/Admin/Downloads/2026-08-30-CARNARVON-T.xlsx")
    d = read_meeting(path)
    bs = split_races(d)
    print(f"{len(bs)} races")
    for b in bs:
        print(f"  R{b.number}  {b.runners:2d} runners  {b.name}")
    print("\n--- race 2 as the engine will see it ---")
    print(bs[1].text[:600])
