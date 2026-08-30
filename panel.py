"""Parse the FootyStats team-comparison panel.

A pasted panel looks like this, and the numbers arrive **concatenated with no
separator at all**::

    AIK Fotboll
    Sweden - Allsvenskan
    League Pos. 7 / 16
    FormResultsPPG
    Overall
    WDLWW
    1.65
    ...
    StatsOverallHomeAway
    Win %47%43%50%
    AVG2.942.713.10
    Scored1.471.001.80
    xG1.491.791.28

`2.942.713.10` has to become 2.94, 2.71, 3.10 — and it is genuinely ambiguous
as a string: `1.542.231.2` could be split 1.54/2.23/1.2 or 1.5/42.2/31.2, and
FootyStats drops trailing zeros so the field widths are not fixed.

The disambiguator is arithmetic, not a guess. **Overall is a weighted average of
Home and Away**, so it must lie between them. That single constraint kills every
wrong tokenisation on every row tested, because the bogus splits throw one value
into the tens or hundreds while Overall stays small.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

# rows of the stats block, with a sane ceiling used only to reject nonsense
ROWS = {
    "Win %": ("win_pct", 100.0),
    "AVG": ("avg_goals", 15.0),
    "Scored": ("scored", 10.0),
    "Conceded": ("conceded", 10.0),
    "BTTS": ("btts_pct", 100.0),
    "CS": ("cs_pct", 100.0),
    "FTS": ("fts_pct", 100.0),
    "xG": ("xg", 8.0),
    "xGA": ("xga", 8.0),
}
COLS = ("overall", "home", "away")

_NUM = re.compile(r"\d{1,3}(?:\.\d{1,2})?")
_POS = re.compile(r"League Pos\.\s*(\d+)\s*/\s*(\d+)")
_RECENT = re.compile(r"Recent\s*:\s*(\d+)\s*Wins?\s*/\s*(\d+)\s*Draws?\s*/\s*(\d+)\s*Loss")
_FORMROW = re.compile(r"^(Overall|Home|Away)$")
_RESULTS = re.compile(r"^[WDL]{1,12}$")
_PPG = re.compile(r"^\d{1,2}\.\d{1,2}$")
_LEAGUE = re.compile(r"^[A-Z][\w .'()-]{2,40}\s+-\s+[\w .'()/-]{2,50}$")


def _split_pcts(s: str):
    parts = re.findall(r"(\d{1,3})\s*%", s)
    return [float(p) for p in parts] if len(parts) == 3 else None


def split_three(s: str, ceiling: float = 15.0):
    """Break a run like `2.942.713.10` into its three numbers.

    Every tokenisation into three numbers is enumerated, then filtered by the
    fact that **Overall sits between Home and Away** — it is their weighted
    average, so it cannot be outside them by more than rounding. Where more
    than one candidate survives, the one whose three parts share a decimal
    width wins, since FootyStats formats a row consistently.
    """
    s = s.strip()
    n = len(s)
    out: list[tuple[str, str, str]] = []

    def walk(i: int, parts: list[str]):
        if len(parts) == 3:
            if i == n:
                out.append((parts[0], parts[1], parts[2]))
            return
        for j in range(i + 1, min(i + 6, n) + 1):
            tok = s[i:j]
            if _NUM.fullmatch(tok):
                walk(j, parts + [tok])

    walk(0, [])
    if not out:
        return None

    scored = []
    for cand in out:
        v = [float(x) for x in cand]
        if any(x > ceiling for x in v):
            continue
        o, h, a = v
        if not (min(h, a) - 0.06 <= o <= max(h, a) + 0.06):
            continue
        widths = {len(x.split(".")[1]) if "." in x else 0 for x in cand}
        scored.append((len(widths), -sum(1 for x in cand if "." in x), v))
    if not scored:
        return None
    scored.sort(key=lambda t: (t[0], t[1]))
    return scored[0][2]


def _clean(x: str) -> str:
    return re.sub(r"\s+", " ", x or "").strip()


def parse_panels(text: str) -> list[dict]:
    """Every team panel in the pasted text, in the order they appear."""
    lines = [_clean(x) for x in text.replace("\r\n", "\n").split("\n")]
    n = len(lines)

    # a panel is anchored on its stats header
    anchors = [i for i, l in enumerate(lines)
               if l.replace(" ", "").lower().startswith("statsoverallhomeaway")]
    panels = []
    for k, a in enumerate(anchors):
        start = anchors[k - 1] if k else 0
        head = lines[start:a]

        league = pos = rank = None
        name = ""
        for j in range(len(head) - 1, -1, -1):
            t = head[j]
            if pos is None:
                m = _POS.search(t)
                if m:
                    pos, rank = int(m.group(1)), int(m.group(2))
                    continue
            if league is None and _LEAGUE.match(t):
                league = t
                # the club name is the last distinct line above the league
                for q in range(j - 1, -1, -1):
                    cand = head[q]
                    if cand and cand != league and not _POS.search(cand) \
                            and not _RECENT.search(cand) \
                            and not cand.lower().startswith(("form", "stats")):
                        name = cand
                        break
                break
        if not name:
            for t in reversed(head):
                if t and not t.lower().startswith(("form", "stats", "overall",
                                                   "home", "away")) \
                        and not _RESULTS.match(t) and not _PPG.match(t) \
                        and not _POS.search(t) and not _RECENT.search(t):
                    name = t
                    break

        rec = None
        for t in head:
            m = _RECENT.search(t)
            if m:
                rec = tuple(int(g) for g in m.groups())
                break

        # form block: Overall / Home / Away, each with results then PPG
        form: dict = {}
        j = 0
        while j < len(head):
            m = _FORMROW.match(head[j])
            if m:
                key = m.group(1).lower()
                res, ppg = "", np.nan
                for q in range(j + 1, min(j + 4, len(head))):
                    if _RESULTS.match(head[q]) and not res:
                        res = head[q]
                    elif _PPG.match(head[q]):
                        ppg = float(head[q])
                        break
                form[key] = {"results": res, "ppg": ppg}
            j += 1

        stats: dict = {}
        end = anchors[k + 1] if k + 1 < len(anchors) else n
        for j in range(a + 1, end):
            t = lines[j]
            # longest label first: "xGA1.57..." also starts with "xG", and
            # matching the shorter one leaves "A1.57...", which cannot be split
            # — so xGA silently vanished from every panel
            for label, (key, ceiling) in sorted(
                    ROWS.items(), key=lambda kv: -len(kv[0])):
                if not t.startswith(label):
                    continue
                rest = t[len(label):].strip()
                vals = _split_pcts(rest) if "%" in rest else split_three(rest, ceiling)
                if vals:
                    stats[key] = dict(zip(COLS, vals))
                break

        panels.append({"team": name, "league": league or "", "pos": pos,
                       "teams_in_league": rank, "recent": rec,
                       "form": form, "stats": stats})
    return panels


def value(panel: dict, key: str, col: str, default=np.nan) -> float:
    d = panel.get("stats", {}).get(key)
    if not d:
        return default
    v = d.get(col, np.nan)
    if not np.isfinite(v):
        v = d.get("overall", default)
    return float(v) if v == v else default


def missing(panel: dict) -> list[str]:
    return [k for k, (key, _) in ROWS.items() if key not in panel.get("stats", {})]


def table(panels: list[dict]) -> pd.DataFrame:
    rows = []
    for p in panels:
        r = {"Team": p["team"], "League": p["league"],
             "Position": p["pos"], "Of": p["teams_in_league"]}
        for w in COLS:
            f = p["form"].get(w, {})
            r[f"Form {w}"] = f.get("results", "")
            r[f"PPG {w}"] = f.get("ppg", np.nan)
        for label, (key, _) in ROWS.items():
            for w in COLS:
                r[f"{label} {w}"] = value(p, key, w)
        rows.append(r)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    import glob
    import sys
    files = sorted(glob.glob(sys.argv[1] if len(sys.argv) > 1
                             else "sample_data/*.txt"))
    for f in files:
        ps = parse_panels(open(f, encoding="utf-8").read())
        print(f"\n=== {f.rsplit('/', 1)[-1]} -> {len(ps)} panels ===")
        for p in ps:
            miss = missing(p)
            print(f"  {p['team']:26s} {p['league']:34s} pos {p['pos']}"
                  f"  missing={miss}")
            print(f"     scored {value(p,'scored','overall'):.2f}/"
                  f"{value(p,'scored','home'):.2f}/{value(p,'scored','away'):.2f}"
                  f"   xG {value(p,'xg','overall'):.2f}/"
                  f"{value(p,'xg','home'):.2f}/{value(p,'xg','away'):.2f}"
                  f"   xGA {value(p,'xga','overall'):.2f}/"
                  f"{value(p,'xga','home'):.2f}/{value(p,'xga','away'):.2f}")
