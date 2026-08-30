"""Parse pasted FootyStats match pages into one row per match.

A pasted page is mostly noise - league tables, top scorers, cards per 90, the
site footer in thirty languages. Four anchors carry everything worth having:

    Saturday Aug 22, 2026 - 7:30pm (Asia/Dubai time)   <- date
    Schwarz-Weiß Bregenz vs SK Sturm Graz II           <- fixture
    Final Results / 3 - 1 / HT / (2 - 1)               <- score
    Data / <home> / <away> / Possession / 51% / 49%    <- the stats block

The names in the `Data` block are the short forms FootyStats uses everywhere
else ("Sturm Graz II"), while the fixture line carries the long forms
("SK Sturm Graz II"). The Data block wins: it is the name that appears
consistently across every page, so it is what the teams are keyed on. Matching
on the fixture line instead would split one team into two.

Pages get pasted more than once - the head-to-head between the two teams being
predicted appears in both teams' sets - so identical matches are de-duplicated
on (date, home, away).
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field

import pandas as pd

_DATE = re.compile(
    r"^\s*(?:Mon|Tues|Wednes|Thurs|Fri|Satur|Sun)day\s+"
    r"(\w{3})\s+(\d{1,2}),\s*(\d{4})", re.I)
_FIXTURE = re.compile(r"^\s*(.+?)\s+vs\s+(.+?)\s*$", re.I)
_SCORE = re.compile(r"^\s*(\d{1,2})\s*-\s*(\d{1,2})\s*$")
_HT = re.compile(r"^\s*\((\d{1,2})\s*-\s*(\d{1,2})\)\s*$")
_PCT = re.compile(r"^\s*(\d{1,3})\s*%\s*$")
_NUM = re.compile(r"^\s*(-?\d+(?:\.\d+)?)\s*$")

# the Data block's rows, in the order FootyStats emits them
STAT_ROWS = {
    "Possession": "possession",
    "Shots": "shots",
    "Shots On Target": "sot",
    "Shots Off Target": "soff",
    "Cards": "cards",
    "Corners": "corners",
    "Fouls": "fouls",
    "Offsides": "offsides",
    "xG": "xg",
}

MONTHS = {m: i + 1 for i, m in enumerate(
    ["jan", "feb", "mar", "apr", "may", "jun",
     "jul", "aug", "sep", "oct", "nov", "dec"])}

# FootyStats names the competition two lines above "Past H2H"
FRIENDLY_HINTS = ("friendl", "club friendlies", "pre-season", "preseason",
                  "testspiel")
CUP_HINTS = ("cup", "pokal", "trophy", "playoff", "play-off", "qualif")


@dataclass
class Match:
    date: str
    competition: str
    home: str
    away: str
    hg: int
    ag: int
    ht_hg: float = float("nan")
    ht_ag: float = float("nan")
    stats: dict = field(default_factory=dict)


def _clean(s: str) -> str:
    return re.sub(r"\s+", " ", s or "").strip()


def _looks_like_team(s: str) -> bool:
    s = _clean(s)
    if not (2 <= len(s) <= 48):
        return False
    # Digits are ordinary in club names - First Vienna FC 1894, Schalke 04,
    # 1860 München - so they cannot be refused outright; rejecting them lost
    # First Vienna entirely and with it one of Sturm Graz II's five matches.
    # What a team name is not is *mostly* digits, which is what a score,
    # a minute or a table row looks like.
    digits = sum(ch.isdigit() for ch in s)
    if digits and digits / len(s) > 0.4:
        return False
    if not any(ch.isalpha() for ch in s):
        return False
    bad = ("stats", "odds", "prediction", "head to head", "fixture", "data",
           "final results", "stadium", "past h2h", "football", "search",
           "market", "average", "league", "table", "form", "goals", "corner",
           "card", "shot", "player", "referee", "http", "www")
    low = s.lower()
    return not any(b in low for b in bad)


def _competition(lines: list[str], i: int) -> str:
    """The league name sits just above the date line, after the country."""
    for j in range(max(0, i - 8), i):
        t = _clean(lines[j])
        if t and t not in {"/", "Past H2H"} and not _DATE.match(lines[j]):
            if t.lower() not in {"football stats by footystats",
                                 "search teams and leagues"}:
                cand = t
    out = []
    for j in range(max(0, i - 8), i):
        t = _clean(lines[j])
        if t and t != "/" and t != "Past H2H" and "footystats" not in t.lower() \
                and "search teams" not in t.lower():
            out.append(t)
    return " / ".join(out[-2:]) if out else ""


def classify(competition: str) -> str:
    low = (competition or "").lower()
    if any(h in low for h in FRIENDLY_HINTS):
        return "Friendly"
    if any(h in low for h in CUP_HINTS):
        return "Cup"
    if not low.strip():
        return "Unknown"
    return "League"


def parse(text: str, return_dropped: bool = False):
    """Matches from pasted pages. With `return_dropped`, also the duplicates.

    The head-to-head between the two teams being predicted appears in *both*
    teams' sets, so ten pasted pages are normally nine distinct matches.
    Counting it twice would double-weight that fixture, so it is dropped - but
    silently dropping it makes the app look like it lost a page, so the
    duplicates are handed back and reported.
    """
    lines = text.replace("\r\n", "\n").split("\n")
    n = len(lines)
    matches: list[Match] = []

    i = 0
    while i < n:
        d = _DATE.match(lines[i])
        if not d:
            i += 1
            continue
        mon = MONTHS.get(d.group(1)[:3].lower())
        if mon is None:
            i += 1
            continue
        date = f"{int(d.group(3)):04d}-{mon:02d}-{int(d.group(2)):02d}"
        competition = _competition(lines, i)

        # the fixture line is the next non-empty line that has " vs "
        fx = None
        for j in range(i + 1, min(i + 6, n)):
            m = _FIXTURE.match(lines[j])
            if m and _looks_like_team(m.group(1)) and _looks_like_team(m.group(2)):
                fx = (_clean(m.group(1)), _clean(m.group(2)))
                break
        if fx is None:
            i += 1
            continue

        # search forward for this page's Final Results and Data blocks, but
        # stop at the next date header so a page without stats cannot borrow
        # the following page's numbers
        end = n
        for j in range(i + 1, n):
            if _DATE.match(lines[j]):
                end = j
                break

        hg = ag = None
        ht_hg = ht_ag = float("nan")
        for j in range(i + 1, end):
            if _clean(lines[j]).lower() == "final results":
                for k in range(j + 1, min(j + 5, end)):
                    s = _SCORE.match(lines[k])
                    if s:
                        hg, ag = int(s.group(1)), int(s.group(2))
                        break
                for k in range(j + 1, min(j + 10, end)):
                    h = _HT.match(lines[k])
                    if h:
                        ht_hg, ht_ag = float(h.group(1)), float(h.group(2))
                        break
                break
        if hg is None:
            i += 1
            continue

        home = away = None
        stats: dict = {}
        for j in range(i + 1, end):
            if _clean(lines[j]) != "Data":
                continue
            cand = [_clean(x) for x in lines[j + 1:j + 4] if _clean(x)]
            if len(cand) >= 2 and _looks_like_team(cand[0]) and _looks_like_team(cand[1]):
                home, away = cand[0], cand[1]
            stats = _read_stats(lines, j, end)
            break

        if home is None:
            home, away = fx
        matches.append(Match(date=date, competition=competition,
                             home=home, away=away, hg=hg, ag=ag,
                             ht_hg=ht_hg, ht_ag=ht_ag, stats=stats))
        i = end

    df, dropped = _frame(matches)
    return (df, dropped) if return_dropped else df


def _read_stats(lines: list[str], j: int, end: int) -> dict:
    """Rows of `label / home value / away value` following a Data header."""
    out: dict = {}
    k = j + 1
    while k < end:
        label = _clean(lines[k])
        if label in STAT_ROWS:
            vals = []
            m = k + 1
            while m < end and len(vals) < 2:
                t = _clean(lines[m])
                if t:
                    p = _PCT.match(t)
                    q = _NUM.match(t)
                    if p:
                        vals.append(float(p.group(1)))
                    elif q:
                        vals.append(float(q.group(1)))
                    else:
                        break
                m += 1
            if len(vals) == 2:
                key = STAT_ROWS[label]
                out[f"h_{key}"], out[f"a_{key}"] = vals
                k = m
                continue
        # the block ends once a clearly different section starts
        if label.lower().startswith(("head to head", "odds market",
                                     "fixture analysis", "prediction stats")):
            break
        k += 1
    return out


def _frame(matches: list[Match]):
    if not matches:
        return pd.DataFrame(), pd.DataFrame()
    rows = []
    for m in matches:
        r = {k: v for k, v in asdict(m).items() if k != "stats"}
        r.update(m.stats)
        r["kind"] = classify(m.competition)
        rows.append(r)
    df = pd.DataFrame(rows)
    key = ["date", "home", "away"]
    dup = df.duplicated(subset=key, keep="first")
    dropped = df[dup].copy()
    df = df[~dup].copy()
    for f in (df, dropped):
        if not f.empty:
            f["date"] = pd.to_datetime(f["date"], errors="coerce")
    df["total_goals"] = df.hg + df.ag
    df["btts"] = ((df.hg > 0) & (df.ag > 0)).astype(int)
    if "h_corners" in df and "a_corners" in df:
        df["total_corners"] = df.h_corners + df.a_corners
    return (df.sort_values("date", ascending=False).reset_index(drop=True),
            dropped.reset_index(drop=True))


def teams(df: pd.DataFrame) -> list[str]:
    """Team names ordered by how many matches they appear in."""
    if df.empty:
        return []
    c = pd.concat([df.home, df.away]).value_counts()
    return list(c.index)



def subject_team(df: pd.DataFrame):
    """Whose matches these are: the team appearing in nearly all of them.

    A team's own five matches contain it five times and each opponent once, so
    the subject is unambiguous. The tie margin is returned as well - if the top
    two counts are level the paste is not one team's form and the caller should
    say so rather than guess.
    """
    if df is None or df.empty:
        return None, 0, 0
    counts = pd.concat([df.home, df.away]).value_counts()
    top = counts.index[0]
    n1 = int(counts.iloc[0])
    n2 = int(counts.iloc[1]) if len(counts) > 1 else 0
    return top, n1, n2


def team_matches(df: pd.DataFrame, team: str) -> pd.DataFrame:
    """One team's matches, flipped so the team is always the `for` side."""
    rows = []
    for _, r in df.iterrows():
        if r.home == team:
            side, opp, gf, ga = "H", r.away, r.hg, r.ag
            pre = ("h_", "a_")
        elif r.away == team:
            side, opp, gf, ga = "A", r.home, r.ag, r.hg
            pre = ("a_", "h_")
        else:
            continue
        row = {"date": r.date, "opponent": opp, "venue": side,
               "gf": gf, "ga": ga, "kind": r.kind,
               "competition": r.competition,
               "result": "W" if gf > ga else ("D" if gf == ga else "L")}
        for key in STAT_ROWS.values():
            row[f"{key}_for"] = r.get(f"{pre[0]}{key}", float("nan"))
            row[f"{key}_against"] = r.get(f"{pre[1]}{key}", float("nan"))
        rows.append(row)
    out = pd.DataFrame(rows)
    return out.sort_values("date", ascending=False).reset_index(drop=True) \
        if not out.empty else out


def warnings_for(df: pd.DataFrame, ts: list[str]) -> list[str]:
    out = []
    if df.empty:
        return ["No matches were read from that text."]
    if "h_xg" not in df.columns or df.h_xg.isna().all():
        out.append("No xG was found — the model will fall back to goals only.")
    for t in ts[:2]:
        k = len(team_matches(df, t))
        if k < 3:
            out.append(f"{t}: only {k} match(es) parsed — too few to model.")
        elif k < 5:
            out.append(f"{t}: {k} matches parsed, fewer than the 5 expected.")
    missing = df[df.get("h_corners", pd.Series(dtype=float)).isna()] \
        if "h_corners" in df else df
    if len(missing):
        out.append(f"{len(missing)} match(es) have no corner data — the "
                   "corners market will use only the rest.")
    return out


if __name__ == "__main__":
    import sys
    path = (sys.argv[1] if len(sys.argv) > 1
            else "sample_data/sturm_graz_ii_vs_rapid_wien_ii.txt")
    d = parse(open(path, encoding="utf-8").read())
    print(f"{len(d)} matches")
    cols = ["date", "home", "away", "hg", "ag", "kind", "h_xg", "a_xg",
            "h_corners", "a_corners"]
    print(d[cols].to_string(index=False))
    ts = teams(d)
    print("\nteams by appearances:", ts[:6])
    for t in ts[:2]:
        print(f"\n--- {t} ---")
        print(team_matches(d, t)[
            ["date", "opponent", "venue", "gf", "ga", "xg_for", "xg_against",
             "corners_for", "corners_against", "kind"]].to_string(index=False))
    print("\nwarnings:", warnings_for(d, ts))
