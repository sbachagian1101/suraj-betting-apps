"""Mauritian (Champ de Mars) gallop-sheet model.

Parses the pasted "GALLOPS for Race N" text, turns each runner's training
lines into features (freshness, workload, work quality, conditions,
consistency, trend, company), scores the race and, once enough results are
stored, fits the feature weights.

The sheet's columns are 200P / 2400P / 2200P / 600M / 400M / 200M.  Only the
600M and 200M clocks are genuinely timed: the 400M is always the midpoint
(600M and 200M average) and the first two split columns are always equal to
(600M - 400M).  So the real signal per gallop is the 600 m time, the final
200 m time and the equipment / comment.
"""
from __future__ import annotations

import json
import math
import re
from dataclasses import asdict, dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
# Vocabulary
# --------------------------------------------------------------------------

# Equipment / manner of the gallop.  Lib = "libre" (free, no urging);
# Leg Acc = light acceleration; Acc = accelerated; Sol = "sollicité"
# (asked for effort); (BT) = barrier trial.
EQUIP_TOKENS = {"lib", "leg", "acc", "sol", "(bt)", "bt", "lang", "oeil", "oeill", "bouch"}
# Seconds added to a 600 m clock so different efforts are comparable.  A free
# (Lib) 33.5 is a better piece of work than a 33.5 under pressure.
EFFORT_PENALTY = {"Lib": 0.0, "Leg Acc": 0.35, "Acc": 0.5, "Sol": 0.8, "Lib (BT)": 0.0}

MONTHS = {m.lower(): i for i, m in enumerate(
    ["January", "February", "March", "April", "May", "June", "July", "August",
     "September", "October", "November", "December"], start=1)}

RACE_HEADER = re.compile(r"GALLOPS\s+for\s+Race\s+(\d+)\s*\[([^\]]*)\]", re.I)
RESULTS_LINE = re.compile(r"^\s*Actual\s+results?\s*:\s*(.+)$", re.I)
HORSE_HEADER = re.compile(r"^\s*(\d{1,2})\s+([A-Z][A-Z0-9' .\-]+?)\s*$")
GALLOP_LINE = re.compile(
    r"^\s*(?P<jockey>.*?)\s*(?P<date>\d{2}\.\d{2}\.\d{2})\s+"
    r"(?P<nums>(?:(?:-|\d+(?:\.\d+)?)\s+){5}(?:-|\d+(?:\.\d+)?))"
    r"(?P<rest>.*)$"
)
SKIP_LINES = ("JOCKEY", "S P L I T", "200P", "SPLIT")


@dataclass
class Gallop:
    day: date
    jockey: str
    t600: float | None
    t400: float | None
    t200: float | None
    equipment: str          # "Lib", "Leg Acc", "Acc", "Sol", "Lib (BT)" ...
    barrier_trial: bool
    comment: str
    company: str | None = None      # worked in company with
    beat: str | None = None         # "bat X"  -> beat X
    beaten_by: str | None = None    # "bab X"  -> beaten by X

    @property
    def first400(self) -> float | None:
        if self.t600 is None or self.t200 is None:
            return None
        return self.t600 - self.t200

    @property
    def finish_ratio(self) -> float | None:
        """Last 200 m divided by the average 200 m of the first 400 m.
        Below 1.0 the horse quickened through the line."""
        f4 = self.first400
        if f4 is None or f4 <= 0 or self.t200 is None:
            return None
        return self.t200 / (f4 / 2.0)

    @property
    def adj600(self) -> float | None:
        if self.t600 is None:
            return None
        return self.t600 + EFFORT_PENALTY.get(self.equipment, 0.4)


@dataclass
class Runner:
    number: int
    name: str
    gallops: list[Gallop] = field(default_factory=list)


@dataclass
class Race:
    race_no: int
    race_date: date
    runners: list[Runner]
    results: list[str] = field(default_factory=list)   # finishing order, names
    label: str = ""

    def to_json(self) -> dict:
        d = asdict(self)
        d["race_date"] = self.race_date.isoformat()
        for r in d["runners"]:
            for g in r["gallops"]:
                g["day"] = g["day"].isoformat()
        return d

    @staticmethod
    def from_json(d: dict) -> "Race":
        runners = []
        for r in d["runners"]:
            gs = [Gallop(**{**g, "day": date.fromisoformat(g["day"])}) for g in r["gallops"]]
            runners.append(Runner(r["number"], r["name"], gs))
        return Race(d["race_no"], date.fromisoformat(d["race_date"]), runners,
                    d.get("results", []), d.get("label", ""))


# --------------------------------------------------------------------------
# Parsing
# --------------------------------------------------------------------------

def _num(tok: str) -> float | None:
    return None if tok == "-" else float(tok)


def _parse_rest(rest: str) -> tuple[str, bool, str, dict]:
    """Split the tail of a gallop line into equipment and comment."""
    toks = rest.split()
    equip: list[str] = []
    i = 0
    while i < len(toks) and toks[i].lower().strip(".,") in EQUIP_TOKENS:
        equip.append(toks[i])
        i += 1
    comment = " ".join(toks[i:]).strip()
    bt = any(t.lower() in ("(bt)", "bt") for t in equip)
    words = [t for t in equip if t.lower() not in ("(bt)", "bt")]
    equipment = " ".join(words) if words else "Lib"
    equipment = {"leg acc": "Leg Acc", "lib": "Lib", "acc": "Acc", "sol": "Sol"}.get(
        equipment.lower(), equipment)
    if bt:
        equipment = f"{equipment} (BT)"
    extra: dict = {}
    m = re.match(r"^(encie|enc|bab|bat)\b\s*(.*)$", comment, re.I)
    if m:
        key, who = m.group(1).lower(), m.group(2).strip()
        if key in ("encie", "enc"):
            extra["company"] = who
        elif key == "bab":
            extra["beaten_by"] = who
        elif key == "bat":
            extra["beat"] = who
    return equipment, bt, comment, extra


def _race_date(label: str, year: int) -> date:
    m = re.search(r"(\d{1,2})\s+([A-Za-z]+)", label)
    if not m:
        raise ValueError(f"Cannot read race date from '{label}'")
    day, mon = int(m.group(1)), MONTHS.get(m.group(2).lower())
    if mon is None:
        raise ValueError(f"Unknown month in '{label}'")
    return date(year, mon, day)


def parse_races(text: str) -> list[Race]:
    """Parse one or more pasted 'GALLOPS for Race N' blocks."""
    races: list[Race] = []
    cur_runners: list[Runner] = []
    cur_header: tuple[int, str] | None = None
    cur_results: list[str] = []
    runner: Runner | None = None

    def flush():
        nonlocal cur_runners, cur_header, cur_results, runner
        if cur_header is None:
            return
        days = [g.day for r in cur_runners for g in r.gallops]
        year = max(days).year if days else date.today().year
        race_no, label = cur_header
        rd = _race_date(label, year)
        if days and rd < max(days):          # gallops ran into the next year
            rd = date(year + 1, rd.month, rd.day)
        races.append(Race(race_no, rd, cur_runners, cur_results, label))
        cur_runners, cur_header, cur_results, runner = [], None, [], None

    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        m = RACE_HEADER.search(line)
        if m:
            flush()
            cur_header = (int(m.group(1)), m.group(2))
            continue
        m = RESULTS_LINE.match(line)
        if m:
            cur_results = [s.strip() for s in re.split(r"\s+-\s+|,|;", m.group(1)) if s.strip()]
            continue
        if any(line.upper().startswith(s) for s in SKIP_LINES):
            continue
        m = GALLOP_LINE.match(line)
        if m and runner is not None:
            nums = m.group("nums").split()
            t600, t400, t200 = _num(nums[3]), _num(nums[4]), _num(nums[5])
            equipment, bt, comment, extra = _parse_rest(m.group("rest"))
            runner.gallops.append(Gallop(
                day=datetime.strptime(m.group("date"), "%d.%m.%y").date(),
                jockey=m.group("jockey").strip(),
                t600=t600, t400=t400, t200=t200,
                equipment=equipment, barrier_trial=bt, comment=comment, **extra))
            continue
        m = HORSE_HEADER.match(line)
        if m and cur_header is not None:
            runner = Runner(int(m.group(1)), m.group(2).strip().upper())
            cur_runners.append(runner)
            continue
    flush()
    return races


# --------------------------------------------------------------------------
# Features
# --------------------------------------------------------------------------

FEATURES = [
    "n_gallops", "gallops_14d", "days_since", "fresh_penalty", "workload",
    "best_adj600", "last_adj600", "mean_adj600", "std600", "trend600", "taper",
    "finish_ratio", "last_finish_ratio", "lib_share", "pressure_share",
    "company_n", "beat_n", "beaten_n", "barrier_trial", "missing_600",
]


def _slope(days: list[float], vals: list[float]) -> float:
    if len(vals) < 2:
        return 0.0
    x, y = np.array(days, float), np.array(vals, float)
    if x.std() == 0:
        return 0.0
    return float(np.polyfit(x, y, 1)[0]) * 7.0     # seconds per week


def runner_features(r: Runner, race_date: date) -> dict:
    gs = sorted(r.gallops, key=lambda g: g.day)
    timed = [g for g in gs if g.t600 is not None and not g.barrier_trial]
    n = len(gs)
    last_day = gs[-1].day if gs else None
    days_since = (race_date - last_day).days if last_day else 30
    g14 = sum(1 for g in gs if (race_date - g.day).days <= 14)
    span = (gs[-1].day - gs[0].day).days if n > 1 else 0
    workload = n / max(span / 7.0, 1.0)             # gallops per week over the prep

    # Freshness: the sweet spot is 3-7 days between the last serious gallop
    # and the race. Very recent (<3) or stale (>9) both cost.
    if days_since < 3:
        fresh_pen = 3 - days_since
    elif days_since > 9:
        fresh_pen = (days_since - 9) / 2.0
    else:
        fresh_pen = 0.0

    adj = [g.adj600 for g in timed]
    raw = [g.t600 for g in timed]
    ratios = [g.finish_ratio for g in timed if g.finish_ratio is not None]
    day_idx = [(g.day - race_date).days for g in timed]
    feats = {
        "n_gallops": n,
        "gallops_14d": g14,
        "days_since": days_since,
        "fresh_penalty": fresh_pen,
        "workload": workload,
        "best_adj600": min(adj) if adj else 40.0,
        "last_adj600": adj[-1] if adj else 40.0,
        "mean_adj600": float(np.mean(adj)) if adj else 40.0,
        "std600": float(np.std(raw)) if len(raw) > 1 else 1.5,
        "trend600": _slope(day_idx, raw),                 # negative = getting faster
        # Taper: how much easier the final timed gallop was than the sharpest
        # one.  Positive = sharp work mid-prep, easy work in race week.
        "taper": (raw[-1] - min(raw)) if raw else 0.0,
        "finish_ratio": float(np.mean(ratios)) if ratios else 1.1,
        "last_finish_ratio": ratios[-1] if ratios else 1.1,
        "lib_share": (sum(1 for g in gs if g.equipment.startswith("Lib")) / n) if n else 0.0,
        "pressure_share": (sum(1 for g in gs if g.equipment.startswith(("Sol", "Acc"))) / n) if n else 0.0,
        "company_n": sum(1 for g in gs if g.company),
        "beat_n": sum(1 for g in gs if g.beat),
        "beaten_n": sum(1 for g in gs if g.beaten_by),
        "barrier_trial": int(any(g.barrier_trial for g in gs)),
        "missing_600": sum(1 for g in gs if g.t600 is None),
    }
    return feats


def race_frame(race: Race) -> pd.DataFrame:
    rows = []
    for r in race.runners:
        f = runner_features(r, race.race_date)
        f.update({"number": r.number, "horse": r.name})
        rows.append(f)
    df = pd.DataFrame(rows).set_index("number")
    if race.results:
        order = {norm(n): i + 1 for i, n in enumerate(race.results)}
        df["finish"] = [order.get(norm(h), 0) for h in df["horse"]]
    return df


def norm(name: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", name.upper())


# --------------------------------------------------------------------------
# Scoring model
# --------------------------------------------------------------------------

# Sign convention: the weight multiplies the within-race z-score of the
# feature; a positive weight means "more of this is better".  The defaults
# are judgment weights (see README), NOT fitted: two races cannot fit
# nineteen numbers.  `fit_weights` replaces them once enough results exist.
DEFAULT_WEIGHTS = {
    "workload": 1.0,          # regular work through the prep (fitness)
    "gallops_14d": 0.4,       # recent work
    "fresh_penalty": -0.8,    # too close to / too far from the race
    "taper": 0.8,             # sharp work mid-prep, easier work in race week
    "best_adj600": -0.6,      # a fast effort-adjusted 600 somewhere in the prep
    "std600": -0.3,           # consistency of the clock
    "trend600": -0.2,         # improving through the prep
    "finish_ratio": -0.3,     # quickens through the line
    "lib_share": 0.2,         # does its work freely
    "beat_n": 0.3,
    "beaten_n": -0.3,
    "missing_600": -0.2,      # short / incomplete work
}


def _z(col: pd.Series) -> pd.Series:
    sd = col.std(ddof=0)
    if not np.isfinite(sd) or sd < 1e-9:
        return col * 0.0
    return (col - col.mean()) / sd


def score_race(race: Race, weights: dict | None = None, temperature: float = 1.0) -> pd.DataFrame:
    """Score every runner; returns the feature frame with score, rank and win %."""
    w = DEFAULT_WEIGHTS if weights is None else weights
    df = race_frame(race)
    score = pd.Series(0.0, index=df.index)
    for k, wt in w.items():
        if k in df:
            score = score + wt * _z(df[k].astype(float))
    df["score"] = score
    ex = np.exp((score - score.max()) / temperature)
    df["win_pct"] = 100 * ex / ex.sum()
    df["rank"] = df["score"].rank(ascending=False, method="first").astype(int)
    return df.sort_values("rank")


def explain(race: Race, weights: dict | None = None) -> pd.DataFrame:
    """Per-runner contribution of each weighted feature (z-score x weight)."""
    w = DEFAULT_WEIGHTS if weights is None else weights
    df = race_frame(race)
    out = pd.DataFrame(index=df.index)
    out["horse"] = df["horse"]
    for k, wt in w.items():
        if k in df:
            out[k] = wt * _z(df[k].astype(float))
    out["score"] = out.drop(columns="horse").sum(axis=1)
    return out.sort_values("score", ascending=False)


# --------------------------------------------------------------------------
# Evaluation and fitting
# --------------------------------------------------------------------------

def evaluate(races: Iterable[Race], weights: dict | None = None) -> dict:
    """Winner log-loss, winner hit rate, placed hit rate, mean winner rank."""
    ll, hits, placed, ranks, n = 0.0, 0, 0, [], 0
    for race in races:
        if not race.results:
            continue
        df = score_race(race, weights)
        if "finish" not in df or (df["finish"] == 1).sum() != 1:
            continue
        n += 1
        winner = df[df["finish"] == 1].iloc[0]
        ll -= math.log(max(winner["win_pct"] / 100, 1e-6))
        ranks.append(int(winner["rank"]))
        hits += int(winner["rank"] == 1)
        top = df[df["rank"] == 1].iloc[0]
        placed += int(1 <= top["finish"] <= 3)
    if n == 0:
        return {"races": 0}
    return {"races": n, "log_loss": ll / n, "winner_hit": hits / n,
            "top_pick_placed": placed / n, "mean_winner_rank": float(np.mean(ranks))}


def fit_weights(races: list[Race], base: dict | None = None, grid=(-1.0, -0.5, 0.0, 0.5, 1.0),
                min_races: int = 15, rounds: int = 3) -> tuple[dict, dict]:
    """Coordinate-descent over a small grid, minimising winner log-loss.

    Refuses (returns the base weights) below `min_races` results because
    anything fitted on fewer is noise dressed as a model.
    """
    scored = [r for r in races if r.results]
    weights = dict(base or DEFAULT_WEIGHTS)
    if len(scored) < min_races:
        return weights, {"fitted": False, "races": len(scored),
                         "reason": f"need at least {min_races} races with results"}
    best = evaluate(scored, weights)["log_loss"]
    for _ in range(rounds):
        improved = False
        for k in list(weights):
            for cand in grid:
                trial = {**weights, k: cand}
                ll = evaluate(scored, trial)["log_loss"]
                if ll < best - 1e-6:
                    best, weights, improved = ll, trial, True
        if not improved:
            break
    return weights, {"fitted": True, "races": len(scored), "log_loss": best}


# --------------------------------------------------------------------------
# Storage
# --------------------------------------------------------------------------

def save_races(races: Iterable[Race], path: Path) -> int:
    """Append races to a JSONL store, replacing any with the same date + number."""
    path = Path(path)
    existing = load_races(path) if path.exists() else []
    keyed = {(r.race_date, r.race_no): r for r in existing}
    for r in races:
        keyed[(r.race_date, r.race_no)] = r
    with path.open("w", encoding="utf-8") as fh:
        for r in sorted(keyed.values(), key=lambda x: (x.race_date, x.race_no)):
            fh.write(json.dumps(r.to_json()) + "\n")
    return len(keyed)


def load_races(path: Path) -> list[Race]:
    path = Path(path)
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as fh:
        return [Race.from_json(json.loads(line)) for line in fh if line.strip()]
