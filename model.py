"""Match model for the Bulgarian Second League built from FootyStats exports.

The engine is a time-weighted Poisson model with team attack and defence
strengths, a home advantage, and the Dixon-Coles correction for low scores.
By default the strengths are fitted to match xG rather than to goals, because
xG is a less noisy read of how a side actually played; the backtest lets you
compare both.

Everything the app shows comes from three functions here:

    fit(matches, as_of, params)   -> Fit        team strengths as of a date
    predict(fit, home, away)      -> dict       1X2, goal lines, scorelines
    walk_forward(matches, params) -> DataFrame  honest out-of-sample record

Nothing post-match leaks into a prediction: a match is only ever priced from
games that finished before its own kick-off day, and the FootyStats
`home_ppg` / `away_ppg` columns (which are season-to-date *after* the match)
are never read.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import poisson

DATA_DIR = Path(__file__).parent / "data"
MAX_GOALS = 10
OUTCOMES = ["Home", "Draw", "Away"]

_KEEP = [
    "timestamp", "date_GMT", "status", "home_team_name", "away_team_name",
    "Game Week", "home_team_goal_count", "away_team_goal_count",
    "team_a_xg", "team_b_xg",
    "odds_ft_home_team_win", "odds_ft_draw", "odds_ft_away_team_win",
    "odds_ft_over25", "odds_btts_yes",
]


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
_FILE_RE = re.compile(r"^(?P<league>.+?)-(?P<kind>matches|teams|players)-(?P<y1>\d{4})-to-(?P<y2>\d{4})-stats\.csv$")


def parse_filename(name: str) -> dict | None:
    """FootyStats names exports `<league>-<kind>-<y1>-to-<y2>-stats.csv`.
    Returns league key, kind and a season label, or None if the name is foreign."""
    m = _FILE_RE.match(Path(name).name)
    if not m:
        return None
    y1, y2 = m["y1"], m["y2"]
    season = y1 if y1 == y2 else f"{y1}-{y2[2:]}"
    return {"league": m["league"], "kind": m["kind"], "season": season}


def league_label(key: str) -> str:
    """'kazakhstan-first-division' -> 'Kazakhstan First Division'. FootyStats sometimes
    repeats the country ('uzbekistan-uzbekistan-super-league'); drop the duplicate."""
    words = key.split("-")
    if len(words) > 1 and words[0] == words[1]:
        words = words[1:]
    return " ".join(words).title()


def discover_leagues(dirs: Iterable[Path]) -> dict[str, list[Path]]:
    """League key -> its *matches* CSVs, across the given folders."""
    out: dict[str, list[Path]] = {}
    for d in dirs:
        if not Path(d).exists():
            continue
        for p in sorted(Path(d).glob("*.csv")):
            info = parse_filename(p.name)
            if info and info["kind"] == "matches":
                out.setdefault(info["league"], []).append(p)
    return out


def load_matches(paths: Iterable[Path | str] | None = None) -> pd.DataFrame:
    """Read one or more FootyStats *matches* CSVs into one tidy frame.

    Both finished and unplayed fixtures are kept; `status` tells them apart.
    Rows from overlapping files are de-duplicated on (timestamp, home, away),
    the later file winning. Files that are not matches exports raise ValueError.
    """
    if paths is None:
        paths = sorted(DATA_DIR.glob("*matches*.csv"))
    frames = []
    for p in paths:
        df = pd.read_csv(p)
        missing = [c for c in _KEEP if c not in df.columns]
        if missing:
            raise ValueError(f"{Path(str(p)).name}: not a FootyStats matches export "
                             f"(missing {', '.join(missing[:3])})")
        df = df[_KEEP].copy()
        info = parse_filename(Path(str(p)).name)
        df["season"] = info["season"] if info else None
        frames.append(df)
    if not frames:
        raise FileNotFoundError("no matches CSVs found")
    m = pd.concat(frames, ignore_index=True)
    m = m.drop_duplicates(["timestamp", "home_team_name", "away_team_name"], keep="last")
    m = m.sort_values("timestamp").reset_index(drop=True)
    m["kickoff"] = pd.to_datetime(m["timestamp"], unit="s", utc=True)
    if m.season.isna().any():           # file not named the FootyStats way: July-June seasons
        yr = m.kickoff.dt.year
        start = np.where(m.kickoff.dt.month >= 7, yr, yr - 1)
        guess = [f"{s}-{str(s + 1)[2:]}" for s in start]
        m["season"] = m.season.where(m.season.notna(), guess)
    m = m.rename(columns={"home_team_name": "home", "away_team_name": "away",
                          "home_team_goal_count": "hg", "away_team_goal_count": "ag",
                          "team_a_xg": "hxg", "team_b_xg": "axg",
                          "odds_ft_home_team_win": "odds_h", "odds_ft_draw": "odds_d",
                          "odds_ft_away_team_win": "odds_a", "Game Week": "week"})
    done = m.status.eq("complete")
    # xG is missing for some games (recorded as 0-0); fall back to goals there.
    m["has_xg"] = done & ((m.hxg > 0) | (m.axg > 0))
    no_xg = done & ~m.has_xg
    m.loc[no_xg, "hxg"] = m.loc[no_xg, "hg"]
    m.loc[no_xg, "axg"] = m.loc[no_xg, "ag"]
    m["result"] = np.where(done, np.select([m.hg > m.ag, m.hg == m.ag], [0, 1], 2), -1)
    return m


def completed(m: pd.DataFrame) -> pd.DataFrame:
    return m[m.status.eq("complete")].reset_index(drop=True)


def fixtures(m: pd.DataFrame) -> pd.DataFrame:
    return m[m.status.eq("incomplete")].reset_index(drop=True)


def market_probs(odds_h: float, odds_d: float, odds_a: float) -> np.ndarray | None:
    """Bookmaker 1X2 odds -> fair probabilities (overround removed pro rata)."""
    o = np.array([odds_h, odds_d, odds_a], dtype=float)
    if not np.all(np.isfinite(o)) or (o <= 1).any():
        return None
    p = 1 / o
    return p / p.sum()


def overround(odds_h: float, odds_d: float, odds_a: float) -> float | None:
    o = np.array([odds_h, odds_d, odds_a], dtype=float)
    if (o <= 1).any():
        return None
    return float((1 / o).sum())


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Params:
    halflife_days: float = 60.0   # weight of a match halves every N days
    ridge: float = 1.0            # shrinkage of strengths toward league average
    rho: float = -0.10            # Dixon-Coles low-score correction
    use_xg: bool = True           # fit to xG (True) or to goals (False)


@dataclass
class Fit:
    teams: list[str]
    attack: np.ndarray
    defence: np.ndarray
    mu: float
    home_adv: float
    params: Params
    n_matches: int
    as_of: pd.Timestamp
    matches_by_team: dict = field(default_factory=dict)

    def index(self, team: str) -> int:
        try:
            return self.teams.index(team)
        except ValueError:
            raise KeyError(f"unknown team {team!r}") from None


def fit(matches: pd.DataFrame, as_of: pd.Timestamp | None = None,
        params: Params = Params(), teams: list[str] | None = None) -> Fit:
    """Fit strengths on completed matches that kicked off before `as_of`."""
    c = completed(matches)
    if as_of is None:
        as_of = c.kickoff.max() + pd.Timedelta(days=1)
    c = c[c.kickoff < as_of]
    if teams is None:
        teams = sorted(set(matches.home) | set(matches.away))
    idx = {t: i for i, t in enumerate(teams)}
    n = len(teams)
    hi = c.home.map(idx).to_numpy()
    ai = c.away.map(idx).to_numpy()
    age_days = (as_of - c.kickoff).dt.total_seconds().to_numpy() / 86400
    w = 0.5 ** (age_days / params.halflife_days)
    hg = (c.hxg if params.use_xg else c.hg).to_numpy(dtype=float)
    ag = (c.axg if params.use_xg else c.ag).to_numpy(dtype=float)

    def nll(x: np.ndarray) -> float:
        att, de = x[:n], x[n:2 * n]
        mu, ha = x[2 * n], x[2 * n + 1]
        lh = np.exp(mu + ha + att[hi] - de[ai])
        la = np.exp(mu + att[ai] - de[hi])
        ll = w * (hg * np.log(lh) - lh + ag * np.log(la) - la)
        return -ll.sum() + params.ridge * (att @ att + de @ de)

    x0 = np.zeros(2 * n + 2)
    x0[2 * n] = np.log(1.3)
    res = minimize(nll, x0, method="L-BFGS-B")
    x = res.x
    counts = pd.concat([c.home, c.away]).value_counts().to_dict()
    return Fit(teams=list(teams), attack=x[:n], defence=x[n:2 * n], mu=float(x[2 * n]),
               home_adv=float(x[2 * n + 1]), params=params, n_matches=len(c),
               as_of=as_of, matches_by_team=counts)


def expected_goals(f: Fit, home: str, away: str) -> tuple[float, float]:
    h, a = f.index(home), f.index(away)
    lh = np.exp(f.mu + f.home_adv + f.attack[h] - f.defence[a])
    la = np.exp(f.mu + f.attack[a] - f.defence[h])
    return float(lh), float(la)


def score_matrix(lh: float, la: float, rho: float) -> np.ndarray:
    g = np.arange(MAX_GOALS + 1)
    P = np.outer(poisson.pmf(g, lh), poisson.pmf(g, la))
    P[0, 0] *= 1 - lh * la * rho
    P[0, 1] *= 1 + lh * rho
    P[1, 0] *= 1 + la * rho
    P[1, 1] *= 1 - rho
    return P / P.sum()


def predict(f: Fit, home: str, away: str) -> dict:
    lh, la = expected_goals(f, home, away)
    P = score_matrix(lh, la, f.params.rho)
    probs = np.array([np.tril(P, -1).sum(), np.trace(P), np.triu(P, 1).sum()])
    tot = np.add.outer(np.arange(MAX_GOALS + 1), np.arange(MAX_GOALS + 1))
    flat = [((i, j), P[i, j]) for i in range(MAX_GOALS + 1) for j in range(MAX_GOALS + 1)]
    flat.sort(key=lambda t: -t[1])
    return {
        "probs": probs,
        "lam_home": lh,
        "lam_away": la,
        "over15": float(P[tot > 1.5].sum()),
        "over25": float(P[tot > 2.5].sum()),
        "over35": float(P[tot > 3.5].sum()),
        "btts": float(P[1:, 1:].sum()),
        "home_clean_sheet": float(P[:, 0].sum()),
        "away_clean_sheet": float(P[0, :].sum()),
        "top_scores": flat[:8],
        "matrix": P,
    }


def ratings_table(f: Fit) -> pd.DataFrame:
    base = np.exp(f.mu)
    df = pd.DataFrame({
        "team": f.teams,
        "attack": np.exp(f.attack) * base,          # goals expected vs average defence, neutral
        "defence": np.exp(-f.defence) * base,       # goals conceded vs average attack, neutral
        "matches": [f.matches_by_team.get(t, 0) for t in f.teams],
    })
    df["rating"] = df.attack - df.defence
    return df.sort_values("rating", ascending=False).reset_index(drop=True)


def fair_odds(p: np.ndarray) -> np.ndarray:
    return 1 / np.clip(p, 1e-9, None)


def kelly(p: float, odds: float) -> float:
    """Full-Kelly stake fraction for a back bet; 0 if there is no edge."""
    b = odds - 1
    if b <= 0:
        return 0.0
    return max(0.0, (p * b - (1 - p)) / b)


# ---------------------------------------------------------------------------
# Backtest
# ---------------------------------------------------------------------------
def walk_forward(matches: pd.DataFrame, params: Params = Params(),
                 start: int = 90) -> pd.DataFrame:
    """Price every completed match from the `start`-th onward using only games
    that finished on earlier days. Returns one row per priced match."""
    c = completed(matches)
    teams = sorted(set(matches.home) | set(matches.away))
    days = (c.timestamp // 86400).to_numpy()
    rows = []
    cache: dict[int, Fit] = {}
    for i in range(start, len(c)):
        r = c.iloc[i]
        mk = market_probs(r.odds_h, r.odds_d, r.odds_a)
        d = days[i]
        if d not in cache:
            cache[d] = fit(c[days < d], as_of=r.kickoff, params=params, teams=teams)
        pr = predict(cache[d], r.home, r.away)
        rows.append({
            "kickoff": r.kickoff, "season": r.season, "home": r.home, "away": r.away,
            "hg": r.hg, "ag": r.ag, "result": int(r.result),
            "m_h": pr["probs"][0], "m_d": pr["probs"][1], "m_a": pr["probs"][2],
            "k_h": mk[0] if mk is not None else np.nan,
            "k_d": mk[1] if mk is not None else np.nan,
            "k_a": mk[2] if mk is not None else np.nan,
            "odds_h": r.odds_h, "odds_d": r.odds_d, "odds_a": r.odds_a,
            "lam_h": pr["lam_home"], "lam_a": pr["lam_away"], "over25": pr["over25"],
        })
    return pd.DataFrame(rows)


def _scores(P: np.ndarray, y: np.ndarray) -> dict:
    P = np.clip(P, 1e-9, 1)
    P = P / P.sum(1, keepdims=True)
    ll = -np.mean(np.log(P[np.arange(len(y)), y]))
    br = np.mean(((P - np.eye(3)[y]) ** 2).sum(1))
    acc = np.mean(P.argmax(1) == y)
    return {"log loss": ll, "Brier": br, "accuracy": acc}


def blend(model: np.ndarray, market: np.ndarray, w: float) -> np.ndarray:
    """Log-linear blend: w on the model, 1-w on the market."""
    B = np.exp(w * np.log(np.clip(model, 1e-9, 1)) + (1 - w) * np.log(np.clip(market, 1e-9, 1)))
    return B / B.sum(1, keepdims=True)


def summarise(bt: pd.DataFrame, blend_w: float = 0.25) -> pd.DataFrame:
    """Score model, market, blend and base rates on matches that had odds."""
    b = bt.dropna(subset=["k_h"])
    y = b.result.to_numpy()
    M = b[["m_h", "m_d", "m_a"]].to_numpy()
    K = b[["k_h", "k_d", "k_a"]].to_numpy()
    base = np.tile(np.bincount(bt.result, minlength=3) / len(bt), (len(b), 1))
    out = {
        "Bookmaker (fair)": _scores(K, y),
        "Model": _scores(M, y),
        f"Blend ({blend_w:.0%} model)": _scores(blend(M, K, blend_w), y),
        "League base rates": _scores(base, y),
    }
    df = pd.DataFrame(out).T
    df.insert(0, "matches", len(b))
    return df


def calibration(bt: pd.DataFrame, prefix: str = "m",
                edges=(0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.75, 1.0)) -> pd.DataFrame:
    """Predicted vs actual frequency, pooling all three outcomes."""
    b = bt.dropna(subset=["k_h"])
    P = b[[f"{prefix}_h", f"{prefix}_d", f"{prefix}_a"]].to_numpy().ravel()
    Y = np.eye(3)[b.result.to_numpy()].ravel()
    cut = pd.cut(P, edges, include_lowest=True)
    g = pd.DataFrame({"bin": cut, "p": P, "y": Y}).groupby("bin", observed=True)
    out = g.agg(n=("y", "size"), predicted=("p", "mean"), actual=("y", "mean")).reset_index()
    out["bin"] = out["bin"].astype(str)
    return out


def flat_stake_roi(bt: pd.DataFrame, min_edge: float = 0.0) -> pd.DataFrame:
    """Back each outcome where model prob exceeds the bookmaker's fair prob by
    `min_edge`; 1 unit flat stakes at the recorded odds."""
    b = bt.dropna(subset=["k_h"])
    rows = []
    for j, (mc, kc, oc, name) in enumerate([("m_h", "k_h", "odds_h", "Home"),
                                            ("m_d", "k_d", "odds_d", "Draw"),
                                            ("m_a", "k_a", "odds_a", "Away")]):
        sel = b[(b[mc] - b[kc]) > min_edge]
        won = sel.result == j
        pnl = np.where(won, sel[oc] - 1, -1.0)
        rows.append({"selection": name, "bets": len(sel), "strike": won.mean() if len(sel) else np.nan,
                     "P&L (units)": pnl.sum(), "ROI": pnl.mean() if len(sel) else np.nan})
    df = pd.DataFrame(rows)
    tot = {"selection": "All", "bets": df.bets.sum(), "strike": np.nan,
           "P&L (units)": df["P&L (units)"].sum(),
           "ROI": df["P&L (units)"].sum() / df.bets.sum() if df.bets.sum() else np.nan}
    return pd.concat([df, pd.DataFrame([tot])], ignore_index=True)


# ---------------------------------------------------------------------------
# Team context for the Predict page
# ---------------------------------------------------------------------------
def recent_form(m: pd.DataFrame, team: str, n: int = 6) -> pd.DataFrame:
    c = completed(m)
    g = c[(c.home == team) | (c.away == team)].tail(n).copy()
    at_home = g.home == team
    g["venue"] = np.where(at_home, "H", "A")
    g["opponent"] = np.where(at_home, g.away, g.home)
    gf = np.where(at_home, g.hg, g.ag)
    ga = np.where(at_home, g.ag, g.hg)
    g["score"] = [f"{a}-{b}" for a, b in zip(gf, ga)]
    g["xG"] = np.where(at_home, g.hxg, g.axg).round(2)
    g["xGA"] = np.where(at_home, g.axg, g.hxg).round(2)
    g["res"] = np.select([gf > ga, gf == ga], ["W", "D"], "L")
    g["date"] = g.kickoff.dt.strftime("%d %b %Y")
    return g[["date", "venue", "opponent", "score", "res", "xG", "xGA"]].iloc[::-1].reset_index(drop=True)


def head_to_head(m: pd.DataFrame, a: str, b: str) -> pd.DataFrame:
    c = completed(m)
    g = c[((c.home == a) & (c.away == b)) | ((c.home == b) & (c.away == a))].copy()
    g["date"] = g.kickoff.dt.strftime("%d %b %Y")
    g["score"] = [f"{x}-{y}" for x, y in zip(g.hg, g.ag)]
    g["xG"] = [f"{x:.2f} - {y:.2f}" for x, y in zip(g.hxg, g.axg)]
    return g[["date", "season", "home", "away", "score", "xG"]].iloc[::-1].reset_index(drop=True)
