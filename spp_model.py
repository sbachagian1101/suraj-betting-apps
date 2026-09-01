"""Ratings and market construction for SoccerPredictorPro.

One distribution, four markets
------------------------------
Everything the app shows is read off a single full-time score matrix and a
single half-time/second-half pair, so 1X2, half-time, HT/FT and Asian handicap
can never contradict each other.

Halves are handled by **splitting** the full-time expectation rather than rating
them independently. Rating half-time directly from a ratio of two team rates
compounds two extremes: on the bundled Iranian data an Esteghlal-Persepolis
fixture came out with half-time lambdas of 1.27/0.80 against a league half-time
average of 0.41/0.38, implying a second half quieter than the first. That is
backwards - second halves carry ~37% more goals - and for some pairings the
implied second half went negative. Splitting by the league's measured first-half
share keeps the two halves coherent by construction.

Markets deliberately absent
---------------------------
No BTTS and no Over/Under. Measured walk-forward on 1,162 matches of the
bundled league, every goals market scored *worse* than simply quoting the base
rate - Over 2.5 at 0.6346 against a 0.6189 base, BTTS at 0.6905 against 0.6679.
The bookmaker cannot beat the base rate on them either. 1X2, half-time and
HT/FT all depend on the *difference* between the two teams' expectations, which
ratings do capture; over/under depends on the *sum*, which nothing in this data
predicts. Offering them would be decoration.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

import soccer_model as sm
import spp_data as spd

MAX_GOALS = 11

# Dixon-Coles low-score correction used when the input cannot support fitting
# one (team files carry no individual scorelines). Chosen by grid search on the
# bundled league: rho = -0.20 lifted P(1-1) from 11.9% to 14.2% against an
# actual 14.4%, and improved 1X2 log-loss from 1.0329 to 1.0270.
DEFAULT_RHO = -0.20

# Fallback first-half share of goals if the upload cannot supply one.
# 0.837 / 1.982 on the bundled league.
DEFAULT_HT_SHARE = 0.42

# Shrinkage strength, in pseudo-matches, pulling a team's rate toward the
# league average. Six was the value that minimised walk-forward log-loss.
PRIOR = 6.0

# Weight multiplier per season back from the most recent one. Without it a
# team-file blend weights purely by matches played, so a completed season three
# years ago (30 matches) outranks the current one (4 matches so far) *and* ties
# last season exactly - no recency preference at all. That contradicts the
# measurement: current-season form alone (1.0481) beat last season alone
# (1.0468 with far worse coverage), two seasons beat one, and all seven seasons
# was worse than two. 0.55 leaves a third season contributing something small,
# which is what "a dead heat with two" looks like.
SEASON_DECAY = 0.55


@dataclass
class Ratings:
    """Per-team venue-split scoring and conceding rates, on a goals scale."""
    teams: list[str]
    atk_home: dict[str, float]      # goals scored per home match
    def_home: dict[str, float]      # goals conceded per home match
    atk_away: dict[str, float]
    def_away: dict[str, float]
    lg_home: float                  # league goals per match by the home side
    lg_away: float
    ht_share: float                 # share of goals scored before half time
    rho: float
    n_matches: dict[str, float] = field(default_factory=dict)
    source: str = ""
    fitted: Any = None              # soccer_model.FittedModel, match route only

    def sample(self, team: str) -> float:
        return float(self.n_matches.get(team, 0.0))


def _shrink(value: float, n: float, league: float, prior: float = PRIOR) -> float:
    """Pull a rate computed from `n` matches toward the league average."""
    if not np.isfinite(value):
        return league
    return float((value * n + league * prior) / (n + prior))


# --- building ratings --------------------------------------------------------

def from_team_files(a: pd.DataFrame) -> Ratings:
    """Blend one row per team per season, weighting by matches actually played.

    A current-season row with four matches played therefore counts for four,
    not for a whole season - which is exactly the behaviour wanted when the
    latest file is a to-date snapshot.
    """
    lg_home = float(np.average(a["goals_scored_per_match_home"],
                               weights=a["matches_played_home"].fillna(1).clip(lower=1)))
    lg_away = float(np.average(a["goals_scored_per_match_away"],
                               weights=a["matches_played_away"].fillna(1).clip(lower=1)))

    # Rank seasons newest-first across the whole upload so every team is decayed
    # on the same scale, even a promoted side missing the earliest season.
    order = {s: i for i, s in enumerate(sorted(a["season_start"].unique(), reverse=True))}

    atk_h, def_h, atk_a, def_a, n_tot = {}, {}, {}, {}, {}
    for team, g in a.groupby("team"):
        decay = g["season_start"].map(order).map(lambda i: SEASON_DECAY ** i).to_numpy(float)
        wh = g["matches_played_home"].fillna(0).clip(lower=0).to_numpy(float) * decay
        wa = g["matches_played_away"].fillna(0).clip(lower=0).to_numpy(float) * decay
        nh, na = float(wh.sum()), float(wa.sum())

        def wavg(col, w, tot, league):
            v = pd.to_numeric(g[col], errors="coerce").to_numpy(float)
            ok = np.isfinite(v) & (w > 0)
            if not ok.any() or tot <= 0:
                return league
            return float(np.average(v[ok], weights=w[ok]))

        atk_h[team] = _shrink(wavg("goals_scored_per_match_home", wh, nh, lg_home), nh, lg_home)
        def_h[team] = _shrink(wavg("goals_conceded_per_match_home", wh, nh, lg_away), nh, lg_away)
        atk_a[team] = _shrink(wavg("goals_scored_per_match_away", wa, na, lg_away), na, lg_away)
        def_a[team] = _shrink(wavg("goals_conceded_per_match_away", wa, na, lg_home), na, lg_home)
        n_tot[team] = nh + na

    return Ratings(teams=sorted(atk_h), atk_home=atk_h, def_home=def_h,
                   atk_away=atk_a, def_away=def_a, lg_home=lg_home, lg_away=lg_away,
                   ht_share=_ht_share_team(a), rho=DEFAULT_RHO, n_matches=n_tot,
                   source="team files")


def _ht_share_team(a: pd.DataFrame) -> float:
    cols = ["goals_scored_per_match_half_time_home", "goals_scored_per_match_half_time_away"]
    if not all(c in a.columns for c in cols):
        return DEFAULT_HT_SHARE
    ht = pd.to_numeric(a[cols[0]], errors="coerce").fillna(0) + \
        pd.to_numeric(a[cols[1]], errors="coerce").fillna(0)
    ft = pd.to_numeric(a["goals_scored_per_match_home"], errors="coerce").fillna(0) + \
        pd.to_numeric(a["goals_scored_per_match_away"], errors="coerce").fillna(0)
    w = a["matches_played"].fillna(0).clip(lower=0)
    if ft.sum() <= 0 or w.sum() <= 0:
        return DEFAULT_HT_SHARE
    share = float(np.average((ht / ft.replace(0, np.nan)).fillna(DEFAULT_HT_SHARE), weights=w))
    return float(np.clip(share, 0.25, 0.60))


def from_match_files(a: pd.DataFrame) -> Ratings:
    """Fit the Dixon-Coles MLE, then expose it through the same interface.

    The match route keeps two things the team route cannot have: rho fitted on
    real scorelines, and time-decay weighting of older matches.
    """
    model = sm.fit(a)
    lg_home = float(a["hg"].mean())
    lg_away = float(a["ag"].mean())
    n_tot = {}
    for t in model.teams:
        n_tot[t] = float((a["home_team_name"] == t).sum() + (a["away_team_name"] == t).sum())
    return Ratings(teams=list(model.teams), atk_home={}, def_home={}, atk_away={}, def_away={},
                   lg_home=lg_home, lg_away=lg_away, ht_share=_ht_share_match(a),
                   rho=float(model.rho), n_matches=n_tot, source="match files", fitted=model)


def _ht_share_match(a: pd.DataFrame) -> float:
    cols = ["home_team_goal_count_half_time", "away_team_goal_count_half_time"]
    if not all(c in a.columns for c in cols):
        return DEFAULT_HT_SHARE
    ht = pd.to_numeric(a[cols[0]], errors="coerce") + pd.to_numeric(a[cols[1]], errors="coerce")
    ft = a["hg"] + a["ag"]
    ok = ht.notna() & (ft > 0)
    if ok.sum() < 20 or ft[ok].sum() <= 0:
        return DEFAULT_HT_SHARE
    return float(np.clip(ht[ok].sum() / ft[ok].sum(), 0.25, 0.60))


def build(kind: str, frame: pd.DataFrame) -> Ratings:
    if kind == spd.KIND_TEAM:
        return from_team_files(frame)
    return from_match_files(frame)


# --- expectations ------------------------------------------------------------

def expected_goals(r: Ratings, home: str, away: str) -> tuple[float, float]:
    """Full-time expected goals for each side.

    Team route: attack times opponent defence, normalised by the league average
    *for that venue*. Normalising the home side by the away average (or vice
    versa) silently inflates one and deflates the other.
    """
    if r.fitted is not None:
        return sm.expected_goals(r.fitted, home, away)
    for t in (home, away):
        if t not in r.atk_home:
            raise KeyError(f"{t!r} does not appear in the loaded data.")
    lh = r.atk_home[home] * r.def_away[away] / max(r.lg_home, 1e-6)
    la = r.atk_away[away] * r.def_home[home] / max(r.lg_away, 1e-6)
    return float(np.clip(lh, 0.05, 6.0)), float(np.clip(la, 0.05, 6.0))


def score_matrix(lh: float, la: float, rho: float) -> np.ndarray:
    return sm.score_matrix(lh, la, rho, MAX_GOALS)


# --- markets -----------------------------------------------------------------

def result_probs(m: np.ndarray) -> dict[str, float]:
    k = m.shape[0]
    x, y = np.meshgrid(np.arange(k), np.arange(k), indexing="ij")
    return {"home": float(m[x > y].sum()), "draw": float(m[x == y].sum()),
            "away": float(m[x < y].sum())}


def top_scorelines(m: np.ndarray, n: int = 6) -> list[tuple[int, int, float]]:
    return sm.top_scorelines(m, n)


def half_matrices(lh: float, la: float, share: float, rho: float = 0.0):
    """Split the full-time expectation into first and second half.

    The halves are left as **plain Poisson**, deliberately. Dixon-Coles `tau`
    was fitted on full-time scorelines, and applying it to each half and again
    at full time corrects the same dependence three times: the joint's
    full-time marginal then drifts off the direct full-time figures (0.529 vs
    0.512 on the first fixture tried). Two independent Poisson halves convolve
    to exactly the full-time Poisson, so the correction can be applied once,
    where it was measured.
    """
    share = float(np.clip(share, 0.05, 0.95))
    ht = score_matrix(lh * share, la * share, rho)
    sh = score_matrix(lh * (1 - share), la * (1 - share), rho)
    return ht, sh


def _raw_htft(ht: np.ndarray, sh: np.ndarray) -> np.ndarray:
    k = ht.shape[0]
    out = np.zeros((3, 3))
    idx = np.arange(k)
    hx, hy = np.meshgrid(idx, idx, indexing="ij")
    hres = np.where(hx > hy, 0, np.where(hx == hy, 1, 2))
    for a in range(k):
        for b in range(k):
            p = ht[a, b]
            if p < 1e-12:
                continue
            fx, fy = np.meshgrid(a + idx, b + idx, indexing="ij")
            fres = np.where(fx > fy, 0, np.where(fx == fy, 1, 2))
            np.add.at(out[hres[a, b]], fres.ravel(), (p * sh).ravel())
    s = out.sum()
    return out / s if s > 0 else out


def htft(ht: np.ndarray, sh: np.ndarray,
         ft_target: dict[str, float] | None = None,
         ht_target: dict[str, float] | None = None,
         iters: int = 12) -> np.ndarray:
    """Joint 3x3 over (half-time result, full-time result).

    Built as ``half-time + second half``. Multiplying a half-time distribution
    by a full-time one is wrong by construction - the full-time result already
    contains the half-time one - and scored 2.0261 against a 1.8860 base rate
    when tried. Convolving the two halves scored 1.8492.

    The raw convolution is then raked (iterative proportional fitting) onto the
    supplied full-time and half-time marginals, so the HT/FT tab can never
    disagree with the 1X2 and half-time tabs about the same fixture. Raking
    preserves the conditional shape - how a half-time state maps onto a
    full-time one - and only moves the totals.
    """
    out = _raw_htft(ht, sh)
    if ft_target is None and ht_target is None:
        return out
    ftv = np.array([ft_target[k] for k in ("home", "draw", "away")]) if ft_target else None
    htv = np.array([ht_target[k] for k in ("home", "draw", "away")]) if ht_target else None
    for _ in range(iters):
        if ftv is not None:
            col = out.sum(0)
            out *= np.where(col > 1e-12, ftv / np.where(col > 1e-12, col, 1), 1.0)[None, :]
        if htv is not None:
            row = out.sum(1)
            out *= np.where(row > 1e-12, htv / np.where(row > 1e-12, row, 1), 1.0)[:, None]
    s = out.sum()
    return out / s if s > 0 else out


HTFT_LABELS = [("H", "H"), ("H", "D"), ("H", "A"),
               ("D", "H"), ("D", "D"), ("D", "A"),
               ("A", "H"), ("A", "D"), ("A", "A")]


def asian_handicap(m: np.ndarray, lines: list[float] | None = None) -> pd.DataFrame:
    """Home-side cover probabilities across the standard handicap ladder.

    Quarter lines split the stake across the two neighbouring half lines, which
    is how they actually settle, so a -0.25 line can half-win or half-lose.
    """
    if lines is None:
        lines = [-2.0, -1.75, -1.5, -1.25, -1.0, -0.75, -0.5, -0.25,
                 0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]
    k = m.shape[0]
    x, y = np.meshgrid(np.arange(k), np.arange(k), indexing="ij")
    margin = (x - y).astype(float)

    def one(line: float) -> tuple[float, float, float]:
        adj = margin + line
        return (float(m[adj > 0].sum()), float(m[adj == 0].sum()), float(m[adj < 0].sum()))

    rows = []
    for line in lines:
        if abs(line * 4) % 2 == 1:                     # quarter line
            w1, p1, l1 = one(line - 0.25)
            w2, p2, l2 = one(line + 0.25)
            win, push, lose = (w1 + w2) / 2, (p1 + p2) / 2, (l1 + l2) / 2
        else:
            win, push, lose = one(line)
        # Stake back on a push, so the meaningful number is the chance of
        # winning given the bet resolves at all.
        live = win + lose
        rows.append({"line": line, "home_win": win, "push": push, "away_win": lose,
                     "home_no_push": win / live if live > 1e-9 else np.nan})
    return pd.DataFrame(rows)


def fair_line(ah: pd.DataFrame) -> float:
    """The handicap at which the fixture is closest to a coin flip."""
    d = ah.dropna(subset=["home_no_push"]).copy()
    if d.empty:
        return 0.0
    return float(d.loc[(d["home_no_push"] - 0.5).abs().idxmin(), "line"])


# --- confidence --------------------------------------------------------------

CONF_HIGH, CONF_MEDIUM, CONF_LOW = "High", "Medium", "Low"
CONF_COLOUR = {CONF_HIGH: "green", CONF_MEDIUM: "orange", CONF_LOW: "red"}

# Separation between the top two probabilities. A 3-way market where the leader
# is 8 points clear of second is a materially different proposition from one
# where they are level, even at the same headline probability.
SEP_HIGH, SEP_MED = 0.18, 0.08
# Combined matches behind the two teams' ratings.
SAMPLE_HIGH, SAMPLE_MED = 40, 18


def confidence(probs: list[float], n_matches: float) -> dict[str, Any]:
    """Band a prediction by how separated it is and how much data backs it.

    This measures the model's *own* certainty. It is not a claim about hit rate:
    on the bundled league the full-time 1X2 model was right 46.5% of the time
    overall, against 48.4% for the bookmaker.
    """
    p = sorted([float(v) for v in probs], reverse=True)
    sep = (p[0] - p[1]) if len(p) > 1 else p[0]

    if sep >= SEP_HIGH:
        by_sep = CONF_HIGH
    elif sep >= SEP_MED:
        by_sep = CONF_MEDIUM
    else:
        by_sep = CONF_LOW

    if n_matches >= SAMPLE_HIGH:
        by_n = CONF_HIGH
    elif n_matches >= SAMPLE_MED:
        by_n = CONF_MEDIUM
    else:
        by_n = CONF_LOW

    order = [CONF_LOW, CONF_MEDIUM, CONF_HIGH]
    band = order[min(order.index(by_sep), order.index(by_n))]   # weakest link wins

    if order.index(by_n) < order.index(by_sep):
        why = (f"limited by data - only {n_matches:.0f} matches behind these two ratings")
    elif sep >= SEP_HIGH:
        why = f"clear favourite - {100 * sep:.0f} points clear of the next outcome"
    elif sep >= SEP_MED:
        why = f"moderate lead - {100 * sep:.0f} points clear of the next outcome"
    else:
        why = f"close to a toss-up - only {100 * sep:.0f} points separate the top two"
    return {"band": band, "colour": CONF_COLOUR[band], "separation": sep,
            "n_matches": float(n_matches), "why": why}


# --- one call for the whole fixture -----------------------------------------

def predict(r: Ratings, home: str, away: str) -> dict[str, Any]:
    lh, la = expected_goals(r, home, away)
    ft = score_matrix(lh, la, r.rho)
    ht_m, sh_m = half_matrices(lh, la, r.ht_share)
    ftp = result_probs(ft)
    htp = result_probs(ht_m)
    joint = htft(ht_m, sh_m, ft_target=ftp, ht_target=htp)
    ah = asian_handicap(ft)
    n = r.sample(home) + r.sample(away)
    # Confidence on the handicap is about how decisively one side is favoured on
    # a LEVEL line. Taking the best cover chance across the whole ladder instead
    # reads off the +2.0 line, which is ~99% for every fixture ever played and
    # made this band say "High" unconditionally.
    level = ah.loc[ah["line"] == 0.0, "home_no_push"]
    p_level = float(level.iloc[0]) if len(level) else 0.5
    return {
        "home": home, "away": away, "lambda_home": lh, "lambda_away": la,
        "ht_share": r.ht_share, "rho": r.rho, "source": r.source,
        "ft": ftp, "ht": htp, "ft_matrix": ft, "ht_matrix": ht_m,
        "htft": joint, "asian": ah, "fair_line": fair_line(ah),
        "top_scorelines": top_scorelines(ft, 6),
        "n_matches": n,
        "conf_ft": confidence([ftp["home"], ftp["draw"], ftp["away"]], n),
        "conf_ht": confidence([htp["home"], htp["draw"], htp["away"]], n),
        "conf_htft": confidence(list(joint.ravel()), n),
        "conf_ah": confidence([p_level, 1 - p_level], n),
        "p_level": p_level,
    }
