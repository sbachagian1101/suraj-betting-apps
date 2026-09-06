"""Prediction model: last-N xG / xGA / ratings -> Poisson match probabilities.

Design notes (kept deliberately simple and transparent):

* Attack rate  = blend of xG-for and goals-for over the last N matches,
  weighted by recency (each older match counts DECAY times the one before).
* Defence rate = blend of xG-against and goals-against, same weighting.
* Both are shrunk toward a league-average rate with a prior worth PRIOR_GAMES
  games, against the *effective* sample size (sum of the recency weights).
* Line-up ratings use a shorter, separate window (N_LINEUP) because who is in
  form changes faster than a team's underlying attack and defence.
* Expected goals for each side follow the usual multiplicative form:
      home = att_home * def_away / AVG * HOME_ADV
      away = att_away * def_home / AVG * AWAY_ADV
* Line-up adjustment: the team's starting XI today is rated by each player's
  average Soccerway rating over the last N matches. The gap between that and
  the team's average rating over the same matches nudges the attack (and the
  opponent's attack) through exp(k * gap).
* Score probabilities come from an independent Poisson grid.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from statistics import mean
from typing import Optional

from soccerway import MatchRecord, Player

LEAGUE_AVG = 1.45          # goals per team per game (top-flight Europe)
PRIOR_GAMES = 2.0          # shrinkage weight in "games"
DECAY = 0.85               # recency weight: match k back counts DECAY**k
N_RATES = 8                # default window for xG / goals rates
N_LINEUP = 4               # default window for player ratings and line-up
HOME_ADV = 1.12
AWAY_ADV = 0.90
XG_WEIGHT = 0.7            # rest is actual goals
RATING_K = 0.35            # attack multiplier per rating point of line-up gap
RATING_K_OPP = 0.20        # opponent attack multiplier per rating point
MAX_GOALS = 8


def _mean(xs) -> Optional[float]:
    xs = [x for x in xs if x is not None]
    return mean(xs) if xs else None


@dataclass
class TeamProfile:
    name: str
    records: list[MatchRecord]          # rates window, most recent first
    lineup_records: list[MatchRecord]   # shorter window for ratings / line-up
    n: int
    n_lineup: int
    weights: list[float]                # recency weight per record
    n_eff: float                        # sum of weights
    form: str
    points: int
    goals_for: float
    goals_against: float
    xg_for: Optional[float]
    xg_against: Optional[float]
    avg_rating: Optional[float]         # over the line-up window
    attack: float
    defence: float
    xg_games: int                       # matches that actually had xG


def recency_weights(n: int, decay: float = DECAY) -> list[float]:
    return [decay ** k for k in range(n)]


def _wmean(values, weights) -> Optional[float]:
    pairs = [(v, w) for v, w in zip(values, weights) if v is not None]
    if not pairs:
        return None
    return sum(v * w for v, w in pairs) / sum(w for _, w in pairs)


def _shrink(sample: Optional[float], n_eff: float) -> float:
    if sample is None or n_eff <= 0:
        return LEAGUE_AVG
    return (sample * n_eff + LEAGUE_AVG * PRIOR_GAMES) / (n_eff + PRIOR_GAMES)


def profile(name: str, records: list[MatchRecord],
            n_rates: Optional[int] = None, n_lineup: Optional[int] = None) -> TeamProfile:
    """Build a team profile.

    ``records`` is most-recent-first. The first ``n_rates`` feed the attack and
    defence rates (recency weighted); the first ``n_lineup`` feed the ratings.
    """
    rates = records[:n_rates] if n_rates else records
    lineup = records[:n_lineup] if n_lineup else rates
    n = len(rates)
    w = recency_weights(n)
    n_eff = sum(w)
    gf = _wmean((r.goals_for for r in rates), w)
    ga = _wmean((r.goals_against for r in rates), w)
    xgf = _wmean((r.xg_for for r in rates), w)
    xga = _wmean((r.xg_against for r in rates), w)
    xg_games = sum(1 for r in rates if r.xg_for is not None)

    def blend(xg, goals):
        if xg is None and goals is None:
            return None
        if xg is None:
            return goals
        if goals is None:
            return xg
        return XG_WEIGHT * xg + (1 - XG_WEIGHT) * goals

    attack = _shrink(blend(xgf, gf), n_eff)
    defence = _shrink(blend(xga, ga), n_eff)
    form = "".join(r.result for r in rates)          # most recent first
    points = sum({"W": 3, "D": 1, "L": 0}[r.result] for r in rates)
    return TeamProfile(
        name=name, records=rates, lineup_records=lineup, n=n, n_lineup=len(lineup),
        weights=w, n_eff=n_eff, form=form, points=points,
        goals_for=gf or 0.0, goals_against=ga or 0.0,
        xg_for=xgf, xg_against=xga,
        avg_rating=_mean(r.team_rating for r in lineup),
        attack=attack, defence=defence, xg_games=xg_games,
    )


# --------------------------------------------------------------------------- #
# Line-up strength
# --------------------------------------------------------------------------- #
@dataclass
class PlayerHistory:
    id: str
    name: str
    role: str
    ratings: list[float] = field(default_factory=list)
    starts: int = 0

    @property
    def avg(self) -> Optional[float]:
        return mean(self.ratings) if self.ratings else None


def player_histories(records: list[MatchRecord]) -> dict[str, PlayerHistory]:
    """Per-player rating history across the team's last N matches."""
    hist: dict[str, PlayerHistory] = {}
    for r in records:
        for p in r.starters:
            h = hist.setdefault(p.id, PlayerHistory(p.id, p.name, p.role))
            h.starts += 1
            if p.rating is not None:
                h.ratings.append(p.rating)
    return hist


@dataclass
class LineupAssessment:
    source: str                                  # "today" or "last match"
    formation: str
    starters: list[Player]
    rated: int                                   # starters with history
    lineup_rating: Optional[float]
    team_recent_rating: Optional[float]
    gap: float                                   # lineup_rating - team_recent_rating
    missing: list[PlayerHistory]                 # regulars not starting today
    newcomers: list[Player]                      # starters without history
    rows: list[dict]                             # for display


def assess_lineup(starters: list[Player], source: str, formation: str,
                  prof: TeamProfile) -> LineupAssessment:
    hist = player_histories(prof.lineup_records)
    team_avg = prof.avg_rating
    vals, rows, newcomers = [], [], []
    for p in starters:
        h = hist.get(p.id)
        avg = h.avg if h else None
        if avg is not None:
            vals.append(avg)
        else:
            newcomers.append(p)
        rows.append({
            "No": p.number, "Player": p.name, "Pos": p.role,
            "Starts (last N)": h.starts if h else 0,
            "Avg rating": round(avg, 2) if avg is not None else None,
            "Ratings": ", ".join(f"{x:.1f}" for x in h.ratings) if h else "",
        })
    lineup_rating = mean(vals) if vals else None
    gap = 0.0
    if lineup_rating is not None and team_avg is not None and len(vals) >= 5:
        gap = lineup_rating - team_avg
    starter_ids = {p.id for p in starters}
    regular_cut = max(1, math.ceil(prof.n_lineup / 2))
    missing = sorted(
        (h for h in hist.values()
         if h.id not in starter_ids and h.starts >= regular_cut),
        key=lambda h: -(h.avg or 0))
    return LineupAssessment(
        source=source, formation=formation, starters=starters, rated=len(vals),
        lineup_rating=lineup_rating, team_recent_rating=team_avg, gap=gap,
        missing=missing, newcomers=newcomers, rows=rows,
    )


# --------------------------------------------------------------------------- #
# Poisson
# --------------------------------------------------------------------------- #
def _pois(lmb: float, k: int) -> float:
    return math.exp(-lmb) * lmb ** k / math.factorial(k)


@dataclass
class Prediction:
    exp_home: float
    exp_away: float
    p_home: float
    p_draw: float
    p_away: float
    p_over25: float
    p_btts: float
    top_scores: list[tuple[str, float]]
    grid: list[list[float]]
    home_mult: float
    away_mult: float


def predict(home: TeamProfile, away: TeamProfile,
            home_lineup: Optional[LineupAssessment] = None,
            away_lineup: Optional[LineupAssessment] = None) -> Prediction:
    exp_home = home.attack * away.defence / LEAGUE_AVG * HOME_ADV
    exp_away = away.attack * home.defence / LEAGUE_AVG * AWAY_ADV

    hg = home_lineup.gap if home_lineup else 0.0
    ag = away_lineup.gap if away_lineup else 0.0
    home_mult = math.exp(RATING_K * hg - RATING_K_OPP * ag)
    away_mult = math.exp(RATING_K * ag - RATING_K_OPP * hg)
    exp_home *= home_mult
    exp_away *= away_mult
    exp_home = min(max(exp_home, 0.2), 5.0)
    exp_away = min(max(exp_away, 0.2), 5.0)

    grid = [[_pois(exp_home, i) * _pois(exp_away, j) for j in range(MAX_GOALS + 1)]
            for i in range(MAX_GOALS + 1)]
    total = sum(map(sum, grid))
    grid = [[x / total for x in row] for row in grid]

    p_home = sum(grid[i][j] for i in range(MAX_GOALS + 1) for j in range(MAX_GOALS + 1) if i > j)
    p_draw = sum(grid[i][i] for i in range(MAX_GOALS + 1))
    p_away = 1 - p_home - p_draw
    p_over = sum(grid[i][j] for i in range(MAX_GOALS + 1) for j in range(MAX_GOALS + 1) if i + j > 2)
    p_btts = sum(grid[i][j] for i in range(1, MAX_GOALS + 1) for j in range(1, MAX_GOALS + 1))
    scores = sorted(((f"{i}-{j}", grid[i][j]) for i in range(MAX_GOALS + 1)
                     for j in range(MAX_GOALS + 1)), key=lambda t: -t[1])[:5]
    return Prediction(exp_home, exp_away, p_home, p_draw, p_away, p_over, p_btts,
                      scores, grid, home_mult, away_mult)


# --------------------------------------------------------------------------- #
# Market comparison
# --------------------------------------------------------------------------- #
def implied(odds: dict) -> Optional[dict]:
    if not odds:
        return None
    raw = {k: 1 / odds[k] for k in ("home", "draw", "away") if odds.get(k)}
    if len(raw) < 3:
        return None
    s = sum(raw.values())
    return {k: v / s for k, v in raw.items()} | {"overround": s - 1}


# --------------------------------------------------------------------------- #
# Insights
# --------------------------------------------------------------------------- #
def insights(home: TeamProfile, away: TeamProfile, pred: Prediction,
             hl: Optional[LineupAssessment], al: Optional[LineupAssessment],
             odds: Optional[dict]) -> list[str]:
    out: list[str] = []

    def team_notes(t: TeamProfile, la: Optional[LineupAssessment], venue: str):
        if t.n == 0:
            out.append(f"**{t.name}**: no finished matches found, league-average rates used.")
            return
        out.append(f"**{t.name}** form {t.form} ({t.points} pts from {t.n}), "
                   f"scoring {t.goals_for:.1f} and conceding {t.goals_against:.1f} per game "
                   f"(recency-weighted, newest match counts most).")
        if t.xg_for is not None:
            diff = t.goals_for - t.xg_for
            if diff > 0.5:
                out.append(f"{t.name} scored {diff:.1f} goals per game *above* their xG "
                           f"({t.xg_for:.2f}); expect some regression.")
            elif diff < -0.5:
                out.append(f"{t.name} scored {abs(diff):.1f} goals per game *below* their xG "
                           f"({t.xg_for:.2f}); they are creating more than the scorelines show.")
            ddiff = t.goals_against - t.xg_against if t.xg_against is not None else 0
            if ddiff > 0.5:
                out.append(f"{t.name} conceded {ddiff:.1f} more per game than their xGA "
                           f"({t.xg_against:.2f}); finishing against them has been unusually clinical.")
            elif ddiff < -0.5:
                out.append(f"{t.name} conceded {abs(ddiff):.1f} fewer per game than their xGA "
                           f"({t.xg_against:.2f}); the defence has been riding its luck.")
        if t.xg_games < t.n:
            out.append(f"{t.name}: only {t.xg_games} of {t.n} matches had xG data; "
                       f"actual goals filled the gap.")
        if la:
            src = "today's confirmed XI" if la.source == "today" else "the last match's XI (today's not published yet)"
            if la.lineup_rating is not None and la.team_recent_rating is not None:
                out.append(f"{t.name} line-up ({la.formation or '?'}, {src}) averages "
                           f"{la.lineup_rating:.2f} across the last {t.n_lineup} games versus a team "
                           f"average of {la.team_recent_rating:.2f} (gap {la.gap:+.2f}).")
            if la.missing:
                names = ", ".join(f"{h.name} ({h.avg:.1f})" if h.avg else h.name for h in la.missing[:4])
                out.append(f"{t.name} regulars not starting: {names}.")
            if la.newcomers:
                names = ", ".join(p.name for p in la.newcomers[:4])
                out.append(f"{t.name} starters with no rating in the last {t.n_lineup} games: {names}.")

    team_notes(home, hl, "home")
    team_notes(away, al, "away")

    fav = max((pred.p_home, "home"), (pred.p_draw, "draw"), (pred.p_away, "away"))
    label = {"home": home.name, "away": away.name, "draw": "the draw"}[fav[1]]
    out.append(f"Model expects {pred.exp_home:.2f} - {pred.exp_away:.2f}; most likely outcome is "
               f"{label} at {fav[0]:.0%}, top scoreline {pred.top_scores[0][0]} "
               f"({pred.top_scores[0][1]:.0%}).")

    imp = implied(odds) if odds else None
    if imp:
        edges = {k: getattr(pred, "p_" + k) - imp[k] for k in ("home", "draw", "away")}
        k, e = max(edges.items(), key=lambda kv: kv[1])
        name = {"home": home.name, "away": away.name, "draw": "Draw"}[k]
        out.append(f"Book (bet365) implies {imp['home']:.0%} / {imp['draw']:.0%} / {imp['away']:.0%}. "
                   f"Biggest model-vs-book gap: {name} {e:+.1%}.")
        if abs(e) >= 0.10:
            out.append("A gap that large after only a few matches usually means the model is "
                       "missing something the book knows (injuries, motivation, schedule). "
                       "Treat it as a question, not a bet.")
        if odds.get("home_open") and odds.get("away_open"):
            mv = []
            for kk, nm in (("home", home.name), ("away", away.name)):
                if odds[kk] and odds[kk + "_open"] and abs(odds[kk] - odds[kk + "_open"]) >= 0.1:
                    mv.append(f"{nm} {odds[kk + '_open']:.2f} -> {odds[kk]:.2f}")
            if mv:
                out.append("Market movement since opening: " + "; ".join(mv) + ".")
    return out
