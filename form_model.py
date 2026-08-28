"""Form score and win probability for every horse, from form alone.

No market anywhere in this model. The meeting export carries no prices, so the
earlier finding — that a trained model could not out-rank the market — does not
apply: there is no market here to lose to. What this has to beat is a dart throw
and each single column used on its own.

## Why the weights are fixed rather than fitted

Fitting was tried first, on 65 races with a result. A 28-feature conditional
logit scored **16.9%** on top pick — *worse* than half a dozen single columns —
and gave negative weights to "better record at this distance" and "better record
on this surface". Both symptoms of one cause: 65 races cannot identify 28 free
parameters, so the fit chases noise and lands on signs that are physically
backwards. Fixed weights cannot overfit, because nothing about those races enters
them.

The choice costs less than it looks. Against 200 random weightings of the same
14 columns, these weights sit *inside* the spread on every measure (top-1 median
0.200, range 0.169–0.246; these weights 0.215). Dropping the single
highest-weighted column entirely moves top-1 from 0.215 to 0.200. **The choice of
columns is doing the work, not the exact numbers on them** — which is the
honest reason not to agonise over the weights.

## What it scored, leave-one-race-out on 65 races

| | top pick wins | its top 3 place | winner in top 3 |
|---|---|---|---|
| dart throw | 9.2% | 25.6% | 33.8% |
| this model | **21.5%** | **42.6%** | **55.4%** |

It clears a dart throw comfortably. Whether it clears its own best single column
(last-start margin, 24.6%) is **not resolvable at this sample size** — three
races separate them.

The one firm structural finding: ranks 1, 2 and 3 perform about the same
(21.5% / 16.9% / 16.9% win, and 44.6% / 43.1% / 40.0% place), then rank 4 falls
away. **The score identifies a top-three group well and cannot reliably order
within it.** Read it that way — the top three are a shortlist, not a ranking.

Win probabilities are calibrated: across six bands the mean gap between
predicted and actual was 0.015, and the most confident band predicted 0.318
against 0.333 actual. Place probabilities are mildly optimistic at the top; see
`DEFAULT_PLACE_SHRINK` for why nothing is done about that yet.

## The red alert

A separate, hand-specified filter sits on top of the ranking — see
`RED_MIN_CAREER_RUNS` and the functions around `red_conditions`. It marks a
shortlisted horse that is experienced, arriving in form, already race-fit and
well ridden. It is **not** part of the score and is **not** validated: 65 races
of results cannot establish whether the combination beats the top three on its
own. It is a way of finding horses that match a stated profile, not evidence
that the profile wins.

## Sample size

65 races. Every rate here carries a 95% interval roughly ±10 points wide. Treat
these as "clearly better than random", not as precise figures.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

# Weights on within-race z-scores. Higher always means a better chance; columns
# where low is better are flipped by SIGN below. Set from racing reasoning:
# recent form and class carry most, connections and course suitability support.
WEIGHTS: dict[str, float] = {
    "ls_mgn": 1.0,      # beaten margin last start — the closest thing to a rating
    "form_mean": 0.9,   # mean recent finishing position
    "pmcar_log": 0.7,   # career prize money — a class proxy
    "pm12_log": 0.7,    # prize money in the last year — current class
    "car_win": 0.6,     # career strike rate
    "12m_win": 0.6,     # strike rate over the last year
    "12m_plc": 0.4,     # place strike rate over the last year
    "jrat": 0.5,        # jockey rating
    "trat": 0.5,        # trainer rating
    "jt_win": 0.3,      # jockey/trainer combination
    "dist_win": 0.3,    # record at today's distance
    "crs_win": 0.2,     # record at today's course
    "cd_win": 0.2,      # record at course AND distance
    "surf_win": 0.3,    # record on today's surface
}

# Columns where a LOWER raw value is better.
SIGN: dict[str, float] = {"ls_mgn": -1.0, "form_mean": -1.0, "ls_pos": -1.0,
                          "bp": -1.0}

LABELS = {
    "ls_mgn": "beaten margin last start", "form_mean": "recent finishing form",
    "pmcar_log": "career prize money", "pm12_log": "prize money, 12 months",
    "car_win": "career strike rate", "12m_win": "strike rate, 12 months",
    "12m_plc": "place strike rate, 12 months", "jrat": "jockey rating",
    "trat": "trainer rating", "jt_win": "jockey/trainer combination",
    "dist_win": "record at this distance", "crs_win": "record at this course",
    "cd_win": "record at course and distance", "surf_win": "record on this surface",
}

# Measured on the 65 races of 2026-08-27 that had a result. Intervals are wide;
# ranks 1-3 are not separable from each other.
RANK_STATS = {
    1: dict(win=0.215, place=0.446, n=65),
    2: dict(win=0.169, place=0.431, n=65),
    3: dict(win=0.169, place=0.400, n=65),
    4: dict(win=0.123, place=0.246, n=65),
    5: dict(win=0.031, place=0.200, n=65),
    6: dict(win=0.066, place=0.230, n=61),
}
BASELINE = dict(win=0.105, place=0.254, top3_has_winner=0.338)
MEASURED = dict(races=65, runners=712, top1_win=0.215, top3_place=0.426,
                top3_has_winner=0.554, calibration_error=0.015,
                place_calibration_error=0.038)

# No shrink by default, and that is a finding rather than an omission.
#
# Plackett-Luce is known to overstate the place chance of the top of the market,
# and it does so here: the most confident band predicts 0.62 against 0.57 actual.
# The obvious fix is a shrink toward the base rate, so it was measured at six
# strengths. Runner-weighted calibration error came out 0.038, 0.033, 0.035,
# 0.036, 0.030, 0.032 for shrink 0.00 to 0.30 — non-monotonic, spanning half a
# point. That is noise, not a signal, and picking the minimum would be tuning a
# parameter on the same 65 races it is scored against.
#
# So the place figures ship uncorrected, with the optimism stated instead: read
# a place probability above ~55% as a few points generous. The parameter is
# still exposed for when there are enough races to settle it.
DEFAULT_PLACE_SHRINK = 0.0
DEFAULT_SIMS = 6000

# ---------------------------------------------------------------------------
# Red alert: a hand-specified filter applied WITHIN the top three on form.
#
# This is the user's rule, not a measured one. It marks a shortlisted horse that
# is experienced, arriving in form, already race-fit, and well ridden:
#
#   1. more than 4 career starts
#   2. placed (1st-3rd) at its last start
#   3. has already had a run this preparation - i.e. not resuming
#   4. its jockey is among the top 3 in the race by jockey rating
#
# All four must hold. It is a flag on top of the ranking, not part of the score,
# and it has NOT been validated against results - there are only 65 races of
# outcomes and 141 runners meet the first three conditions, far too few to say
# whether the combination beats the top three on its own.
RED_MIN_CAREER_RUNS = 4      # strictly more than this
RED_MAX_LAST_FINISH = 3      # "placed" last start
RED_TOP_JOCKEYS = 3          # jockey inside the race's top N by rating
RED_WITHIN_TOP = 3           # only the top N on form are eligible

# How the alert actually did on the 65 races that have results. Reported so the
# rule can be watched, NOT as evidence that it works - see the caveats in the
# app's Method tab. n=27 is far too small, and all 65 races are a single day.
RED_MEASURED = dict(
    alerts=27, alert_win=0.370, alert_place=0.630,
    others=168, other_win=0.155, other_place=0.393,
    win_p=0.0138, place_p=0.0345,
    alert_win_ci=(0.215, 0.558), alert_place_ci=(0.442, 0.785),
    other_win_ci=(0.108, 0.217), other_place_ci=(0.322, 0.468),
    fired_in_races=29, of_races=73, share_of_shortlist=0.15,
)

RED_LABELS = {
    "runs": "more than {n} career starts",
    "placed": "placed at its last start",
    "fit": "already had a run this preparation",
    "jockey": "jockey in the race's top {n} by rating",
}


def places_paid(n_runners: int) -> int:
    """Standard terms: 3 places for 8+, 2 for 5-7, win only under 5."""
    if n_runners >= 8:
        return 3
    if n_runners >= 5:
        return 2
    return 1


def has_run_this_prep(form: Any) -> bool:
    """True when the horse has already raced this preparation.

    Racing & Sports form strings run **oldest to newest**, with `x` marking a
    spell, so a trailing `x` means the most recent event was a break and the
    horse resumes today. Verified two ways on the 27 August card:

    * the rightmost digit matched the independent "last start" column for
      **726 of 726** runners that had both;
    * every one of the 105 horses whose form ends in `x` had been off for more
      than 60 days (median 124), against none of the 626 that do not (median 17).

    A horse with no figures at all is a first-starter, which also fails.
    """
    if not isinstance(form, str):
        return False
    if not any(ch.isdigit() for ch in form):
        return False
    return not form.rstrip().endswith("x")


def jockey_rank(race: pd.DataFrame) -> np.ndarray:
    """Rank the field by jockey rating, 1 = best. Higher rating is better.

    Ties share the best rank, so in a race where seven jockeys are rated alike
    they are all "top 3" - which is the honest answer, because in that race the
    rating does not separate them. Unrated jockeys sort last rather than
    accidentally winning the tie.
    """
    r = pd.to_numeric(race.get("jrat"), errors="coerce")
    if r is None or r.isna().all():
        return np.full(len(race), len(race) + 1, dtype=int)
    return r.fillna(-np.inf).rank(ascending=False, method="min").to_numpy(dtype=int)


def red_conditions(race: pd.DataFrame, *,
                   min_runs: int = RED_MIN_CAREER_RUNS,
                   max_last: int = RED_MAX_LAST_FINISH,
                   top_jockeys: int = RED_TOP_JOCKEYS) -> pd.DataFrame:
    """The four conditions, per runner, before the top-N restriction."""
    race = race.reset_index(drop=True)
    runs = pd.to_numeric(race.get("car_runs"), errors="coerce")
    last = pd.to_numeric(race.get("ls_pos"), errors="coerce")
    jr = jockey_rank(race)
    return pd.DataFrame({
        "runs": (runs > min_runs).fillna(False).to_numpy(dtype=bool),
        "placed": (last <= max_last).fillna(False).to_numpy(dtype=bool),
        "fit": np.array([has_run_this_prep(v) for v in race.get("Form L5",
                                                                pd.Series(dtype=object))],
                        dtype=bool),
        "jockey": jr <= int(top_jockeys),
    })


def _z(frame: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Within-race z-scores. Racing is a ranking problem: what matters is a
    figure relative to today's opponents, and it also removes the scale
    differences between one country's data and another's."""
    X = frame[cols].astype(float)
    X = X.fillna(X.median())
    X = X.fillna(0.0)                       # a column empty for the whole race
    m, s = X.mean(), X.std(ddof=0)
    return ((X - m) / s.replace(0.0, np.nan)).fillna(0.0)


def contributions(race: pd.DataFrame) -> pd.DataFrame:
    """Each column's signed, weighted contribution to every runner's score."""
    cols = list(WEIGHTS)
    Z = _z(race, cols)
    total = sum(WEIGHTS.values())
    out = pd.DataFrame(index=race.index)
    for c in cols:
        out[c] = Z[c] * SIGN.get(c, 1.0) * WEIGHTS[c] / total
    return out


def raw_score(race: pd.DataFrame) -> np.ndarray:
    return contributions(race).sum(axis=1).values


def win_probability(score: np.ndarray) -> np.ndarray:
    """Softmax within the race.

    Left at temperature 1.0 because that is what measured as calibrated: across
    six probability bands the mean absolute gap between predicted and actual was
    0.014, and the most confident band predicted 0.302 against 0.333 actual.
    """
    s = np.asarray(score, dtype=float)
    e = np.exp(s - s.max())
    return e / e.sum()


def place_probability(p_win: np.ndarray, places: int, *,
                      shrink: float = DEFAULT_PLACE_SHRINK,
                      sims: int = DEFAULT_SIMS, seed: int = 0) -> np.ndarray:
    """P(finishing inside the paying places), by Plackett-Luce sampling.

    Finishing orders are drawn with each horse's chance proportional to its win
    probability (the Gumbel top-k trick), then the result is shrunk toward the
    field's base place rate to correct the known optimism at the top.
    """
    p = np.clip(np.asarray(p_win, dtype=float), 1e-12, None)
    p = p / p.sum()
    n = len(p)
    k = int(np.clip(places, 1, n))
    if n == 1:
        return np.ones(1)
    rng = np.random.default_rng(seed)
    g = rng.gumbel(size=(sims, n))
    order = np.argsort(-(np.log(p) + g), axis=1)[:, :k]
    hits = np.zeros(n)
    np.add.at(hits, order.ravel(), 1.0)
    raw = hits / sims
    base = k / n
    return raw + float(np.clip(shrink, 0.0, 1.0)) * (base - raw)


def form_score(score: np.ndarray) -> np.ndarray:
    """Raw score rescaled to 0-100 within the race, best on 100.

    A relative figure by construction: 100 means "top rated here", not "good in
    absolute terms". The bottom is pinned at 10 rather than 0 so that a wide
    race and a close one do not look identical.
    """
    s = np.asarray(score, dtype=float)
    lo, hi = s.min(), s.max()
    if not np.isfinite(lo) or hi - lo < 1e-12:
        return np.full(len(s), 50.0)
    return 10.0 + 90.0 * (s - lo) / (hi - lo)


def why(race: pd.DataFrame, i: int, top: int = 3) -> str:
    """The columns pushing one runner up or down, strongest first."""
    c = contributions(race).iloc[i]
    ranked = c.reindex(c.abs().sort_values(ascending=False).index)
    up = [f"{LABELS.get(k, k)} (+{v:.2f})" for k, v in ranked.items() if v > 0.01][:top]
    dn = [f"{LABELS.get(k, k)} ({v:.2f})" for k, v in ranked.items() if v < -0.01][:top]
    bits = []
    if up:
        bits.append("for: " + ", ".join(up))
    if dn:
        bits.append("against: " + ", ".join(dn))
    return "; ".join(bits) if bits else "nothing separates this runner from the field"


def rate_race(race: pd.DataFrame, *, places: int | None = None,
              shrink: float = DEFAULT_PLACE_SHRINK,
              sims: int = DEFAULT_SIMS, seed: int = 0,
              min_runs: int = RED_MIN_CAREER_RUNS,
              max_last: int = RED_MAX_LAST_FINISH,
              top_jockeys: int = RED_TOP_JOCKEYS,
              within_top: int = RED_WITHIN_TOP) -> pd.DataFrame:
    """Score one race. Returns a table ranked best first.

    `race` must be the runners of a single race, one row each.
    """
    race = race.reset_index(drop=True)
    n = len(race)
    if n == 0:
        return pd.DataFrame()
    if places is None:
        places = places_paid(n)

    s = raw_score(race)
    pw = win_probability(s)
    pp = place_probability(pw, places, shrink=shrink, sims=sims, seed=seed)
    fs = form_score(s)

    out = pd.DataFrame({
        "Tab": race["tab"].astype("Int64"),
        "Horse": race["horse"],
        "Form score": fs,
        "Win%": 100 * pw,
        "Place%": 100 * pp,
        "Fair win $": 1.0 / np.clip(pw, 1e-9, None),
        "Fair place $": 1.0 / np.clip(pp, 1e-9, None),
        "Last 5": race.get("Form L5"),
        "BP": race.get("bp"),
        "_raw": s,
    })
    out["Why"] = [why(race, i) for i in range(n)]

    cond = red_conditions(race, min_runs=min_runs, max_last=max_last,
                          top_jockeys=top_jockeys)
    for c in cond.columns:
        out["_c_" + c] = cond[c].to_numpy()
    out["_jrank"] = jockey_rank(race)

    out = out.sort_values("_raw", ascending=False).reset_index(drop=True)
    out.insert(0, "Rank", np.arange(1, n + 1))

    # The alert only applies inside the shortlist: a horse outside the top N is
    # not a selection, so flagging it would be noise.
    eligible = out["Rank"] <= int(within_top)
    met = out[["_c_" + c for c in cond.columns]].all(axis=1)
    out["Alert"] = np.where(eligible & met, "\U0001F534", "")
    out["Checks"] = [
        f"{int(m)}/4" if e else "—"
        for m, e in zip(out[["_c_" + c for c in cond.columns]].sum(axis=1), eligible)]

    hist = out["Rank"].map(lambda r: RANK_STATS.get(int(r), {}).get("win"))
    out["Historic win% at this rank"] = [100 * v if v is not None else np.nan
                                         for v in hist]
    return out.drop(columns=["_raw"])


def red_detail(table: pd.DataFrame, *, min_runs: int = RED_MIN_CAREER_RUNS,
               top_jockeys: int = RED_TOP_JOCKEYS) -> pd.DataFrame:
    """Condition-by-condition breakdown for the shortlisted runners."""
    eligible = table[table["Checks"] != "—"]
    rows = []
    for _, r in eligible.iterrows():
        rows.append({
            "Rank": r["Rank"], "Tab": r["Tab"], "Horse": r["Horse"],
            RED_LABELS["runs"].format(n=min_runs): "✅" if r["_c_runs"] else "❌",
            RED_LABELS["placed"]: "✅" if r["_c_placed"] else "❌",
            RED_LABELS["fit"]: "✅" if r["_c_fit"] else "❌",
            RED_LABELS["jockey"].format(n=top_jockeys):
                ("✅" if r["_c_jockey"] else "❌") + f"  (rated #{int(r['_jrank'])})",
            "Alert": r["Alert"] or "—",
        })
    return pd.DataFrame(rows)


def red_summary(table: pd.DataFrame) -> str:
    hits = table[table["Alert"] != ""]
    if hits.empty:
        return ("No shortlisted runner meets all four alert conditions in this "
                "race. That is the common case — across the 27 August card only "
                "a minority of races produced one.")
    names = " · ".join(f"#{int(r['Tab'])} {r['Horse']}" for _, r in hits.iterrows())
    return f"🔴 **Alert: {names}** — meets all four conditions."


def rate_meeting(df: pd.DataFrame, **kw) -> dict[str, pd.DataFrame]:
    """Score every race in a parsed meeting frame."""
    return {rid: rate_race(g, **kw) for rid, g in df.groupby("race_id", sort=False)}


def summary_line(table: pd.DataFrame, meta: dict[str, Any] | None = None) -> str:
    if table.empty:
        return "No runners read for this race."
    top = table.iloc[0]
    return (f"Top rated **#{int(top['Tab'])} {top['Horse']}** — form score "
            f"{top['Form score']:.0f}, win {top['Win%']:.1f}%, place "
            f"{top['Place%']:.1f}%. On the 65 races this was measured against, "
            f"the top-rated horse won {100*RANK_STATS[1]['win']:.0f}% of the "
            f"time against {100*BASELINE['win']:.0f}% for a dart throw — but "
            f"ranks 1, 2 and 3 all performed about the same, so read the top "
            f"three as a group rather than an order.")


def style(table: pd.DataFrame):
    """Green for the top three, since that group is what the score resolves."""
    def row_colour(row):
        if row.get("Alert"):
            # Red outranks the green shortlist shading: an alert row is the one
            # thing on the table the reader is meant to spot first.
            return ["background-color: rgba(231, 76, 60, 0.30); "
                    "font-weight: 600"] * len(row)
        if row["Rank"] <= 3:
            bg = "rgba(46, 204, 113, 0.22)"
        elif row["Rank"] <= 5:
            bg = "rgba(52, 152, 219, 0.12)"
        else:
            bg = "rgba(128, 128, 128, 0.05)"
        return [f"background-color: {bg}"] * len(row)

    fmt = {"Form score": "{:.0f}", "Win%": "{:.1f}%", "Place%": "{:.1f}%",
           "Fair win $": "${:.2f}", "Fair place $": "${:.2f}",
           "Historic win% at this rank": "{:.1f}%"}
    fmt = {k: v for k, v in fmt.items() if k in table.columns}
    hide = [c for c in table.columns if c.startswith("_")]
    st = table.style.apply(row_colour, axis=1).format(fmt, na_rep="—")
    return st.hide(hide, axis="columns") if hide else st
