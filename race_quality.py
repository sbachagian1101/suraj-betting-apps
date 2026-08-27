"""Which races are worth playing at all.

The Place Finder answers "which runner". This module answers the question that
comes first: **is this a race where winners and placegetters can be found?**

It exists because of what a full-strength model test showed. Trained on 2,959
Racing & Sports races and scored on races it had never seen, a 117-feature
LightGBM was a *worse* ranker than the market price at every depth:

    finding the winner          top 1    top 2    top 3
    market price                0.370    0.534    0.664
    model (win-trained)         0.350    0.534    0.679
    model (place-trained)       0.324    0.522    0.667

So there is no ranking edge to be had. What there *is* — and it is large — is a
difference between races. Split by the favourite's price and the field size, the
favourite's place strike rate runs from 78% down to 50%. That spread is bigger
than anything the model could add to the ordering, and unlike the model it needs
no training data at all: the price and the field size are already on the page.

Adding the model's opinion as a fifth condition moved the place rate about a
point while discarding 82 of 566 races — inside the noise — so it is reported as
a note, never as a gate. The tiers are the market's own numbers.

**These are strike rates, not profit.** A 78% place strike into a market that
already prices it at 78% still loses the overround. This module tells you where
the outcome is predictable, which is what was asked for; it does not claim those
races are profitable.

Measured on the 1,480 most recent races (the back half of the data, none of it
used to choose the thresholds beyond the price/field split itself), and checked
across four successive time folds — the tier order held in 3 or 4 of 4 folds on
both win and place.
"""
from __future__ import annotations

from typing import Any

import numpy as np

PRIME, STRONG, FAIR, SKIP = "PRIME", "STRONG", "FAIR", "SKIP"
TIER_ORDER = [PRIME, STRONG, FAIR, SKIP]

# Favourite's price thresholds. The dominant signal by a wide margin: the
# favourite wins 54.7% of races when it is under $2.50 and 22.6% at $4 or more.
PRIME_MAX_ODDS = 2.50
PLAYABLE_MAX_ODDS = 4.00
# Field band that reads best. Not a difficulty measure on its own - fields under
# 8 pay only two places, so their lower place rate is arithmetic, not a warning.
GOOD_FIELD = (8, 10)

# Measured on 1,480 held-out races. `fav_win` / `fav_place` are the market
# favourite's strike rates; `top3_*` describe a three-runner shortlist taken
# straight down the market.
TIER_STATS: dict[str, dict[str, Any]] = {
    PRIME: dict(races=466, coverage=0.315, fav_win=0.547, fav_place=0.783,
                fav_win_ci=(0.50, 0.59), fav_place_ci=(0.74, 0.82),
                top3_has_winner=0.843, top3_any_place=0.985,
                top3_two_place=0.622, top3_avg_placed=1.71,
                avg_field=8.2, avg_places=2.53,
                label="Prime", blurb="short, dominant favourite"),
    STRONG: dict(races=310, coverage=0.209, fav_win=0.377, fav_place=0.745,
                 fav_win_ci=(0.33, 0.43), fav_place_ci=(0.69, 0.79),
                 top3_has_winner=0.687, top3_any_place=0.965,
                 top3_two_place=0.645, top3_avg_placed=1.71,
                 avg_field=9.1, avg_places=3.00,
                 label="Strong", blurb="clear favourite in a three-place field"),
    FAIR: dict(races=386, coverage=0.261, fav_win=0.313, fav_place=0.598,
               fav_win_ci=(0.27, 0.36), fav_place_ci=(0.55, 0.65),
               top3_has_winner=0.668, top3_any_place=0.927,
               top3_two_place=0.482, top3_avg_placed=1.44,
               avg_field=10.5, avg_places=2.67,
               label="Fair", blurb="clear favourite, but an awkward field size"),
    SKIP: dict(races=318, coverage=0.215, fav_win=0.226, fav_place=0.500,
               fav_win_ci=(0.18, 0.28), fav_place_ci=(0.45, 0.55),
               top3_has_winner=0.506, top3_any_place=0.862,
               top3_two_place=0.327, top3_avg_placed=1.21,
               avg_field=11.7, avg_places=2.92,
               label="Skip", blurb="no favourite the market is confident in"),
}

# Place strike rate by market rank within each tier. This is what a shortlist
# actually returns: the second and third picks are worth far less than the first
# everywhere, and in a SKIP race the third pick places less than a third of the
# time.
RANK_STATS: dict[str, dict[int, dict[str, float]]] = {
    PRIME:  {1: dict(win=0.547, place=0.783, avg_odds=1.97),
             2: dict(win=0.206, place=0.539, avg_odds=4.99),
             3: dict(win=0.090, place=0.393, avg_odds=8.34)},
    STRONG: {1: dict(win=0.377, place=0.745, avg_odds=3.13),
             2: dict(win=0.184, place=0.503, avg_odds=4.28),
             3: dict(win=0.126, place=0.465, avg_odds=6.03)},
    FAIR:   {1: dict(win=0.313, place=0.598, avg_odds=3.14),
             2: dict(win=0.207, place=0.464, avg_odds=4.62),
             3: dict(win=0.148, place=0.381, avg_odds=6.50)},
    SKIP:   {1: dict(win=0.226, place=0.500, avg_odds=4.72),
             2: dict(win=0.160, place=0.393, avg_odds=5.60),
             3: dict(win=0.119, place=0.314, avg_odds=6.83)},
}

BASELINE = dict(fav_win=0.382, fav_place=0.666, top3_has_winner=0.693)

BADGE = {PRIME: "🟢", STRONG: "🔵", FAIR: "🟡", SKIP: "🔴"}


def best_price(runner: dict[str, Any]) -> float:
    """Shortest usable price for a runner, or inf when it has none.

    Betfair first, then the bookmaker column, then TAB. The 999 sentinel the
    parser writes for a missing price must never be read as a real quote - it
    would turn every unpriced field into a SKIP for the wrong reason.
    """
    for key in ("bf_odds", "book_odds", "tab_odds"):
        try:
            v = float(runner.get(key) or 0.0)
        except (TypeError, ValueError):
            continue
        if 1.0 < v < 900.0:
            return v
    return float("inf")


def favourite_price(active: list[dict[str, Any]]) -> float:
    """The shortest price in the race."""
    if not active:
        return float("inf")
    return min(best_price(r) for r in active)


def classify(fav_odds: float, n_runners: int) -> str:
    """Tier from the favourite's price and the field size."""
    if not np.isfinite(fav_odds) or fav_odds <= 1.0:
        return SKIP
    if fav_odds < PRIME_MAX_ODDS:
        return PRIME
    if fav_odds < PLAYABLE_MAX_ODDS:
        lo, hi = GOOD_FIELD
        return STRONG if lo <= n_runners <= hi else FAIR
    return SKIP


def assess(active: list[dict[str, Any]],
           result: dict[str, Any] | None = None) -> dict[str, Any]:
    """Classify a race and attach the measured rates for its tier.

    `result` is optional. When supplied, the model's opinion of the favourite is
    reported as a note - it is not part of the classification, because it did not
    survive validation as a gate.
    """
    n = len(active)
    fav_odds = favourite_price(active)
    tier = classify(fav_odds, n)
    stats = TIER_STATS[tier]

    out = {
        "tier": tier,
        "label": stats["label"],
        "badge": BADGE[tier],
        "blurb": stats["blurb"],
        "runners": n,
        "fav_odds": fav_odds,
        "fav_priced": bool(np.isfinite(fav_odds)),
        "stats": stats,
        "rank_stats": RANK_STATS[tier],
        "baseline": BASELINE,
        "model_agrees": None,
        "model_fav_rank": None,
    }

    if result is not None and n:
        # The model indexes whatever list it was given. `active` must be that
        # same list, so a length mismatch means the caller filtered scratchings
        # on one side only - report nothing rather than a rank that points at
        # the wrong horse.
        # `order` is a numpy array, so `arr or []` would raise "truth value is
        # ambiguous" - test it against None explicitly.
        raw = result.get("order")
        try:
            order = [] if raw is None else [int(i) for i in raw]
        except (TypeError, ValueError):
            order = []
        if len(order) == n:
            prices = [best_price(r) for r in active]
            fav_i = int(np.argmin(prices))
            if fav_i in order:
                rank = order.index(fav_i) + 1
                out["model_fav_rank"] = rank
                out["model_agrees"] = rank <= 2
    return out


def headline(a: dict[str, Any]) -> str:
    """One-line verdict for the top of the tab."""
    s = a["stats"]
    if not a["fav_priced"]:
        return ("⚪ **No prices parsed** — the tier is decided by the favourite's "
                "price, so this race cannot be graded. Paste a page that includes "
                "the market.")
    return (f"{a['badge']} **{a['label'].upper()}** — {a['blurb']}. "
            f"Favourite ${a['fav_odds']:.2f}, {a['runners']} runners. "
            f"In races like this the favourite won **{100*s['fav_win']:.0f}%** "
            f"and placed **{100*s['fav_place']:.0f}%** of the time.")


def detail(a: dict[str, Any]) -> str:
    """The supporting numbers, including what this tier costs you in coverage."""
    s, b = a["stats"], a["baseline"]
    bits = [
        f"Measured on **{s['races']} held-out races** ({100*s['coverage']:.0f}% "
        f"of all cards fall in this tier).",
        f"Favourite placed {100*s['fav_place']:.1f}% "
        f"(95% CI {100*s['fav_place_ci'][0]:.0f}–{100*s['fav_place_ci'][1]:.0f}%), "
        f"against **{100*b['fav_place']:.1f}% across all races**.",
        f"A three-deep shortlist held the winner {100*s['top3_has_winner']:.0f}% "
        f"of the time and returned {s['top3_avg_placed']:.2f} placegetters on "
        f"average, with at least two placing in "
        f"{100*s['top3_two_place']:.0f}% of races.",
    ]
    if a["tier"] == SKIP:
        bits.append("**This is the tier to leave alone.** Nothing about it is "
                    "unbettable — it is simply where the outcome is least "
                    "predictable, and skipping it is the single largest "
                    "improvement available.")
    if a["model_fav_rank"] is not None:
        verb = "agrees" if a["model_agrees"] else "disagrees"
        bits.append(f"_Note: the form model {verb} — it ranks the favourite "
                    f"#{a['model_fav_rank']}. This is shown for interest only; "
                    "as a filter it moved the place rate about a point while "
                    "discarding a sixth of the races, which the sample cannot "
                    "distinguish from noise._")
    return " ".join(bits)


def expected_rate(a: dict[str, Any], market_rank: int, kind: str = "place") -> float | None:
    """Measured strike rate for a given market rank in this tier.

    Returns None beyond the third market pick, where the sample thins out and a
    number would imply precision the data does not support.
    """
    r = a["rank_stats"].get(int(market_rank))
    return None if r is None else float(r[kind])


def summary_table(a: dict[str, Any]):
    """Per-market-rank expectations, as a small DataFrame for display."""
    import pandas as pd
    rows = []
    for rank, st in sorted(a["rank_stats"].items()):
        rows.append({
            "Market rank": rank,
            "Historic win%": 100 * st["win"],
            "Historic place%": 100 * st["place"],
            "Typical $": st["avg_odds"],
        })
    return pd.DataFrame(rows)
