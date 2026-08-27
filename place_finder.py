"""Placegetter shortlist for HarnessPredict.

Ported from the thoroughbred app, where the criteria were measured against a real
Wolverhampton card (six races, 16 placegetters). What did and did not travel:

**Kept, because the reasoning is structural rather than code-specific**

* *Require the market to agree.* Taking the model's top 5 and keeping only those
  the market also rates in its top 3 was the single biggest lift there -
  precision 33.3% -> 47.1% at the same ~3 selections per race. Betting markets
  being hard to beat is not a thoroughbred peculiarity, so the gate is on by
  default here too.
* *Cap the selections.* Coverage is not the goal; a shortlist you can bet is.
* *Use the right number of places.* A top-3 probability overstates the real place
  chance when only two places pay, so the module switches to `Top2%` itself.
* *Shrink toward the base rate.* Pulling toward `places / runners` corrects
  over-confidence at the top and under-confidence at the bottom in one move, with
  no curve fitted to a small sample. Principled for any code.

**Dropped by default, because the number came from thoroughbreds**

* *The F/M >= 2.0 exclusion.* On that card, horses the form model rated at twice
  the market's opinion placed 1 time in 20. There is **no harness evidence for
  that threshold**, so it ships switched off. The slider is there if you want to
  experiment, and the column is shown either way.

Nothing here is a proven edge for harness racing. It is a sensible starting
method transplanted from a neighbouring code, and it says so in the app.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

DEFAULT_TOP_N = 5          # model shortlist the consensus is drawn from
DEFAULT_MARKET_TOP = 3     # must also be this well rated by the market
DEFAULT_MAX_PICKS = 3      # hard cap on final selections
DEFAULT_FM_MAX = 99.0      # OFF: the 2.0 figure is thoroughbred-derived
DEFAULT_SHRINK = 0.15

STATUS_QUALIFY = "✅ SELECTION"
STATUS_FM = "⚠️ F/M filter"
STATUS_MARKET = "⚠️ market disagrees"
STATUS_RESERVE = "➖ reserve (over cap)"
STATUS_OUT = "— outside model top N"

# Column labels for this code.
RUNNER_LABEL = "Horse"
DRAW_LABEL = "Gate"
CONNECTION_LABEL = "Driver"


def places_paid(n_runners: int) -> int:
    """Standard place terms: 3 places for 8+, 2 for 5-7, none under 5."""
    if n_runners >= 8:
        return 3
    if n_runners >= 5:
        return 2
    return 0


def market_rank(active: list[dict[str, Any]]) -> np.ndarray:
    """Rank runners by market price, shortest first (1 = market favourite).

    Betfair is preferred and TAB is the fallback. Missing prices (the 999
    sentinel) sort last, so an unpriced runner never slips into the market's top
    group and quietly becomes a selection.
    """
    prices = []
    for r in active:
        for key in ("bf_odds", "book_odds", "tab_odds"):
            v = float(r.get(key) or 999.0)
            if 1.0 < v < 900:
                prices.append(v)
                break
        else:
            prices.append(9999.0)
    order = np.argsort(np.asarray(prices, dtype=float), kind="stable")
    ranks = np.empty(len(active), dtype=int)
    ranks[order] = np.arange(1, len(active) + 1)
    return ranks


def fund_market_ratio(result: dict[str, Any]) -> np.ndarray:
    """Fundamental probability divided by market probability.

    The harness model does not return this directly, so it is derived. Values
    well above 1 mean the form model rates the horse far higher than the market.
    """
    if "fund_mkt_ratio" in result:
        return np.asarray(result["fund_mkt_ratio"], dtype=float)
    pf = np.asarray(result["p_fund"], dtype=float)
    pm = np.clip(np.asarray(result["p_mkt"], dtype=float), 1e-9, None)
    return pf / pm


def place_probability(result: dict[str, Any], places: int) -> np.ndarray:
    """Probability of finishing inside the paying places, per runner."""
    if places <= 1:
        return np.asarray(result["p_win"], dtype=float)
    if places == 2:
        return np.asarray(result["top2"], dtype=float)
    return np.asarray(result["top3"], dtype=float)


def shrink_to_base(p: np.ndarray, places: int, n_runners: int,
                   k: float = DEFAULT_SHRINK) -> np.ndarray:
    """Pull probabilities toward the field's base place rate.

    base = places / runners is what a dart throw would score. Shrinking toward it
    lifts the under-confident tail and trims the over-confident head in one
    principled move, with no curve fitted to a small sample.
    """
    p = np.asarray(p, dtype=float)
    if n_runners <= 0 or places <= 0:
        return p
    base = places / n_runners
    k = float(np.clip(k, 0.0, 1.0))
    return p + k * (base - p)


def build(active: list[dict[str, Any]], result: dict[str, Any], *,
          top_n: int = DEFAULT_TOP_N, market_top: int = DEFAULT_MARKET_TOP,
          max_picks: int = DEFAULT_MAX_PICKS, fm_max: float = DEFAULT_FM_MAX,
          shrink: float = DEFAULT_SHRINK, places: int | None = None,
          place_odds: dict[int, float] | None = None) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Build the place table, ranked by the model, with a qualify/exclude status.

    `result["order"]` indexes the model's own active list, and `predict` filters
    scratchings itself - so that list is preferred over whatever the caller
    passed, otherwise a scratching would shift every row by one.
    """
    active = list(result.get("runners") or active)
    n = len(active)
    if places is None:
        places = places_paid(n)
    raw = place_probability(result, max(places, 1))
    adj = shrink_to_base(raw, places, n, shrink)
    fm = fund_market_ratio(result)
    mrank = market_rank(active)
    order = list(result["order"])          # model ranking, best first

    rows = []
    taken = 0
    for rank, i in enumerate(order, start=1):
        r = active[i]
        if rank > top_n:
            status = STATUS_OUT
        elif fm[i] >= fm_max:
            status = STATUS_FM
        elif mrank[i] > market_top:
            status = STATUS_MARKET
        elif taken >= max_picks:
            status = STATUS_RESERVE
        else:
            status = STATUS_QUALIFY
            taken += 1
        p = float(adj[i])
        row = {
            "Rank": rank,
            "Tab": r.get("tab"),
            RUNNER_LABEL: r.get("horse"),
            DRAW_LABEL: r.get("gate") or r.get("box") or None,
            CONNECTION_LABEL: r.get("driver", ""),
            "Trainer": r.get("trainer", ""),
            "Win%": 100 * float(result["p_win"][i]),
            "Place% (raw)": 100 * float(raw[i]),
            "Place% (adj)": 100 * p,
            "Fair place $": 1.0 / max(p, 1e-9),
            "F/M": float(fm[i]),
            "Mkt rank": int(mrank[i]),
            "Conf": int(result["conf"][i]),
            "Status": status,
        }
        if place_odds:
            o = place_odds.get(int(r.get("tab") or -1))
            if o and o > 1.0:
                row["Your place $"] = float(o)
                row["Place edge"] = p * float(o) - 1.0
        rows.append(row)

    df = pd.DataFrame(rows)
    meta = {
        "runners": n,
        "places": places,
        "base_rate": (places / n) if n else 0.0,
        "top_n": top_n,
        "market_top": market_top,
        "max_picks": max_picks,
        "fm_max": fm_max,
        "shrink": shrink,
        "qualifiers": int((df["Status"] == STATUS_QUALIFY).sum()),
        "fm_excluded": int((df["Status"] == STATUS_FM).sum()),
        "market_excluded": int((df["Status"] == STATUS_MARKET).sum()),
        "reserves": int((df["Status"] == STATUS_RESERVE).sum()),
        "prob_source": "Top2%" if places == 2 else ("Top3%" if places == 3 else "Win%"),
        "no_place_market": places == 0,
    }
    return df, meta


def style(df: pd.DataFrame):
    """Green for selections, blue for reserves, amber for exclusions."""
    def row_colour(row):
        if row["Status"] == STATUS_QUALIFY:
            bg = "rgba(46, 204, 113, 0.24)"
        elif row["Status"] == STATUS_RESERVE:
            bg = "rgba(52, 152, 219, 0.14)"
        elif row["Status"] in (STATUS_FM, STATUS_MARKET):
            bg = "rgba(241, 196, 15, 0.16)"
        else:
            bg = "rgba(128, 128, 128, 0.05)"
        return [f"background-color: {bg}"] * len(row)

    styler = df.style.apply(row_colour, axis=1)
    fmt = {c: "{:.1f}%" for c in ("Win%", "Place% (raw)", "Place% (adj)") if c in df.columns}
    fmt.update({c: "${:.2f}" for c in ("Fair place $", "Your place $") if c in df.columns})
    fmt["F/M"] = "{:.2f}x"
    if "Place edge" in df.columns:
        fmt["Place edge"] = "{:+.3f}"
    return styler.format(fmt)


def summary_line(meta: dict[str, Any]) -> str:
    if meta["no_place_market"]:
        return (f"Only {meta['runners']} runners — bookmakers do not usually pay a place "
                "dividend under five. Treat this as a win-only race.")
    bits = [f"{meta['runners']} runners → **{meta['places']} places paid**, so the "
            f"place probability is taken from **{meta['prob_source']}**.",
            f"**{meta['qualifiers']} selection(s)** (cap {meta['max_picks']})."]
    drops = []
    if meta["market_excluded"]:
        drops.append(f"{meta['market_excluded']} dropped because the market does not rate "
                     f"them inside its top {meta['market_top']}")
    if meta["fm_excluded"]:
        drops.append(f"{meta['fm_excluded']} dropped by the F/M ≥ "
                     f"{meta['fm_max']:.1f}× filter")
    if meta["reserves"]:
        drops.append(f"{meta['reserves']} held back as reserve(s) by the cap")
    if drops:
        bits.append(f"From the model's top {meta['top_n']}: " + "; ".join(drops) + ".")
    bits.append(f"A dart throw places {100 * meta['base_rate']:.1f}% of the time.")
    return " ".join(bits)
