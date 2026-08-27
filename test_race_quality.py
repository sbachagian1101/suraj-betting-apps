"""Regression tests for the race quality tiers.

Pins the boundaries (a race must not change tier because a price moved a cent
the wrong side of a comparison), the price-selection order, the missing-price
sentinel, and the internal consistency of the measured tables - a tier whose
numbers drift out of order would quietly mislead rather than fail.

    python test_race_quality.py     # expect: PASS <n>  FAIL 0
"""
import os

import numpy as np

import horse_model as hm
import horse_parser as hp
import race_quality as rq

_HERE = os.path.dirname(os.path.abspath(__file__))
FIXTURE = os.path.join(_HERE, "tests_fixture_tamworth_r4.txt")


class Checker:
    def __init__(self):
        self.passes = 0
        self.fails = []

    def check(self, label, got, want, ok=None):
        ok = (got == want) if ok is None else ok
        if ok:
            self.passes += 1
        else:
            self.fails.append(f"  {label}: got {got!r}, want {want!r}")

    def true(self, label, cond):
        self.check(label, cond, True, bool(cond))


def runner(**kw):
    d = dict(tab=1, horse="X", bf_odds=None, book_odds=None, tab_odds=None)
    d.update(kw)
    return d


def main():
    c = Checker()

    # ---- classification boundaries --------------------------------------
    c.check("$2.49 in a small field is PRIME", rq.classify(2.49, 8), rq.PRIME)
    c.check("$2.50 is no longer PRIME", rq.classify(2.50, 8), rq.STRONG)
    c.check("PRIME ignores field size (big)", rq.classify(2.00, 16), rq.PRIME)
    c.check("PRIME ignores field size (small)", rq.classify(2.00, 5), rq.PRIME)
    c.check("$3.99 with 8 runners is STRONG", rq.classify(3.99, 8), rq.STRONG)
    c.check("$3.99 with 10 runners is STRONG", rq.classify(3.99, 10), rq.STRONG)
    c.check("$3.99 with 7 runners is FAIR", rq.classify(3.99, 7), rq.FAIR)
    c.check("$3.99 with 11 runners is FAIR", rq.classify(3.99, 11), rq.FAIR)
    c.check("$4.00 is SKIP", rq.classify(4.00, 9), rq.SKIP)
    c.check("a long favourite is SKIP", rq.classify(9.0, 9), rq.SKIP)
    c.check("no price at all is SKIP", rq.classify(float("inf"), 9), rq.SKIP)
    c.check("a nonsense price is SKIP", rq.classify(1.0, 9), rq.SKIP)
    c.check("every tier is a known tier",
            {rq.classify(o, n) for o in (1.5, 2.5, 3.5, 5.0) for n in (6, 9, 14)}
            <= set(rq.TIER_ORDER), True)

    # ---- price selection -------------------------------------------------
    c.check("Betfair preferred", rq.best_price(
        runner(bf_odds=2.0, book_odds=3.0, tab_odds=4.0)), 2.0)
    c.check("falls back to the book", rq.best_price(
        runner(book_odds=3.0, tab_odds=4.0)), 3.0)
    c.check("falls back to TAB", rq.best_price(runner(tab_odds=4.0)), 4.0)
    c.true("the 999 sentinel is not a price",
           not np.isfinite(rq.best_price(runner(bf_odds=999.0))))
    c.check("999 on Betfair still finds TAB",
            rq.best_price(runner(bf_odds=999.0, tab_odds=3.5)), 3.5)
    c.true("no price at all is infinite",
           not np.isfinite(rq.best_price(runner())))
    c.true("a junk price is ignored",
           not np.isfinite(rq.best_price(runner(bf_odds="n/a"))))
    c.true("a price of 1.0 is ignored",
           not np.isfinite(rq.best_price(runner(bf_odds=1.0))))

    field = [runner(tab=1, bf_odds=5.0), runner(tab=2, bf_odds=2.2),
             runner(tab=3, tab_odds=3.0), runner(tab=4)]
    c.check("favourite is the shortest in the race", rq.favourite_price(field), 2.2)
    c.check("an empty field has no favourite",
            rq.favourite_price([]), float("inf"))

    # ---- assess ----------------------------------------------------------
    a = rq.assess(field)
    c.check("that race is PRIME", a["tier"], rq.PRIME)
    c.check("runner count carried through", a["runners"], 4)
    c.check("favourite price carried through", a["fav_odds"], 2.2)
    c.true("flagged as priced", a["fav_priced"])
    c.true("headline names the tier", "PRIME" in rq.headline(a))
    c.true("headline quotes the price", "$2.20" in rq.headline(a))
    c.true("detail cites the sample size",
           str(rq.TIER_STATS[rq.PRIME]["races"]) in rq.detail(a))
    c.true("detail compares against the all-races baseline",
           "across all races" in rq.detail(a))

    unpriced = rq.assess([runner(tab=1), runner(tab=2)])
    c.check("an unpriced race is SKIP", unpriced["tier"], rq.SKIP)
    c.true("and says the prices are missing",
           "No prices parsed" in rq.headline(unpriced))
    c.true("an empty field does not crash", rq.assess([])["runners"] == 0)

    skip = rq.assess([runner(tab=1, bf_odds=6.0), runner(tab=2, bf_odds=7.0)])
    c.true("the SKIP tier says to leave it alone",
           "leave alone" in rq.detail(skip))

    # ---- the model note is a note, never a gate --------------------------
    # The app filters scratchings, then predicts on that list - so the model's
    # indices refer to `active`. Any test that predicts on the full field would
    # be testing a calling convention the app does not use.
    header, runners, _ = hp.parse(open(FIXTURE, encoding="utf-8").read())
    active = [r for r in runners if not r.get("scratched")]
    result = hm.predict(active, header)
    with_model = rq.assess(active, result)
    without = rq.assess(active)
    c.check("the model does not change the tier",
            with_model["tier"], without["tier"])
    c.true("without a result there is no model note",
           without["model_fav_rank"] is None)
    c.true("with a result the model rank is reported",
           with_model["model_fav_rank"] is not None)
    c.true("the model rank is a real rank",
           1 <= with_model["model_fav_rank"] <= len(active))
    c.check("agreement means top two", with_model["model_agrees"],
            with_model["model_fav_rank"] <= 2)
    c.true("the note is flagged as interest only",
           "for interest only" in rq.detail(with_model))
    # A caller that filters on one side only must get silence, not a wrong rank.
    c.true("fixture really has scratchings",
           any(r.get("scratched") for r in runners))
    mismatched = rq.assess(runners, result)
    c.true("a length mismatch suppresses the model note",
           mismatched["model_fav_rank"] is None)
    c.true("but the tier is still returned",
           mismatched["tier"] in rq.TIER_ORDER)
    c.true("real race classifies to a known tier",
           with_model["tier"] in rq.TIER_ORDER)

    # ---- expected rates --------------------------------------------------
    c.check("rank 1 place rate for PRIME",
            rq.expected_rate(a, 1), rq.RANK_STATS[rq.PRIME][1]["place"])
    c.check("win rate is available too",
            rq.expected_rate(a, 1, "win"), rq.RANK_STATS[rq.PRIME][1]["win"])
    c.true("beyond the third pick returns nothing",
           rq.expected_rate(a, 4) is None)
    c.true("rank 0 returns nothing", rq.expected_rate(a, 0) is None)

    t = rq.summary_table(a)
    c.check("summary has one row per ranked pick", len(t), 3)
    c.true("summary is ordered by market rank",
           list(t["Market rank"]) == [1, 2, 3])

    # ---- the measured tables must stay internally consistent -------------
    for t1, t2 in zip(rq.TIER_ORDER, rq.TIER_ORDER[1:]):
        s1, s2 = rq.TIER_STATS[t1], rq.TIER_STATS[t2]
        c.true(f"{t1} finds more winners than {t2}", s1["fav_win"] > s2["fav_win"])
        c.true(f"{t1} shortlist holds the winner more than {t2}",
               s1["top3_has_winner"] > s2["top3_has_winner"])
    c.true("PRIME places best", rq.TIER_STATS[rq.PRIME]["fav_place"]
           == max(s["fav_place"] for s in rq.TIER_STATS.values()))
    c.true("SKIP places worst", rq.TIER_STATS[rq.SKIP]["fav_place"]
           == min(s["fav_place"] for s in rq.TIER_STATS.values()))
    c.true("coverage sums to one",
           abs(sum(s["coverage"] for s in rq.TIER_STATS.values()) - 1.0) < 0.01)
    c.true("race counts sum to the stated sample",
           sum(s["races"] for s in rq.TIER_STATS.values()) == 1480)

    for tier, ranks in rq.RANK_STATS.items():
        vals = [ranks[r]["place"] for r in sorted(ranks)]
        c.true(f"{tier}: place rate falls down the market",
               vals == sorted(vals, reverse=True))
        wins = [ranks[r]["win"] for r in sorted(ranks)]
        c.true(f"{tier}: win rate falls down the market",
               wins == sorted(wins, reverse=True))
        odds = [ranks[r]["avg_odds"] for r in sorted(ranks)]
        c.true(f"{tier}: prices lengthen down the market",
               odds == sorted(odds))
        for r, st in ranks.items():
            c.true(f"{tier} rank {r}: placing is at least as likely as winning",
                   st["place"] >= st["win"])
    for tier, s in rq.TIER_STATS.items():
        c.true(f"{tier}: favourite places at least as often as it wins",
               s["fav_place"] >= s["fav_win"])
        c.true(f"{tier}: confidence interval brackets the estimate",
               s["fav_place_ci"][0] <= s["fav_place"] <= s["fav_place_ci"][1])
        c.true(f"{tier}: at-least-one beats at-least-two",
               s["top3_any_place"] > s["top3_two_place"])
        c.true(f"{tier}: average placed is within the three named",
               0 < s["top3_avg_placed"] <= 3)
    c.true("every tier has a badge", set(rq.BADGE) == set(rq.TIER_ORDER))
    c.true("every tier has rank stats", set(rq.RANK_STATS) == set(rq.TIER_ORDER))
    c.true("the baseline sits between the extremes",
           rq.TIER_STATS[rq.SKIP]["fav_place"] < rq.BASELINE["fav_place"]
           < rq.TIER_STATS[rq.PRIME]["fav_place"])

    print(f"PASS {c.passes}  FAIL {len(c.fails)}")
    if c.fails:
        print("\n".join(c.fails))
    return 1 if c.fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
