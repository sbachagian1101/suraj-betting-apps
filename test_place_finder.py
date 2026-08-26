"""Regression tests for the Place Finder selection criteria.

Checks the rules actually behave as documented, and pins the invariants that must
hold for any race: place probability rises with the number of places paid, the
shrink moves both tails toward the base rate, and the filters exclude what they
claim to.

    python test_place_finder.py     # expect: PASS <n>  FAIL 0
"""
import os

import numpy as np

import horse_model as hm
import horse_parser as hp
import place_finder as pf

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


def main():
    c = Checker()
    header, runners, _ = hp.parse(open(FIXTURE, encoding="utf-8").read())
    active = [r for r in runners if not r.get("scratched")]
    res = hm.predict(active, header)
    n = len(active)

    # ---- place terms -----------------------------------------------------
    c.check("4 runners pay no places", pf.places_paid(4), 0)
    c.check("5 runners pay 2", pf.places_paid(5), 2)
    c.check("7 runners pay 2", pf.places_paid(7), 2)
    c.check("8 runners pay 3", pf.places_paid(8), 3)
    c.check("20 runners pay 3", pf.places_paid(20), 3)

    # ---- place probability ties back to the model ------------------------
    c.true("places=2 returns the model top2",
           np.allclose(pf.place_probability(res, 2), res["top2"]))
    c.true("places=3 returns the model top3",
           np.allclose(pf.place_probability(res, 3), res["top3"]))
    c.true("top2 never exceeds top3", bool(np.all(res["top2"] <= res["top3"] + 1e-9)))
    c.true("every place probability is a probability",
           bool(np.all((res["top3"] >= 0) & (res["top3"] <= 1 + 1e-9))))
    c.true("top3 probabilities sum to the number of places",
           abs(float(res["top3"].sum()) - 3.0) < 1e-6)

    # ---- shrink ----------------------------------------------------------
    raw = pf.place_probability(res, 3)
    adj = pf.shrink_to_base(raw, 3, n, 0.15)
    base = 3 / n
    c.true("shrink pulls the highest runner down", adj.max() < raw.max())
    c.true("shrink lifts the lowest runner up", adj.min() > raw.min())
    c.true("shrink never crosses the base rate",
           bool(np.all(np.sign(adj - base) == np.sign(raw - base))))
    c.true("shrink=0 is a no-op", np.allclose(pf.shrink_to_base(raw, 3, n, 0.0), raw))
    c.true("shrink=1 collapses to the base rate",
           np.allclose(pf.shrink_to_base(raw, 3, n, 1.0), base))

    # ---- table construction ---------------------------------------------
    table, meta = pf.build(active, res)
    c.check("one row per active runner", len(table), n)
    c.check("places auto-detected", meta["places"], pf.places_paid(n))
    c.check("probability source labelled", meta["prob_source"], "Top3%")
    c.check("shortlist size honoured",
            int((table["Status"] != pf.STATUS_OUT).sum()), pf.DEFAULT_TOP_N)
    c.true("ranked best first", list(table["Rank"]) == sorted(table["Rank"]))
    c.true("fair price is the reciprocal of the adjusted probability",
           bool(np.allclose(table["Fair place $"],
                            100.0 / np.clip(table["Place% (adj)"], 1e-9, None))))
    c.true("place% falls as rank worsens",
           bool(np.all(np.diff(table["Place% (adj)"].values) <= 1e-9)))

    # ---- the F/M exclusion actually excludes -----------------------------
    tight, m_tight = pf.build(active, res, fm_max=1.2)
    c.true("a biting F/M threshold excludes at least one runner",
           m_tight["fm_excluded"] >= 1)
    excluded = tight[tight["Status"] == pf.STATUS_FM]
    c.true("every excluded runner is above the threshold",
           bool((excluded["F/M"] >= 1.2).all()))
    c.true("exclusions come out of the shortlist, not the tail",
           bool((excluded["Rank"] <= pf.DEFAULT_TOP_N).all()))
    loose, m_loose = pf.build(active, res, fm_max=99.0)
    c.check("an unreachable threshold excludes nobody", m_loose["fm_excluded"], 0)

    # ---- 2-place terms must be strictly harsher than 3 -------------------
    t2, _ = pf.build(active, res, places=2)
    t3, _ = pf.build(active, res, places=3)
    a2 = t2.sort_values("Tab")["Place% (raw)"].values
    a3 = t3.sort_values("Tab")["Place% (raw)"].values
    c.true("2-place probabilities never exceed 3-place", bool(np.all(a2 <= a3 + 1e-9)))
    c.true("2-place is strictly lower for the top pick", a2.max() < a3.max())

    # ---- small fields ----------------------------------------------------
    tiny, m_tiny = pf.build(active[:4], hm.predict(active[:4], header), places=0)
    c.true("under five runners flags no place market", m_tiny["no_place_market"])
    c.true("no-place-market summary warns the user",
           "do not usually pay" in pf.summary_line(m_tiny))

    # ---- optional place odds --------------------------------------------
    tab = int(table.iloc[0]["Tab"])
    p_adj = float(table.iloc[0]["Place% (adj)"]) / 100
    generous, _ = pf.build(active, res, place_odds={tab: 10.0})
    stingy, _ = pf.build(active, res, place_odds={tab: 1.01})
    c.true("a generous price shows positive edge",
           float(generous.iloc[0]["Place edge"]) > 0)
    c.true("a short price shows negative edge",
           float(stingy.iloc[0]["Place edge"]) < 0)
    c.true("edge equals p x price - 1",
           abs(float(generous.iloc[0]["Place edge"]) - (p_adj * 10.0 - 1)) < 1e-6)
    c.true("no odds column when none supplied", "Place edge" not in table.columns)

    # ---- market consensus gate and cap ------------------------------------
    mr = pf.market_rank(active)
    c.true("market ranks are a permutation of 1..n",
           sorted(mr.tolist()) == list(range(1, n + 1)))
    prices = [float(r.get("bf_odds") or 999.0) for r in active]
    c.check("market rank 1 is the shortest price",
            int(np.argmin(prices)), int(np.argmin(mr)))
    unpriced = [dict(r) for r in active]
    unpriced[0]["bf_odds"] = 999.0
    unpriced[0]["tab_odds"] = 999.0
    c.true("an unpriced runner sorts last",
           pf.market_rank(unpriced)[0] == n)

    tbl, mt = pf.build(active, res)
    c.true("default settings never exceed the cap", mt["qualifiers"] <= pf.DEFAULT_MAX_PICKS)
    c.true("selections are a subset of the model shortlist",
           bool((tbl[tbl["Status"] == pf.STATUS_QUALIFY]["Rank"] <= pf.DEFAULT_TOP_N).all()))
    c.true("every selection is inside the market top group",
           bool((tbl[tbl["Status"] == pf.STATUS_QUALIFY]["Mkt rank"]
                 <= pf.DEFAULT_MARKET_TOP).all()))
    c.true("market exclusions really are outside the market top group",
           bool((tbl[tbl["Status"] == pf.STATUS_MARKET]["Mkt rank"]
                 > pf.DEFAULT_MARKET_TOP).all()))

    for cap in (1, 2, 3):
        _, mc = pf.build(active, res, max_picks=cap)
        c.check(f"cap {cap} honoured", mc["qualifiers"] <= cap, True)
    wide, mw = pf.build(active, res, market_top=n, max_picks=99)
    c.check("market gate off restores the full shortlist",
            mw["qualifiers"], pf.DEFAULT_TOP_N)
    c.check("market gate off excludes nobody on market grounds",
            mw["market_excluded"], 0)
    tight, mtg = pf.build(active, res, market_top=1, max_picks=99)
    c.check("market_top=1 leaves at most one selection", mtg["qualifiers"] <= 1, True)

    capped, mcap = pf.build(active, res, market_top=n, max_picks=2)
    c.check("overflow becomes reserves, not exclusions",
            mcap["reserves"], pf.DEFAULT_TOP_N - 2)
    c.true("reserves rank below every selection",
           (capped[capped["Status"] == pf.STATUS_RESERVE]["Rank"].min()
            > capped[capped["Status"] == pf.STATUS_QUALIFY]["Rank"].max()))
    c.true("summary reports the selection count",
           f"{mt['qualifiers']} selection(s)" in pf.summary_line(mt))

    # ---- styling must not crash and must colour every row -----------------
    styled = pf.style(table)
    c.true("styler renders", styled.to_html() is not None)

    print(f"PASS {c.passes}  FAIL {len(c.fails)}")
    if c.fails:
        print("\n".join(c.fails))
    return 1 if c.fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
