"""Regression tests for the harness Place Finder.

Pins the rules and the invariants that must hold for any race: place probability
rises with the number of places paid, the shrink moves both tails toward the base
rate, the filters exclude what they claim to, and - the one most likely to bite -
the table stays index-aligned with the model when a race has scratchings.

    python test_place_finder.py     # expect: PASS <n>  FAIL 0
"""
import os

import numpy as np

import harness_model as hm
import harness_parser as hp
import place_finder as pf

_HERE = os.path.dirname(os.path.abspath(__file__))
FIXTURE = os.path.join(_HERE, "tests_fixture_gloucesterpark_r2.txt")


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
    res = hm.predict(runners, header)
    active = res["runners"]
    n = len(active)

    # ---- scratchings must not shift the table --------------------------
    c.check("model dropped the scratchings", n,
            len([r for r in runners if not r.get("scratched")]))
    c.true("fixture actually contains scratchings",
           any(r.get("scratched") for r in runners))
    # Passing the FULL field must still produce the active table, because
    # build() prefers result["runners"].
    t_full, m_full = pf.build(runners, res)
    t_act, m_act = pf.build(active, res)
    c.check("passing all runners still yields the active field", len(t_full), n)
    c.true("both callers give the same table",
           list(t_full["Tab"]) == list(t_act["Tab"]))
    c.true("no scratched runner appears in the table",
           not set(t_full["Tab"]) & {r["tab"] for r in runners if r.get("scratched")})
    top = t_full.iloc[0]
    c.check("top row is the model's own top pick", int(top["Tab"]),
            int(active[res["order"][0]]["tab"]))
    c.check("top row carries that runner's name", top[pf.RUNNER_LABEL],
            active[res["order"][0]]["horse"])

    # ---- place terms ----------------------------------------------------
    c.check("4 runners pay no places", pf.places_paid(4), 0)
    c.check("5 runners pay 2", pf.places_paid(5), 2)
    c.check("7 runners pay 2", pf.places_paid(7), 2)
    c.check("8 runners pay 3", pf.places_paid(8), 3)
    c.check("this 7-runner field pays 2", m_act["places"], 2)
    c.check("so the probability comes from Top2%", m_act["prob_source"], "Top2%")

    # ---- place probability ties back to the model -----------------------
    c.true("places=2 returns the model top2",
           np.allclose(pf.place_probability(res, 2), res["top2"]))
    c.true("places=3 returns the model top3",
           np.allclose(pf.place_probability(res, 3), res["top3"]))
    c.true("places=1 returns the win probability",
           np.allclose(pf.place_probability(res, 1), res["p_win"]))
    c.true("top2 never exceeds top3", bool(np.all(res["top2"] <= res["top3"] + 1e-9)))
    c.true("top2 sums to two places", abs(float(res["top2"].sum()) - 2.0) < 1e-6)
    c.true("top3 sums to three places", abs(float(res["top3"].sum()) - 3.0) < 1e-6)

    # ---- F/M is derived, since this model does not return it ------------
    fm = pf.fund_market_ratio(res)
    c.check("one ratio per runner", len(fm), n)
    c.true("ratio equals p_fund / p_mkt",
           np.allclose(fm, np.asarray(res["p_fund"]) / np.asarray(res["p_mkt"])))
    c.true("all ratios positive", bool(np.all(fm > 0)))

    # ---- market rank ----------------------------------------------------
    mr = pf.market_rank(active)
    c.true("ranks are a permutation of 1..n",
           sorted(mr.tolist()) == list(range(1, n + 1)))
    prices = [float(r.get("bf_odds") or 999.0) for r in active]
    c.check("rank 1 is the shortest price", int(np.argmin(prices)), int(np.argmin(mr)))
    unpriced = [dict(r) for r in active]
    for k in ("bf_odds", "book_odds", "tab_odds"):
        unpriced[0][k] = 999.0
    c.true("an unpriced runner sorts last", pf.market_rank(unpriced)[0] == n)

    # ---- shrink ---------------------------------------------------------
    raw = pf.place_probability(res, 2)
    adj = pf.shrink_to_base(raw, 2, n, 0.15)
    base = 2 / n
    c.true("shrink pulls the highest down", adj.max() < raw.max())
    c.true("shrink lifts the lowest up", adj.min() > raw.min())
    c.true("shrink never crosses the base rate",
           bool(np.all(np.sign(adj - base) == np.sign(raw - base))))
    c.true("shrink=0 is a no-op", np.allclose(pf.shrink_to_base(raw, 2, n, 0.0), raw))
    c.true("shrink=1 collapses to the base rate",
           np.allclose(pf.shrink_to_base(raw, 2, n, 1.0), base))

    # ---- table ----------------------------------------------------------
    c.check("one row per active runner", len(t_act), n)
    c.true("ranked best first", list(t_act["Rank"]) == sorted(t_act["Rank"]))
    c.true("fair price is the reciprocal of the adjusted probability",
           bool(np.allclose(t_act["Fair place $"],
                            100.0 / np.clip(t_act["Place% (adj)"], 1e-9, None))))
    c.true("place% falls as rank worsens",
           bool(np.all(np.diff(t_act["Place% (adj)"].values) <= 1e-9)))
    for col in (pf.RUNNER_LABEL, pf.DRAW_LABEL, pf.CONNECTION_LABEL, "Trainer"):
        c.true(f"column {col} present", col in t_act.columns)

    # ---- the F/M filter ships OFF for harness ---------------------------
    c.check("F/M default is off", pf.DEFAULT_FM_MAX, 99.0)
    c.check("so nothing is F/M-excluded by default", m_act["fm_excluded"], 0)
    tight, m_tight = pf.build(active, res, fm_max=2.0, market_top=n, max_picks=99)
    c.true("turning it on does exclude runners", m_tight["fm_excluded"] >= 1)
    c.true("every exclusion is above the threshold",
           bool((tight[tight["Status"] == pf.STATUS_FM]["F/M"] >= 2.0).all()))
    c.true("exclusions come from the shortlist, not the tail",
           bool((tight[tight["Status"] == pf.STATUS_FM]["Rank"] <= pf.DEFAULT_TOP_N).all()))

    # ---- consensus gate and cap -----------------------------------------
    c.true("default never exceeds the cap", m_act["qualifiers"] <= pf.DEFAULT_MAX_PICKS)
    sel = t_act[t_act["Status"] == pf.STATUS_QUALIFY]
    c.true("selections sit inside the model shortlist",
           bool((sel["Rank"] <= pf.DEFAULT_TOP_N).all()))
    c.true("selections sit inside the market top group",
           bool((sel["Mkt rank"] <= pf.DEFAULT_MARKET_TOP).all()))
    c.true("market exclusions really are outside it",
           bool((t_act[t_act["Status"] == pf.STATUS_MARKET]["Mkt rank"]
                 > pf.DEFAULT_MARKET_TOP).all()))
    for cap in (1, 2, 3):
        _, mc = pf.build(active, res, max_picks=cap)
        c.true(f"cap {cap} honoured", mc["qualifiers"] <= cap)
    _, mw = pf.build(active, res, market_top=n, max_picks=99)
    c.check("gate off restores the full shortlist",
            mw["qualifiers"] + mw["fm_excluded"], pf.DEFAULT_TOP_N)
    _, mcap = pf.build(active, res, market_top=n, max_picks=2)
    c.check("overflow becomes reserves, not exclusions",
            mcap["qualifiers"] + mcap["reserves"] + mcap["fm_excluded"],
            pf.DEFAULT_TOP_N)

    # ---- small fields ----------------------------------------------------
    _, m_tiny = pf.build(active[:4], res, places=0)
    c.true("under five runners flags no place market", m_tiny["no_place_market"])
    c.true("and says so", "do not usually pay" in pf.summary_line(m_tiny))

    # ---- optional place odds --------------------------------------------
    tab = int(t_act.iloc[0]["Tab"])
    p_adj = float(t_act.iloc[0]["Place% (adj)"]) / 100
    generous, _ = pf.build(active, res, place_odds={tab: 10.0})
    stingy, _ = pf.build(active, res, place_odds={tab: 1.01})
    c.true("a generous price shows positive edge",
           float(generous.iloc[0]["Place edge"]) > 0)
    c.true("a short price shows negative edge",
           float(stingy.iloc[0]["Place edge"]) < 0)
    c.true("edge equals p x price - 1",
           abs(float(generous.iloc[0]["Place edge"]) - (p_adj * 10.0 - 1)) < 1e-6)
    c.true("no odds columns when none supplied", "Place edge" not in t_act.columns)

    c.true("styler renders", pf.style(t_act).to_html() is not None)
    c.true("summary reports the selection count",
           f"{m_act['qualifiers']} selection(s)" in pf.summary_line(m_act))

    print(f"PASS {c.passes}  FAIL {len(c.fails)}")
    if c.fails:
        print("\n".join(c.fails))
    return 1 if c.fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
