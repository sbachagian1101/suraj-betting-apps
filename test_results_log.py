"""Regression tests for the results ledger and threshold tuner.

The important one is `selections_for_race` agreeing with `place_finder.build`.
The ledger re-derives selections from stored columns so old races can be
re-scored under new thresholds; if those two ever drift apart, every historic
number silently becomes wrong.

    python test_results_log.py     # expect: PASS <n>  FAIL 0
"""
import os

import numpy as np
import pandas as pd

import horse_model as hm
import horse_parser as hp
import place_finder as pf
import results_log as rl

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


def synthetic(n_races=45, seed=7):
    """A ledger of plausible races so the tuner can be exercised end to end."""
    rng = np.random.default_rng(seed)
    rows = []
    for r in range(n_races):
        n = int(rng.integers(7, 13))
        places = rl_places(n)
        strength = np.sort(rng.random(n))[::-1]
        strength = strength / strength.sum()
        order = rng.choice(n, size=n, replace=False, p=strength)
        rid = f"Synthetic_R{r}"
        for rank, i in enumerate(np.argsort(-strength), start=1):
            tab = i + 1
            fin = int(np.where(order == i)[0][0]) + 1
            rows.append({
                "race_id": rid, "date": "2026-01-01", "track": "Synthetic",
                "race_no": r, "field_size": n, "places_paid": places,
                "tab": tab, "horse": f"H{tab}", "model_rank": rank,
                "win_pct": 100 * strength[i],
                "place_pct_raw": min(99.0, 300 * strength[i]),
                "place_pct_adj": min(99.0, 300 * strength[i]),
                "fm": float(rng.uniform(0.2, 3.0)),
                "mkt_rank": int(rng.permutation(n)[rank - 1] + 1),
                "bf_odds": float(1 / max(strength[i], 0.01)),
                "tab_odds": float(1 / max(strength[i], 0.01)),
                "conf": 3, "status_at_prediction": "",
                "finish_pos": fin, "placed": int(fin <= places),
            })
    led = pd.DataFrame(rows, columns=rl.LEDGER_COLUMNS)
    # Mark what the default rule would have selected, as the app does.
    for rid, d in led.groupby("race_id"):
        picks = rl.selections_for_race(d, pf.DEFAULT_TOP_N, pf.DEFAULT_MARKET_TOP,
                                       pf.DEFAULT_FM_MAX, pf.DEFAULT_MAX_PICKS)
        led.loc[(led.race_id == rid) & (led.tab.isin(picks)),
                "status_at_prediction"] = pf.STATUS_QUALIFY
    return led


def rl_places(n):
    return 3 if n >= 8 else 2


def main():
    c = Checker()
    header, runners, _ = hp.parse(open(FIXTURE, encoding="utf-8").read())
    active = [r for r in runners if not r.get("scratched")]
    res = hm.predict(active, header)
    table, meta = pf.build(active, res)

    # ---- snapshot --------------------------------------------------------
    snap = rl.snapshot(header, active, res, table, meta)
    c.check("snapshot has the ledger schema", list(snap.columns), rl.LEDGER_COLUMNS)
    c.check("one row per active runner", len(snap), len(active))
    c.check("one race id", snap["race_id"].nunique(), 1)
    c.true("results start empty", snap["placed"].isna().all())
    c.true("race id carries the track", "Tamworth" in snap["race_id"].iloc[0])
    c.check("field size recorded", int(snap["field_size"].iloc[0]), len(active))
    c.check("places recorded", int(snap["places_paid"].iloc[0]), meta["places"])

    # ---- the critical consistency check ----------------------------------
    live = sorted(int(t) for t in table[table["Status"] == pf.STATUS_QUALIFY]["Tab"])
    rederived = sorted(rl.selections_for_race(
        snap, top_n=pf.DEFAULT_TOP_N, market_top=pf.DEFAULT_MARKET_TOP,
        fm_max=pf.DEFAULT_FM_MAX, max_picks=pf.DEFAULT_MAX_PICKS))
    c.check("ledger re-derives the app's selections exactly", rederived, live)
    for tn, mt, fm, mp in [(3, 2, 1.5, 2), (6, 4, 3.0, 3), (5, 99, 99.0, 5)]:
        a = sorted(int(t) for t in pf.build(
            active, res, top_n=tn, market_top=mt, fm_max=fm, max_picks=mp
        )[0].query("Status == @pf.STATUS_QUALIFY")["Tab"])
        b = sorted(rl.selections_for_race(snap, tn, mt, fm, mp))
        c.check(f"agreement at ({tn},{mt},{fm},{mp})", b, a)

    # ---- merge -----------------------------------------------------------
    led = rl.merge(rl.empty_ledger(), snap)
    c.check("merge into empty", len(led), len(snap))
    led2 = rl.merge(led, snap)
    c.check("re-logging the same race replaces, not duplicates", len(led2), len(snap))
    other = snap.copy()
    other["race_id"] = "Other_R1"
    led3 = rl.merge(led2, other)
    c.check("a different race is appended", led3["race_id"].nunique(), 2)

    # ---- record_result ---------------------------------------------------
    rid = snap["race_id"].iloc[0]
    tabs = [int(t) for t in snap["tab"]]
    done = rl.record_result(led, rid, tabs[:3])
    m = done["race_id"] == rid
    c.check("every runner is now settled", int(done.loc[m, "placed"].isna().sum()), 0)
    c.check("three placed in a 3-place race", int(done.loc[m, "placed"].sum()), 3)
    c.check("winner gets finish position 1",
            int(done.loc[m & (done["tab"] == tabs[0]), "finish_pos"].iloc[0]), 1)
    c.check("an unlisted runner is unplaced",
            int(done.loc[m & (done["tab"] == tabs[-1]), "placed"].iloc[0]), 0)
    two = led.copy()
    two["places_paid"] = 2
    done2 = rl.record_result(two, rid, tabs[:3])
    c.check("2-place terms place only two", int(done2.loc[m, "placed"].sum()), 2)
    try:
        rl.record_result(led, "no-such-race", [1, 2, 3])
        c.true("unknown race raises", False)
    except KeyError:
        c.true("unknown race raises", True)

    # ---- scoring ---------------------------------------------------------
    syn = synthetic(45)
    c.check("synthetic ledger is fully settled", int(syn["placed"].isna().sum()), 0)
    sc = rl.score_params(syn, top_n=5, market_top=3, fm_max=2.0, max_picks=3)
    c.true("precision is a proportion", 0.0 <= sc["precision"] <= 1.0)
    c.true("hits never exceed selections", sc["hits"] <= sc["selections"])
    c.check("every race counted", sc["races"], syn["race_id"].nunique())
    wide = rl.score_params(syn, top_n=99, market_top=99, fm_max=99.0, max_picks=99)
    c.true("an open rule selects more than a strict one",
           wide["selections"] > sc["selections"])
    c.true("cap is respected in scoring",
           rl.score_params(syn, 5, 3, 2.0, 2)["picks_per_race"] <= 2.0)

    # ---- tuner -----------------------------------------------------------
    small = syn[syn["race_id"].isin(sorted(syn["race_id"].unique())[:2])]
    c.check("tuner refuses two races", rl.tune(small)["ok"], False)
    t = rl.tune(syn)
    c.true("tuner runs on a real ledger", t["ok"])
    c.check("tuner reports the race count", t["races"], 45)
    c.true("tuned params are inside the grid",
           all(t["params"][k] in v for k, v in rl.DEFAULT_GRID.items()))
    c.true("cross-validated precision is a proportion",
           0.0 <= t["cv_precision"] <= 1.0)
    c.true("cross-validated selections were actually made", t["cv_selections"] > 0)
    c.true("45 races is flagged as not yet fully trustworthy", not t["trustworthy"])
    big = rl.tune(synthetic(rl.CONFIDENT_RACES + 5, seed=11))
    c.true("a large ledger is trustworthy", big["trustworthy"])
    c.true("tuner never suggests a rule that never bets",
           t["in_sample"]["picks_per_race"] >= 1.0)

    # ---- performance and readiness --------------------------------------
    perf = rl.performance(done)
    c.check("performance counts the race", perf["races"], 1)
    c.true("log-loss is finite", np.isfinite(perf["logloss"]))
    c.true("brier in range", 0.0 <= perf["brier"] <= 1.0)
    c.check("empty ledger reports no races", rl.performance(rl.empty_ledger())["races"], 0)
    psyn = rl.performance(syn)
    c.true("confidence interval brackets the point estimate",
           psyn["ci_low"] <= psyn["precision"] <= psyn["ci_high"])

    c.check("0 races reads as empty", rl.readiness(0)[0], "empty")
    c.check("3 races reads as thin", rl.readiness(3)[0], "thin")
    c.check("20 races reads as monitor", rl.readiness(20)[0], "monitor")
    c.check("50 races unlocks tuning", rl.readiness(50)[0], "tune")
    c.check("100 races reads as confident", rl.readiness(100)[0], "confident")
    c.true("the monitor message names the tuning threshold",
           str(rl.MIN_RACES_TO_TUNE) in rl.readiness(20)[1])

    # ---- round trip through CSV -----------------------------------------
    csv = done.to_csv(index=False)
    back = pd.read_csv(pd.io.common.StringIO(csv))
    c.check("CSV round trip keeps the schema", list(back.columns), rl.LEDGER_COLUMNS)
    c.check("CSV round trip keeps every row", len(back), len(done))
    c.check("CSV round trip preserves results",
            int(back["placed"].sum()), int(done["placed"].sum()))

    print(f"PASS {c.passes}  FAIL {len(c.fails)}")
    if c.fails:
        print("\n".join(c.fails))
    return 1 if c.fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
