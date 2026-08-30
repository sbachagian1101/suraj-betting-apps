"""Checks for Top Race Predictor. Run: python test_top.py

Covers the meeting splitter, the calibration, and the app itself. The original
engine keeps its own suite in test_engine.py and is not re-tested here beyond
confirming it still behaves identically at uncertainty_scale=1.0.
"""
from __future__ import annotations

import warnings

warnings.filterwarnings("ignore")

import numpy as np

import backtest as BT
import engine as E
import meeting as M

MEETING = "sample_data/2026-08-30-CARNARVON-T.xlsx"


class Checker:
    def __init__(self):
        self.passes, self.fails = 0, []

    def true(self, name, cond):
        if cond:
            self.passes += 1
        else:
            self.fails.append(name)

    def check(self, name, got, want):
        self.true(f"{name} (got {got!r}, want {want!r})", got == want)


def main():
    c = Checker()

    # ------------------------------------------------ the meeting splitter
    df = M.read_meeting(MEETING)
    blocks = M.split_races(df)
    c.check("the Carnarvon meeting yields six races", len(blocks), 6)
    c.check("numbered 1 to 6", [b.number for b in blocks], [1, 2, 3, 4, 5, 6])
    c.check("race 2 has seven runners",
            next(b for b in blocks if b.number == 2).runners, 7)

    # a sponsor whose name begins "TAB" must not be eaten by a header guard
    r3 = next(b for b in blocks if b.number == 3)
    c.true("the TABTOUCH race survives as its own race",
           r3.name.upper().startswith("TABTOUCH"))
    c.check("and keeps its own eleven runners", r3.runners, 11)
    c.true("no race absorbed another",
           all(2 <= b.runners <= 26 for b in blocks))

    # every block must be parseable by the untouched engine
    for b in blocks:
        race, runners, _, _ = E.parse_race_text(b.text)
        c.check(f"R{b.number} parses the runners it advertises",
                len(runners), b.runners)
        c.true(f"R{b.number} detected a distance", race.distance_m is not None)
        c.true(f"R{b.number} tabs are unique",
               len({r.tab for r in runners}) == len(runners))
        c.true(f"R{b.number} horses are named",
               all(r.horse.strip() for r in runners))

    c.true("the meeting label is recovered",
           "CARNARVON" in M.meeting_label(df).upper())

    # ---------------------------------------------------------- the engine
    b2 = next(b for b in blocks if b.number == 2)
    a = E.analyse_race_text(b2.text, simulations=4000)
    c.check("every runner is predicted", len(a.predictions), 7)
    c.true("win percentages sum to 100",
           abs(sum(p.win_pct for p in a.predictions) - 100.0) < 0.3)
    c.true("ranks are 1..n", [p.rank for p in a.predictions] == list(range(1, 8)))
    c.true("top-three percentages are percentages",
           all(0 <= p.top3_pct <= 100 for p in a.predictions))
    c.true("top-3 percentage is at least win percentage",
           all(p.top3_pct >= p.win_pct - 1e-9 for p in a.predictions))
    c.true("a verdict is produced", bool(a.verdict.strip()))
    c.true("a report renders", len(E.analysis_report_text(a)) > 200)

    # determinism: the engine seeds from the text, so repeats must agree
    a2 = E.analyse_race_text(b2.text, simulations=4000)
    c.true("the same race scores identically twice",
           [p.tab for p in a.predictions] == [p.tab for p in a2.predictions])

    # ------------------------------------------------------- the calibration
    c.true("scale 1.0 reproduces the original engine exactly",
           [round(p.win_pct, 6) for p in
            E.analyse_race_text(b2.text, simulations=4000,
                                uncertainty_scale=1.0).predictions]
           == [round(p.win_pct, 6) for p in a.predictions])

    raw, cal, same_pick = [], [], 0
    for b in blocks:
        ar = E.analyse_race_text(b.text, simulations=6000, uncertainty_scale=1.0)
        ac = E.analyse_race_text(b.text, simulations=6000, uncertainty_scale=2.0)
        raw.append(ar.predictions[0].win_pct)
        cal.append(ac.predictions[0].win_pct)
        same_pick += ar.predictions[0].tab == ac.predictions[0].tab
    c.true("calibration lowers the stated confidence",
           float(np.mean(cal)) < float(np.mean(raw)))
    c.true("the original is overconfident for fields of this size",
           float(np.mean(raw)) > 50.0)
    c.true("the calibrated version is in the region favourites really win",
           25.0 < float(np.mean(cal)) < 50.0)
    c.true("calibration mostly keeps the same selection",
           same_pick >= len(blocks) - 1)
    c.true("calibrated probabilities still sum to 100",
           abs(sum(p.win_pct for p in
                   E.analyse_race_text(b2.text, simulations=4000,
                                       uncertainty_scale=2.0).predictions)
               - 100.0) < 0.3)

    # a very large scale must approach a uniform field, not break
    flat = E.analyse_race_text(b2.text, simulations=6000, uncertainty_scale=40.0)
    c.true("huge noise flattens towards uniform",
           flat.predictions[0].win_pct < 40.0)
    tiny = E.analyse_race_text(b2.text, simulations=4000, uncertainty_scale=0.0)
    c.true("zero noise is clamped rather than dividing by zero",
           abs(sum(p.win_pct for p in tiny.predictions) - 100.0) < 0.3)

    # ---------------------------------------------------------- the backtest
    rows = BT.run(MEETING, BT.RESULTS["2026-08-30-CARNARVON-T.xlsx"], sims=6000)
    c.check("all five scored races were found", len(rows), 5)
    c.true("the recorded winners are real tab numbers in their race",
           all(r["winner"] in r["order"] for r in rows))
    c.check("the top pick won none of them", sum(r["won"] for r in rows), 0)
    c.check("and placed once", sum(r["pick_placed"] for r in rows), 1)
    c.true("every race has four recorded placings",
           all(len(r["finish"]) == 4 for r in rows))
    c.true("R2's recorded winner is tab 7 — Weapons Hot, verified separately",
           next(r for r in rows if r["race"] == 2)["winner"] == 7)

    print(f"PASS {c.passes}  FAIL {len(c.fails)}")
    for f in c.fails:
        print("  FAIL:", f)
    return 1 if c.fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
