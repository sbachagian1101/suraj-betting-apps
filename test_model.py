"""Checks for FT Score Predictor. Run: python test_model.py"""
from __future__ import annotations

import glob
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

import backtest as BT
import model as MD
import panel as PN

SAMPLES = sorted(glob.glob("sample_data/*.txt"))


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

    def close(self, name, got, want, tol=1e-9):
        self.true(f"{name} (got {got!r}, want {want!r})", abs(got - want) <= tol)


def main():
    c = Checker()
    c.true("there are sample panels", len(SAMPLES) == 5)

    # ------------------------------------------- the concatenated-number split
    cases = [
        ("2.942.713.10", [2.94, 2.71, 3.10]),
        ("3.173.602.63", [3.17, 3.60, 2.63]),
        ("1.491.791.28", [1.49, 1.79, 1.28]),
        ("0.830.501.25", [0.83, 0.50, 1.25]),
        ("1.21.131.28", [1.20, 1.13, 1.28]),      # trailing zero dropped
        ("1.821.741.9", [1.82, 1.74, 1.90]),      # and again
        ("0.930.631.2", [0.93, 0.63, 1.20]),
        ("3.243.253.22", [3.24, 3.25, 3.22]),     # three near-identical
        ("1.561.571.56", [1.56, 1.57, 1.56]),
        ("4.694.634.75", [4.69, 4.63, 4.75]),
        ("1.542.231.2", [1.54, 2.23, 1.20]),      # from the Austrian panel
    ]
    for s, want in cases:
        got = PN.split_three(s)
        c.true(f"{s} splits to {want} (got {got})",
               got is not None and all(abs(a - b) < 1e-9
                                       for a, b in zip(got, want)))

    c.true("a run that cannot be three numbers is refused",
           PN.split_three("abc") is None)
    c.true("and one that breaks Overall-between-Home-and-Away is refused",
           PN.split_three("9.001.001.10") is None)
    c.check("percentages split on the sign",
            PN._split_pcts("47%43%50%"), [47.0, 43.0, 50.0])
    c.check("including a zero", PN._split_pcts("11%0%25%"), [11.0, 0.0, 25.0])
    c.check("and a hundred", PN._split_pcts("100%0%50%"), [100.0, 0.0, 50.0])

    # ------------------------------------------------------------- the panels
    for f in SAMPLES:
        ps = PN.parse_panels(open(f, encoding="utf-8").read())
        name = f.replace("\\", "/").rsplit("/", 1)[-1]
        c.check(f"{name} yields two panels", len(ps), 2)
        for p in ps:
            c.true(f"{name}: {p['team'] or '?'} is named", bool(p["team"]))
            c.true(f"{name}: {p['team']} names its league", bool(p["league"]))
            c.check(f"{name}: {p['team']} has every stat row",
                    PN.missing(p), [])
            for key in ("scored", "conceded", "xg", "xga", "avg_goals"):
                for col in PN.COLS:
                    v = PN.value(p, key, col)
                    c.true(f"{name}: {p['team']} {key}.{col} is a number",
                           np.isfinite(v))
                o = PN.value(p, key, "overall")
                h = PN.value(p, key, "home")
                a = PN.value(p, key, "away")
                c.true(f"{name}: {p['team']} {key} — Overall lies between "
                       f"Home and Away ({h}, {o}, {a})",
                       min(h, a) - 0.07 <= o <= max(h, a) + 0.07)
            for key in ("win_pct", "btts_pct", "cs_pct", "fts_pct"):
                for col in PN.COLS:
                    v = PN.value(p, key, col)
                    c.true(f"{name}: {p['team']} {key}.{col} is a percentage",
                           0 <= v <= 100)

    # a couple of values checked against the source by hand
    aik = PN.parse_panels(
        open("sample_data/aik_vs_hammarby.txt", encoding="utf-8").read())
    c.check("the home panel is AIK", aik[0]["team"], "AIK Fotboll")
    c.check("the away panel is Hammarby", aik[1]["team"], "Hammarby IF")
    c.check("AIK's league position", aik[0]["pos"], 7)
    c.check("out of", aik[0]["teams_in_league"], 16)
    c.close("AIK scores 1.00 at home", PN.value(aik[0], "scored", "home"), 1.00)
    c.close("and 1.80 away", PN.value(aik[0], "scored", "away"), 1.80)
    c.close("Hammarby's away xG", PN.value(aik[1], "xg", "away"), 2.08)
    c.close("and their away xGA", PN.value(aik[1], "xga", "away"), 1.19)
    c.check("AIK's home form", aik[0]["form"]["home"]["results"], "LLLWL")
    c.close("and its PPG", aik[0]["form"]["home"]["ppg"], 1.29)

    hon = PN.parse_panels(
        open("sample_data/honefoss_vs_follo.txt", encoding="utf-8").read())
    c.check("a panel with no league position still parses",
            hon[0]["pos"], None)
    c.check("and carries its recent record", hon[0]["recent"], (10, 2, 4))
    c.close("Hønefoss away xG survives the dropped zero",
            PN.value(hon[0], "xg", "away"), 1.90)

    c.true("junk text yields no panels", PN.parse_panels("hello\nworld") == [])
    c.true("empty text yields no panels", PN.parse_panels("") == [])

    # ---------------------------------------------------------------- rates
    H, A = aik[0], aik[1]
    r = MD.rates(H, A, venue_weight=1.0, xg_weight=0.0)
    c.close("at full venue weight the home rate pairs home-scored with "
            "away-conceded", r["lh"], (1.00 + 1.25) / 2, 1e-9)
    c.close("and the away rate pairs away-scored with home-conceded",
            r["la"], (1.38 + 1.71) / 2, 1e-9)
    rx = MD.rates(H, A, venue_weight=1.0, xg_weight=1.0)
    c.close("on xG the home rate is home xG against away xGA",
            rx["lh"], (1.79 + 1.19) / 2, 1e-9)
    r0 = MD.rates(H, A, venue_weight=0.0, xg_weight=0.0)
    c.close("at zero venue weight it falls back to the overall columns",
            r0["lh"], (1.47 + 0.83) / 2, 1e-9)
    rm = MD.rates(H, A, venue_weight=0.5, xg_weight=0.0)
    c.true("half venue weight lands between the two",
           min(r["lh"], r0["lh"]) <= rm["lh"] <= max(r["lh"], r0["lh"]))

    # venue already carries home advantage, so no extra factor is applied:
    # swapping the two panels must not simply mirror the rates
    rr = MD.rates(A, H, venue_weight=1.0, xg_weight=0.0)
    c.true("swapping the sides gives genuinely different rates, because each "
           "team is now read at the other venue",
           abs(rr["lh"] - r["la"]) > 0.01)

    # ---------------------------------------------------------- the matrix
    m = MD.score_matrix(1.4, 1.1)
    c.close("the grid is a distribution", float(m.sum()), 1.0, 1e-9)
    c.true("with no negative cells", bool((m >= -1e-12).all()))
    mk = MD.markets(m)
    c.close("1X2 sums to one", mk["home"] + mk["draw"] + mk["away"], 1.0, 1e-9)
    c.true("over 1.5 is at least over 2.5", mk["over15"] >= mk["over25"])
    c.true("over 2.5 is at least over 3.5", mk["over25"] >= mk["over35"])
    c.close("expected goals matches the rates", mk["exp_goals"], 2.5, 0.03)

    lop = MD.markets(MD.score_matrix(2.6, 0.5))
    c.true("a much stronger home side is favoured", lop["home"] > lop["away"])
    c.true("clearly so", lop["home"] > 0.65)
    even = MD.markets(MD.score_matrix(1.5, 1.5))
    c.close("equal rates give an even game", even["home"], even["away"], 1e-9)

    nb = MD.score_matrix(1.5, 1.5, dispersion=4.0)
    c.close("the over-dispersed grid is still a distribution",
            float(nb.sum()), 1.0, 1e-9)
    # over-dispersion fattens BOTH tails, and at these rates the low one wins:
    # 0-0 climbs sharply while the over-lines drift slightly down. Asserting
    # the opposite is what caught a wrong comment in the model.
    c.true("over-dispersion raises the chance of 0-0",
           float(nb[0, 0]) > float(MD.score_matrix(1.5, 1.5)[0, 0]))
    c.true("and nudges the over-lines down rather than up",
           MD.markets(nb)["over25"] < even["over25"])
    c.close("while leaving expected goals alone",
            MD.markets(nb)["exp_goals"], even["exp_goals"], 0.05)

    tops = MD.top_scores(m, 5)
    c.check("five scorelines come back", len(tops), 5)
    c.true("ordered by probability",
           [p for _, p in tops] == sorted([p for _, p in tops], reverse=True))
    c.close("the most likely scoreline probability matches the grid",
            tops[0][1], float(m[tops[0][0][0], tops[0][0][1]]), 1e-12)
    c.true("no single scoreline is anywhere near certain", tops[0][1] < 0.30)
    c.close("a score off the end of the grid has probability zero",
            MD.score_prob(m, 99, 0), 0.0)

    pr = MD.predict(H, A)
    c.true("predict returns a pick", isinstance(pr["pick"], tuple))
    c.close("whose probability matches the grid", pr["pick_prob"],
            float(pr["matrix"][pr["pick"][0], pr["pick"][1]]), 1e-12)

    # -------------------------------------------------------- the backtest
    bt = BT.run()
    c.check("all five results are scored", len(bt), 5)
    c.true("every match has a probability for the actual score",
           bool((bt.p_actual >= 0).all()))
    c.true("the model beats a league-average Poisson on scoreline log-loss",
           bt.logloss.mean() < BT.baseline_logloss())
    c.true("it gets at least half the 1X2 calls", bt.res_ok.sum() >= 3)
    c.true("expected goals are in a believable range",
           2.0 < bt.exp.mean() < 5.0)
    c.true("no match is given a silly rate",
           bool(((bt.lh > 0.2) & (bt.lh < 5) & (bt.la > 0.2) & (bt.la < 5)).all()))

    print(f"PASS {c.passes}  FAIL {len(c.fails)}")
    for f in c.fails:
        print("  FAIL:", f)
    return 1 if c.fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
