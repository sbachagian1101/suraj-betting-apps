"""Checks for Match Insight. Run: python test_model.py"""
from __future__ import annotations

import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

import methods as X
import metrics as M
import parser as P

SAMPLE = "sample_data/sturm_graz_ii_vs_rapid_wien_ii.txt"
SAMPLE_HOME = "sample_data/home_sturm_graz_ii.txt"
SAMPLE_AWAY = "sample_data/away_rapid_wien_ii.txt"
HOME, AWAY = "Sturm Graz II", "Rapid Wien II"


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
    text = open(SAMPLE, encoding="utf-8").read()
    df, dropped = P.parse(text, return_dropped=True)

    # ------------------------------------------------------------ parsing
    c.check("ten pages are read", len(df) + len(dropped), 10)
    c.check("nine distinct matches survive", len(df), 9)
    c.check("one page is a duplicate", len(dropped), 1)
    c.true("the duplicate is the head-to-head, kept once",
           len(df[(df.home == HOME) & (df.away == AWAY)]) == 1)
    c.true("and the dropped page is that same fixture",
           bool((dropped.iloc[0].home == HOME) and (dropped.iloc[0].away == AWAY)))
    c.true("a plain parse still returns just the frame",
           isinstance(P.parse(text), pd.DataFrame))
    c.true("every match has a date", df.date.notna().all())
    c.true("scores are non-negative", ((df.hg >= 0) & (df.ag >= 0)).all())
    c.true("xG was read for every match", df.h_xg.notna().all() and df.a_xg.notna().all())
    c.true("corners were read for every match", df.h_corners.notna().all())
    c.true("possession pairs sum to about 100",
           bool((abs(df.h_possession + df.a_possession - 100) <= 2).all()))
    c.true("shots split into on and off target",
           bool((abs(df.h_sot + df.h_soff - df.h_shots) <= 1).all()))

    # a club with digits in its name must survive — this cost a real match
    c.true("First Vienna FC 1894 is parsed as a team",
           "First Vienna" in set(df.home) | set(df.away))
    c.true("both teams have five matches",
           len(P.team_matches(df, HOME)) == 5 and len(P.team_matches(df, AWAY)) == 5)
    c.true("a name that is mostly digits is refused",
           not P._looks_like_team("3 - 1") and not P._looks_like_team("2026"))
    c.true("a name with a few digits is accepted",
           P._looks_like_team("First Vienna FC 1894")
           and P._looks_like_team("Schalke 04"))

    # a specific match, end to end
    row = df[(df.home == "First Vienna")].iloc[0]
    c.check("First Vienna beat Sturm 4-0", (int(row.hg), int(row.ag)), (4, 0))
    c.close("with 2.13 xG", float(row.h_xg), 2.13, 1e-9)
    c.close("and Sturm 0.84", float(row.a_xg), 0.84, 1e-9)

    tmh = P.team_matches(df, HOME)
    c.true("team view flips away matches so gf is always the team's",
           int(tmh[tmh.opponent == "First Vienna"].iloc[0].gf) == 0)
    c.true("and the opponent's goals land in ga",
           int(tmh[tmh.opponent == "First Vienna"].iloc[0].ga) == 4)
    c.true("venue is recorded", set(tmh.venue) <= {"H", "A"})

    # parsing is robust to junk
    c.true("empty text yields an empty frame", P.parse("").empty)
    c.true("junk text yields an empty frame", P.parse("hello\nworld\n123").empty)
    c.true("a truncated page is skipped rather than half-read",
           P.parse("Saturday Aug 22, 2026 - 7:30pm\nA vs B\n").empty)

    # ------------------------------------------- one box, one team
    dh = P.parse(open(SAMPLE_HOME, encoding="utf-8").read())
    da = P.parse(open(SAMPLE_AWAY, encoding="utf-8").read())
    c.check("the home box holds five matches", len(dh), 5)
    c.check("the away box holds five matches", len(da), 5)

    sh, h1, h2 = P.subject_team(dh)
    sa, a1, a2 = P.subject_team(da)
    c.check("the home box names its team", sh, HOME)
    c.check("the away box names its team", sa, AWAY)
    c.true("the home team is in every one of its matches", h1 == 5)
    c.true("and stands clear of the next-most-seen side", h1 - h2 >= 4)
    c.true("the away team likewise", a1 == 5 and a1 - a2 >= 4)
    c.check("an empty box names nobody", P.subject_team(None)[0], None)
    c.check("so does an empty frame", P.subject_team(pd.DataFrame())[0], None)

    pooled = pd.concat([dh, da], ignore_index=True).drop_duplicates(
        subset=["date", "home", "away"])
    c.check("pooling the two boxes gives nine distinct matches", len(pooled), 9)
    c.check("with one overlap — the head-to-head",
            len(dh) + len(da) - len(pooled), 1)
    c.true("and the head-to-head is still form for both sides",
           len(P.team_matches(pooled, HOME)) == 5
           and len(P.team_matches(pooled, AWAY)) == 5)

    c.check("a friendly is classified", P.classify("Austria / Club Friendlies"),
            "Friendly")
    c.check("a cup tie is classified", P.classify("Austria / ÖFB Cup"), "Cup")
    c.check("a league is classified", P.classify("Austria / 2. Liga"), "League")

    # ------------------------------------------------------------ metrics
    base = M.sample_baselines(df)
    c.true("baseline goals are plausible", 0.5 < base["goals"] < 4.0)
    c.true("baseline xG is plausible", 0.5 < base["xg"] < 4.0)
    c.true("home advantage is shrunk into a sane band",
           0.85 <= base["home_adv"] <= 1.45)

    ph = M.team_profile(df, HOME, base)
    pa = M.team_profile(df, AWAY, base)
    c.check("five matches for the home team", ph["n"], 5)
    c.true("weights are positive and at most one",
           bool(((ph["matches"].weight > 0) & (ph["matches"].weight <= 1)).all()))
    c.true("the newest match carries full weight",
           abs(ph["matches"].weight.max() - 1.0) < 1e-9)
    c.true("the oldest match carries the least",
           ph["matches"].sort_values("date").weight.iloc[0]
           == ph["matches"].weight.min())

    # shrinkage must actually move the rate toward the mean
    c.true("shrinking pulls the attack rate toward the sample mean",
           abs(ph["xgf_s"] - base["xg"]) < abs(ph["xgf"] - base["xg"]) + 1e-12)
    c.true("and the defence rate too",
           abs(ph["xga_s"] - base["xg"]) < abs(ph["xga"] - base["xg"]) + 1e-12)
    c.close("a fully shrunk rate with no data is the prior",
            M._shrink(float("nan"), 1.4, 0.0), 1.4, 1e-12)
    c.true("more data means less shrinkage",
           abs(M._shrink(3.0, 1.0, 40.0) - 3.0) < abs(M._shrink(3.0, 1.0, 2.0) - 3.0))

    c.true("attack strength is positive", ph["atk_strength"] > 0)
    c.true("defence strength is positive", ph["def_strength"] > 0)
    c.true("indices are finite for both teams",
           all(np.isfinite([ph["atk_strength"], ph["def_strength"],
                            pa["atk_strength"], pa["def_strength"]])))

    # a friendly must be able to matter less
    kw = {"League": 1.0, "Friendly": 0.0, "Cup": 1.0, "Unknown": 1.0}
    d2 = df.copy()
    d2.loc[d2.index[0], "kind"] = "Friendly"
    w_full = M.match_weights(P.team_matches(df, HOME))
    tm2 = P.team_matches(d2, HOME)
    w_less = M.match_weights(tm2, kind_weight=kw)
    c.true("downweighting friendlies lowers the total weight",
           float(w_less.sum()) < float(w_full.sum()))

    lh, la = M.expected_goals(ph, pa, base)
    c.true("expected goals are in a sane range",
           0.15 <= lh <= 6 and 0.15 <= la <= 6)
    ch, ca = M.expected_corners(ph, pa, base)
    c.true("expected corners are in a sane range", 0.5 <= ch + ca <= 24)

    # ------------------------------------------------------------ methods
    ctx = X.build_context(df, ph, pa, base, sims=8000)
    res = X.run_all(ctx)
    c.true("at least ten methods are implemented", len(X.METHODS) >= 10)
    c.check("every method returned something", len(res), len(X.METHODS))
    errs = [r for r in res if "error" in r]
    c.true(f"no method raised ({[e['method'] for e in errs]})", not errs)

    for r in res:
        n = r["method"]
        c.true(f"{n}: 1X2 sums to one",
               abs(r["home"] + r["draw"] + r["away"] - 1.0) < 2e-3)
        c.true(f"{n}: probabilities are in range",
               all(0 <= r[k] <= 1 for k in ("home", "draw", "away")))
        c.true(f"{n}: BTTS is a probability", 0 <= r.get("btts", 0.5) <= 1)
        c.true(f"{n}: over 1.5 is at least over 2.5",
               r.get("over15", 1) >= r.get("over25", 0) - 1e-9)
        c.true(f"{n}: over 2.5 is at least over 3.5",
               r.get("over25", 1) >= r.get("over35", 0) - 1e-9)
        if "matrix" in r:
            m = r["matrix"]
            c.close(f"{n}: the score matrix sums to one", float(m.sum()), 1.0, 1e-6)
            c.true(f"{n}: no negative cells", bool((m >= -1e-12).all()))

    # a lopsided fixture must produce a lopsided answer
    strong = dict(ph, xgf_s=3.0, xga_s=0.6, gf_s=3.0, ga_s=0.6)
    weak = dict(pa, xgf_s=0.6, xga_s=3.0, gf_s=0.6, ga_s=3.0)
    ctx2 = X.build_context(df, strong, weak, base, sims=4000)
    r2 = X.run_all(ctx2)
    e2 = X.ensemble(r2)
    c.true("a much stronger home side is favoured", e2["home"] > e2["away"])
    c.true("clearly so", e2["home"] > 0.55)
    ctx3 = X.build_context(df, weak, strong, base, sims=4000)
    e3 = X.ensemble(X.run_all(ctx3))
    c.true("and the reverse fixture flips", e3["away"] > e3["home"])

    # symmetry: identical teams with home advantage removed
    even = dict(ph, xgf_s=1.4, xga_s=1.4, gf_s=1.4, ga_s=1.4, ppg=1.5,
                atk_weakness=0.0, def_weakness=0.0)
    b2 = dict(base, home_adv=1.0)
    ctx4 = X.build_context(df, even, dict(even), b2, sims=4000)
    m4 = X.m_poisson_xg(ctx4)["matrix"]
    o4 = X.outcomes(m4)
    c.close("identical teams with no home edge draw level",
            o4["home"], o4["away"], 1e-6)

    ens = X.ensemble(res)
    c.close("the ensemble 1X2 sums to one",
            ens["home"] + ens["draw"] + ens["away"], 1.0, 1e-9)
    c.true("the ensemble has a score matrix", "matrix" in ens)
    c.close("which sums to one", float(ens["matrix"].sum()), 1.0, 1e-6)
    c.true("the spread across methods is reported", ens["spread"] >= 0)
    c.check("the ensemble counts its methods", ens["n_methods"], len(res))

    # weighting a single method to dominance must move the ensemble to it
    tgt = res[0]
    w = {r["method"]: (100.0 if r["method"] == tgt["method"] else 1e-6)
         for r in res}
    c.true("weighting one method dominates the ensemble",
           abs(X.ensemble(res, w)["home"] - tgt["home"]) < 0.01)

    cm = X.corners_markets(5.0, 4.0, 8.5)
    c.close("corners over and under sum to one", cm["over"] + cm["under"], 1.0, 1e-9)
    c.true("a higher line means a lower over",
           X.corners_markets(5.0, 4.0, 12.5)["over"]
           < X.corners_markets(5.0, 4.0, 6.5)["over"])
    c.true("more expected corners means a higher over",
           X.corners_markets(8.0, 8.0, 8.5)["over"]
           > X.corners_markets(3.0, 3.0, 8.5)["over"])

    tops = X.top_scores(ens["matrix"], 5)
    c.check("five scorelines are returned", len(tops), 5)
    c.true("they are ordered by probability",
           [p for _, p in tops] == sorted([p for _, p in tops], reverse=True))
    c.true("each looks like a scoreline", all("-" in s for s, _ in tops))

    # outcomes() must agree with a hand-checked matrix
    m = np.zeros((9, 9))
    m[2, 1] = 0.5      # home win, BTTS, 3 goals
    m[1, 1] = 0.3      # draw, BTTS, 2 goals
    m[0, 2] = 0.2      # away win, no BTTS, 2 goals
    o = X.outcomes(m)
    c.close("hand-checked home win", o["home"], 0.5, 1e-9)
    c.close("hand-checked draw", o["draw"], 0.3, 1e-9)
    c.close("hand-checked away win", o["away"], 0.2, 1e-9)
    c.close("hand-checked BTTS", o["btts"], 0.8, 1e-9)
    c.close("hand-checked over 2.5", o["over25"], 0.5, 1e-9)

    print(f"PASS {c.passes}  FAIL {len(c.fails)}")
    for f in c.fails:
        print("  FAIL:", f)
    return 1 if c.fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
