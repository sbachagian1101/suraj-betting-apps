"""Regression tests for the soccer data loader and Dixon-Coles model.

Two kinds of check:

* **Data integrity** - expectations read off the raw CSVs by hand, so the loader
  and the files are independent of each other.
* **Model behaviour** - mathematical invariants that must hold for any input
  (probabilities summing to one, home advantage having the right sign), plus
  golden values pinning the current fit so an accidental change is visible.

    python test_model.py     # expect: PASS <n>  FAIL 0
"""
import glob
import os

import numpy as np
import pandas as pd

import soccer_data as sd
import soccer_model as sm

_HERE = os.path.dirname(os.path.abspath(__file__))
SAMPLE = sorted(glob.glob(os.path.join(_HERE, "sample_data", "*.csv")))

# Ground truth counted directly from the three Latvian Virsliga CSVs.
EXPECT_DATA = {
    "raw_rows": 540,            # 180 fixtures x 3 seasons
    "complete": 491,            # 2026 is still in progress
    "seasons": 3,
    "teams": 12,                # 10 per season, with promotion/relegation
    "sentinels_cleared": 17,    # `-1` entries across shot/foul columns
    "xg_rows_dropped": 3,       # both teams recorded 0.00 xG
}
EXPECT_TEAMS = ["Auda", "BFC Daugavpils", "FS Jelgava", "Grobiņa", "Liepāja",
                "Metta - LU", "Ogre United", "Riga", "Rīgas FS", "Super Nova",
                "Tukums", "Valmiera / BSS"]


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

    def close(self, label, got, want, tol):
        self.check(f"{label} (±{tol})", got, want, abs(float(got) - float(want)) <= tol)


def test_data(c: Checker):
    raw = sum(len(pd.read_csv(f)) for f in SAMPLE)
    c.check("raw rows across files", raw, EXPECT_DATA["raw_rows"])

    df, notes = sd.load_frames(SAMPLE, names=[os.path.basename(f) for f in SAMPLE])
    c.check("completed matches", len(df), EXPECT_DATA["complete"])
    c.check("seasons", int(df["season"].nunique()), EXPECT_DATA["seasons"])
    c.check("teams", len(sd.teams_of(df)), EXPECT_DATA["teams"])
    c.check("team list", sd.teams_of(df), EXPECT_TEAMS)
    c.check("incomplete excluded",
            any("not marked complete" in n for n in notes), True)
    c.check(f"{EXPECT_DATA['sentinels_cleared']} sentinels reported",
            any(f"{EXPECT_DATA['sentinels_cleared']} `-1`" in n for n in notes), True)
    c.check(f"{EXPECT_DATA['xg_rows_dropped']} bad-xG rows reported",
            any(f"xG for {EXPECT_DATA['xg_rows_dropped']} match" in n for n in notes), True)

    # No -1 may survive anywhere in the numeric columns.
    surviving = int(sum(int((df[col] == -1).sum())
                        for col in sd.NUMERIC_SENTINEL if col in df.columns))
    c.check("no -1 sentinels survive cleaning", surviving, 0)
    # And no match may keep a 0/0 xG pair.
    both0 = int(((df["team_a_xg"] == 0) & (df["team_b_xg"] == 0)).sum())
    c.check("no both-zero xG rows survive", both0, 0)

    c.check("dates sorted ascending", df["date"].is_monotonic_increasing, True)
    c.check("no duplicate fixtures",
            int(df.duplicated(["date", "home_team_name", "away_team_name"]).sum()), 0)
    c.check("scores are integers",
            bool(df["hg"].dtype.kind in "iu" and df["ag"].dtype.kind in "iu"), True)

    # League baselines, hand-checked against the raw files.
    c.close("home goals per match", df["hg"].mean(), 1.699, 0.01)
    c.close("away goals per match", df["ag"].mean(), 1.322, 0.01)
    c.close("BTTS rate", float(((df.hg > 0) & (df.ag > 0)).mean()), 0.523, 0.01)
    c.close("Over 2.5 rate", float((df.hg + df.ag > 2.5).mean()), 0.568, 0.01)
    return df


def test_model(c: Checker, df):
    m = sm.fit(df)
    c.check("model fitted on all matches", m.n_matches, len(df))
    c.check("attack params sum to zero", True, True,
            abs(float(m.attack.sum())) < 1e-6)
    c.check("defence params sum to zero", True, True,
            abs(float(m.defence.sum())) < 1e-6)
    c.check("home advantage is positive", True, True, m.gamma > 0)
    c.check("response weights sum to 1", True, True,
            abs(sum(m.weights.values()) - 1.0) < 1e-9)
    c.check("goals keep half the weight", round(m.weights["goals"], 3), 0.5)
    c.close("SoT conversion rate", m.conversion, 0.335, 0.01)

    # Golden values pinning the current fit.
    c.close("mu", m.mu, 0.2809, 0.01)
    c.close("gamma", m.gamma, 0.1411, 0.01)
    c.close("home advantage in goals", m.home_advantage_goals, 0.201, 0.02)

    p = sm.predict(m, "Auda", "Liepāja")
    c.close("1X2 sums to 1", p["home_win"] + p["draw"] + p["away_win"], 1.0, 1e-9)
    c.close("BTTS sums to 1", p["btts_yes"] + p["btts_no"], 1.0, 1e-9)
    c.close("O/U 2.5 sums to 1", p["over_25"] + p["under_25"], 1.0, 1e-9)
    c.check("score matrix sums to 1", True, True,
            abs(float(p["matrix"].sum()) - 1.0) < 1e-9)
    c.check("every probability in [0,1]", True, True,
            all(0.0 <= p[k] <= 1.0 for k in
                ("home_win", "draw", "away_win", "btts_yes", "over_25")))
    c.check("Over 1.5 >= Over 2.5 >= Over 3.5", True, True,
            p["over_15"] >= p["over_25"] >= p["over_35"])
    c.close("Auda expected goals", p["lambda_home"], 1.69, 0.05)
    c.close("Liepāja expected goals", p["lambda_away"], 1.07, 0.05)
    c.close("Auda win probability", p["home_win"], 0.519, 0.02)

    # Home advantage must actually favour the home side: swapping venue
    # has to lower the same team's win probability.
    rev = sm.predict(m, "Liepāja", "Auda")
    c.check("venue swap reduces Auda's win chance", True, True,
            rev["away_win"] < p["home_win"])

    # A stronger opponent must lower a team's expected goals.
    tt = sd.team_table(df, m)
    best_def = tt.sort_values("Defence idx").iloc[0]["Team"]
    worst_def = tt.sort_values("Defence idx").iloc[-1]["Team"]
    if best_def != "Auda" and worst_def != "Auda":
        vs_best = sm.expected_goals(m, "Auda", best_def)[0]
        vs_worst = sm.expected_goals(m, "Auda", worst_def)[0]
        c.check("scores fewer against the best defence", True, True, vs_best < vs_worst)

    c.check("unknown team raises", True, True, _raises(m, "Nowhere United", "Auda"))

    # Market blending: full weight on either side must return that side.
    mp = np.array([0.5, 0.3, 0.2]); kp = np.array([0.2, 0.3, 0.5])
    c.check("blend w=0 returns model", True, True,
            np.allclose(sm.blend(mp, kp, 0.0), mp, atol=1e-9))
    c.check("blend w=1 returns market", True, True,
            np.allclose(sm.blend(mp, kp, 1.0), kp, atol=1e-9))
    c.close("de-vig removes the margin", sm.devig_1x2(2.0, 3.5, 4.0).sum(), 1.0, 1e-9)
    return m


def _raises(m, home, away):
    try:
        sm.expected_goals(m, home, away)
        return False
    except KeyError:
        return True


def test_backtest(c: Checker, df):
    """The backtest must be strictly out-of-sample and beat naive baselines."""
    bt = sm.walk_forward(df)
    ev = sm.evaluate(bt)
    c.check("backtest produced predictions", True, True, ev["n"] > 300)
    c.check("beats league base rates on log-loss", True, True,
            ev["logloss_1x2"] < ev["logloss_baserate"])
    c.check("beats league base rates on RPS", True, True,
            ev["rps_1x2"] < ev["rps_baserate"])
    c.check("beats a coin flip on Over 2.5", True, True, ev["logloss_o25"] < 0.6931)
    c.check("1X2 accuracy above 55%", True, True, ev["acc_1x2"] > 0.55)
    c.check("probabilities are valid", True, True,
            bool(((bt[["p_H", "p_D", "p_A"]].sum(axis=1) - 1).abs() < 1e-9).all()))
    # Being *behind* the closing market is expected; assert we are at least close.
    if "market_logloss" in ev:
        c.check("within 0.10 log-loss of the bookmaker", True, True,
                ev["logloss_1x2"] - ev["market_logloss"] < 0.10)
    print(f"    backtest: n={ev['n']} 1X2 ll={ev['logloss_1x2']:.4f} "
          f"rps={ev['rps_1x2']:.4f} acc={ev['acc_1x2']:.3f} | "
          f"market ll={ev.get('market_logloss', float('nan')):.4f} | "
          f"baserate ll={ev['logloss_baserate']:.4f}")
    print(f"    BTTS ll={ev['logloss_btts']:.4f}  Over2.5 ll={ev['logloss_o25']:.4f}")
    return ev, bt


def test_bet_signal(c: Checker, model, df, bt):
    """The 1X2-plus-scorelines agreement rule.

    Mostly boundary work: the threshold is strict ("> 45%"), a draw as the most
    likely score is disagreement rather than a weak confirmation, and the away
    side must be treated symmetrically with the home side.
    """
    def P(h, d, a, t1, t2, home="HOME", away="AWAY"):
        return {"home_win": h, "draw": d, "away_win": a, "home": home, "away": away,
                "top_scorelines": [(t1[0], t1[1], 0.12), (t2[0], t2[1], 0.10),
                                   (0, 0, 0.05)]}

    c.check("2-0 is a home win", sm.scoreline_outcome(2, 0), "H")
    c.check("0-2 is an away win", sm.scoreline_outcome(0, 2), "A")
    c.check("1-1 is a draw", sm.scoreline_outcome(1, 1), "D")
    c.check("0-0 is a draw", sm.scoreline_outcome(0, 0), "D")

    g = sm.bet_signal(P(.50, .25, .25, (2, 0), (1, 0)))
    c.check("two home wins on top gives green", g["level"], sm.BET_GREEN)
    c.check("and names the home team", g["team"], "HOME")
    y = sm.bet_signal(P(.50, .25, .25, (2, 0), (1, 1)))
    c.check("win then draw gives yellow", y["level"], sm.BET_YELLOW)

    # the threshold is strict, as specified: "> 45%"
    c.check("exactly at the threshold does not fire", True, True,
            sm.bet_signal(P(.45, .30, .25, (2, 0), (1, 0))) is None)
    c.check("a hair above the threshold fires", True, True,
            sm.bet_signal(P(.4501, .30, .25, (2, 0), (1, 0))) is not None)
    c.check("below the threshold does not fire", True, True,
            sm.bet_signal(P(.44, .31, .25, (2, 0), (1, 0))) is None)
    c.check("a custom threshold is honoured", True, True,
            sm.bet_signal(P(.50, .25, .25, (2, 0), (1, 0)), min_win_prob=0.6) is None)

    # a draw as the MOST likely score is disagreement, not a weak confirmation
    c.check("draw on top gives no signal", True, True,
            sm.bet_signal(P(.50, .25, .25, (1, 1), (2, 0))) is None)
    c.check("second score for the other side gives no signal", True, True,
            sm.bet_signal(P(.50, .25, .25, (2, 0), (0, 1))) is None)
    c.check("two draws give no signal", True, True,
            sm.bet_signal(P(.50, .25, .25, (1, 1), (0, 0))) is None)

    # the away side is handled symmetrically
    ga = sm.bet_signal(P(.23, .25, .52, (0, 2), (0, 1)))
    c.check("away can go green", ga["level"], sm.BET_GREEN)
    c.check("and names the away team", ga["team"], "AWAY")
    c.check("away side recorded", ga["side"], "A")
    ya = sm.bet_signal(P(.23, .25, .52, (0, 1), (1, 1)))
    c.check("away can go yellow", ya["level"], sm.BET_YELLOW)

    # degenerate input must not raise
    c.check("no scorelines gives no signal", True, True,
            sm.bet_signal({"home_win": .9, "draw": .05, "away_win": .05,
                           "home": "A", "away": "B", "top_scorelines": []}) is None)
    c.check("one scoreline gives no signal", True, True,
            sm.bet_signal({"home_win": .9, "draw": .05, "away_win": .05,
                           "home": "A", "away": "B",
                           "top_scorelines": [(2, 0, .2)]}) is None)
    c.check("no signal renders as empty text", sm.bet_signal_text(None), "")
    c.check("green text names the team", True, True,
            "HOME" in sm.bet_signal_text(g))
    c.check("yellow text mentions the draw", True, True,
            "draw" in sm.bet_signal_text(y))

    # a real prediction must flow through unchanged
    teams = sd.teams_of(df)
    live = sm.predict(model, teams[0], teams[1])
    sig = sm.bet_signal(live)
    c.check("a real prediction yields a valid signal or none", True, True,
            sig is None or sig["level"] in (sm.BET_GREEN, sm.BET_YELLOW))
    if sig:
        c.check("the signal names one of the two playing", True, True,
                sig["team"] in (live["home"], live["away"]))
        c.check("the signal's probability matches the model", True, True,
                abs(sig["p_win"] - (live["home_win"] if sig["side"] == "H"
                                    else live["away_win"])) < 1e-12)

    # the backtest must carry the scorelines the signal needs
    c.check("walk_forward keeps the top two scorelines", True, True,
            {"s1_h", "s1_a", "s2_h", "s2_a"} <= set(bt.columns))
    c.check("scorelines are non-negative goal counts", True, True,
            bool((bt[["s1_h", "s1_a", "s2_h", "s2_a"]] >= 0).all().all()))

    # replaying the rule on held-out matches must reproduce the published gap
    fired = {"green": [0, 0], "yellow": [0, 0]}
    for r in bt.itertuples():
        sig = sm.bet_signal({"home_win": r.p_H, "draw": r.p_D, "away_win": r.p_A,
                             "home": r.home, "away": r.away,
                             "top_scorelines": [(int(r.s1_h), int(r.s1_a), 0.0),
                                                (int(r.s2_h), int(r.s2_a), 0.0)]})
        if sig:
            b = fired[sig["level"]]
            b[0] += 1
            b[1] += int(sig["side"] == r.result)
    c.check("green fires on held-out matches", True, True, fired["green"][0] > 0)
    c.check("green wins more often than yellow", True, True,
            fired["green"][1] / max(fired["green"][0], 1)
            > fired["yellow"][1] / max(fired["yellow"][0], 1))
    print(f"    replayed: green {fired['green'][1]}/{fired['green'][0]}, "
          f"yellow {fired['yellow'][1]}/{fired['yellow'][0]}")

    # the published measurements must stay self-consistent
    M = sm.BET_MEASURED
    c.check("green beats yellow", True, True, M["green_won"] > M["yellow_won"])
    c.check("green beats backing every >45% side", True, True,
            M["green_won"] > M["over45_won"])
    c.check("yellow is worse than backing every >45% side", True, True,
            M["yellow_won"] < M["over45_won"])
    c.check("yellow underperformed what the model said", True, True,
            M["yellow_won"] < M["yellow_model_said"])
    c.check("yellow draws roughly double", True, True,
            M["yellow_drew"] > 1.5 * M["over45_drew"])
    c.check("green and yellow sum to either", True, True,
            M["green_n"] + M["yellow_n"] == M["either_n"])
    c.check("signals are a subset of the >45% sides", True, True,
            M["either_n"] < M["over45_n"])
    c.check("intervals bracket their estimates", True, True,
            M["green_ci"][0] <= M["green_won"] <= M["green_ci"][1]
            and M["yellow_ci"][0] <= M["yellow_won"] <= M["yellow_ci"][1])
    c.check("p-values are probabilities", True, True,
            0 < M["green_vs_yellow_p"] < 1 and 0 < M["green_vs_over45_p"] < 1)


def main():
    if not SAMPLE:
        print("No sample data found in sample_data/ - cannot run tests.")
        return 1
    c = Checker()
    print("--- data integrity")
    df = test_data(c)
    print("--- model behaviour")
    m = test_model(c, df)
    print("--- walk-forward backtest (this refits per matchday, ~30s)")
    _, bt = test_backtest(c, df)
    print("--- bet signal")
    test_bet_signal(c, m, df, bt)
    print(f"\nPASS {c.passes}  FAIL {len(c.fails)}")
    if c.fails:
        print("\n".join(c.fails))
    return 1 if c.fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
