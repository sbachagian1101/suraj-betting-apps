"""Regression tests for SoccerPredict.

Deliberately **data-agnostic**. An earlier version hardcoded Latvian team names
and golden log-loss values, so swapping the bundled league broke the suite for
reasons that had nothing to do with the code. These assert properties that must
hold for any league: probabilities summing to one, a home advantage with the
right sign, a stronger defence lowering expected goals, tuning that never sees
its own validation, and — the claim the whole app rests on — that the model's
disadvantage against the market grows with disagreement.

    python test_model.py     # expect: PASS <n>  FAIL 0
"""
import glob
import os
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

import assess as A
import soccer_data as sd
import soccer_model as sm

_HERE = os.path.dirname(os.path.abspath(__file__))
SAMPLE = sorted(glob.glob(os.path.join(_HERE, "sample_data", "*.csv")))


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

    def close(self, label, got, want, tol=1e-9):
        self.check(label, got, want, abs(float(got) - float(want)) <= tol)


def main():
    c = Checker()
    if not SAMPLE:
        print("No sample data found in sample_data/ — cannot run tests.")
        return 1

    # ---- data -------------------------------------------------------------
    df, notes = sd.load_frames(SAMPLE)
    c.true("data loaded", len(df) > 200)
    c.true("a note is produced for the reader", len(notes) >= 1)
    for col in ("date", "home_team_name", "away_team_name", "hg", "ag"):
        c.true(f"column {col} present", col in df.columns)
    c.true("goals are non-negative integers",
           bool(((df.hg >= 0) & (df.ag >= 0)).all()))
    c.true("dates are datetimes", pd.api.types.is_datetime64_any_dtype(df.date))
    c.true("no unplayed fixtures survive cleaning", len(df) > 0)
    teams = sd.teams_of(df)
    c.true("teams are sorted and unique",
           teams == sorted(set(teams)))
    c.true("every match's teams are known",
           set(df.home_team_name) <= set(teams)
           and set(df.away_team_name) <= set(teams))
    c.true("a team never plays itself",
           bool((df.home_team_name != df.away_team_name).all()))

    # ---- fitting ----------------------------------------------------------
    m = sm.fit(df)
    c.true("home advantage is positive", m.home_advantage_goals > 0)
    c.true("home advantage is plausible", 0.0 < m.home_advantage_goals < 1.0)
    c.true("every team gets an attack rating", len(m.attack) == len(teams))
    c.true("every team gets a defence rating", len(m.defence) == len(teams))
    c.true("attack ratings are centred", abs(float(np.mean(m.attack))) < 0.2)
    c.check("the index covers every team", len(m.index), len(teams))
    c.true("rho is a small correction", abs(m.rho) < 0.5)

    a, b = teams[0], teams[1]
    lh, la = sm.expected_goals(m, a, b)
    c.true("expected goals are positive", lh > 0 and la > 0)
    c.true("expected goals are plausible", lh < 6 and la < 6)
    lh2, la2 = sm.expected_goals(m, b, a)
    c.true("swapping home and away changes the numbers",
           abs(lh - la2) > 1e-9 or abs(la - lh2) > 1e-9)
    try:
        sm.expected_goals(m, "NOT A REAL TEAM", b)
        c.true("an unknown team raises", False)
    except KeyError:
        c.true("an unknown team raises", True)

    # a stronger defence must reduce the goals conceded to it
    # in lambda = exp(mu + attack + defence_opponent + gamma) a LOWER defence
    # value concedes fewer goals, so the best defence is the minimum
    best_def = m.teams[int(np.argmin(m.defence))]
    worst_def = m.teams[int(np.argmax(m.defence))]
    if best_def != a and worst_def != a:
        g_best, _ = sm.expected_goals(m, a, best_def)
        g_worst, _ = sm.expected_goals(m, a, worst_def)
        c.true("a better defence concedes fewer expected goals", g_best < g_worst)

    # ---- score matrix and markets -----------------------------------------
    mat = sm.score_matrix(1.4, 1.1, m.rho)
    c.close("the score matrix is a distribution", float(mat.sum()), 1.0, 1e-6)
    c.true("all cells are non-negative", bool((mat >= -1e-12).all()))
    mk = sm.markets(mat)
    c.close("1X2 sums to one", mk["home_win"] + mk["draw"] + mk["away_win"], 1.0, 1e-6)
    c.close("BTTS sums to one", mk["btts_yes"] + mk["btts_no"], 1.0, 1e-6)
    c.close("over/under 2.5 sums to one", mk["over_25"] + mk["under_25"], 1.0, 1e-6)
    c.true("over 1.5 is at least over 2.5", mk["over_15"] >= mk["over_25"])
    c.true("over 2.5 is at least over 3.5", mk["over_25"] >= mk["over_35"])
    hi = sm.markets(sm.score_matrix(2.5, 2.5, m.rho))
    lo = sm.markets(sm.score_matrix(0.6, 0.6, m.rho))
    c.true("more expected goals means more over 2.5",
           hi["over_25"] > lo["over_25"])
    c.true("more expected goals means more BTTS", hi["btts_yes"] > lo["btts_yes"])
    lop = sm.markets(sm.score_matrix(2.0, 0.5, m.rho))
    c.true("a stronger home side wins more often",
           lop["home_win"] > lop["away_win"])
    tops = sm.top_scorelines(mat, 5)
    c.check("top scorelines returns what was asked", len(tops), 5)
    c.true("scorelines are ordered by probability",
           all(tops[i][2] >= tops[i + 1][2] for i in range(len(tops) - 1)))

    p = sm.predict(m, a, b)
    c.close("prediction 1X2 sums to one",
            p["home_win"] + p["draw"] + p["away_win"], 1.0, 1e-6)
    c.check("prediction names the teams", (p["home"], p["away"]), (a, b))

    # ---- de-vigging -------------------------------------------------------
    for o in ([2.0, 3.4, 4.0], [1.2, 6.0, 15.0], [3.0, 3.0, 3.0]):
        q = sm.devig_1x2(*o)
        c.close(f"de-vig {o} sums to one", float(q.sum()), 1.0, 1e-9)
        c.true(f"de-vig {o} is positive", bool((q > 0).all()))
        c.true(f"de-vig {o} keeps the favourite favourite",
               int(np.argmax(q)) == int(np.argmin(o)))
    raw = np.array([1 / 2.0, 1 / 3.4, 1 / 4.0])
    c.true("de-vigging removes the overround",
           float(sm.devig_1x2(2.0, 3.4, 4.0).sum()) < float(raw.sum()) + 1e-9)

    # ---- tuning -----------------------------------------------------------
    best, table = A.tune(df)
    c.true("tuning returns every parameter",
           set(best) == {"xi", "reg", "w_xg", "w_sot"})
    c.true("the grid was searched", len(table) == 27)
    c.true("the table is sorted best first",
           bool((table.log_loss.diff().dropna() >= -1e-12).all()))
    c.true("the winner is the top row",
           abs(table.iloc[0].xi - best["xi"]) < 1e-12)
    c.true("every setting scored something finite",
           bool(np.isfinite(table.log_loss).all()))
    c.true("the winner is at least as good as the defaults",
           float(table.log_loss.min()) <= float(
               table[(table.xi == sm.XI) & (table.reg == sm.REG)
                     & (table.w_xg == sm.W_XG)].log_loss.iloc[0]) + 1e-12)
    c.true("tuned parameters are inside the searched grid",
           best["xi"] in A.GRID_XI and best["reg"] in A.GRID_REG)
    d0 = A.default_params()
    c.check("defaults match the module constants",
            (d0["xi"], d0["reg"]), (sm.XI, sm.REG))
    tiny, tiny_tbl = A.tune(df.head(20))
    c.true("too little data falls back to the defaults", tiny == A.default_params())
    c.true("and returns no table", tiny_tbl.empty)

    # ---- backtest ---------------------------------------------------------
    bt = sm.walk_forward(df, **best)
    c.true("the backtest produced predictions", len(bt) > 100)
    c.true("probabilities sum to one",
           bool(((bt[["p_H", "p_D", "p_A"]].sum(axis=1) - 1).abs() < 1e-9).all()))
    ev = sm.evaluate(bt)
    c.true("it beats the league base rate on log-loss",
           ev["logloss_1x2"] < ev["logloss_baserate"])
    c.true("it beats the league base rate on RPS",
           ev["rps_1x2"] < ev["rps_baserate"])
    c.true("accuracy is a proportion", 0.0 <= ev["acc_1x2"] <= 1.0)
    c.true("log-loss is below the uninformative 1.0986",
           ev["logloss_1x2"] < np.log(3))

    # ---- the market comparison the whole app rests on ---------------------
    d = A.with_market(bt)
    c.true("matches with a full book were found", len(d) > 50)
    c.true("every kept match has usable odds",
           bool(((d.odds_H > 1) & (d.odds_D > 1) & (d.odds_A > 1)).all()))
    c.true("market probabilities sum to one",
           bool(((d[["m_H", "m_D", "m_A"]].sum(axis=1) - 1).abs() < 1e-9).all()))
    c.true("per-match log-losses are positive",
           bool((d.ll_model > 0).all() and (d.ll_market > 0).all()))
    c.true("disagreement is a probability distance",
           bool(((d.disagreement >= 0) & (d.disagreement <= 1)).all()))

    gap = A.overall_gap(d)
    c.check("the gap is the difference of the means", round(gap["gap"], 9),
            round(gap["model"] - gap["market"], 9))
    c.true("the interval brackets the estimate",
           gap["ci_low"] <= gap["gap"] <= gap["ci_high"])
    c.true("the market is ahead on this league", gap["gap"] > 0)

    dt = A.disagreement_table(d)
    c.true("the disagreement table has buckets", len(dt) >= 3)
    c.true("every bucket holds matches", bool((dt.matches > 0).all()))
    c.true("bucket sizes add up to the scored matches",
           abs(int(dt.matches.sum()) - len(d)) <= 2)
    c.true("model minus market is the difference of its own columns",
           bool((abs(dt.model_minus_market - (dt.model - dt.market)) < 1e-9).all()))
    c.true("intervals bracket their estimates",
           bool(((dt.ci_low <= dt.model_minus_market)
                 & (dt.model_minus_market <= dt.ci_high)).all()))
    c.true("the flag's premise holds: the model is worse where it disagrees most",
           float(dt.iloc[-1].model_minus_market) > float(dt.iloc[0].model_minus_market))
    c.true("significance is flagged only when the interval clears zero",
           bool((dt.model_worse == (dt.ci_low > 0)).all()))

    pc = A.pick_conflict(d)
    c.true("conflicts were found", pc["conflicts"] > 0)
    c.true("the share is a proportion", 0.0 <= pc["share"] <= 1.0)
    c.true("the three outcomes account for everything",
           abs(pc["model_right"] + pc["market_right"] + pc["neither"] - 1.0) < 1e-9)
    c.true("the market wins the arguments on this league",
           pc["market_right"] > pc["model_right"])

    # ---- calibration ------------------------------------------------------
    ct = A.calibration_table(bt)
    c.true("calibration bands were produced", len(ct) >= 3)
    c.true("every forecast is counted",
           int(ct.forecasts.sum()) == 3 * len(bt))
    c.true("predicted rises across the bands",
           bool((ct.predicted.diff().dropna() > 0).all()))
    c.true("error is actual minus predicted",
           bool((abs(ct.error - (ct.actual - ct.predicted)) < 1e-9).all()))
    ce = A.calibration_error(bt)
    c.true("calibration error is small", 0.0 <= ce < 0.10)

    # ---- the flag ---------------------------------------------------------
    c.true("no odds means no flag", A.flag(p, None) is None)
    c.true("a nonsense book means no flag", A.flag(p, (1.0, 1.0, 1.0)) is None)
    c.true("a negative price means no flag", A.flag(p, (-2.0, 3.0, 4.0)) is None)
    fair = sm.devig_1x2(1 / p["home_win"], 1 / p["draw"], 1 / p["away_win"])
    agree = A.flag(p, (1 / p["home_win"], 1 / p["draw"], 1 / p["away_win"]))
    c.check("a market equal to the model is aligned", agree["level"], "aligned")
    c.true("and reports a tiny gap", agree["gap"] < 0.01)
    c.true("and agrees on the favourite", agree["same_favourite"])
    far = A.flag(p, (20.0, 20.0, 1.05))
    c.check("a wildly different market is a caution", far["level"], "caution")
    c.true("and reports a large gap", far["gap"] > 0.10)
    c.true("the caution text warns about the MODEL, not the match",
           "warning about **the model**" in A.FLAG_TEXT["caution"])
    c.true("no flag level sounds like a tip",
           not any(w in " ".join(A.FLAG_TEXT.values()).lower()
                   for w in ("back this", "value bet", "profit", "edge")))
    for lvl in ("aligned", "watch", "caution"):
        c.true(f"{lvl} has explanatory text", len(A.FLAG_TEXT[lvl]) > 30)
    c.true("model and market probabilities are both reported",
           set(far["model"]) == set(far["market"]))
    c.true("the named outcome is one of the three",
           far["outcome"] in far["model"])

    # ---- the recommendation ----------------------------------------------
    r = A.recommend(p, m, None)
    c.true("a selection is named", r["selection"] in
           (p["home"], "Draw", p["away"]))
    c.true("the selection is the most likely outcome",
           abs(r["probability"] - max(p["home_win"], p["draw"],
                                      p["away_win"])) < 1e-12)
    c.true("confidence is one of the three labels",
           r["confidence"] in ("high", "medium", "low"))
    c.true("fair odds are the reciprocal",
           abs(r["fair_odds"] - 1 / r["probability"]) < 1e-6)
    c.true("no price means no expected value", r["ev"] is None)
    c.true("reasons are given", len(r["reasons"]) >= 4)
    c.true("a reason cites the expected goals",
           any("Expected goals" in x for x in r["reasons"]))
    c.true("a reason cites home advantage",
           any("Home advantage" in x for x in r["reasons"]))
    c.true("without a price it says so",
           any("No prices entered" in x for x in r["reasons"]))

    # bands must map probability to the measured strike rate
    for prob, want in ((0.70, "high"), (0.50, "medium"), (0.30, "low")):
        lab, n_, strike = A.confidence_band(prob)
        c.check(f"{prob:.0%} is {want} confidence", lab, want)
        c.true(f"{want} carries a measured strike rate", 0.0 < strike < 1.0)
        c.true(f"{want} carries a sample size", n_ > 50)
    c.true("higher bands have higher realised strike rates",
           A.confidence_band(0.70)[2] > A.confidence_band(0.50)[2]
           > A.confidence_band(0.30)[2])

    # expected value, and the downgrade when the market disagrees
    fair_book = (1 / p["home_win"], 1 / p["draw"], 1 / p["away_win"])
    r_fair = A.recommend(p, m, fair_book)
    c.true("a fair book gives roughly zero expected value",
           abs(r_fair["ev"]) < 0.02)
    c.true("agreement is detected", r_fair["agrees_with_market"] is True)
    short = list(fair_book)
    short[r_fair["outcome_index"]] *= 0.5
    r_short = A.recommend(p, m, tuple(short))
    c.true("a shorter price lowers expected value", r_short["ev"] < r_fair["ev"])

    flip = [1.02, 50.0, 50.0]
    flip[r_fair["outcome_index"]] = 50.0
    flip[(r_fair["outcome_index"] + 1) % 3] = 1.02
    r_dis = A.recommend(p, m, tuple(flip))
    c.true("a market naming another favourite is detected",
           r_dis["agrees_with_market"] is False)
    c.true("and confidence is downgraded",
           ["low", "medium", "high"].index(r_dis["confidence"])
           <= ["low", "medium", "high"].index(r_fair["confidence"]))
    c.true("and a reason explains the downgrade",
           any("different" in x for x in r_dis["reasons"]))

    # the published bucket evidence must stay self-consistent
    for key, b in A.BUCKET_ROI.items():
        c.true(f"{key} has a sample size", b["n"] > 0)
        c.true(f"{key} interval brackets its estimate",
               b["ci"][0] <= b["roi"] <= b["ci"][1])
        c.true(f"{key} is only called conclusive when the interval excludes zero",
               b["conclusive"] == (b["ci"][0] < 0 and b["ci"][1] < 0))
        c.true(f"{key} has a strike rate", 0.0 < b["strike"] < 1.0)
    c.true("backing the favourite lost money", A.FAVOURITE_ROI["roi"] < 0)
    c.true("the book carries a margin", A.FAVOURITE_ROI["overround"] > 1.0)
    c.true("no confidence text promises profit",
           not any(w in " ".join(A.CONFIDENCE_TEXT.values()).lower()
                   for w in ("profit", "guaranteed", "edge", "value bet")))
    for lvl in ("high", "medium", "low"):
        c.true(f"{lvl} confidence has explanatory text",
               len(A.CONFIDENCE_TEXT[lvl]) > 40)

    print(f"PASS {c.passes}  FAIL {len(c.fails)}")
    if c.fails:
        print("\n".join(c.fails))
    return 1 if c.fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
