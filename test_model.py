"""Regression tests for the model.

The maths is the product here, so most of these pin mathematical properties that
must hold regardless of the data: probabilities that sum to one inside each
race, a de-vig that actually removes the overround, a conditional logit that
recovers a signal it was given, and place probabilities that sum to the number
of places paid.

    python test_model.py     # expect: PASS <n>  FAIL 0
"""
import os

import numpy as np
import pandas as pd

import data as D
import models as M
import predict as P

_HERE = os.path.dirname(os.path.abspath(__file__))
RACE = "D:/01_PREDICTION MODELS/20260818-lingfield-r06.xlsx"


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
    rng = np.random.default_rng(0)

    # ---- Shin de-vig ------------------------------------------------------
    for odds in ([2.0, 3.0, 6.0], [1.5, 10.0, 12.0, 20.0],
                 [2.5] * 4, [1.2, 15.0, 40.0, 60.0, 90.0]):
        p = D.shin_devig(np.array(odds, dtype=float))
        c.close(f"de-vig of {odds} sums to 1", p.sum(), 1.0, 1e-9)
        c.true(f"de-vig of {odds} is all positive", bool((p > 0).all()))
        c.true(f"de-vig of {odds} keeps the favourite favourite",
               int(np.argmax(p)) == int(np.argmin(odds)))
        raw = 1 / np.array(odds, dtype=float)
        c.true(f"de-vig of {odds} shrinks the book",
               p.max() <= raw.max() / raw.sum() + 1e-9)
    fair = D.shin_devig(np.array([2.0, 2.0]))
    c.close("a fair two-horse book splits evenly", fair[0], 0.5, 1e-6)
    big = D.shin_devig(np.array([1.1, 30.0, 40.0]))
    c.close("a lopsided book still sums to 1", big.sum(), 1.0, 1e-9)

    # ---- grouping ---------------------------------------------------------
    rid = np.array(["a", "a", "a", "b", "b", "c"])
    sl = M.race_slices(rid)
    c.check("three races found", len(sl), 3)
    c.check("first race has three runners", sl[0].stop - sl[0].start, 3)
    c.check("last race has one runner", sl[2].stop - sl[2].start, 1)
    c.check("slices cover every row", sum(s.stop - s.start for s in sl), 6)

    s = np.array([1.0, 2.0, 3.0, 0.0, 0.0, 5.0])
    p = M.softmax_by_race(s, sl)
    for i, sc in enumerate(sl):
        c.close(f"race {i} softmax sums to 1", p[sc].sum(), 1.0, 1e-12)
    c.true("softmax preserves ordering", p[2] > p[1] > p[0])
    c.close("a one-runner race gets probability 1", p[5], 1.0, 1e-12)
    q = M.normalise_by_race(np.array([2.0, 2.0, 4.0, 1.0, 3.0, 9.0]), sl)
    c.close("normalise sums to 1", q[0] + q[1] + q[2], 1.0, 1e-12)
    c.true("normalise keeps proportions", abs(q[2] / q[0] - 2.0) < 1e-9)

    # ---- conditional logit recovers a signal it was given -----------------
    n_races, k = 400, 3
    rows, ys, rids = [], [], []
    beta_true = np.array([1.5, -0.8, 0.0])
    for r in range(n_races):
        m = rng.integers(6, 11)
        X = rng.normal(size=(m, k))
        u = X @ beta_true + rng.gumbel(size=m)
        w = int(np.argmax(u))
        rows.append(X)
        ys.append(np.eye(m, dtype=int)[w])
        rids += [f"r{r}"] * m
    X = np.vstack(rows)
    y = np.concatenate(ys)
    cl = M.ConditionalLogit(l2=1e-4).fit(X, y, np.array(rids))
    c.true("logit converged", cl.converged_)
    c.true("logit recovers a positive weight", cl.beta_[0] > 0.8)
    c.true("logit recovers a negative weight", cl.beta_[1] < -0.4)
    c.true("logit leaves a useless feature near zero", abs(cl.beta_[2]) < 0.35)
    pp = cl.predict_proba(X, np.array(rids))
    slx = M.race_slices(np.array(rids))
    for sc in slx[:20]:
        c.close("logit probabilities sum to 1 per race", pp[sc].sum(), 1.0, 1e-9)
    ll_fit = M.race_logloss(pp, y, slx)
    ll_flat = M.race_logloss(
        M.softmax_by_race(np.zeros(len(y)), slx), y, slx)
    c.true("a fitted logit beats a uniform guess", ll_fit < ll_flat)

    # ---- blending ---------------------------------------------------------
    a = M.softmax_by_race(rng.normal(size=len(y)), slx)
    b = M.softmax_by_race(rng.normal(size=len(y)), slx)
    bl = M.blend_log([a, b], np.array([0.5, 0.5]), slx)
    for sc in slx[:20]:
        c.close("blend sums to 1 per race", bl[sc].sum(), 1.0, 1e-9)
    only_a = M.blend_log([a, b], np.array([1.0, 0.0]), slx)
    c.true("weight 1 on one model reproduces it", np.allclose(only_a, a, atol=1e-9))
    c.true("blend weights are renormalised",
           np.allclose(M.blend_log([a, b], np.array([2.0, 2.0]), slx), bl, atol=1e-9))
    grid = list(M._simplex_grid(3, 6))
    c.true("simplex grid vectors all sum to 1",
           all(abs(g.sum() - 1) < 1e-9 for g in grid))
    c.true("simplex grid includes the corners",
           any(np.allclose(g, [1, 0, 0]) for g in grid))
    w = M.fit_blend_weights([a, b], y, slx)
    c.close("fitted weights sum to 1", w.sum(), 1.0, 1e-9)
    c.true("fitted weights are non-negative", bool((w >= 0).all()))

    # ---- Plackett-Luce ----------------------------------------------------
    pw = M.softmax_by_race(np.array([2.0, 1.0, 0.5, 0.0, -1.0]), [slice(0, 5)])
    for k_ in (1, 2, 3):
        pl = M.place_probabilities(pw, [slice(0, 5)], np.full(5, k_), sims=40000)
        c.close(f"place probabilities sum to {k_}", pl.sum(), k_, 0.05)
        c.true(f"places={k_}: place chance >= win chance", bool((pl >= pw - 0.02).all()))
        c.true(f"places={k_}: ordering follows win probability",
               bool(np.all(np.diff(pl) <= 0.02)))
    one = M.place_probabilities(pw, [slice(0, 5)], np.full(5, 1), sims=40000)
    c.true("places=1 reproduces the win probabilities",
           np.allclose(one, pw, atol=0.02))
    solo = M.place_probabilities(np.array([1.0]), [slice(0, 1)], np.array([1]))
    c.close("a lone runner always places", solo[0], 1.0)

    # ---- metrics ----------------------------------------------------------
    perfect = np.array([1 - 1e-9, 5e-10, 5e-10])
    c.true("log-loss of a perfect call is ~0",
           M.race_logloss(perfect, np.array([1, 0, 0]), [slice(0, 3)]) < 1e-6)
    third = M.race_logloss(np.array([1/3, 1/3, 1/3]), np.array([1, 0, 0]),
                           [slice(0, 3)])
    c.close("log-loss of a uniform 3-way is log 3", third, np.log(3), 1e-9)
    c.check("top-1 hit", M.top_k_hit(np.array([.6, .3, .1]), np.array([1, 0, 0]),
                                     [slice(0, 3)], 1), 1.0)
    c.check("top-1 miss", M.top_k_hit(np.array([.1, .3, .6]), np.array([1, 0, 0]),
                                      [slice(0, 3)], 1), 0.0)
    c.check("top-3 catches a third pick",
            M.top_k_hit(np.array([.1, .3, .6]), np.array([1, 0, 0]),
                        [slice(0, 3)], 3), 1.0)

    # ---- features are genuinely within-race -------------------------------
    fake = pd.DataFrame({
        "race_id": ["a"] * 4 + ["b"] * 3,
        "field_size": [4] * 4 + [3] * 3,
        "Handicap Rating": [70, 80, 90, 60, 50, 55, 45],
        "Career Runs": [10, 20, 30, 40, 5, 6, 7],
        "Prize Money": [1000] * 4 + [2000] * 3,     # constant inside each race
    })
    X, names = D.build_features(fake, cols=["Handicap Rating", "Career Runs",
                                            "Prize Money"], with_market=False)
    zr = X["z_Handicap Rating"].to_numpy()
    c.close("z-scores have mean 0 in race a", zr[:4].mean(), 0.0, 1e-9)
    c.close("z-scores have mean 0 in race b", zr[4:].mean(), 0.0, 1e-9)
    c.true("the best-rated horse has the top z-score in its race",
           int(np.argmax(zr[:4])) == 2)
    c.true("a race-constant column is passed through raw, not zeroed",
           "race_Prize Money" in names)
    c.true("and keeps its value", X["race_Prize Money"].iloc[0] == 1000)
    c.true("ranks are percentiles", bool(((X["r_Career Runs"] > 0)
                                          & (X["r_Career Runs"] <= 1)).all()))

    # ---- the shipped bundle ----------------------------------------------
    B = P.load_bundle(os.path.join(_HERE, "model_bundle.joblib"))
    c.true("bundle carries feature names", len(B["feature_names"]) > 100)
    c.check("logit has one weight per feature", len(B["logit_beta"]),
            len(B["feature_names"]))
    c.close("blend weights sum to 1", float(np.sum(B["weights"])), 1.0, 1e-9)
    c.close("fundamental weights sum to 1",
            float(np.sum(B["weights_fundamentals"])), 1.0, 1e-9)
    c.check("four blend weights with the market", len(B["weights"]), 4)
    c.check("three without it", len(B["weights_fundamentals"]), 3)
    V = B["validation"]
    c.true("the blend beat the market on log-loss",
           V["blend_logloss"] < V["market_logloss"])
    c.close("the stated edge matches the two log-losses",
            V["market_logloss"] - V["blend_logloss"], V["logloss_edge"], 1e-4)
    c.true("the edge interval excludes zero", V["edge_ci"][0] > 0)
    c.true("the interval brackets the estimate",
           V["edge_ci"][0] <= V["logloss_edge"] <= V["edge_ci"][1])
    c.true("the ROI interval brackets the estimate",
           V["value_roi_ci"][0] <= V["value_roi"] <= V["value_roi_ci"][1])
    c.true("ROI degrades as the price worsens",
           V["value_roi"] > V["roi_at_5pct_worse"] > V["roi_at_10pct_worse"]
           > V["roi_at_20pct_worse"])
    c.true("the honest note that top-1 is worse than the market",
           V["blend_top1"] < V["market_top1"])

    # ---- end to end on a real race ---------------------------------------
    if os.path.exists(RACE):
        race = pd.read_excel(RACE)
        t = P.score_race(race, B, sims=4000)
        n = len(race)
        c.check("one row per runner", len(t), n)
        c.close("win probabilities sum to 100", t["Win %"].sum(), 100.0, 1e-6)
        c.true("ranked best first",
               bool(np.all(np.diff(t["Win %"].to_numpy()) <= 1e-9)))
        c.check("ranks are 1..n", list(t["Rank"]), list(range(1, n + 1)))
        c.true("every probability is positive", bool((t["Win %"] > 0).all()))
        c.close("place probabilities sum to the places paid",
                t["Place %"].sum() / 100.0, t.attrs["places_paid"], 0.05)
        c.true("place chance is at least win chance",
               bool((t["Place %"] >= t["Win %"] - 0.5).all()))
        c.true("fair price is the reciprocal",
               np.allclose(t["Fair $"], 100.0 / t["Win %"], rtol=1e-6))
        c.true("edge equals probability x odds - 1",
               np.allclose(t["Edge"], t["Win %"] / 100 * t["Odds"] - 1, atol=1e-9))
        c.true("the market blend was used on a fully priced race",
               "market" in t.attrs["mode"])
        # turning the market off must change the answer, and only that
        t2 = P.score_race(race, B, use_market=False, sims=4000)
        c.true("fundamentals-only mode is reported", "fundamental" in t2.attrs["mode"])
        c.close("and still sums to 100", t2["Win %"].sum(), 100.0, 1e-6)
        c.true("the two modes disagree somewhere",
               not np.allclose(sorted(t["Win %"]), sorted(t2["Win %"]), atol=1e-6))
        # a race with no book must fall back rather than fail
        noodds = race.copy()
        noodds[D.ODDS] = np.nan
        t3 = P.score_race(noodds, B, sims=4000)
        c.true("a race with no prices falls back to fundamentals",
               "fundamental" in t3.attrs["mode"])
        c.close("and still produces a distribution", t3["Win %"].sum(), 100.0, 1e-6)
    else:
        print("  (race file not found — skipped the end-to-end checks)")

    # ---- attrs must survive the trip to the browser -----------------------
    # Streamlit ships a DataFrame as Arrow, and pandas writes `DataFrame.attrs`
    # into the Arrow schema metadata AS JSON. Python's json.loads accepts NaN;
    # JavaScript's JSON.parse does not. A NaN here therefore fails nowhere in
    # Python and blows up client-side as an unreadable SyntaxError mid-page.
    # This is the regression test for that.
    import re

    import pyarrow as pa

    def js_safe(frame):
        md = pa.Table.from_pandas(frame).schema.metadata
        if not md or b"pandas" not in md:
            return True, ""
        raw = md[b"pandas"].decode()
        bad = re.search(r":\s*(NaN|Infinity|-Infinity)\b", raw)
        return (bad is None), (bad.group(0) if bad else "")

    if os.path.exists(RACE):
        race_full = pd.read_excel(RACE)
        # a race where ONE runner has no price - the case that broke the app
        partial = race_full.copy()
        partial.loc[partial.index[0], D.ODDS] = np.nan
        for label, frame in (("fully priced", race_full), ("partly priced", partial)):
            tt = P.score_race(frame, B, sims=3000)
            c.true(f"{label}: no NaN anywhere in attrs",
                   not any(isinstance(v, float) and np.isnan(v)
                           for v in tt.attrs.values()))
            shown = tt.drop(columns=[x for x in tt.columns if x.startswith("_")])
            ok_js, found = js_safe(shown)
            c.true(f"{label}: arrow metadata is valid JavaScript JSON{'' if ok_js else ' — found '+found}",
                   ok_js)
        pt = P.score_race(partial, B, sims=3000)
        c.true("a partly priced race falls back to fundamentals",
               pt.attrs["has_book"] is False)
        c.true("and reports overround as None, not NaN",
               pt.attrs["overround"] is None)
        c.check("and counts the priced runners",
                pt.attrs["priced_runners"], len(race_full) - 1)
        c.true("and is flagged as outside the validated configuration",
               pt.attrs["validated_config"] is False)
        ft = P.score_race(race_full, B, sims=3000)
        c.true("a fully priced race reports a real overround",
               isinstance(ft.attrs["overround"], float)
               and ft.attrs["overround"] > 1.0)
        c.true("and is the validated configuration",
               ft.attrs["validated_config"] is True)
        c.true("turning the market off leaves the validated flag false",
               P.score_race(race_full, B, use_market=False,
                            sims=3000).attrs["validated_config"] is False)

    # ---- the CSV column-shift trap ----------------------------------------
    # The CSV export ends every data row with a trailing comma, so rows carry
    # 129 fields against a 128-name header. Pandas resolves that by promoting
    # the first column to the index, shifting every column one left: 'Horse
    # Name' ends up holding ages and 'Best Fixed Odds' holding carried weights.
    # Nothing raises. This is the regression test for that silent corruption.
    import csv as _csv
    import tempfile

    hdr = ["Num", "Horse Name", "Age", "Best Fixed Odds"]
    rows = [[1, "Alpha", 5, 2.5], [2, "Bravo", 4, 3.5], [3, "Charlie", 6, 9.0]]
    with tempfile.TemporaryDirectory() as tmp:
        good = os.path.join(tmp, "good.csv")
        with open(good, "w", newline="", encoding="utf-8") as fh:
            w = _csv.writer(fh)
            w.writerow(hdr)
            w.writerows(rows)
        trap = os.path.join(tmp, "trap.csv")
        with open(trap, "w", encoding="utf-8") as fh:
            fh.write(",".join(hdr) + "\n")
            for r in rows:                       # the trailing comma
                fh.write(",".join(str(x) for x in r) + ",\n")

        g = D.read_race_file(good)
        c.check("a clean csv reads its names", list(g["Horse Name"]),
                ["Alpha", "Bravo", "Charlie"])
        c.check("and its odds", list(g["Best Fixed Odds"]), [2.5, 3.5, 9.0])

        naive = pd.read_csv(trap)
        c.true("the trap really does corrupt a naive read",
               list(naive["Horse Name"]) != ["Alpha", "Bravo", "Charlie"])

        t_ = D.read_race_file(trap)
        c.check("the safe reader recovers the names", list(t_["Horse Name"]),
                ["Alpha", "Bravo", "Charlie"])
        c.check("and the odds", list(t_["Best Fixed Odds"]), [2.5, 3.5, 9.0])

        # a genuinely misaligned file must be refused, not silently scored
        broken = os.path.join(tmp, "broken.csv")
        with open(broken, "w", encoding="utf-8") as fh:
            fh.write("Horse Name,Age\n")
            fh.write("5,Alpha\n6,Bravo\n7,Charlie\n")
        try:
            D.read_race_file(broken)
            c.true("numeric horse names are rejected", False)
        except ValueError as exc:
            c.true("numeric horse names are rejected", "misaligned" in str(exc))

    c.true("an xlsx path still routes to the excel reader",
           D.read_race_file(RACE)["Horse Name"].notna().all()
           if os.path.exists(RACE) else True)

    # ---- jockey-only mode -------------------------------------------------
    jc = D.jockey_columns(D.load()) if False else B["jockey_columns"]
    c.true("the jockey column set is a real subset",
           0 < len(jc) < len(B["base_columns"]))
    c.true("every jockey column is about the jockey",
           all(x.lower().startswith("jockey") for x in jc))
    c.true("nothing about the horse leaked in",
           not any(k in x.lower() for x in jc
                   for k in ("career", "track", "distance", "trainer",
                             "barrier", "weight carried", "handicap")))
    c.true("the apprentice claim is included",
           any("claim" in x.lower() for x in jc))
    c.check("jockey logit has one weight per jockey feature",
            len(B["jockey_logit_beta"]), len(B["jockey_feature_names"]))
    c.close("jockey blend weights sum to one",
            float(np.sum(B["jockey_weights"])), 1.0, 1e-9)
    c.close("jockey market weights sum to one",
            float(np.sum(B["jockey_weights_market"])), 1.0, 1e-9)

    Jv = B["jockey_validation"]
    c.true("jockey beats a dart throw", Jv["top1"] > Jv["dart_throw_top1"])
    c.true("by roughly double", Jv["top1"] > 1.7 * Jv["dart_throw_top1"])
    c.true("but loses to the market", Jv["top1"] < Jv["market_top1"])
    c.true("and to the full model", Jv["top1"] < Jv["full_top1"])
    c.true("its log-loss is worse than the market's",
           Jv["logloss"] > Jv["market_logloss"])
    c.true("blending put most weight on the market",
           Jv["market_weight_when_blended"] >= 0.85)
    c.true("top-3 beats top-1", Jv["top3"] > Jv["top1"])
    c.true("the drivers are named", len(Jv["top_drivers"]) >= 4)
    c.true("ROI leads the drivers", "ROI" in Jv["top_drivers"][0])

    if os.path.exists(RACE):
        race_x = D.read_race_file(RACE)
        tj = P.score_race(race_x, B, feature_set="jockey", sims=3000)
        ta = P.score_race(race_x, B, feature_set="all", sims=3000)
        c.check("jockey mode scores every runner", len(tj), len(race_x))
        c.close("jockey probabilities sum to 100", tj["Win %"].sum(), 100.0, 1e-6)
        c.true("jockey mode is ranked", list(tj["Rank"]) == sorted(tj["Rank"]))
        c.check("the mode is reported", tj.attrs["feature_set"], "jockey")
        c.check("and for the full model", ta.attrs["feature_set"], "all")
        c.true("jockey mode is never the validated configuration",
               tj.attrs["validated_config"] is False)
        c.true("the two modes actually differ",
               not np.allclose(sorted(tj["Win %"]), sorted(ta["Win %"]), atol=1e-6))
        try:
            P.score_race(race_x, B, feature_set="trainer")
            c.true("an unknown feature set is rejected", False)
        except ValueError:
            c.true("an unknown feature set is rejected", True)

    print(f"PASS {c.passes}  FAIL {len(c.fails)}")
    if c.fails:
        print("\n".join(c.fails))
    return 1 if c.fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
