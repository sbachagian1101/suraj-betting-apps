"""Regression tests for RacingScorePredictor.

The point of this app is fidelity to a written specification, so most of these
check the framework's own rules rather than the parser's output:

* base weights are the document's, and adjusted weights always renormalise to
  exactly 100 for every distance band, surface, race type and profile;
* a missing category leaves the denominator instead of scoring zero - the rule
  the document is most emphatic about, and the easiest to get wrong;
* untested going is neutral, not a penalty;
* class evidence is taken once, not stacked;
* odds never move the score.

    python test_scoring.py     # expect: PASS <n>  FAIL 0
"""
import copy
import os

import numpy as np

import rs_parser as rp
import scoring as sc

_HERE = os.path.dirname(os.path.abspath(__file__))
FIXTURE = os.path.join(_HERE, "tests_fixture_greyville_r7.txt")


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

    def close(self, label, got, want, tol=1e-6):
        self.check(label, got, want, abs(float(got) - float(want)) <= tol)


def main():
    c = Checker()
    header, runners, warns = rp.parse(
        open(FIXTURE, encoding="utf-8", errors="replace").read())
    active = [r for r in runners if not r.get("scratched")]

    # ---- the parse the scoring rests on ----------------------------------
    c.check("track read", header["track"], "Greyville")
    c.check("race number read", header["race_no"], 7)
    c.check("distance read", header["distance_m"], 1200)
    c.check("surface read", header["surface"], "TURF")
    c.check("going read", header["going"], "SOFT")
    c.check("ten runners on the page", len(runners), 10)
    c.check("one scratching detected", len(runners) - len(active), 1)
    c.check("the scratching is PEMBURY",
            next(r["horse"] for r in runners if r["scratched"]), "PEMBURY")
    c.true("every active runner has a name",
           all(r["horse"] for r in active))
    c.true("tabs are unique", len({r["tab"] for r in runners}) == len(runners))

    # OHR feeds the second-heaviest category, so its recovery is pinned.
    c.true("every active runner has an official rating",
           all(sc._f(r.get("ohr")) > 0 for r in active))
    lupin = next(r for r in runners if r["tab"] == 1)
    c.check("per-run ratings recovered in order",
            [x.get("ohr") for x in lupin["recent_runs"]], [62, 63, 65, 66, 67])
    c.check("preparation run numbers recovered",
            [x.get("prep_run") for x in lupin["recent_runs"]], [21, 20, 19, 18, 17])
    c.check("long-form dates converted to days",
            [x.get("days_ago") for x in lupin["recent_runs"]],
            [40, 51, 67, 86, 133])
    c.check("apprentice claim read",
            next(r["claim"] for r in runners if r["tab"] == 4), 4.0)
    c.true("head-to-head meetings found",
           sum(len(r.get("h2h") or []) for r in runners) > 20)
    m = lupin["h2h"][0]
    c.check("a h2h meeting holds two runners", len(m["runners"]), 2)
    c.true("h2h carries finish, margin and weight",
           all(k in m["runners"][0] for k in ("finish", "horse", "margin", "weight")))

    # ---- the document's own numbers --------------------------------------
    c.check("fifteen categories", len(sc.CATEGORIES), 15)
    c.close("base weights total 100", sum(sc.BASE_WEIGHTS.values()), 100.0)
    for k, w in (("recent_form", 15), ("ability", 14), ("pace", 10),
                 ("class_", 9), ("distance", 9), ("surface", 8),
                 ("sectionals", 6), ("weight", 6), ("h2h", 6), ("course", 4),
                 ("barrier", 3), ("fitness", 3), ("jockey", 3), ("trainer", 2),
                 ("trip", 2)):
        c.check(f"base weight of {k}", sc.BASE_WEIGHTS[k], w)

    c.check("sprint band", sc.distance_band(1200), "<=1200")
    c.check("mile band", sc.distance_band(1600), "1201-1600")
    c.check("middle band", sc.distance_band(2000), "1601-2000")
    c.check("staying band", sc.distance_band(2400), ">2000")
    c.check("AW is synthetic", sc.surface_key("AW"), "SYNTHETIC")
    c.check("turf is turf", sc.surface_key("TURF"), "TURF")
    c.check("dirt is dirt", sc.surface_key("DIRT"), "DIRT")

    # section 4 spot-checks against the printed table
    c.check("draw matters most in sprints",
            sc.DISTANCE_MULT["<=1200"]["barrier"], 1.35)
    c.check("draw matters least when staying",
            sc.DISTANCE_MULT[">2000"]["barrier"], 0.6)
    c.check("stamina peaks over a trip",
            sc.DISTANCE_MULT[">2000"]["distance"], 1.35)
    c.check("pace peaks in sprints", sc.DISTANCE_MULT["<=1200"]["pace"], 1.25)
    c.true("the mile band is the unmodified baseline",
           all(v == 1.0 for v in sc.DISTANCE_MULT["1201-1600"].values()))

    # ---- weights always renormalise to exactly 100 -----------------------
    for band, dist in (("<=1200", 1000), ("1201-1600", 1400),
                       ("1601-2000", 1800), (">2000", 2400)):
        for surf in ("TURF", "AW", "DIRT"):
            for rt in ("handicap", "wfa", "group", "maiden"):
                for prof in ("overall", "win", "place"):
                    w = sc.adjusted_weights(
                        {"distance_m": dist, "surface": surf}, prof, rt)
                    c.close(f"{band}/{surf}/{rt}/{prof} totals 100",
                            sum(w.values()), 100.0, 1e-9)
                    c.true(f"{band}/{surf}/{rt}/{prof} all positive",
                           all(v > 0 for v in w.values()))
    base = sc.adjusted_weights({"distance_m": 1400, "surface": "TURF"},
                               "overall", "handicap")
    wfa = sc.adjusted_weights({"distance_m": 1400, "surface": "TURF"},
                              "overall", "wfa")
    c.true("weight-for-age cuts the weight category",
           wfa["weight"] < base["weight"] * 0.6)
    grp = sc.adjusted_weights({"distance_m": 1400, "surface": "TURF"},
                              "overall", "group")
    c.true("group races lift ability", grp["ability"] > base["ability"])
    win = sc.adjusted_weights({"distance_m": 1400, "surface": "TURF"}, "win")
    plc = sc.adjusted_weights({"distance_m": 1400, "surface": "TURF"}, "place")
    c.true("win profile leans on ability", win["ability"] > plc["ability"])
    c.true("place profile leans on head-to-head", plc["h2h"] > win["h2h"])
    c.true("place profile leans on suitability", plc["distance"] > win["distance"])

    # ---- the missing-data rule, which the document is emphatic about -----
    acc = sc.Acc()
    s, ok, _ = acc.out()
    c.true("a category with no evidence is unavailable", not ok)
    c.check("and scores zero rather than a number", s, 0.0)
    acc2 = sc.Acc()
    acc2.add(3.0, 6.0)
    s2, ok2, _ = acc2.out()
    c.true("a category with evidence is available", ok2)
    c.close("and is scaled to ten", s2, 5.0)
    acc3 = sc.Acc()
    acc3.add(2.0, 4.0)
    acc3.skip("nothing for this sub-parameter")
    s3, _, _ = acc3.out()
    c.close("a skipped sub-parameter leaves the denominator", s3, 5.0)

    res = sc.score_race(runners, header)
    rows = res["rows"]
    c.check("only active runners are scored", len(rows), len(active))
    c.true("no scratched runner is scored",
           "PEMBURY" not in {r["horse"] for r in rows})

    for row in rows:
        num = sum(x["contribution"] for x in row["categories"].values())
        den = row["available_weight"]
        c.close(f"#{row['tab']} final = 100 x contributions / available weight",
                row["final"], 100.0 * num / den, 1e-9)
        c.true(f"#{row['tab']} unavailable categories contribute nothing",
               all(x["contribution"] == 0.0
                   for x in row["categories"].values() if not x["available"]))
        c.true(f"#{row['tab']} available weight excludes missing categories",
               abs(den - sum(x["weight"] for x in row["categories"].values()
                             if x["available"])) < 1e-9)
        c.true(f"#{row['tab']} score is on the 0-100 scale",
               0.0 <= row["final"] <= 100.0)
        c.true(f"#{row['tab']} every category score is 0-10",
               all(0.0 <= x["score"] <= 10.0 + 1e-9
                   for x in row["categories"].values()))
        c.true(f"#{row['tab']} confidence is a percentage",
               0.0 <= row["confidence"] <= 100.0)

    c.true("ranked best first",
           all(rows[i]["final"] >= rows[i + 1]["final"] for i in range(len(rows) - 1)))
    c.check("ranks are 1..n", [r["rank"] for r in rows],
            list(range(1, len(rows) + 1)))
    c.close("the best horse indexes at 100", rows[0]["field_index"], 100.0, 1e-9)
    c.true("no index exceeds 100", all(r["field_index"] <= 100.0 + 1e-9 for r in rows))

    # dropping a category must rescale, not punish
    solo = copy.deepcopy(active[0])
    solo["Soft_starts"] = solo["Soft_win"] = solo["Soft_plc"] = 0
    solo["Good_starts"] = solo["Firm_starts"] = solo["Heavy_starts"] = 0
    ctx = sc.field_context(active, header)
    w = sc.adjusted_weights(header)
    scored = sc.score_runner(solo, ctx, w, 2.0)
    c.true("an untested horse still scores a going category",
           scored["categories"]["surface"]["available"])
    c.true("untested going is neutral, not zero",
           scored["categories"]["surface"]["score"] > 2.0)

    # ---- class evidence is taken once, never stacked ---------------------
    c.true("G1 outranks G3", sc.class_rank("G1") > sc.class_rank("G3"))
    c.true("a benchmark ladder is ordered",
           sc.class_rank("BM90") > sc.class_rank("BM64"))
    c.true("C1 outranks C5", sc.class_rank("C1") > sc.class_rank("C5"))
    c.true("a maiden is the bottom", sc.class_rank("MDN") <= sc.class_rank("C5"))
    c.true("an unreadable class is None", sc.class_rank("???") is None)
    stacked = copy.deepcopy(active[0])
    for run in stacked["recent_runs"]:
        run["finish"] = 1
        run["race_class"] = "G1"
    s_cls, ok_cls, _ = sc.cat_class(stacked, ctx, 2.0)
    c.true("class is available with evidence", ok_cls)
    c.true("five higher-class wins do not exceed the category maximum",
           s_cls <= 10.0 + 1e-9)

    # ---- the market must never touch the score ---------------------------
    priced = copy.deepcopy(runners)
    for r in priced:
        r["tab_odds"] = 1.01
        r["bf_odds"] = 1.01
    res2 = sc.score_race(priced, header)
    c.true("rewriting every price leaves every score identical",
           all(abs(a["final"] - b["final"]) < 1e-9
               for a, b in zip(res["rows"], res2["rows"])))

    # ---- pace, per section 7 ---------------------------------------------
    c.check("five running styles", len(sc.RUNNING_STYLES), 5)
    fake = [{"tab": i, "horse": f"H{i}", "recent_runs":
             [{"settle_pos": p, "field_size": 10, "finish": 5}]}
            for i, p in enumerate([1, 1, 2, 5, 9], start=1)]
    ctx2 = sc.field_context(fake, header)
    exp = (1.0 * sum(1 for s in ctx2["styles"].values() if s == 1)
           + 0.7 * sum(1 for s in ctx2["styles"].values() if s == 2)
           + 0.35 * sum(1 for s in ctx2["styles"].values() if s == 3))
    c.close("pace pressure index matches the published formula", ctx2["ppi"], exp)
    c.check("a horse that leads is a Leader",
            sc.running_style({"recent_runs": [
                {"settle_pos": 1, "field_size": 12, "finish": 1}]}), 1)
    c.check("a horse that trails is a Backmarker",
            sc.running_style({"recent_runs": [
                {"settle_pos": 12, "field_size": 12, "finish": 6}]}), 5)
    c.check("no in-running data defaults to Midfield",
            sc.running_style({"recent_runs": []}), 4)

    # ---- confidence, per section 16 --------------------------------------
    allok = {k: True for k in sc.BASE_WEIGHTS}
    deb = sc.confidence({"Car_starts": 0, "dslr": 20}, allok)
    exp20 = sc.confidence({"Car_starts": 25, "dslr": 20}, allok)
    c.true("a debutant is less confident than an exposed horse", deb < exp20)
    c.true("a debutant sits in the document's band", 25 <= deb <= 40)
    half = sc.confidence({"Car_starts": 25, "dslr": 20},
                         {k: (i % 2 == 0) for i, k in enumerate(sc.BASE_WEIGHTS)})
    c.true("less evidence means less confidence", half < exp20)
    c.true("confidence never reaches certainty", exp20 <= 97.0)

    # ---- probabilities ----------------------------------------------------
    p = sc.win_probabilities(rows, spread=14.0)
    c.close("probabilities sum to one", float(p.sum()), 1.0, 1e-9)
    c.true("the top-rated horse is the most likely", int(np.argmax(p)) == 0)
    c.true("all probabilities are positive", bool((p > 0).all()))
    tight = sc.win_probabilities(rows, spread=5.0)
    wide = sc.win_probabilities(rows, spread=30.0)
    c.true("a tighter spread concentrates probability", tight[0] > wide[0])
    c.check("an empty field returns nothing", len(sc.win_probabilities([])), 0)

    # ---- degenerate input -------------------------------------------------
    c.check("a single runner cannot be ranked",
            sc.score_race(active[:1], header)["rows"], [])
    c.check("an empty field cannot be ranked",
            sc.score_race([], header)["rows"], [])
    blank = {"tab": 99, "horse": "BLANK", "recent_runs": [], "h2h": []}
    ctx3 = sc.field_context([blank, active[0]], header)
    got = sc.score_runner(blank, ctx3, w, 2.0)
    c.true("a horse with no form still produces a score",
           0.0 <= got["final"] <= 100.0)
    c.true("and is marked low confidence", got["confidence"] < 60.0)

    print(f"PASS {c.passes}  FAIL {len(c.fails)}")
    if c.fails:
        print("\n".join(c.fails))
    return 1 if c.fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
