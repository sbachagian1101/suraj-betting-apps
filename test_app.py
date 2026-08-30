"""App-level checks. Run: python test_app.py

A Streamlit app with a fatal error deeper in the script still serves HTTP 200
and still renders its first tab, so the tabs that do the work are driven here.
"""
from __future__ import annotations

import warnings

warnings.filterwarnings("ignore")

import numpy as np
from streamlit.testing.v1 import AppTest


def main():
    fails, passes = [], 0

    def check(name, cond):
        nonlocal passes
        if cond:
            passes += 1
        else:
            fails.append(name)

    def bail(at):
        for e in at.exception:
            print("EXCEPTION:", e.value)
        print(f"PASS {passes}  FAIL {len(fails)}")
        return 1

    at = AppTest.from_file("app.py", default_timeout=300).run()
    check("the app starts without an exception", not at.exception)
    if at.exception:
        return bail(at)

    check("it names itself", any("Match Insight" in t.value for t in at.title))
    check("four tabs are present", len(at.tabs) >= 4)
    check("it asks for a paste first", len(at.info) >= 1)
    check("a text area is offered", len(at.text_area) >= 1)

    # ---------------------------------------------- load the bundled example
    labels = [b.label for b in at.button]
    check("a sample loader is offered",
          any("bundled example" in l for l in labels))
    at.button[0].click().run()
    check("loading the example does not raise", not at.exception)
    if at.exception:
        return bail(at)

    check("matches were parsed and shown", len(at.dataframe) >= 3)
    check("the teams can be chosen", len(at.selectbox) >= 2)
    opts = at.selectbox[0].options
    check("both teams are selectable", len(opts) >= 2)

    body = " ".join([m.value for m in at.markdown] + [c.value for c in at.caption])
    check("the parsed-data tab explains the indices",
          "Attack strength" in body or "attack strength" in body.lower())
    check("shrinkage is disclosed", "shrunk" in body.lower())

    # ------------------------------------------------------ predict button
    pb = [i for i, b in enumerate(at.button) if "Predict" in b.label]
    check("a Predict match button exists", len(pb) == 1)
    if not pb:
        return bail(at)
    at.button[pb[0]].click().run()
    check("pressing Predict does not raise", not at.exception)
    if at.exception:
        return bail(at)

    check("it confirms the prediction is ready",
          any("Prediction ready" in s.value for s in at.success))

    body = " ".join(
        [m.value for m in at.markdown] + [c.value for c in at.caption]
        + [w.value for w in at.warning] + [s.value for s in at.success])
    for market in ("Both teams to score", "Over 2.5", "Correct score",
                   "Corners", "What each method said"):
        check(f"the {market} section rendered", market in body)

    check("the disagreement caveat is shown",
          "not* a confidence interval" in body or
          "not a confidence interval" in body.lower())

    # the per-method table must be complete and coherent
    tables = [d.value for d in at.dataframe]
    meth = [t for t in tables if "Method" in getattr(t, "columns", [])]
    check("a per-method table rendered", len(meth) == 1)
    if meth:
        mt = meth[0]
        check("at least ten methods are listed", len(mt) >= 10)
        s = mt["1"] + mt["X"] + mt["2"]
        check("every method's 1X2 sums to 100",
              bool((s.sub(100).abs() < 0.5).all()))
        check("no method reports a negative probability",
              bool((mt[["1", "X", "2"]] >= -1e-9).all().all()))

    # the score matrix must be a real distribution
    grids = [t for t in tables
             if hasattr(t, "shape") and t.shape[0] >= 8 and t.shape[1] >= 8
             and all(str(c)[-1].isdigit() for c in t.columns)]
    check("a correct-score grid rendered", len(grids) >= 1)
    if grids:
        g = np.asarray(grids[0], dtype=float)
        check("the grid sums to about 100%", abs(g.sum() - 100.0) < 1.0)
        check("no negative cells in the grid", bool((g >= -1e-9).all()))

    check("the derived-indices table renders without an Arrow type error",
          any("Attack strength" in list(getattr(t, "index", []))
              for t in tables))

    # ------------------------------------------------- changing the weights
    at.slider[0].set_value(20).run()
    check("changing the recency half-life does not raise", not at.exception)
    at.slider[3].set_value(0.0).run()
    check("zeroing the friendly weight does not raise", not at.exception)

    # ---------------------------------------------------------- the method tab
    check("the method tab lists the methods",
          "Dixon–Coles" in body or "Dixon" in body)
    check("it states there is no measured track record",
          "no measured track record" in body.lower()
          or "not been validated" in body.lower())

    print(f"PASS {passes}  FAIL {len(fails)}")
    for f in fails:
        print("  FAIL:", f)
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
