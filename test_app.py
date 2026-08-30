"""App-level checks. Run: python test_app.py

A Streamlit app with a fatal error deeper in the script still serves HTTP 200
and still renders its first tab, so the tabs that do the work are driven here.
"""
from __future__ import annotations

import warnings

warnings.filterwarnings("ignore")

import numpy as np
from streamlit.testing.v1 import AppTest


class _BlockMatplotlib:
    """Make matplotlib un-importable for the duration of a test.

    Streamlit Cloud does not install matplotlib, and this machine does. That
    difference is the whole bug: `Styler.background_gradient` imports it
    lazily, so the score grid rendered here and raised ImportError there, below
    the fold of the Prediction tab where a quick look does not reach. Hiding
    the module locally is the only way this class of failure gets caught before
    a deploy.
    """

    def find_module(self, name, path=None):
        return self if name.split(".")[0] == "matplotlib" else None

    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] == "matplotlib":
            raise ImportError(f"{name} is blocked for this test")
        return None

    def __enter__(self):
        import sys
        # pandas.io.formats.style decides `has_mpl` ONCE, at import time. If an
        # earlier test in this process already imported it while matplotlib was
        # available, the flag stays True and blocking matplotlib does nothing -
        # which is exactly how this test passed against the broken code. Purge
        # the pandas style modules too so the flag is worked out again.
        self._saved = {k: v for k, v in sys.modules.items()
                       if k.split(".")[0] == "matplotlib"
                       or k.startswith("pandas.io.formats.style")}
        for k in self._saved:
            del sys.modules[k]
        sys.meta_path.insert(0, self)
        return self

    def __exit__(self, *a):
        import sys
        sys.meta_path.remove(self)
        sys.modules.update(self._saved)
        return False


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
    check("two text areas are offered — one per team", len(at.text_area) == 2)
    check("it will not proceed until both are filled",
          any("box" in i.value for i in at.info))

    # ---------------------------------------------- load the bundled example
    labels = [b.label for b in at.button]
    check("a sample loader is offered",
          any("bundled example" in l for l in labels))
    check("and a way to clear both boxes", any("Clear both" in l for l in labels))
    at.button[0].click().run()
    check("the loader fills both boxes",
          all(t.value.strip() for t in at.text_area))
    check("loading the example does not raise", not at.exception)
    if at.exception:
        return bail(at)

    check("matches were parsed and shown", len(at.dataframe) >= 3)
    heads = " ".join(h.value for h in at.subheader)
    check("pages read and distinct matches are both reported",
          "pages read" in heads and "distinct matches" in heads)
    check("the overlap between the boxes is called out, not silently dropped",
          any("both" in i.value and "counted once" in i.value
              for i in at.info))

    # the box decides the side; there is no dropdown to disagree with it
    mets = {m.label: m.value for m in at.metric}
    check("the home side is named from the home box",
          mets.get("Home") == "Sturm Graz II")
    check("the away side is named from the away box",
          mets.get("Away") == "Rapid Wien II")
    check("no home/away dropdown remains to contradict the boxes",
          not any(sb.label in ("Home team", "Away team") for sb in at.selectbox))

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

    # ---------------------------------------- swapping the boxes swaps sides
    at3 = AppTest.from_file("app.py", default_timeout=300).run()
    at3.button[0].click().run()
    h_text = at3.text_area[0].value
    a_text = at3.text_area[1].value
    at3.text_area[0].set_value(a_text).run()
    at3.text_area[1].set_value(h_text).run()
    check("swapping the two boxes does not raise", not at3.exception)
    if not at3.exception:
        m3 = {m.label: m.value for m in at3.metric}
        check("the sides swap with the boxes",
              m3.get("Home") == "Rapid Wien II" and m3.get("Away") == "Sturm Graz II")

    # one team's form in both boxes is refused rather than modelled
    at4 = AppTest.from_file("app.py", default_timeout=300).run()
    at4.button[0].click().run()
    same = at4.text_area[0].value
    at4.text_area[1].set_value(same).run()
    check("the same team in both boxes does not raise", not at4.exception)
    if not at4.exception:
        check("and is refused with a clear message",
              any("Both boxes" in e.value for e in at4.error))

    # ------------------------------------ the deployed environment has no
    # matplotlib; prove the app does not need it
    import importlib
    with _BlockMatplotlib():
        try:
            importlib.import_module("matplotlib")
            check("matplotlib really is blocked for this test", False)
        except ImportError:
            check("matplotlib really is blocked for this test", True)
        at2 = AppTest.from_file("app.py", default_timeout=300).run()
        at2.button[0].click().run()
        pb2 = [i for i, b in enumerate(at2.button) if "Predict" in b.label]
        if pb2:
            at2.button[pb2[0]].click().run()
        check("the whole prediction path runs without matplotlib",
              not at2.exception)
        if at2.exception:
            for e in at2.exception:
                print("EXCEPTION (no-matplotlib):", e.value)
        else:
            grids2 = [d.value for d in at2.dataframe
                      if hasattr(d.value, "shape") and d.value.shape[0] >= 8
                      and d.value.shape[1] >= 8]
            check("and still renders the shaded score grid", len(grids2) >= 1)

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
