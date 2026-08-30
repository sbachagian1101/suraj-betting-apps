"""App-level checks. Run: python test_app.py

A Streamlit app with a fatal error deeper in the script still serves HTTP 200
and still renders its first tab, so the tab that does the work is driven here —
and the whole path is run again with matplotlib hidden, because that is the
difference between this machine and Streamlit Cloud.
"""
from __future__ import annotations

import warnings

warnings.filterwarnings("ignore")

import numpy as np
from streamlit.testing.v1 import AppTest

SAMPLE = "sample_data/aik_vs_hammarby.txt"


class BlockMatplotlib:
    """Hide matplotlib, the way Streamlit Cloud does.

    pandas.io.formats.style decides `has_mpl` once at import time, so the
    pandas style modules have to be purged too — otherwise an earlier test in
    the same process has already set the flag and blocking the module does
    nothing at all.
    """

    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] == "matplotlib":
            raise ImportError(f"{name} is blocked for this test")
        return None

    def __enter__(self):
        import sys
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
    check("it names itself", any("FT Score Predictor" in t.value for t in at.title))
    check("five tabs are present", len(at.tabs) >= 5)
    check("two panel boxes are offered", len(at.text_area) == 2)
    check("it asks for a panel first", len(at.info) >= 1)

    # the accuracy tab must stand on its own before anything is pasted
    body0 = " ".join(m.value for m in at.markdown)
    check("the five-match record is readable straight away",
          "five matches" in body0.lower() or "5" in body0)

    # ------------------------------------------------------- paste a fixture
    text = open(SAMPLE, encoding="utf-8").read()
    at = AppTest.from_file("app.py", default_timeout=300).run()
    at.text_area[0].set_value(text).run()
    check("pasting both panels into one box does not raise", not at.exception)
    if at.exception:
        return bail(at)

    heads = " ".join(h.value for h in at.subheader)
    check("both teams are named, home first",
          "AIK Fotboll" in heads and "Hammarby IF" in heads)
    check("it says the panels came from one box",
          any("home box" in c.value for c in at.caption))

    body = " ".join(
        [m.value for m in at.markdown] + [c.value for c in at.caption])
    for section in ("The score matrix", "Most likely scorelines", "Markets",
                    "Total goals"):
        check(f"the {section} section rendered", section in body)

    tables = [d.value for d in at.dataframe]
    grids = [t for t in tables if hasattr(t, "shape")
             and t.shape[0] >= 10 and t.shape[1] >= 10]
    check("a score grid rendered", len(grids) >= 1)
    if grids:
        g = np.asarray(grids[0], dtype=float)
        check("the grid sums to about 100%", abs(g.sum() - 100.0) < 1.0)
        check("no negative cells", bool((g >= -1e-9).all()))
        check("no single scoreline is close to certain", g.max() < 30.0)

    tops = [t for t in tables if "Score" in getattr(t, "columns", [])]
    check("the most-likely-scoreline table rendered", len(tops) == 1)
    if tops:
        tt = tops[0]
        check("it lists ten scorelines", len(tt) == 10)
        check("ordered by probability",
              list(tt["Probability %"]) == sorted(tt["Probability %"],
                                                 reverse=True))
        check("each row is a scoreline", bool(tt["Score"].str.contains("–").all()))

    # ---------------------------------------------------- swapping the sides
    at.toggle[0].set_value(True).run()
    check("swapping home and away does not raise", not at.exception)
    if not at.exception:
        h2 = " ".join(h.value for h in at.subheader)
        check("and the order really swaps",
              h2.index("Hammarby") < h2.index("AIK"))
        at.toggle[0].set_value(False).run()

    # ------------------------------------------------------ the sliders work
    at.slider[0].set_value(1.0).run()
    check("full venue weight does not raise", not at.exception)
    at.slider[1].set_value(1.0).run()
    check("full xG weight does not raise", not at.exception)
    at.slider[0].set_value(0.0).run()
    check("zero venue weight does not raise", not at.exception)

    # -------------------------------------------------------- the accuracy tab
    body = " ".join([m.value for m in at.markdown])
    check("the accuracy tab reports the exact-scoreline count",
          "Exact scoreline" in " ".join(m.label for m in at.metric))
    check("and says five matches cannot certify a model",
          "five matches" in body.lower())
    check("and flags that the sample is one-sided",
          "won all five" in body.lower() or "one-sided" in body.lower())
    check("the method tab explains the concatenation trap",
          "concatenated" in body.lower())
    check("and that no home-advantage factor is applied",
          "home-advantage" in body.lower() or "home advantage" in body.lower())

    # -------------------------------- Streamlit Cloud has no matplotlib
    import importlib
    with BlockMatplotlib():
        try:
            importlib.import_module("matplotlib")
            check("matplotlib really is blocked", False)
        except ImportError:
            check("matplotlib really is blocked", True)
        at2 = AppTest.from_file("app.py", default_timeout=300).run()
        at2.text_area[0].set_value(text).run()
        check("the whole scoring path runs without matplotlib", not at2.exception)
        if at2.exception:
            for e in at2.exception:
                print("EXCEPTION (no-matplotlib):", e.value)
        else:
            g2 = [d.value for d in at2.dataframe
                  if hasattr(d.value, "shape") and d.value.shape[0] >= 10]
            check("and still renders the shaded grid", len(g2) >= 1)

    print(f"PASS {passes}  FAIL {len(fails)}")
    for f in fails:
        print("  FAIL:", f)
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
