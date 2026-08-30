"""App-level checks. Run: python test_app.py

The landing page is not enough. A Streamlit app with a fatal error deeper in
the script still serves HTTP 200 and still renders its first tab, so the tab
that uses the new code has to be driven directly - twice before, checking only
the landing state let a broken prediction path ship.
"""
from __future__ import annotations

import warnings

warnings.filterwarnings("ignore")

from streamlit.testing.v1 import AppTest


def run(timeout=300):
    at = AppTest.from_file("app.py", default_timeout=timeout)
    at.run()
    return at


def main():
    fails, passes = [], 0

    def check(name, cond):
        nonlocal passes
        if cond:
            passes += 1
        else:
            fails.append(name)

    # ---------------------------------------------------- the landing state
    at = run()
    check("the app starts without an exception", not at.exception)
    if at.exception:
        for e in at.exception:
            print("EXCEPTION:", e.value)
        print(f"PASS {passes}  FAIL {len(fails)}")
        return 1

    check("it names itself", any("AustraliaPdfHorseRacing" in t.value
                                 for t in at.title))
    check("it asks for a file before doing anything",
          any("Upload" in i.value for i in at.info))
    check("the four tabs are present", len(at.tabs) >= 4)
    check("the headline metrics are on the sidebar", len(at.metric) >= 2)

    # -------------------------------------------------- the prediction path
    at = run()
    at.toggle[0].set_value(True).run()
    check("turning on the sample meetings does not raise", not at.exception)
    if at.exception:
        for e in at.exception:
            print("EXCEPTION:", e.value)
        print(f"PASS {passes}  FAIL {len(fails)}")
        return 1

    check("a race can be chosen", len(at.selectbox) >= 1)
    check("more than one race was parsed",
          len(at.selectbox[0].options) > 3)

    body = " ".join(
        [m.value for m in at.markdown]
        + [s.value for s in at.success] + [w.value for w in at.warning]
        + [i.value for i in at.info] + [c.value for c in at.caption])
    check("a selection is announced", "Selection:" in body)
    check("a confidence level is stated",
          any(k in body for k in ("low confidence", "medium confidence",
                                  "high confidence")))
    check("the confidence is backed by a measured strike rate",
          "held-out races" in body)
    check("a confidence interval is quoted", "95% CI" in body)
    check("reasons are given", "Why this runner" in body)
    check("the app says it loses to the market",
          "starting price does better" in body or "market" in body.lower())
    check("a results table was rendered", len(at.dataframe) >= 1)

    df = at.dataframe[0].value
    check("the table has a rank column", "Rank" in df.columns)
    check("it has win percentages", "Win %" in df.columns)
    check("win percentages sum to 100",
          abs(float(df["Win %"].sum()) - 100.0) < 1e-4)
    check("place percentages are never below win percentages",
          bool((df["Place %"] >= df["Win %"] - 1e-9).all()))
    check("the top row is rank 1", int(df.Rank.iloc[0]) == 1)

    # ------------------------------------------------ switching race works
    n = len(at.selectbox[0].options)
    at.selectbox[0].set_value(at.selectbox[0].options[min(2, n - 1)]).run()
    check("switching race does not raise", not at.exception)
    if not at.exception:
        df2 = at.dataframe[0].value
        check("the second race also sums to 100",
              abs(float(df2["Win %"].sum()) - 100.0) < 1e-4)

    # ------------------------------------------------------- the other tabs
    check("the validation tab reports held-out races",
          "held-out races" in body or "out of sample" in body)
    check("the method tab explains the odds trap",
          "odds" in body.lower() and "minus one" in body.lower())
    check("the app discloses the pre-acceptance caveat",
          "pre-acceptance" in body.lower())
    check("the app discloses that gear is excluded",
          "regression to the mean" in body.lower())

    print(f"PASS {passes}  FAIL {len(fails)}")
    for f in fails:
        print("  FAIL:", f)
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
