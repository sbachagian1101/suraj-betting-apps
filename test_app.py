"""App-level checks. Run: python test_app.py

A Streamlit app with a fatal error deeper in the script still serves HTTP 200
and still renders its first tab, so the tab that does the work is driven here.
"""
from __future__ import annotations

import warnings

warnings.filterwarnings("ignore")

from streamlit.testing.v1 import AppTest

import meeting as M

M_BLOCKS = M.split_races(M.read_meeting(
    "sample_data/2026-08-30-CARNARVON-T.xlsx"))


def main():
    fails, passes = [], 0

    def check(name, cond):
        nonlocal passes
        if cond:
            passes += 1
        else:
            fails.append(name)

    at = AppTest.from_file("app.py", default_timeout=300).run()
    check("the app starts without an exception", not at.exception)
    if at.exception:
        for e in at.exception:
            print("EXCEPTION:", e.value)
        print(f"PASS {passes}  FAIL {len(fails)}")
        return 1

    check("it names itself", any("Top Race Predictor" in t.value for t in at.title))
    check("the five tabs are present", len(at.tabs) >= 5)
    check("it asks for input first", len(at.info) >= 1)

    # the accuracy tab must be readable before anything is loaded
    body0 = " ".join([m.value for m in at.markdown])
    check("the honest record shows without loading a race",
          "0 of 5" in body0 or "won 0" in body0.lower())

    # ------------------------------------------------ drive the real path
    at = AppTest.from_file("app.py", default_timeout=300).run()
    at.toggle[0].set_value(True).run()          # bundled Carnarvon meeting
    check("loading the sample meeting does not raise", not at.exception)
    if at.exception:
        for e in at.exception:
            print("EXCEPTION:", e.value)
        print(f"PASS {passes}  FAIL {len(fails)}")
        return 1

    check("a race selector appeared", len(at.selectbox) >= 1)
    opts = at.selectbox[0].options
    check("all six Carnarvon races are offered", len(opts) == 6)
    check("the TABTOUCH race is one of them",
          any("TABTOUCH" in o.upper() for o in opts))

    check("a results table was rendered", len(at.dataframe) >= 1)
    df = at.dataframe[0].value
    check("the table is ranked", list(df["Rank"]) == sorted(df["Rank"]))
    check("win percentages sum to 100",
          abs(float(df["Win %"].sum()) - 100.0) < 0.5)
    check("top-3 is never below win", bool((df["Top-3 %"] >= df["Win %"] - 1e-9).all()))

    body = " ".join(
        [m.value for m in at.markdown] + [s.value for s in at.success]
        + [w.value for w in at.warning] + [c.value for c in at.caption]
        + [i.value for i in at.info])
    check("a selection is announced", "win" in body.lower())
    check("the calibration is disclosed", "favourites" in body.lower())
    # AppTest exposes no download_button accessor, so check the exports are
    # at least constructed without raising, via the report the button serves
    import engine as _E
    check("a full text report can be produced",
          len(_E.analysis_report_text(
              _E.analyse_race_text(
                  M_BLOCKS[0].text, simulations=2000))) > 200)

    # calibration is on by default and must be reversible
    cal_on = float(df["Win %"].max())
    at.toggle[1].set_value(False).run()         # turn calibration off
    check("switching to the original engine does not raise", not at.exception)
    if not at.exception:
        raw = float(at.dataframe[0].value["Win %"].max())
        check("the original engine is more confident than the calibrated one",
              raw > cal_on)
        warn = " ".join(w.value for w in at.warning)
        check("and the app says so", "Original engine" in warn or "60" in warn)

    # a different race must also work
    at.toggle[1].set_value(True).run()
    at.selectbox[0].set_value(opts[2]).run()
    check("switching race does not raise", not at.exception)
    if not at.exception:
        check("the new race also sums to 100",
              abs(float(at.dataframe[0].value["Win %"].sum()) - 100.0) < 0.5)

    print(f"PASS {passes}  FAIL {len(fails)}")
    for f in fails:
        print("  FAIL:", f)
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
