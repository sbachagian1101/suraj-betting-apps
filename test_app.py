"""Checks for OddsPredictor.  Run: python test_app.py

A Streamlit app with a fatal error further down the script still serves
HTTP 200 and still renders its header, so the buttons that do the work are
driven here, and the whole path runs again with matplotlib hidden -- the
difference between a development machine and Streamlit Cloud.
"""
from __future__ import annotations

import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

from streamlit.testing.v1 import AppTest

import odds_model as MD
import odds_parser as PS

IPSW = Path("samples/ipswich_cl1.txt")
MURR = Path("samples/murray_bridge_cl1.txt")

# Graded results. These are FINISHING ORDERS, taken from the user, and are
# deliberately not in the sample files -- the header line in those files
# ("1,11,6,4" / "8,9,12,2") looks like a first four and is not one.
RESULTS = {"ipswich": [6, 11, 12, 2], "murray": [1, 13, 12, 7]}


class BlockMatplotlib:
    """Hide matplotlib the way Streamlit Cloud does.

    pandas.io.formats.style fixes `has_mpl` at import time, so those modules
    must be purged too or an earlier import leaves the flag True and the
    block does nothing at all.
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


def count_charts(at):
    found = []

    def walk(node):
        for ch in getattr(node, "children", {}).values():
            if (type(ch).__name__ == "UnknownElement"
                    and getattr(ch, "type", "") == "vega_lite_chart"):
                found.append(ch)
            walk(ch)
    walk(at._tree)
    return len(found)


def main():
    fails, passes = [], 0

    def check(name, cond):
        nonlocal passes
        if cond:
            passes += 1
        else:
            fails.append(name)

    # ------------------------------------------------------------- parser
    ip = PS.parse(IPSW.read_text(encoding="utf-8"))
    mb = PS.parse(MURR.read_text(encoding="utf-8"))

    check("Ipswich yields 12 priced runners", len(ip.active) == 12)
    check("Ipswich yields 5 scratchings", len(ip.scratched) == 5)
    check("Murray Bridge yields 13 priced runners", len(mb.active) == 13)
    check("Murray Bridge yields 2 scratchings", len(mb.scratched) == 2)
    check("no runner is parsed twice",
          len({r.number for r in ip.runners}) == len(ip.runners))

    # the bug that ate every second runner: a bare number matches the price
    # pattern, so the next runner's number was consumed as a fifth price
    check("consecutive runner numbers all survive",
          {8, 12, 9, 2, 1, 5, 7, 6, 13, 4, 10, 11, 14}
          == {r.number for r in mb.active})

    t = {r.number: r for r in mb.active}
    check("prices land in the right four slots",
          t[8].fixed_win == 4.20 and t[8].fixed_place == 1.80
          and t[8].tote_win == 4.80 and t[8].tote_place == 2.10)
    check("the opening and top prices are read",
          t[9].opening == 14.0 and t[9].top == 14.0)
    check("a name with an apostrophe survives", t[5].name.startswith("GOODLOOKIN"))
    check("a country suffix survives", "(NZ)" in t[11].name)
    check("race conditions are read",
          mb.distance == 1400 and mb.going == "Soft5")
    check("the meeting is read", "Murray Bridge" in mb.meeting)
    check("the class is read", mb.race_class.startswith("CL1"))

    # the header number list is NOT a result
    check("the header runner list is flagged as a note, not a result",
          any("stewards" in n.lower() for n in mb.notes))
    fin = RESULTS["murray"]
    check("and that header list really is not the finishing order",
          [8, 9, 12, 2] != fin)
    check("nothing on the Race object claims to be a result",
          not hasattr(mb, "result"))

    # ------------------------------------------------------------- model
    lines, meta = MD.analyse(ip)
    check("every priced runner is scored", len(lines) == 12)
    check("win probabilities sum to 1",
          abs(sum(x.p_win for x in lines) - 1.0) < 1e-6)
    check("they come back sorted", [x.p_win for x in lines]
          == sorted([x.p_win for x in lines], reverse=True))
    check("the margin is stripped, not left in",
          meta["book_fixed"] > 1.15 and abs(sum(x.p_fixed for x in lines)
                                            - 1.0) < 1e-6)
    check("place probabilities total about the places paid",
          abs(sum(x.p_place for x in lines) - meta["places_paid"]) < 0.05)
    check("a 12-runner field pays three places", meta["places_paid"] == 3)
    check("a 6-runner field pays two", MD.paid_places(6) == 2)
    check("a 4-runner field pays one", MD.paid_places(4) == 1)
    check("best price is never worse than the fixed price",
          all(x.best_win >= x.fixed_win - 1e-9 for x in lines))
    check("fair price is the reciprocal of the probability",
          all(abs(x.fair_win * x.p_win - 1.0) < 1e-6 for x in lines))
    check("EV agrees with probability times price",
          all(abs(x.ev_win - (x.p_win * x.best_win - 1.0)) < 1e-9
              for x in lines))
    check("Kelly is never negative",
          all(x.kelly_win >= 0 and x.kelly_place >= 0 for x in lines))
    check("confidence sits on 0-100", 0 <= meta["confidence"] <= 100)

    # the fake-edge bug: place probability from one book, price from the
    # other, produced "back a 0.7% chance at $37.50 place"
    rec = meta["recommendation"]
    if rec["action"] != "No bet":
        L, m = rec["line"], rec["market"]
        p = L.p_win if m == "Win" else L.p_place
        check("a recommendation is never made on a hopeless runner",
              p >= (0.06 if m == "Win" else 0.15))
        check("the recommended stake is capped",
              0 < rec["points"] <= meta["max_points"])
        check("the recommendation carries a positive edge", rec["ev"] > 0)
    else:
        check("a no-bet still names the top rating", rec["line"] is not None)
        passes += 2

    check("no runner under the floor is ever recommended",
          all(not (x.p_place < 0.15 and rec.get("market") == "Place"
                   and rec["line"].number == x.number) for x in lines))

    # tote weight actually moves the answer
    a = MD.analyse(ip, tote_weight=0.0)[0][0].p_win
    b = MD.analyse(ip, tote_weight=1.0)[0][0].p_win
    check("the tote weight changes the numbers", abs(a - b) > 1e-6)

    # the graded record the app claims in its sidebar
    for key, race in (("ipswich", ip), ("murray", mb)):
        ls, _ = MD.analyse(race)
        order = [x.number for x in ls]
        rank = order.index(RESULTS[key][0]) + 1
        check(f"{key}: the winner is still ranked where the app says",
              rank == (1 if key == "ipswich" else 6))

    check("insights come back", len(MD.insights(lines, meta, ip)) >= 3)
    check("bands are ordered", MD.band(90)[0] == "High"
          and MD.band(10)[0] == "Very low")

    # empty and junk input must not explode
    check("junk parses to nothing rather than crashing",
          len(PS.parse("hello world\n\n123").active) == 0)
    check("a two-runner race still analyses",
          len(MD.analyse(PS.parse(
              "1\nALPHA\nOpening Win2.00Top Win2.00\n2.00\n1.20\n"
              "2.10Arrow\n1.25\n2\nBETA\nOpening Win2.00Top Win2.00\n"
              "1.80\n1.15\n1.90Arrow\n1.20\n"))[0]) == 2)

    # --------------------------------------------------------------- app
    at = AppTest.from_file("app.py", default_timeout=900).run()
    check("the app starts without an exception", not at.exception)
    if at.exception:
        for e in at.exception:
            print("EXCEPTION:", e.value)
        print(f"PASS {passes}  FAIL {len(fails)}")
        return 1

    check("it names itself",
          any("OddsPredictor" in m.value for m in at.markdown))
    check("the measured record is stated up front",
          any("lost" in m.value.lower() for m in at.markdown))
    src = Path("app.py").read_text(encoding="utf-8")
    check("there is exactly one paste box", src.count("st.text_area(") == 1)
    check("the paste box is the shared one", src.count('key="op_raw"') == 1)
    check("there are no tabs -- everything on one page",
          "st.tabs(" not in src)
    check("a Parse and a Predict button exist",
          any(b.label == "Parse" for b in at.button)
          and any(b.label == "Predict" for b in at.button))

    # load -> parse -> predict, the real path
    at = AppTest.from_file("app.py", default_timeout=900).run()
    at.button[0].click().run()                       # load Ipswich
    check("loading a sample does not raise", not at.exception)
    pb = [i for i, b in enumerate(at.button) if b.label == "Parse"][0]
    at.button[pb].click().run()
    check("parsing does not raise", not at.exception)
    check("it reports how many runners were priced",
          any("priced runners" in s.value for s in at.success))
    check("the scratchings are named",
          any("Scratched" in c.value for c in at.caption))

    pr = [i for i, b in enumerate(at.button) if b.label == "Predict"][0]
    at.button[pr].click().run()
    check("predicting does not raise", not at.exception)
    if at.exception:
        for e in at.exception:
            print("EXCEPTION (predict):", e.value)
        print(f"PASS {passes}  FAIL {len(fails)}")
        return 1

    body = " ".join(m.value for m in at.markdown)
    check("a betting recommendation is printed",
          "Betting recommendation" in body)
    check("the stake is given in points", "points" in body or "point" in body)
    check("a confidence level is shown", "onfidence" in body)
    check("the recommendation states win or place",
          "WIN &mdash;" in body or "PLACE &mdash;" in body
          or "No bet" in body)
    tbl = [d.value for d in at.dataframe]
    check("the full runner table renders", len(tbl) >= 1)
    got = [t for t in tbl if "Win %" in getattr(t, "columns", [])]
    check("it carries probabilities, fair price and EV",
          got and {"Win %", "Fair", "Best", "EV %"} <= set(got[0].columns))
    # Altair charts land in the AppTest tree as UnknownElement carrying the
    # proto type "vega_lite_chart", not under any at.<name> accessor.
    check("all four charts rendered", count_charts(at) == 4)
    check("insights rendered", "Market margin" in body)
    check("the confidence caveat is present",
          any("not** a hit rate" in i.value or "not a hit rate" in i.value
              for i in at.info))

    # predict without parsing first must still work
    at2 = AppTest.from_file("app.py", default_timeout=900).run()
    at2.button[1].click().run()                      # load Murray Bridge
    pr2 = [i for i, b in enumerate(at2.button) if b.label == "Predict"][0]
    at2.button[pr2].click().run()
    check("Predict alone parses and predicts", not at2.exception)
    check("and it produced a recommendation",
          any("Betting recommendation" in m.value for m in at2.markdown))

    # ------------------------------- Streamlit Cloud has no matplotlib
    import importlib
    with BlockMatplotlib():
        try:
            importlib.import_module("matplotlib")
            check("matplotlib really is blocked", False)
        except ImportError:
            check("matplotlib really is blocked", True)
        at3 = AppTest.from_file("app.py", default_timeout=900).run()
        at3.button[0].click().run()
        pr3 = [i for i, b in enumerate(at3.button) if b.label == "Predict"]
        if pr3:
            at3.button[pr3[0]].click().run()
        check("the whole path runs without matplotlib", not at3.exception)
        check("and still draws every chart", count_charts(at3) == 4)
        if at3.exception:
            for e in at3.exception:
                print("EXCEPTION (no-matplotlib):", e.value)

    print(f"PASS {passes}  FAIL {len(fails)}")
    for f in fails:
        print("  FAIL:", f)
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
