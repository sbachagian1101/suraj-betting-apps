"""Checks for WorksheetPredictor.  Run: python test_app.py

A Streamlit app with a fatal error further down still serves HTTP 200 and
renders its header, so the buttons that do the work are driven here, and the
whole path runs again with matplotlib hidden - the difference between a
development machine and Streamlit Cloud.

The claims the app prints about its own accuracy are re-derived from the
bundled data rather than trusted, so a stale calibration.json fails the build.
"""
from __future__ import annotations

import json
import math
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

from streamlit.testing.v1 import AppTest

import ws_model as MD
import ws_parser as PS

SAMPLES = Path("samples")
RESULTS = json.loads((SAMPLES / "results.json").read_text(encoding="utf-8"))
CAL = MD.load_calibration()


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


def sb(at, label):
    """Select a selectbox by label. AppTest orders main-body widgets before
    sidebar ones, so positional indexing here is a trap."""
    for x in at.selectbox:
        if x.label == label:
            return x
    raise AssertionError(f"no selectbox labelled {label!r}")


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


def graded():
    """[(region, live_runners, first_four)] for every graded race."""
    out = []
    for f in sorted(SAMPLES.glob("*.csv")):
        mt = PS.parse(f.read_text(encoding="utf-8-sig"),
                      PS.meeting_name_from_filename(f.name))
        res = RESULTS.get(f.stem, [])
        for rc in mt.races:
            if rc.index - 1 < len(res) and len(rc.live) >= 2:
                out.append((mt.region, rc.live, res[rc.index - 1]))
    return out


def main():
    fails, passes = [], 0

    def check(name, cond):
        nonlocal passes
        if cond:
            passes += 1
        else:
            fails.append(name)

    # ------------------------------------------------------------- parser
    rip = PS.parse((SAMPLES / "ripon.csv").read_text(encoding="utf-8-sig"),
                   "Ripon")
    gow = PS.parse((SAMPLES / "gowran-park.csv").read_text(
        encoding="utf-8-sig"), "Gowran Park")

    check("Ripon splits into 7 races", len(rip.races) == 7)
    check("Gowran Park splits into 8 races", len(gow.races) == 8)
    check("runners are read", sum(len(r.runners) for r in rip.races) == 61)
    check("scratchings are detected",
          any(x.scratched for r in rip.races for x in r.runners))
    check("a scratching never reaches the live list",
          all(not x.scratched for r in rip.races for x in r.live))
    r1 = rip.races[0].runners
    check("saddlecloths and names survive",
          r1[0].tab == 7 and r1[0].horse == "COSMOS RAJ")
    check("the numeric columns are read",
          abs(r1[0].per - 39.5) < 1e-9 and abs(r1[0].fr - 35.0) < 1e-9)
    check("the R&S price is read", abs((r1[0].div or 0) - 2.50) < 1e-9)
    check("first-up is flagged from RFS=FU",
          any(x.first_up for r in rip.races for x in r.runners))
    check("an apostrophe in a name survives",
          any("'" in x.horse for r in rip.races for x in r.runners))
    check("regions are detected",
          PS.region_for("Ripon") == "UK"
          and PS.region_for("Wodonga") == "AUS"
          and PS.region_for("Gowran Park") == "IRE"
          and PS.region_for("Deauville") == "FR")
    check("an unknown track falls back to OTHER",
          PS.region_for("Nowhere Downs") == "OTHER")
    check("the filename helper strips the R&S boilerplate",
          PS.meeting_name_from_filename(
              "Tuesday, 01st September 2026 - Ripon Races Worksheets (1).csv")
          == "Ripon")
    check("a tab-separated paste also parses",
          len(PS.parse("Tab\tHorse\tRFS\tDLS\t12m\tBRR\tFORM\tCOND\tCONS\t"
                       "BP\tJOCK\tJC\tFR\tEM\tPER\tDIV\n"
                       "1\tALPHA\t3\t10\t30\t30\t0\t0\t0\t0\t0\t0\t30\t0\t"
                       "60\t$1.60\n"
                       "2\tBETA\t4\t12\t28\t28\t0\t0\t0\t0\t0\t0\t28\t2\t"
                       "40\t$2.40\n").races) == 1)
    check("junk does not crash the parser", len(PS.parse("hello\nworld").races)
          == 0)

    # ---- the re-ingest bug: switching race bounced back to Parse/Predict
    # st.file_uploader returns the SAME file on every rerun, so the ingest
    # block ran again on every widget change and popped the parsed meeting.
    data = (SAMPLES / "ripon.csv").read_bytes()
    fresh1, fp1, text1, name1 = PS.ingest_upload(
        None, "Tuesday - Ripon Races Worksheets.csv", data)
    check("a new upload is ingested once", fresh1 and text1 and name1 == "Ripon")
    fresh2, fp2, _, _ = PS.ingest_upload(
        fp1, "Tuesday - Ripon Races Worksheets.csv", data)
    check("the SAME file on a later rerun is not re-ingested",
          not fresh2 and fp2 == fp1)
    fresh3, fp3, _, _ = PS.ingest_upload(
        fp1, "Tuesday - Ripon Races Worksheets.csv",
        data + bytes([10]) + b"1,X" + bytes([10]))
    check("an edited file of the same name IS re-ingested",
          fresh3 and fp3 != fp1)
    fresh4, _, _, _ = PS.ingest_upload(fp1, "other.csv", data)
    check("a different filename is re-ingested", fresh4)
    check("an empty upload does not crash",
          PS.ingest_upload(None, "x.csv", b"")[0])

    # -------------------------------------------------------------- model
    live = rip.races[0].live
    rows, meta = MD.analyse_race(live, "UK", CAL)
    check("every live runner is scored", len(rows) == len(live))
    check("win probabilities sum to 1",
          abs(sum(r.prob for r in rows) - 1.0) < 1e-9)
    check("they come back sorted",
          [r.prob for r in rows] == sorted([r.prob for r in rows],
                                           reverse=True))
    check("place is never below win",
          all(r.place >= r.prob - 1e-9 for r in rows))
    check("place probabilities total about the places paid",
          abs(sum(r.place for r in rows) - meta["places"]) < 0.35)
    check("fair price is the reciprocal",
          all(abs(r.fair_win * r.prob - 1.0) < 1e-6 for r in rows))
    check("confidence is on 0-100", 0 <= meta["confidence"] <= 100)

    # the headline promise: calibration must NEVER reorder the field
    for reg in ("AUS", "UK", "IRE", "FR", "OTHER"):
        a = [r.tab for r in MD.analyse_race(live, reg, CAL)[0]]
        b = [x.tab for x in sorted(live, key=lambda y: -y.per)]
        check(f"{reg}: the running order is unchanged", a == b)

    check("Australia is sharpened, the UK flattened",
          MD.temperature(CAL, "AUS")[0] < 1.0
          < MD.temperature(CAL, "UK")[0])
    check("AUS and UK are marked validated",
          MD.temperature(CAL, "AUS")[1] and MD.temperature(CAL, "UK")[1])
    check("FR and IRE are marked provisional",
          not MD.temperature(CAL, "FR")[1]
          and not MD.temperature(CAL, "IRE")[1])

    # ------------------------------------- re-derive the app's own claims
    G = graded()
    check("the bundled data still holds 61 graded races", len(G) == 61)

    hit = sum(1 for _, L, res in G
              if max(L, key=lambda x: x.per).tab == res[0])
    check("top pick really wins what the app claims",
          abs(hit / len(G) - CAL["record"]["top_pick_win"]) < 0.005)

    t3 = sum(1 for _, L, res in G
             if res[0] in [x.tab for x in sorted(L, key=lambda y: -y.per)[:3]])
    check("top three really hold the winner as claimed",
          abs(t3 / len(G) - CAL["record"]["top_three"]) < 0.005)

    # calibration: the claimed probability must match reality, per region
    agg = {}
    for reg, L, res in G:
        rows_, _ = MD.analyse_race(L, reg, CAL)
        a = agg.setdefault(reg, [0, 0.0, 0, 0.0, 0])
        a[0] += 1
        a[1] += rows_[0].prob
        a[2] += rows_[0].tab == res[0]
        a[3] += rows_[0].place
        a[4] += rows_[0].tab in res[:3]
    for reg, (n, cw, hw, cp, hp) in agg.items():
        check(f"{reg}: the win number is calibrated within 10 points",
              abs(cw / n - hw / n) < 0.10)
        check(f"{reg}: the place number is calibrated within 10 points",
              abs(cp / n - hp / n) < 0.10)
    tot = [sum(agg[r][i] for r in agg) for i in range(5)]
    check("overall the win number is within 3 points",
          abs(tot[1] / tot[0] - tot[2] / tot[0]) < 0.03)
    check("overall the place number is within 3 points",
          abs(tot[3] / tot[0] - tot[4] / tot[0]) < 0.03)

    # published PER is worse than uniform -- the reason this app exists
    def ll(fn):
        s = n = 0
        for reg, L, res in G:
            tabs = [x.tab for x in L]
            if res[0] not in tabs:
                continue
            s -= math.log(max(fn(reg, L)[tabs.index(res[0])], 1e-9))
            n += 1
        return s / n
    pub = ll(lambda reg, L: MD.calibrate([x.per for x in L], 1.0))
    uni = ll(lambda reg, L: [1 / len(L)] * len(L))
    cal = ll(lambda reg, L: MD.calibrate(
        [x.per for x in L], MD.temperature(CAL, reg)[0]))
    check("published PER really is worse than uniform", pub > uni)
    check("and the calibrated version beats both", cal < uni < pub)
    check("the app quotes the published log-loss correctly",
          abs(pub - CAL["record"]["logloss_published"]) < 0.01)

    # staking
    rec = MD.recommend(rows, meta)
    check("without a price there is no stake and no EV",
          rec["points"] is None and rec["ev"] is None)
    check("but it still names a break-even price", rec["fair"] > 1.0)
    rec2 = MD.recommend(rows, meta, price=rec["fair"] * 1.5, max_points=5.0)
    check("a generous price produces a positive-EV stake",
          rec2["action"] == "Back" and rec2["ev"] > 0
          and 0 < rec2["points"] <= 5.0)
    rec3 = MD.recommend(rows, meta, price=1.01)
    check("a hopeless price is refused", rec3["action"] == "No bet")
    check("insights come back", len(MD.insights(rows, meta, CAL)) >= 3)
    check("bands are ordered",
          MD.band(90)[0] == "High" and MD.band(5)[0] == "Very low")

    # ---------------------------------------------------------------- app
    at = AppTest.from_file("app.py", default_timeout=900).run()
    check("the app starts without an exception", not at.exception)
    if at.exception:
        for e in at.exception:
            print("EXCEPTION:", e.value)
        print(f"PASS {passes}  FAIL {len(fails)}")
        return 1

    src = Path("app.py").read_text(encoding="utf-8")
    check("there is exactly one worksheet box", src.count("st.text_area(") == 1)
    check("no tabs -- everything on one page", "st.tabs(" not in src)
    check("it names itself",
          any("WorksheetPredictor" in m.value for m in at.markdown))
    check("the measured record is in the sidebar",
          any("random" in m.value for m in at.markdown))
    check("Parse and Predict both exist",
          any(b.label == "Parse" for b in at.button)
          and any(b.label == "Predict" for b in at.button))

    # load -> parse -> predict
    at = AppTest.from_file("app.py", default_timeout=900).run()
    sb(at, "Bundled meeting").set_value("ripon").run()
    lb = [i for i, b in enumerate(at.button)
          if "Load bundled" in b.label][0]
    at.button[lb].click().run()
    check("loading a bundled meeting does not raise", not at.exception)
    pb = [i for i, b in enumerate(at.button) if b.label == "Parse"][0]
    at.button[pb].click().run()
    check("parsing does not raise", not at.exception)
    check("it reports the races found",
          any("races" in s.value for s in at.success))

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
    check("it states win, place or no bet",
          "WIN &mdash;" in body or "PLACE &mdash;" in body
          or "No bet" in body)
    check("a confidence level is shown", "confidence" in body.lower())
    check("the break-even price is offered when no price is entered",
          "needs better than" in body or "points" in body)
    tbl = [d.value for d in at.dataframe]
    got = [t for t in tbl if "Calibrated %" in getattr(t, "columns", [])]
    check("the runner table renders with both columns",
          got and {"R&S %", "Calibrated %", "Place %", "Fair win"}
          <= set(got[0].columns))
    check("all four charts rendered", count_charts(at) == 4)
    check("the confidence caveat is present",
          any("not** a hit rate" in i.value or "not a hit rate" in i.value
              for i in at.info))

    # switching race must keep the prediction AND show a different field
    rsel = sb(at, "Race")
    check("a race picker appears", rsel is not None and len(rsel.options) > 1)

    def shown(a):
        d = [x.value for x in a.dataframe
             if "Calibrated %" in getattr(x.value, "columns", [])]
        return tuple(d[0]["Horse"]) if d else ()

    fields = []
    for opt in rsel.options[:4]:
        sb(at, "Race").set_value(opt).run()
        check(f"{opt}: switching race does not raise", not at.exception)
        b = " ".join(m.value for m in at.markdown)
        check(f"{opt}: the prediction is still on screen",
              "Betting recommendation" in b)
        check(f"{opt}: the parsed meeting survived the rerun",
              "wp_mt" in at.session_state and "wp_done" in at.session_state)
        fields.append(shown(at))
    check("each race shows a different field", len(set(fields)) == len(fields))

    # any unrelated rerun must also leave the prediction standing
    sb(at, "Region").set_value("UK").run()
    check("changing an unrelated setting keeps the prediction",
          not at.exception
          and any("Betting recommendation" in m.value for m in at.markdown))

    # a provisional region must say so
    at2 = AppTest.from_file("app.py", default_timeout=900).run()
    sb(at2, "Region").set_value("IRE").run()
    sb(at2, "Bundled meeting").set_value("gowran-park").run()
    lb2 = [i for i, b in enumerate(at2.button) if "Load bundled" in b.label][0]
    at2.button[lb2].click().run()
    pr2 = [i for i, b in enumerate(at2.button) if b.label == "Predict"][0]
    at2.button[pr2].click().run()
    check("a provisional region predicts without raising", not at2.exception)
    check("and it warns that the constant is provisional",
          any("provisional" in i.value.lower() for i in at2.info))

    # -------------------------------- Streamlit Cloud has no matplotlib
    import importlib
    with BlockMatplotlib():
        try:
            importlib.import_module("matplotlib")
            check("matplotlib really is blocked", False)
        except ImportError:
            check("matplotlib really is blocked", True)
        at3 = AppTest.from_file("app.py", default_timeout=900).run()
        sb(at3, "Bundled meeting").set_value("wodonga").run()
        lb3 = [i for i, b in enumerate(at3.button)
               if "Load bundled" in b.label][0]
        at3.button[lb3].click().run()
        pr3 = [i for i, b in enumerate(at3.button) if b.label == "Predict"]
        if pr3:
            at3.button[pr3[0]].click().run()
        check("the whole path runs without matplotlib", not at3.exception)
        if at3.exception:
            for e in at3.exception:
                print("EXCEPTION (no-matplotlib):", e.value)
        else:
            check("and still draws every chart", count_charts(at3) == 4)

    print(f"PASS {passes}  FAIL {len(fails)}")
    for f in fails:
        print("  FAIL:", f)
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
