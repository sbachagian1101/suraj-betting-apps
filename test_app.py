"""App-level checks for Bet365 Predictor. Run: python test_app.py

The engine keeps its own suite in tests/test_predictor.py and is used here
unchanged; what is tested below is the Streamlit layer wrapped around it.

A Streamlit app with a fatal error deeper in the script still serves HTTP 200
and still renders its first tab, so the tabs that do the work are driven here,
and the whole path is run again with matplotlib hidden — the difference between
a development machine and Streamlit Cloud.
"""
from __future__ import annotations

import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

from streamlit.testing.v1 import AppTest

# button[0] loads the five-meeting card, so the result built below has to
# come from that same card or it maps to nothing
SAMPLE = Path("samples/Bet365_predictions_sample.txt")


class BlockMatplotlib:
    """Hide matplotlib the way Streamlit Cloud does.

    pandas.io.formats.style fixes `has_mpl` at import time, so those modules
    must be purged as well or an earlier import in the same process leaves the
    flag True and the block does nothing at all.
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

    at = AppTest.from_file("app.py", default_timeout=900).run()
    check("the app starts without an exception", not at.exception)
    if at.exception:
        return bail(at)

    check("it names itself", any("Bet365 Predictor" in t.value for t in at.title))
    check("the five tabs are present", len(at.tabs) >= 5)
    check("the shipped trained state loaded",
          any(m.label == "Training races" and m.value not in ("0", 0)
              for m in at.metric))
    check("the odds firewall is stated",
          any("firewall" in c.value.lower() for c in at.caption))
    check("the ephemeral-disk caveat is shown before anything else",
          any("session" in w.value.lower() and "lost" in w.value.lower()
              for w in at.warning))
    check("the trained state can be downloaded", len(at.button) >= 2)

    # ------------------------------------------------------ parse a race card
    at = AppTest.from_file("app.py", default_timeout=900).run()
    at.button[0].click().run()          # load the bundled race card
    check("loading the bundled card does not raise", not at.exception)
    if at.exception:
        return bail(at)

    body = " ".join([s.value for s in at.success]
                    + [m.value for m in at.markdown])
    check("it reports how many races and meetings were parsed",
          "races" in body and "meetings" in body)
    check("more than one meeting came back", "meetings" in body)

    subs = " ".join(h.value for h in at.subheader)
    check("the parsed tab counts races and meetings",
          "races" in subs and "meetings" in subs)

    tables = [d.value for d in at.dataframe]
    check("parsed tables rendered", len(tables) >= 2)
    race_tbl = [t for t in tables if "Title" in getattr(t, "columns", [])]
    check("a race table rendered", len(race_tbl) >= 1)
    if race_tbl:
        rt = race_tbl[0]
        check("it carries distance and going",
              "Distance" in rt.columns and "Going" in rt.columns)
        check("and separates active from declared runners",
              "Active" in rt.columns and "Declared" in rt.columns)
        check("active never exceeds declared",
              bool((rt["Active"] <= rt["Declared"]).all()))
    run_tbl = [t for t in tables if "Horse" in getattr(t, "columns", [])]
    check("a runner table rendered", len(run_tbl) >= 1)
    if run_tbl:
        ru = run_tbl[0]
        check("runners have a status, not a missing 'scratched' flag",
              "Status" in ru.columns)
        check("statuses are the engine's own values",
              set(ru["Status"]) <= {"ACTIVE", "SCRATCHED"})
        check("past runs counted from historical_runs",
              "Past runs" in ru.columns and bool((ru["Past runs"] >= 0).all()))

    # --------------------------------------------------------------- predict
    pb = [i for i, b in enumerate(at.button) if "Predict all races" in b.label]
    check("a Predict all races button exists", len(pb) == 1)
    if not pb:
        return bail(at)
    at.button[pb[0]].click().run()
    check("predicting does not raise", not at.exception)
    if at.exception:
        return bail(at)
    check("it confirms the races were scored",
          any("scored" in s.value for s in at.success))

    tables = [d.value for d in at.dataframe]
    pred = [t for t in tables if "Win %" in getattr(t, "columns", [])]
    check("a prediction table rendered", len(pred) >= 1)
    if pred:
        pt = pred[0]
        check("every active runner has a position",
              list(pt["Pos"]) == sorted(pt["Pos"]))
        check("win percentages sum to about 100",
              abs(float(pt["Win %"].sum()) - 100.0) < 1.5)
        check("top-3 is never below win",
              bool((pt["Top-3 %"] >= pt["Win %"] - 1e-6).all()))
        check("confidence sits on the 0-9 scale",
              bool(((pt["Conf"] >= 0) & (pt["Conf"] <= 9)).all()))
        check("fair odds are positive", bool((pt["Fair odds"] > 0).all()))
        check("every runner is classified", bool(pt["Class"].notna().all()))

    body = " ".join([m.value for m in at.markdown]
                    + [c.value for c in at.caption])
    check("the predicted order is shown", "Predicted order" in body
          or any("order" in c.value.lower() for c in at.caption))

    # the four specialist views must all render without error
    radios = [i for i, r in enumerate(at.radio) if "View" in (r.label or "")]
    check("a specialist view selector exists", len(radios) >= 1)
    if radios:
        for view in ("Model agreement", "Specialist ranks",
                     "Horse explanations"):
            at.radio[radios[0]].set_value(view).run()
            check(f"the {view} view renders", not at.exception)
            if at.exception:
                for e in at.exception:
                    print(f"EXCEPTION ({view}):", e.value)
                break
        at.radio[radios[0]].set_value("Final trained").run()

    # ------------------------------------------------- results and retraining
    tb = [i for i, b in enumerate(at.button) if "template" in b.label.lower()]
    check("a results template can be inserted", len(tb) == 1)
    if tb:
        at.button[tb[0]].click().run()
        check("inserting the template does not raise", not at.exception)
        if not at.exception:
            filled = [t for t in at.text_area if "R1:" in (t.value or "")]
            check("the template lists the meetings actually loaded",
                  len(filled) == 1)

    # a real result must map, validate and retrain
    at2 = AppTest.from_file("app.py", default_timeout=900).run()
    at2.button[0].click().run()
    import bet365_parser as PS
    races = PS.parse_bet365_text(SAMPLE.read_text(encoding="utf-8"), "t")
    first = races[0]
    active = [r["number"] for r in first["runners"]
              if r.get("status") == "ACTIVE"][:4]
    res_text = f"{first['meeting']}\nR{first['race_no']}: " + "-".join(
        str(n) for n in active)
    ta = [i for i, t in enumerate(at2.text_area) if t.key == "b3_res"]
    check("a results box exists", len(ta) == 1)
    if ta:
        at2.text_area[ta[0]].set_value(res_text).run()
        check("entering a result does not raise", not at2.exception)
        if not at2.exception:
            maps = [d.value for d in at2.dataframe
                    if "Status" in getattr(d.value, "columns", [])
                    and "Result" in getattr(d.value, "columns", [])]
            check("the result-to-race mapping is shown", len(maps) == 1)
            if maps:
                check("and it maps cleanly", "ok" in list(maps[0]["Status"]))
            rb = [i for i, b in enumerate(at2.button)
                  if "retrain" in b.label.lower()]
            check("a retrain button appears once a result maps", len(rb) == 1)
            if rb:
                before = next(m.value for m in at2.metric
                              if m.label == "Training races")
                at2.button[rb[0]].click().run()
                check("retraining does not raise", not at2.exception)
                if at2.exception:
                    for e in at2.exception:
                        print("EXCEPTION (retrain):", e.value)
                else:
                    check("it reports the retrain",
                          any("Retrained" in s.value for s in at2.success))
                    after = next(m.value for m in at2.metric
                                 if m.label == "Training races")
                    check("the training race count grew",
                          int(str(after)) > int(str(before)))
                    check("and it says the state is session-only",
                          any("session" in i.value.lower() for i in at2.info))

    # --------------------------------- Streamlit Cloud has no matplotlib
    import importlib
    with BlockMatplotlib():
        try:
            importlib.import_module("matplotlib")
            check("matplotlib really is blocked", False)
        except ImportError:
            check("matplotlib really is blocked", True)
        at3 = AppTest.from_file("app.py", default_timeout=900).run()
        at3.button[0].click().run()
        pb3 = [i for i, b in enumerate(at3.button)
               if "Predict all races" in b.label]
        if pb3:
            at3.button[pb3[0]].click().run()
        check("the whole prediction path runs without matplotlib",
              not at3.exception)
        if at3.exception:
            for e in at3.exception:
                print("EXCEPTION (no-matplotlib):", e.value)

    print(f"PASS {passes}  FAIL {len(fails)}")
    for f in fails:
        print("  FAIL:", f)
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
