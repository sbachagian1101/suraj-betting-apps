"""Regression test for harness_parser against a real R&S harness paste.

Ground truth is read directly off the Racing & Sports Gloucester Park R2 page,
not copied from parser output, so the two are independent.

    python test_parser.py     # expect: PASS <n>  FAIL 0

If a page ever parses to fewer runners than its field table shows, save the
paste as a new fixture and add it to FIXTURES below.
"""
import os

import harness_parser as hp

_HERE = os.path.dirname(os.path.abspath(__file__))


def _load(name):
    return open(os.path.join(_HERE, name), encoding="utf-8").read()


# Gloucester Park R2, 2130m ALL WEATHER GOOD, 2yo. 9 runners, 2 scratched.
EXPECT_GP = {
    1: dict(name="ASTERI ADELPHOS", driver="GARY HALL JNR", trainer="JUSTIN PRENTICE",
            tab=10.0, bf=18.5, form="6x22", dri=0.22, dri_plc=0.54, dri_n=50,
            trn=0.16, trn_plc=0.36, dt=0.36, dt_n=810, dls=15, runs=3,
            age=2, sex="Colt", career=(0, 2, 3), crs=(0, 0, 1), dist=(0, 0, 0),
            last_fin=2, last_margin=11.8, last_sp=1.35, last_hcp="SR2",
            ohr=44, imr=120.24, mra=0.34),
    2: dict(name="KRESCENDO", driver="JOCELYN YOUNG", trainer="JOCELYN YOUNG",
            tab=101.0, bf=210.0, form="9", dri=0.12, trn=0.12, dt=0.17,
            dls=15, runs=1, career=(0, 0, 1),
            last_fin=9, last_margin=22.6, last_sp=20.0, last_hcp="SR4", imr=122.55),
    3: dict(name="FRONTMAN", driver="KYLE SYMINGTON", trainer="K L HOWLETT",
            tab=71.0, bf=180.0, form="673", dri=0.06, trn=0.26, dt=0.31, dt_n=16,
            dls=15, runs=3, sex="Gelding", career=(0, 1, 3), crs=(0, 0, 1),
            last_fin=3, last_margin=14.4, last_sp=14.0, last_hcp="FR5", imr=120.38),
    4: dict(name="THEMARASSHADOW", driver="AIDEN DE CAMPO", trainer="AIDEN DE CAMPO",
            tab=4.2, bf=6.8, form="415x", dri=0.20, trn=0.20, dt=0.21, dt_n=2420,
            dls=113, runs=3, career=(1, 0, 3),
            last_fin=5, last_margin=9.3, last_sp=5.50, last_hcp="FR7",
            imr=116.93, mra=-0.69),
    5: dict(name="RIP TIDE", driver="RYAN WARWICK", trainer="TERRY FERGUSON",
            tab=26.0, bf=55.0, form="32x26", dri=0.08, trn=0.04, dt=0.05, dt_n=43,
            dls=18, runs=4, career=(0, 3, 4), crs=(0, 3, 4), dist=(0, 1, 2),
            last_fin=6, last_margin=57.6, last_sp=41.0, last_hcp="SR1", imr=120.50),
    6: dict(name="ACKNOWLEDGEMENT", driver="CODY WALLRODT", trainer="PETER ANDERSON",
            tab=151.0, bf=300.0, form="64635", dri=0.06, trn=0.06, dt=0.06, dt_n=110,
            dls=3, runs=5, career=(0, 1, 9), crs=(0, 1, 2),
            last_fin=5, last_margin=6.6, last_sp=13.0, last_hcp="FR4", imr=119.58),
    7: dict(name="IGNITE THE FURNACE", driver="TRENT WHEELER", trainer="SARAH WALL",
            tab=1.15, bf=1.27, form="1x23", dri=0.02, trn=0.08, dt=0.08, dt_n=64,
            dls=18, runs=3, sex="Gelding", career=(1, 2, 3), crs=(0, 2, 2),
            last_fin=3, last_margin=9.5, last_sp=34.0, last_hcp="FR9",
            imr=117.76, ohr=47),
    # Scratched, and the page shows no price line at all for this runner.
    8: dict(name="ALLAMERICAN EAGLE", scratched=True, driver="TRENT WHEELER",
            trainer="PETER ANDERSON", form="x9627", dri=0.02, trn=0.06, runs=0),
    # Scratched; price prefix is the unusual composite token "USR|GRS".
    9: dict(name="SPYCHIEF", scratched=True, driver="LIAM ELLIOTT",
            trainer="RYAN BELL", form="x6338", dri=0.02, trn=0.06, runs=0),
}

EXPECT_HEADER_GP = dict(track="Gloucester Park", race_no=2, distance_m=2130,
                        surface="ALL WEATHER", going="GOOD", race_type="2yo",
                        fastest_time="2:30.20")

FIXTURES = [
    ("Gloucester Park R2 - 2yo 2130m",
     "tests_fixture_gloucesterpark_r2.txt", EXPECT_GP, EXPECT_HEADER_GP),
]

_WPS_KEYS = {"career": "career", "crs": "course", "dist": "distance",
             "cd": "course_distance", "aw": "aw", "fu": "first_up"}


def close(a, b, tol=1e-6):
    try:
        return abs(float(a) - float(b)) <= tol
    except (TypeError, ValueError):
        return False


def run_fixture(title, filename, expect, exp_header, fails, counter):
    header, runners, warnings = hp.parse(_load(filename))

    def check(tab, label, got, want, ok):
        if ok:
            counter[0] += 1
        else:
            fails.append(f"  [{title}] #{tab} {label}: got {got!r}, want {want!r}")

    print(f"--- {title}")
    print("HEADER:", header)
    print("WARNINGS:", warnings or "none")

    for key, want in exp_header.items():
        check(0, f"header.{key}", header.get(key), want, header.get(key) == want)
    # Every runner in the field table must survive parsing.
    check(0, "runner count", len(runners), len(expect), len(runners) == len(expect))

    by_tab = {r["tab"]: r for r in runners}
    for tab, exp in expect.items():
        r = by_tab.get(tab)
        if r is None:
            fails.append(f"  [{title}] #{tab} missing runner entirely")
            continue
        check(tab, "horse", r.get("horse"), exp["name"], r.get("horse") == exp["name"])
        check(tab, "scratched", bool(r.get("scratched")), exp.get("scratched", False),
              bool(r.get("scratched")) is exp.get("scratched", False))
        check(tab, "gate", r.get("gate"), tab, r.get("gate") == tab)
        for key, field in (("driver", "driver"), ("trainer", "trainer"),
                           ("form", "form"), ("sex", "sex")):
            if key in exp:
                check(tab, field, r.get(field), exp[key], r.get(field) == exp[key])
        for key, field in (("tab", "tab_odds"), ("bf", "bf_odds"),
                           ("dri", "driver_win"), ("dri_plc", "driver_place"),
                           ("trn", "trainer_win"), ("trn_plc", "trainer_place"),
                           ("dt", "driver_trainer_win")):
            if key in exp:
                check(tab, field, r.get(field), exp[key], close(r.get(field), exp[key]))
        for key, field in (("dri_n", "driver_l50_n"), ("dt_n", "driver_trainer_n"),
                           ("dls", "dls"), ("age", "age"), ("last_fin", "last_fin")):
            if key in exp:
                check(tab, field, r.get(field), exp[key], r.get(field) == exp[key])
        if "runs" in exp:
            got = len(r.get("recent_runs", []))
            check(tab, "recent run count", got, exp["runs"], got == exp["runs"])
        for key, prefix in _WPS_KEYS.items():
            if key in exp:
                got = (r.get(f"{prefix}_wins"), r.get(f"{prefix}_places"),
                       r.get(f"{prefix}_starts"))
                check(tab, f"{prefix} W-P-S", got, exp[key], got == exp[key])
        runs = r.get("recent_runs", [])
        last = runs[0] if runs else {}
        for key, field in (("last_margin", "margin"), ("last_sp", "sp"),
                           ("imr", "imr"), ("mra", "mile_rate_adj")):
            if key in exp:
                check(tab, f"last run {field}", last.get(field), exp[key],
                      close(last.get(field), exp[key]))
        if "last_hcp" in exp:
            check(tab, "last run hcp", last.get("hcp"), exp["last_hcp"],
                  last.get("hcp") == exp["last_hcp"])
        if "ohr" in exp:
            got = next((x["ohr"] for x in runs if x.get("ohr")), None)
            check(tab, "latest OHR", got, exp["ohr"], got == exp["ohr"])
    # Barrier trials must never be counted as completed runs.
    for tab, r in by_tab.items():
        for run in r.get("recent_runs", []):
            check(tab, "no barrier trial in runs", run.get("field"), ">=2",
                  int(run.get("field", 0)) >= 2)
    print()


def main():
    fails, counter = [], [0]
    for title, filename, expect, exp_header in FIXTURES:
        run_fixture(title, filename, expect, exp_header, fails, counter)
    print(f"PASS {counter[0]}  FAIL {len(fails)}")
    if fails:
        print("\n".join(fails))
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
