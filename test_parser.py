"""Regression tests for greyhound_parser against real R&S clipboard pastes.

Two fixtures, because R&S serve more than one column set for the same Enhanced
Form page:

* Geelong R9  - carries a WT column, prices in a trailing unnamed column.
* Q Lakeside R9 - no WT column at all; prizemoney and bookmaker columns instead.

The second layout is why runners were once silently dropped, so both are pinned.
"""
import greyhound_parser as gp

import os
_HERE = os.path.dirname(os.path.abspath(__file__))


def _load(name):
    return open(os.path.join(_HERE, name), encoding="utf-8").read()


# Ground truth read directly off the Racing & Sports page.
EXPECT_GEELONG = {
    1: dict(name="FADE OUT", form="41544", bf=22.0, trn_w=0.14, trn_p=0.36, best=23.31,
            career="17%-17%-6", crs="0%-0%-2", dist="17%-17%-6", cd="0%-0%-2", dls=7,
            box=1, box_ws=(0, 0), last_fin=4, last_mgn=5.1, last_mrk=0.80, last_split=8.68, last_settle=4),
    2: dict(name="SONIA'S SHARK", scratched=True, form="85676", trn_w=0.34, trn_p=0.60),
    3: dict(name="RAPID WATERBOY", form="17658", bf=400.0, trn_w=0.12, trn_p=0.30, best=23.20,
            career="2%-21%-42", crs="4%-25%-24", dist="2%-20%-41", cd="4%-25%-24", dls=8,
            box=3, box_ws=(0, 1), last_fin=8, last_mgn=12.5, last_mrk=0.86, last_split=9.05, last_settle=8),
    4: dict(name="SCARY STORM", form="42743", bf=1.62, trn_w=0.18, trn_p=0.56, best=22.88,
            career="11%-44%-9", crs="0%-100%-1", dist="0%-100%-1", cd="0%-100%-1", dls=31,
            box=4, box_ws=(0, 1), last_fin=3, last_mgn=14.0, last_mrk=1.08, last_split=5.29, last_settle=8),
    5: dict(name="SPEEDY JUDE", form="3466x", bf=6.6, trn_w=0.10, trn_p=0.38, best=22.93,
            career="12%-50%-8", dls=279,
            box=5, box_ws=(0, 0), last_fin=6, last_mgn=14.2, last_mrk=1.11, last_split=5.36, last_settle=5),
    6: dict(name="TRACKMAN TIM", form="35442", bf=5.5, trn_w=0.16, trn_p=0.40, best=23.01,
            career="8%-42%-12", crs="0%-40%-5", dist="10%-40%-10", cd="0%-50%-4", dls=4,
            box=6, box_ws=(0, 2), last_fin=2, last_mgn=4.3, last_mrk=0.29, last_split=4.03, last_settle=4),
    7: dict(name="SASSY BELLE", form="52x57", bf=95.0, trn_w=0.04, trn_p=0.30, best=23.04,
            career="10%-30%-10", crs="20%-40%-5", dist="14%-43%-7", cd="33%-67%-3", dls=18,
            box=7, box_ws=(0, 0), last_fin=7, last_mgn=12.9, last_mrk=1.30, last_split=6.76, last_settle=5),
    8: dict(name="PATS FOR ARTHUR", scratched=True, form="86717", trn_w=0.12, trn_p=0.30, best=23.59),
    9: dict(name="SNOOPY", form="78746", bf=150.0, trn_w=0.04, trn_p=0.14, best=22.92,
            career="5%-26%-19", crs="5%-26%-19", dist="5%-26%-19", cd="5%-26%-19", dls=7,
            box=2, box_ws=(1, 3), last_fin=6, last_mgn=8.4, last_mrk=1.03, last_split=8.74, last_settle=6),
}

# Q Lakeside R9 550m - the layout with no WT column.
EXPECT_QLAKESIDE = {
    1: dict(name="NANGAR ZARA", form="41215", bf=1.53, tab=1.5, trn_w=0.28, trn_p=0.54, best=30.39,
            career="9-5-26", crs="3-2-9", dist="2-2-11", cd="2-1-4", dls=9,
            box=1, box_ws=(2, 4), last_fin=5, last_mgn=10.5, last_mrk=0.75, last_split=9.60, last_settle=3),
    2: dict(name="LAUNCH MODE", form="22428", bf=15.5, tab=9.5, trn_w=0.14, trn_p=0.54,
            career="6-8-32", crs="0-1-2", dist="3-4-20", cd="0-0-0", dls=23,
            box=2, box_ws=(1, 5), last_fin=8, last_mgn=9.0, last_mrk=0.65, last_split=5.27, last_settle=6),
    3: dict(name="HOOKED ON DOLLY", form="23148", bf=50.0, tab=23.0, trn_w=0.16, trn_p=0.50,
            career="3-9-21", crs="2-3-8", dist="1-3-7", cd="0-0-0", dls=9,
            box=3, box_ws=(1, 2), last_fin=8, last_mgn=8.8, last_mrk=1.20, last_split=9.82, last_settle=6),
    4: dict(name="SCOTCH ROCKET", form="23113", bf=42.0, tab=26.0, trn_w=0.24, trn_p=0.48, best=31.08,
            career="9-16-42", crs="0-0-1", dist="7-8-24", cd="0-0-1", dls=6,
            box=4, box_ws=(1, 8), last_fin=3, last_mgn=6.5, last_mrk=0.54, last_split=9.46, last_settle=6),
    5: dict(name="SPRING FOCUS", form="53542", bf=180.0, tab=51.0, trn_w=0.14, trn_p=0.42, best=30.84,
            career="2-14-40", crs="1-6-23", dist="0-5-13", cd="0-2-5", dls=7,
            box=5, box_ws=(0, 5), last_fin=2, last_mgn=4.8, last_mrk=0.34, last_split=9.25, last_settle=4),
    6: dict(name="ALLEY ACRES", scratched=True, form="81587", trn_w=0.14, trn_p=0.40),
    7: dict(name="TRUE HORNET", form="24564", bf=18.5, tab=14.0, trn_w=0.16, trn_p=0.42,
            career="2-2-10", crs="0-1-2", dist="2-1-6", cd="0-0-0", dls=6,
            box=7, box_ws=(1, 1), last_fin=4, last_mgn=3.3, last_mrk=0.23, last_settle=5),
    8: dict(name="BLOW BAYOU", form="51261", bf=3.9, tab=3.3, trn_w=0.18, trn_p=0.56, best=30.52,
            career="8-7-23", crs="4-5-15", dist="1-0-2", cd="1-0-1", dls=7,
            box=8, box_ws=(1, 4), last_fin=1, last_mgn=4.8, last_mrk=0.00, last_split=9.06, last_settle=2),
}

FIXTURES = [
    ("Geelong R9 - WT-column layout", "tests_fixture_geelong_r9.txt", EXPECT_GEELONG),
    ("Q Lakeside R9 - no-WT layout", "tests_fixture_qlakeside_r9.txt", EXPECT_QLAKESIDE),
]


def close(a, b, tol=1e-6):
    try:
        return abs(float(a) - float(b)) <= tol
    except Exception:
        return False


def run_fixture(title, filename, expect, fails, counter):
    header, runners, warnings = gp.parse(_load(filename))

    def check(tab, label, got, want, ok):
        if ok:
            counter[0] += 1
        else:
            fails.append(f"  [{title}] #{tab} {label}: got {got!r}, want {want!r}")

    print(f"--- {title}")
    print("HEADER:", header)
    print("WARNINGS:", warnings or "none")
    # Every runner in the field table must survive parsing.
    check(0, "runner count", len(runners), len(expect), len(runners) == len(expect))
    by_tab = {r["tab"]: r for r in runners}
    for tab, exp in expect.items():
        r = by_tab.get(tab)
        if r is None:
            fails.append(f"  [{title}] #{tab} missing runner entirely")
            continue
        check(tab, "name", r.get("horse"), exp["name"], r.get("horse") == exp["name"])
        if exp.get("scratched"):
            check(tab, "scratched", r.get("scratched"), True, bool(r.get("scratched")))
        if "form" in exp:
            check(tab, "form", r.get("form"), exp["form"], r.get("form") == exp["form"])
        if "bf" in exp:
            check(tab, "bf_odds", r.get("bf_odds"), exp["bf"], close(r.get("bf_odds"), exp["bf"]))
        if "tab" in exp:
            check(tab, "tab_odds", r.get("tab_odds"), exp["tab"], close(r.get("tab_odds"), exp["tab"]))
        if "trn_w" in exp:
            check(tab, "trainer_win", r.get("trainer_win"), exp["trn_w"], close(r.get("trainer_win"), exp["trn_w"]))
            check(tab, "trainer_place", r.get("trainer_place"), exp["trn_p"], close(r.get("trainer_place"), exp["trn_p"]))
        if "best" in exp:
            check(tab, "tra_dist_best", r.get("tra_dist_best"), exp["best"], close(r.get("tra_dist_best"), exp["best"]))
        for key, field in (("career", "career_rec"), ("crs", "course_rec"), ("dist", "distance_rec"), ("cd", "course_distance_rec")):
            if key in exp:
                check(tab, field, r.get(field), exp[key], r.get(field) == exp[key])
        if "dls" in exp:
            check(tab, "dls", r.get("dls"), exp["dls"], r.get("dls") == exp["dls"])
        if "box" in exp:
            check(tab, "box", r.get("box"), exp["box"], r.get("box") == exp["box"])
        if "box_ws" in exp:
            bs = r.get("box_stats", {}).get(r.get("box", 0), {})
            got = (bs.get("wins", -1), bs.get("starts", -1))
            check(tab, "box W-S", got, exp["box_ws"], got == exp["box_ws"])
        runs = r.get("recent_runs", [])
        last = runs[0] if runs else {}
        for key, field in (("last_fin", "finish"), ("last_mgn", "margin"), ("last_mrk", "mrk_delta"),
                           ("last_split", "first_split"), ("last_settle", "settle_pos")):
            if key in exp:
                check(tab, field, last.get(field), exp[key], close(last.get(field, -999), exp[key]))
    print()


def main():
    fails, counter = [], [0]
    for title, filename, expect in FIXTURES:
        run_fixture(title, filename, expect, fails, counter)
    print(f"PASS {counter[0]}  FAIL {len(fails)}")
    if fails:
        print(chr(10).join(fails))
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
