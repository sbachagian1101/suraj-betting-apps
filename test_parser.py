"""Regression test for greyhound_parser against the real Geelong R9 clipboard paste."""
import greyhound_parser as gp

import os
RAW = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "tests_fixture_geelong_r9.txt"), encoding="utf-8").read()

# Ground truth read directly off the Racing & Sports page.
EXPECT = {
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


def close(a, b, tol=1e-6):
    try:
        return abs(float(a) - float(b)) <= tol
    except Exception:
        return False


def main():
    header, runners, warnings = gp.parse(RAW)
    fails, passes = [], 0

    def check(tab, label, got, want, ok):
        nonlocal passes
        if ok:
            passes += 1
        else:
            fails.append(f"  #{tab} {label}: got {got!r}, want {want!r}")

    print("HEADER:", header)
    print("WARNINGS:", warnings or "none")
    by_tab = {r["tab"]: r for r in runners}
    for tab, exp in EXPECT.items():
        r = by_tab.get(tab)
        if r is None:
            fails.append(f"  #{tab} missing runner entirely")
            continue
        check(tab, "name", r.get("horse"), exp["name"], r.get("horse") == exp["name"])
        if exp.get("scratched"):
            check(tab, "scratched", r.get("scratched"), True, bool(r.get("scratched")))
        if "form" in exp:
            check(tab, "form", r.get("form"), exp["form"], r.get("form") == exp["form"])
        if "bf" in exp:
            check(tab, "bf_odds", r.get("bf_odds"), exp["bf"], close(r.get("bf_odds"), exp["bf"]))
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

    print(f"\nPASS {passes}  FAIL {len(fails)}")
    if fails:
        print("\n".join(fails))
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
