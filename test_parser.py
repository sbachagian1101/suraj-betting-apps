"""Regression test for horse_parser against a real R&S thoroughbred paste.

Ground truth is read directly off the Racing & Sports Tamworth R4 page, not
copied from parser output, so the two are independent of each other.

Run this whenever Racing & Sports change their page markup:

    python test_parser.py     # expect: PASS <n>  FAIL 0

If you hit a page that parses to fewer runners than the field table shows, save
the paste as a new fixture and add it to FIXTURES below.
"""
import os

import horse_parser as hp

_HERE = os.path.dirname(os.path.abspath(__file__))


def _load(name):
    return open(os.path.join(_HERE, name), encoding="utf-8").read()


# Tamworth R4, 1400m TURF SOFT 5, MDN. 13 runners, 3 scratched.
EXPECT_TAMWORTH = {
    1: dict(name="AKUMA", wt=59.0, bp=8, jockey="KODY NESTOR", jrat=3.2,
            trainer="SALLY TORRENS", trat=3.2, tab=23.0, bf=28.0, form="58x74",
            jky_win=0.08, trn_win=0.04, dslr=11, runs=5, age=5, sex="Gelding",
            car=(0, 1, 12), m12=(0, 1, 9), crs=(0, 1, 1), dist=(0, 0, 3),
            good=(0, 1, 8), soft=(0, 0, 4), fu=(0, 0, 4),
            last_fin=4, last_margin=3.7, last_sp=8.5, runup=3),
    2: dict(name="GRIKO", scratched=True, wt=59.0, jockey="GABRIELLE JOHNSTON",
            claim=2.0, jrat=2.8, trainer="STEPHEN GLEESON", trat=1.9,
            form="00", jky_win=0.18, trn_win=0.04),
    # No jockey declared: the empty cell must not shift the trainer/rating columns.
    3: dict(name="MUSIC MAESTRO", scratched=True, wt=59.0, jockey="", jrat=0.0,
            trainer="TROY O'NEILE", trat=3.1, form="579x8", trn_win=0.04),
    4: dict(name="PUISSANCE POWER", wt=59.0, bp=5, jockey="DIEGO MONTES DE OCA",
            jrat=2.0, trainer="MICHAEL MULHOLLAND", trat=3.5, tab=3.5, bf=4.4,
            form="56x22", jky_win=0.10, trn_win=0.18, dslr=8, runs=4,
            car=(0, 3, 8), m12=(0, 2, 2), dist=(0, 0, 3), good=(0, 1, 6),
            last_fin=2, last_margin=1.3, last_sp=2.6, runup=3),
    5: dict(name="PURPLE PRINCE", scratched=True, wt=59.0, jockey="OLIVIA DALTON",
            claim=2.0, jrat=3.2, trainer="JEREMY SYLVESTER", trat=2.0, form="28364"),
    6: dict(name="TAKETHEMONEYANDRUN", wt=59.0, bp=0, jockey="ANNA ROPER",
            claim=0.0, jrat=3.1, trainer="SALLY TORRENS", trat=3.2, tab=21.0,
            bf=32.0, form="045x7", jky_win=0.06, trn_win=0.04, dslr=11, runs=5,
            car=(0, 5, 14), m12=(0, 4, 11), dist=(0, 3, 5), soft=(0, 1, 6),
            last_fin=7, last_margin=3.2, last_sp=31.0, runup=2),
    # Page shows `Tab$10` (no Betfair market) -> bf_odds falls back to that price.
    7: dict(name="ALBERT'S PICK", wt=57.0, bp=4, jockey="GRACE PALMER", claim=3.0,
            jrat=2.6, trainer="TROY O'NEILE", trat=3.1, tab=10.0, bf=10.0,
            form="73558", jky_win=0.04, trn_win=0.04, dslr=21, runs=5,
            car=(0, 3, 15), m12=(0, 3, 14), dist=(0, 1, 5), heavy=(0, 0, 1),
            last_fin=8, last_margin=18.6, last_sp=7.0, runup=8),
    8: dict(name="DYNEEMA", wt=57.0, bp=7, jockey="JAKE PRACEY-HOLMES", jrat=3.9,
            trainer="MARK MILTON", trat=1.9, tab=23.0, bf=34.0, form="76",
            jky_win=0.14, trn_win=0.04, dslr=8, runs=2, car=(0, 0, 2),
            last_fin=6, last_margin=8.1, last_sp=41.0, runup=3),
    9: dict(name="LIGHTNING PRINCESS", wt=57.0, bp=10, jockey="JENNY DUGGAN",
            jrat=3.2, trainer="IKY FOUSTOK", trat=2.3, tab=126.0, bf=170.0,
            form="x7080", jky_win=0.04, trn_win=0.10, dslr=21, runs=4,
            car=(0, 0, 6), m12=(0, 0, 4), last_fin=12, last_margin=16.0, last_sp=151.0, runup=5),
    10: dict(name="REDALUCA GIRL", wt=57.0, bp=2, jockey="JASMINE URQUHART-WARREN",
             claim=3.0, jrat=1.8, trainer="MARK MILTON", trat=1.9, tab=16.0,
             bf=20.0, form="24566", jky_win=0.08, trn_win=0.04, dslr=30, runs=5,
             car=(0, 9, 21), m12=(0, 7, 12), soft=(0, 5, 12), heavy=(0, 2, 2),
             last_fin=6, last_margin=5.3, last_sp=17.0, runup=9),
    11: dict(name="SHAKESPEARE'S GIRL", wt=57.0, bp=6, jockey="DONOVAN DILLON",
             jrat=2.5, trainer="PATRICK CLEAVE", trat=1.9, tab=9.0, bf=9.6,
             form="x33x6", jky_win=0.04, trn_win=0.02, dslr=16, runs=3,
             car=(0, 2, 7), dist=(0, 2, 3), last_fin=6, last_margin=3.7, last_sp=4.8, runup=2),
    12: dict(name="SIDE QUEST", wt=57.0, bp=9, jockey="BEN LOOKER", jrat=3.8,
             trainer="HOLLY WILLIAMS", trat=1.7, tab=3.9, bf=4.7, form="62423",
             jky_win=0.10, trn_win=0.04, dslr=19, runs=4, car=(0, 3, 6),
             crs=(0, 1, 2), last_fin=3, last_margin=4.5, last_sp=4.6, runup=7),
    13: dict(name="TRALEE", wt=57.0, bp=3, jockey="RORY HUTCHINGS", jrat=4.0,
             trainer="MELISSA DENNETT", trat=3.0, tab=4.0, bf=4.6, form="57x33",
             jky_win=0.18, trn_win=0.10, dslr=16, runs=4, car=(0, 2, 4),
             crs=(0, 1, 3), good=(0, 2, 4), last_fin=3, last_margin=0.7, last_sp=8.5, runup=3),
}

EXPECT_HEADER_TAMWORTH = dict(track="Tamworth", race_no=4, distance_m=1400,
                              surface="TURF", going="SOFT", going_rating=5,
                              race_type="MDN")

FIXTURES = [
    ("Tamworth R4 - MDN 1400m soft",
     "tests_fixture_tamworth_r4.txt", EXPECT_TAMWORTH, EXPECT_HEADER_TAMWORTH),
]

_WPS_KEYS = {"car": "Car", "m12": "M12", "crs": "Crs", "dist": "Dist",
             "cd": "CrsDist", "good": "Good", "soft": "Soft",
             "heavy": "Heavy", "fu": "FU"}


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
        for key, field in (("wt", "wt"), ("jrat", "jrat"), ("trat", "trat"),
                           ("claim", "claim"), ("tab", "tab_odds"), ("bf", "bf_odds"),
                           ("jky_win", "jky_win"), ("trn_win", "trn_win"),
                           ("last_margin", "last_margin"), ("last_sp", "last_sp")):
            if key in exp:
                check(tab, field, r.get(field), exp[key], close(r.get(field), exp[key]))
        for key, field in (("bp", "bp"), ("dslr", "dslr"), ("last_fin", "last_fin"),
                           ("age", "age"), ("runup", "runup")):
            if key in exp:
                check(tab, field, r.get(field), exp[key], r.get(field) == exp[key])
        for key, field in (("jockey", "jockey"), ("trainer", "trainer"),
                           ("form", "form"), ("sex", "sex")):
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
