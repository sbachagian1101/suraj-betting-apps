"""Regression tests for the meeting parser and the form model.

Pins the things that would break quietly rather than loudly: a triplet read as
its first number only, a form string where `0` is treated as a win, a race whose
rows run past the blank line into the next race, probabilities that stop summing
to one, and a score whose ordering does not match the probabilities it produces.

Runs against real meeting files when they are present in Downloads, and against
a built-in synthetic meeting when they are not — so the suite still means
something on a machine that does not have the spreadsheets.

    python test_form_model.py     # expect: PASS <n>  FAIL 0
"""
import glob
import io
import os

import numpy as np
import pandas as pd

import form_model as fm
import meeting_parser as mp

REAL = sorted(glob.glob(os.path.expanduser(
    "C:/Users/Admin/Downloads/2026-08-27-*-T.xlsx")))


class Checker:
    def __init__(self):
        self.passes = 0
        self.fails = []

    def check(self, label, got, want, ok=None):
        ok = (got == want) if ok is None else ok
        if ok:
            self.passes += 1
        else:
            self.fails.append(f"  {label}: got {got!r}, want {want!r}")

    def true(self, label, cond):
        self.check(label, cond, True, bool(cond))

    def close(self, label, got, want, tol=1e-9):
        self.check(label, got, want, abs(float(got) - float(want)) <= tol)


def synthetic_grid(n_races=2, n_runners=6):
    """A meeting laid out exactly like the real export, built from scratch."""
    rows = []
    for r in range(1, n_races + 1):
        rows.append([r, None, f"Race {r} Handicap"] + [None] * 18)
        rows.append([None, None, "Type : 4U HCP"] + [None] * 18)
        rows.append([f"1{r}:00", None, f"{1200 + 100*r}m, TURF G"] + [None] * 18)
        rows.append([None] * 21)
        rows.append(list(mp.HEADER))
        for t in range(1, n_runners + 1):
            rows.append([
                t, f"HORSE {r}{t}", "12345"[:5], t,
                f"{t}-{t*2}-{10+t}", f"{t}-{t*3}-{40+t}", f"{t}-{t*2}-{t}", 10 + t,
                f"{t}-{t}-{t}", 2.0 + t / 10, 3.0 + t / 10, f"${100*t}",
                f"{t}-{t}-{t}", f"{t}-{t*2}-{t}", f"{t}-{t}-{t}", f"{t}-{t}-{t}",
                f"${1000*t}", f"{t}-{t}.5L-ABC5-1200m-4", f"{t}-{t}-{t}",
                f"{t}-{t}-{t}", f"{t}-{t}-{t}"])
        rows += [[None] * 21, [None] * 21]
    return pd.DataFrame(rows)


def main():
    c = Checker()

    # ---- triplet splitting ------------------------------------------------
    c.check("triplet splits to three numbers",
            mp.split_triplet("12-28-40"), (12.0, 28.0, 40.0))
    c.check("a zero triplet is zeros, not missing",
            mp.split_triplet("0-0-0"), (0.0, 0.0, 0.0))
    c.check("whitespace tolerated", mp.split_triplet(" 5-10-20 "),
            (5.0, 10.0, 20.0))
    for bad in ("", "-", "abc", "1-2", "1-2-3-4", None, 12.0):
        got = mp.split_triplet(bad)
        c.true(f"{bad!r} yields NaNs", all(np.isnan(x) for x in got))
    c.true("a triplet is never read as its first number alone",
           mp.split_triplet("12-28-40")[2] == 40.0)

    # ---- form figures -----------------------------------------------------
    c.check("0 counts as a bad run, not a win", mp.form_figures("0"), (10.0, 1))
    c.check("a win is 1", mp.form_figures("1"), (1.0, 1))
    c.check("spells are skipped", mp.form_figures("x1x"), (1.0, 1))
    c.check("mean over the digits present", mp.form_figures("123"), (2.0, 3))
    c.check("count reflects runs read", mp.form_figures("12345")[1], 5)
    m, k = mp.form_figures("-")
    c.true("a first-starter has no figure", np.isnan(m) and k == 0)
    m, k = mp.form_figures(None)
    c.true("a missing form string has no figure", np.isnan(m) and k == 0)
    c.true("0 is worse than 9", mp.form_figures("0")[0] > mp.form_figures("9")[0])

    # ---- filename handling ------------------------------------------------
    c.check("track name from filename",
            mp.track_name("2026-08-27-FFOS-LAS-T.xlsx"), "FFOS-LAS")
    c.check("single-word track",
            mp.track_name("2026-08-27-NAVAN-T.xlsx"), "NAVAN")
    c.check("date from filename",
            mp.meeting_date("2026-08-27-NAVAN-T.xlsx"), "2026-08-27")
    c.true("a filename without a date gives None",
           mp.meeting_date("NAVAN-T.xlsx") is None)

    # ---- parsing a synthetic meeting --------------------------------------
    grid = synthetic_grid(n_races=3, n_runners=7)
    rows = mp.parse_grid(grid, "TESTTRACK", "2026-08-27")
    df = mp.to_frame(rows)
    c.check("every runner of every race read", len(df), 21)
    c.check("three races found", df.race_id.nunique(), 3)
    c.true("blocks do not bleed into each other",
           all(len(g) == 7 for _, g in df.groupby("race_id")))
    c.true("tabs are 1..n in each race",
           all(sorted(g.tab.tolist()) == list(range(1, 8))
               for _, g in df.groupby("race_id")))
    c.check("race numbers recovered", sorted(df.race.unique().tolist()), [1, 2, 3])
    c.check("distance recovered for race 2",
            float(df[df.race == 2].dist.iloc[0]), 1400.0)
    c.check("surface recovered", df.surface.iloc[0], "TURF")
    c.true("no parser warnings on a clean meeting", mp.warnings_for(df) == [])
    c.true("career triplet became three columns",
           {"car_win", "car_plc", "car_runs"} <= set(df.columns))
    c.check("career runs read from the third slot",
            float(df.car_runs.iloc[0]), 41.0)
    c.true("prize money parsed past the dollar sign",
           float(df.pmcar.iloc[0]) == 1000.0)
    c.true("last-start margin parsed", float(df.ls_mgn.iloc[0]) == 1.5)
    c.true("surface record chosen for turf",
           np.allclose(df.surf_win.values, df.turf_win.values))

    # a race that is short one tab must be reported, not silently accepted
    short = mp.to_frame(mp.parse_grid(synthetic_grid(1, 6), "T", None))
    short = short[short.tab != 3]
    c.true("a missing tab raises a warning",
           any("not 1.." in w for w in mp.warnings_for(short)))

    # ---- scoring ----------------------------------------------------------
    one = df[df.race_id == df.race_id.iloc[0]]
    t = fm.rate_race(one)
    n = len(one)
    c.check("one row per runner", len(t), n)
    c.check("ranks are 1..n", list(t["Rank"]), list(range(1, n + 1)))
    c.close("win probabilities sum to 100", t["Win%"].sum(), 100.0, 1e-6)
    c.true("win probability falls with rank",
           bool(np.all(np.diff(t["Win%"].values) <= 1e-9)))
    c.true("every win probability is positive", bool((t["Win%"] > 0).all()))
    c.close("best form score is 100", t["Form score"].max(), 100.0, 1e-6)
    c.close("worst form score is 10", t["Form score"].min(), 10.0, 1e-6)
    c.true("form score falls with rank",
           bool(np.all(np.diff(t["Form score"].values) <= 1e-9)))
    c.true("fair win price is the reciprocal",
           bool(np.allclose(t["Fair win $"], 100.0 / t["Win%"])))
    c.true("every runner has an explanation",
           bool(t["Why"].astype(str).str.len().gt(0).all()))
    c.true("the top pick's explanation cites a real column",
           any(lbl in t["Why"].iloc[0] for lbl in fm.LABELS.values()))

    # ---- place probability -----------------------------------------------
    pw = fm.win_probability(np.array([2.0, 1.0, 0.5, 0.0, -1.0]))
    for k in (1, 2, 3):
        pp = fm.place_probability(pw, k, sims=20000, seed=1)
        c.close(f"place probabilities sum to {k}", pp.sum(), k, 0.05)
        c.true(f"places={k}: place chance is at least win chance",
               bool(np.all(pp >= pw - 0.02)))
        c.true(f"places={k}: ordering matches win order",
               bool(np.all(np.diff(pp) <= 0.02)))
    c.true("places=1 reproduces the win probabilities",
           bool(np.allclose(fm.place_probability(pw, 1, sims=40000, seed=2),
                            pw, atol=0.02)))
    c.true("a single runner always places",
           float(fm.place_probability(np.array([1.0]), 1)[0]) == 1.0)
    c.true("shrink pulls toward the base rate",
           fm.place_probability(pw, 3, shrink=1.0, sims=4000).std() < 1e-6)
    c.check("no shrink by default", fm.DEFAULT_PLACE_SHRINK, 0.0)

    # ---- place terms ------------------------------------------------------
    c.check("4 runners is win only", fm.places_paid(4), 1)
    c.check("5 runners pay 2", fm.places_paid(5), 2)
    c.check("7 runners pay 2", fm.places_paid(7), 2)
    c.check("8 runners pay 3", fm.places_paid(8), 3)

    # ---- degenerate races -------------------------------------------------
    c.true("an empty race returns an empty table", fm.rate_race(one.head(0)).empty)
    solo = fm.rate_race(one.head(1))
    c.check("a one-runner race still scores", len(solo), 1)
    c.close("and gets the whole probability", float(solo["Win%"].iloc[0]), 100.0, 1e-6)
    flat = one.copy()
    for col in fm.WEIGHTS:
        if col in flat.columns:
            flat[col] = 1.0
    ft = fm.rate_race(flat)
    c.true("identical runners get identical probabilities",
           float(ft["Win%"].max() - ft["Win%"].min()) < 1e-6)
    c.true("and a mid form score", abs(float(ft["Form score"].iloc[0]) - 50.0) < 1e-6)

    # ---- the published constants must stay self-consistent ----------------
    c.true("top-rated beats a dart throw",
           fm.RANK_STATS[1]["win"] > fm.BASELINE["win"])
    c.true("top-rated places more than a dart throw",
           fm.RANK_STATS[1]["place"] > fm.BASELINE["place"])
    c.true("rank 1 wins more than rank 5",
           fm.RANK_STATS[1]["win"] > fm.RANK_STATS[5]["win"])
    c.true("ranks 1-3 sum to the top-3 winner coverage",
           abs(sum(fm.RANK_STATS[r]["win"] for r in (1, 2, 3))
               - fm.MEASURED["top3_has_winner"]) < 0.01)
    for r, st in fm.RANK_STATS.items():
        c.true(f"rank {r}: placing is at least as likely as winning",
               st["place"] >= st["win"])
    c.true("styler renders", fm.style(t).to_html() is not None)
    c.true("summary names the top pick", str(int(t.iloc[0]["Tab"])) in
           fm.summary_line(t))
    c.true("summary warns the top three are a group",
           "group" in fm.summary_line(t))

    # ---- against the real meetings, when they are on disk ------------------
    if REAL:
        real = mp.load(REAL)
        c.check("all nine meetings parse", len(REAL), 9)
        c.check("716 runners read", len(real), 716)
        c.check("66 races read", real.race_id.nunique(), 66)
        c.true("no runner lost to a blank name",
               bool(real.horse.astype(str).str.len().gt(0).all()))
        c.true("every race has contiguous tabs", mp.warnings_for(real) == [])
        tables = fm.rate_meeting(real)
        c.check("every race scored", len(tables), 66)
        c.true("every race's win probabilities sum to 100",
               all(abs(v["Win%"].sum() - 100) < 1e-6 for v in tables.values()))
        c.true("no NaN form scores",
               all(v["Form score"].notna().all() for v in tables.values()))
        big = max(tables.values(), key=len)
        c.true("the 26-runner field scores without trouble", len(big) >= 26)
        # Races must group by meeting: sorting on race number alone would
        # interleave the nine cards and make the selector unusable.
        order = mp.races(real)
        tracks = [real[real.race_id == r]["track"].iloc[0] for r in order]
        runs = [t for i, t in enumerate(tracks) if i == 0 or t != tracks[i - 1]]
        c.check("every race appears exactly once", len(order),
                real.race_id.nunique())
        c.check("each meeting is one contiguous run, not interleaved",
                len(runs), len(set(runs)))
        c.check("all nine meetings present", len(set(runs)), 9)
        for tk in sorted(set(tracks)):
            nums = [int(real[real.race_id == r]["race"].iloc[0])
                    for r in order
                    if real[real.race_id == r]["track"].iloc[0] == tk]
            c.check(f"{tk}: races ascend", nums, sorted(nums))
        c.check("first race is the first meeting's race 1", order[0],
                sorted(set(tracks))[0] + "_R1")

    else:
        print("  (real meeting files not found — skipped those checks)")

    print(f"PASS {c.passes}  FAIL {len(c.fails)}")
    if c.fails:
        print("\n".join(c.fails))
    return 1 if c.fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
