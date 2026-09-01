"""Regression tests for the parser, the rating and the Streamlit script.

Run with:  python -m pytest -q
"""

from __future__ import annotations

import io
import os

import pytest

import rating
import rs_parser
from rs_parser import GREYHOUND, HARNESS, THOROUGHBRED

HERE = os.path.dirname(os.path.abspath(__file__))

BULLI = "bulli_r6_2026-09-01.txt"
QLAKE = "qlakeside_r7_2026-09-01.txt"
HSHM = "horsham_r8_2026-09-01.txt"
DEAU = "deauville_r1_2026-09-01.txt"
CABO = "cabourg_r4_2026-09-01.txt"
ALL = [BULLI, QLAKE, HSHM, DEAU, CABO]


def load(name: str) -> rs_parser.Race:
    return rs_parser.parse(
        io.open(os.path.join(HERE, "fixtures", name), encoding="utf-8").read())


# --- header -------------------------------------------------------------------

@pytest.mark.parametrize("fx,code,track,rno,dist,surface,going,grade", [
    (BULLI, GREYHOUND, "Bulli", 6, 340, "AW", "GOOD", "5th Grade"),
    (QLAKE, GREYHOUND, "Q Lakeside", 7, 457, "AW", "GOOD", "5th Grade"),
    (HSHM, GREYHOUND, "Horsham", 8, 410, "AW", "GOOD", "Free For All"),
    (DEAU, THOROUGHBRED, "Deauville", 1, 1600, "TURF", "GOOD", "HCP CL2"),
    (CABO, HARNESS, "Cabourg", 4, 2750, "SAND", "GOOD", "F19"),
])
def test_header(fx, code, track, rno, dist, surface, going, grade):
    r = load(fx)
    assert r.code == code
    assert (r.track, r.race_no, r.dist_m) == (track, rno, dist)
    assert (r.surface, r.going, r.grade) == (surface, going, grade)
    assert r.race_date is not None and r.race_date.isoformat() == "2026-09-01"


@pytest.mark.parametrize("fx,tc", [
    (BULLI, "BULI"), (QLAKE, "QLAK"), (HSHM, "HSHM"),
    (DEAU, "DEA"), (CABO, "CBRG")])
def test_track_code_prefers_a_code_matching_the_track_name(fx, tc):
    """Frequency alone picks CTYA at Deauville - thoroughbreds travel."""
    assert load(fx).track_code() == tc


@pytest.mark.parametrize("fx,live,scr", [
    (BULLI, 8, 2), (QLAKE, 5, 4), (HSHM, 6, 1), (DEAU, 15, 1), (CABO, 15, 0)])
def test_field_sizes(fx, live, scr):
    r = load(fx)
    assert len(r.field_) == live
    assert sum(1 for x in r.runners if x.scratched) == scr


# --- runner block layout ------------------------------------------------------

def test_greyhound_name_is_the_dog_not_the_trainer():
    """Box 1 at Bulli is EXPLORE, *trained by* Frank Micallef."""
    one = next(x for x in load(BULLI).field_ if x.tab == 1)
    assert (one.name, one.trainer) == ("EXPLORE", "FRANK MICALLEF")


def test_thoroughbred_splits_weight_barrier_jockey_and_trainer():
    """The greyhound layout reads the WEIGHT as the trainer here."""
    calas = next(x for x in load(DEAU).field_ if x.tab == 1)
    assert calas.name == "CALAS"
    assert calas.weight == 61.5
    assert calas.barrier == 10
    assert calas.jockey == "MAXIME GUYON"
    assert calas.trainer == "PIA BRANDT"


def test_thoroughbred_strips_a_jockey_claim_from_the_name():
    c = next(x for x in load(DEAU).field_ if x.tab == 13)
    assert c.jockey == "MAXENCE MARQUETTE"      # printed as "(-0.5kg)"


def test_harness_splits_driver_and_trainer():
    """The greyhound layout reads the DRIVER as the trainer here."""
    m = next(x for x in load(CABO).field_ if x.tab == 1)
    assert m.name == "MENTIONNAL"
    assert m.driver == "FLORIAN DESMIGNEUX"
    assert m.trainer == "JULIEN RAFFESTIN"
    assert m.handler == "FLORIAN DESMIGNEUX"


def test_names_with_punctuation_and_country_suffix():
    assert any(x.name == "WHERE'S HARRY" for x in load(BULLI).runners)
    assert any(x.name == "ADOBE RAVE (NZ)" for x in load(QLAKE).runners)
    assert any(x.name == "PURPLE LION (IRE)" for x in load(DEAU).runners)


def test_every_live_runner_has_a_number_a_price_and_runs():
    for fx in ALL:
        for x in load(fx).field_:
            assert x.tab is not None, f"{fx}: {x.name} has no tab number"
            assert x.odds and x.odds > 1.0, f"{fx}: {x.name} has no price"
            assert x.runs, f"{fx}: {x.name} has no runs"


# --- record panel (glued labels) ---------------------------------------------

def test_glued_record_labels_are_peeled():
    meeki = next(x for x in load(HSHM).field_ if x.tab == 1)
    assert meeki.record("Career") == (23, 8, 3, 2)
    assert meeki.record("Dist") == (14, 7, 2, 1)
    assert (meeki.win_pct, meeki.place_pct) == (35.0, 57.0)


def test_zero_records_are_kept_not_dropped():
    ace = next(x for x in load(BULLI).field_ if x.tab == 2)
    assert ace.record("Course") == (0, 0, 0, 0)
    assert ace.record("Turf") == (7, 3, 2, 1)


def test_going_records_are_available_for_thoroughbred():
    calas = next(x for x in load(DEAU).field_ if x.tab == 1)
    assert calas.record("Good") == (7, 2, 1, 1)
    assert calas.record("Soft") == (6, 2, 1, 0)
    assert calas.record("Heavy") == (1, 0, 0, 0)


# --- run table ----------------------------------------------------------------

def test_greyhound_run_row_columns_are_read_by_header():
    first = next(x for x in load(BULLI).field_ if x.tab == 1).runs[0]
    assert (first.pos, first.field_size) == (7, 7)
    assert first.margin == pytest.approx(7.9)
    assert (first.track, first.dist_m, first.surface) == ("RIST", 324, "T")
    assert (first.box, first.sp, first.sectional) == (8, 41.0, 7.27)
    assert first.days_ago == 5


def test_empty_sectional_does_not_shift_columns():
    run = next(x for x in load(BULLI).field_ if x.tab == 1).runs[1]
    assert run.sectional is None
    assert run.sp == 23.0
    assert run.beat_or_beaten_by == "FLASH BOOM BANG"


def test_thoroughbred_run_row_carries_weight_barrier_and_jockey():
    run = next(x for x in load(DEAU).field_ if x.tab == 1).runs[0]
    assert (run.pos, run.field_size) == (4, 20)
    assert run.margin == pytest.approx(1.7)
    assert (run.weight, run.box) == (59.0, 20)
    assert run.jockey == "C SOUMILLON"          # printed as "(0.5kg)"
    assert (run.dist_m, run.surface, run.going) == (1600, "T", "G")


def test_a_win_is_stored_as_a_negative_margin():
    g = next(x for x in load(BULLI).field_ if x.tab == 1).runs[-1]
    assert (g.pos, g.margin) == (1, pytest.approx(-2.0))
    t = next(x for x in load(DEAU).field_ if x.tab == 1).runs[2]
    assert (t.pos, t.margin) == (1, pytest.approx(-0.4))


def test_impossible_field_size_is_flagged_not_believed():
    r = load(HSHM)
    bad = [run for x in r.field_ for run in x.runs if run.field_size_suspect]
    assert len(bad) == 5
    assert all(run.field_size is None for run in bad)
    assert any("impossible" in w for w in r.warnings)


def test_spell_marker_lines_are_not_parsed_as_runs():
    one = next(x for x in load(BULLI).field_ if x.tab == 1)
    assert all(run.run_date is not None for run in one.runs)


def test_straight_and_foreign_tracks_are_classified():
    dyna = next(x for x in load(BULLI).field_ if x.tab == 6)
    assert all(run.track_kind == "straight" for run in dyna.runs)   # all RIST
    adobe = next(x for x in load(QLAKE).field_ if x.tab == 1)
    assert all(run.track_kind == "foreign" for run in adobe.runs)   # NZD


# --- harness specifics --------------------------------------------------------

def test_harness_margins_are_metres_converted_to_lengths():
    m = next(x for x in load(CABO).field_ if x.tab == 1)
    first = m.runs[0]
    assert (first.margin_raw, first.margin_unit) == (48.2, "m")
    assert first.margin == pytest.approx(48.2 / rs_parser.METRES_PER_LENGTH)


def test_dqg_is_a_non_finish_not_a_result():
    m = next(x for x in load(CABO).field_ if x.tab == 1)
    dq = [r for r in m.runs if r.disqualified]
    assert len(dq) == 1
    assert dq[0].dq_code == "DQG"
    assert dq[0].pos is None
    assert dq[0].margin is None                  # the 99m sentinel is discarded
    assert dq[0].margin_raw == pytest.approx(rs_parser.SENTINEL_METRES)
    assert not dq[0].counts_as_form


def test_a_blank_finishing_position_is_also_a_non_finish():
    """MOLITOR FLIGNY 30-Nov-2025 prints an empty FP with the 99m sentinel.
    Stripping the row would delete that empty cell and shift every column."""
    mf = next(x for x in load(CABO).field_ if x.tab == 2)
    assert len(mf.runs) == 7
    blank = [r for r in mf.runs if r.run_date and r.run_date.isoformat() == "2025-11-30"]
    assert len(blank) == 1
    assert blank[0].disqualified and blank[0].dq_code == "NR"
    assert blank[0].track == "FRCT"              # columns did NOT shift
    assert blank[0].dist_m == 2400


def test_dq_counts_and_rates():
    r = load(CABO)
    milord = next(x for x in r.field_ if x.tab == 9)
    assert milord.dq_count == 6 and len(milord.runs) == 10
    assert milord.dq_rate == pytest.approx(0.6)
    planchette = next(x for x in r.field_ if x.tab == 13)
    assert planchette.dq_count == 5


def test_varying_run_table_columns_within_one_page():
    """Cabourg ships two column sets: runners 1-3 and 8 have no Draw column,
    the rest do.  A column map built once for the page mis-reads half of it."""
    r = load(CABO)
    no_draw = next(x for x in r.field_ if x.tab == 1)
    with_draw = next(x for x in r.field_ if x.tab == 4)
    assert no_draw.runs[0].track == "CBRG" and no_draw.runs[0].sp == 46.0
    assert with_draw.runs[0].track == "BSUR"
    handicapped = [run for run in with_draw.runs if run.handicap_m]
    assert handicapped and handicapped[0].handicap_m == -25.0


def test_harness_distance_handicap_is_read():
    jiel = next(x for x in load(CABO).field_ if x.tab == 5)
    assert any(run.handicap_m == 25.0 for run in jiel.runs)


# --- robustness ---------------------------------------------------------------

def test_garbage_input_does_not_raise():
    r = rs_parser.parse("hello world\nnothing to see here")
    assert r.field_ == [] and r.warnings


@pytest.mark.parametrize("fx", ALL)
def test_parser_is_insensitive_to_line_endings(fx):
    raw = io.open(os.path.join(HERE, "fixtures", fx), encoding="utf-8").read()
    a, b = rs_parser.parse(raw), rs_parser.parse(raw.replace("\n", "\r\n"))
    assert [x.name for x in a.field_] == [x.name for x in b.field_]
    assert [len(x.runs) for x in a.field_] == [len(x.runs) for x in b.field_]


# --- rating -------------------------------------------------------------------

@pytest.mark.parametrize("fx", ALL)
def test_probabilities_are_a_distribution(fx):
    race = load(fx)
    rated, _ = rating.rate(race)
    assert len(rated) == len(race.field_)
    assert sum(r.p_final for r in rated) == pytest.approx(1.0)
    assert sum(r.p_model for r in rated) == pytest.approx(1.0)
    assert sum(r.p_market for r in rated) == pytest.approx(1.0)
    assert all(0.0 < r.p_final < 1.0 for r in rated)
    assert rated == sorted(rated, key=lambda r: -r.p_final)


@pytest.mark.parametrize("fx", ALL)
def test_place_probabilities_are_ordered_and_bounded(fx):
    rated, _ = rating.rate(load(fx))
    for r in rated:
        assert r.p_final <= r.p_top2 + 1e-9 <= r.p_top3 + 1e-9 <= 1.0 + 1e-9
    assert sum(r.p_top2 for r in rated) == pytest.approx(2.0)
    assert sum(r.p_top3 for r in rated) == pytest.approx(min(3.0, float(len(rated))))


@pytest.mark.parametrize("fx", ALL)
def test_margins_are_on_a_sane_scale(fx):
    """A field-size credit calibrated for greyhounds turns a 4th-of-20 into a
    NEGATIVE average beaten margin, i.e. an apparent winner."""
    race = load(fx)
    rated, _ = rating.rate(race)
    beaten = [r for r in rated if r.runner.career_wins < r.runner.career_starts / 2]
    assert beaten, "expected some mostly-beaten runners"
    assert any(r.avg_margin > 0 for r in beaten)
    assert all(-20 < r.avg_margin < 90 for r in rated)


def test_greyhound_selections_match_the_written_analysis():
    """The three races the model was developed on. Bulli and Horsham were
    correct; Q Lakeside picked the one that ran 3rd. Pinned so a change to the
    parser or the rating is visible, not silent."""
    assert rating.rate(load(BULLI))[0][0].name == "LIZZIE LONG LEGS"
    assert rating.rate(load(QLAKE))[0][0].name == "DAWN SURE CAN"
    assert rating.rate(load(HSHM))[0][0].name == "PAW PALMER"


def test_code_specific_terms_are_present_and_absent():
    g = rating.rate(load(BULLI))[0][0].terms
    t = rating.rate(load(DEAU))[0][0].terms
    h = rating.rate(load(CABO))[0][0].terms
    assert {"early speed", "box", "vacant box"} <= set(g)
    assert "barrier" not in g and "reliability" not in g
    assert "barrier" in t and "early speed" not in t and "reliability" not in t
    assert "reliability" in h and "barrier" not in h and "box" not in h
    for terms in (g, t, h):
        assert {"form", "class", "conversion", "distance", "course",
                "surface", "going", "layoff"} <= set(terms)


def test_thoroughbred_weight_adjustment_moves_the_margin():
    race = load(DEAU)
    light = rating.rate(race, rating.replace(
        rating.defaults_for(THOROUGHBRED), lengths_per_kg=0.0))[0]
    heavy = rating.rate(race, rating.replace(
        rating.defaults_for(THOROUGHBRED), lengths_per_kg=1.2))[0]
    # HARPER carried 59.0 in most recent runs and is set to 54.5 today, so a
    # bigger weight scale should IMPROVE its adjusted margin.
    a = next(x for x in light if x.tab == 9).avg_margin
    b = next(x for x in heavy if x.tab == 9).avg_margin
    assert b < a


def test_harness_dqg_runs_are_excluded_from_the_margin_average():
    race = load(CABO)
    rated, _ = rating.rate(race)
    milord = next(x for x in rated if x.tab == 9)
    assert milord.used_runs == 4          # 10 runs, 6 of them non-finishes
    assert milord.terms["reliability"] < -1.0


def test_harness_reliability_penalty_scales_with_the_dq_rate():
    race = load(CABO)
    rated, _ = rating.rate(race)
    by = {x.tab: x.terms["reliability"] for x in rated}
    assert by[9] < by[13] < by[10]        # 7/10 worse than 5/10 worse than 0/10


def test_harness_handicap_credit_helps_a_horse_that_gave_ground():
    race = load(CABO)
    none_ = rating.rate(race, rating.replace(
        rating.defaults_for(HARNESS), handicap_credit=0.0))[0]
    full = rating.rate(race, rating.replace(
        rating.defaults_for(HARNESS), handicap_credit=1.0))[0]
    a = next(x for x in none_ if x.tab == 11).avg_margin
    b = next(x for x in full if x.tab == 11).avg_margin
    assert b < a


def test_market_weight_of_one_ignores_the_market():
    rated, _ = rating.rate(load(DEAU), rating.replace(
        rating.defaults_for(THOROUGHBRED), market_weight=1.0))
    for r in rated:
        assert r.p_final == pytest.approx(r.p_model)


def test_missing_price_falls_back_to_model_only():
    race = load(HSHM)
    race.field_[0].odds = None
    rated, notes = rating.rate(race)
    assert any("no market blend" in n for n in notes)
    assert all(r.p_final == pytest.approx(r.p_model) for r in rated)


def test_shrinkage_prior_tames_a_two_start_record():
    """WHO'S IDEA is 0 from 2 at 457m. A weak prior turns that into a huge
    penalty; that specification cost a live race."""
    race = load(QLAKE)
    weak = rating.rate(race, rating.replace(
        rating.defaults_for(GREYHOUND), prior_starts=7.0))[0]
    strong = rating.rate(race, rating.replace(
        rating.defaults_for(GREYHOUND), prior_starts=15.0))[0]
    pw = next(r for r in weak if r.tab == 4).terms["distance"]
    ps = next(r for r in strong if r.tab == 4).terms["distance"]
    assert pw < ps < 0 and abs(ps) < 0.5 * abs(pw)


def test_vacant_box_term_is_off_by_default_and_wired_when_enabled():
    """Q Lakeside runs boxes 1,4,5,6,8 with 2,3,7 empty."""
    race = load(QLAKE)
    assert {r.terms["vacant box"] for r in rating.rate(race)[0]} == {0.0}
    on = {r.tab: r.terms["vacant box"] for r in rating.rate(
        race, rating.replace(rating.defaults_for(GREYHOUND), k_gap=0.5))[0]}
    assert on[4] == pytest.approx(1.0)    # boxes 2 and 3 vacant inside box 4
    assert on[8] == pytest.approx(0.5)    # box 7 vacant inside box 8
    assert on[5] == on[6] == on[1] == pytest.approx(0.0)


def test_code_defaults_differ_by_code():
    assert (rating.defaults_for(GREYHOUND).spread
            < rating.defaults_for(THOROUGHBRED).spread
            < rating.defaults_for(HARNESS).spread)
    assert (rating.defaults_for(GREYHOUND).k_field
            > rating.defaults_for(THOROUGHBRED).k_field)


# --- streamlit script ---------------------------------------------------------

def test_streamlit_app_runs_clean():
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(os.path.join(HERE, "app.py"), default_timeout=120)
    at.run()
    assert not at.exception, at.exception


@pytest.mark.parametrize("fx", [HSHM, DEAU, CABO])
def test_streamlit_app_renders_every_code(fx):
    from streamlit.testing.v1 import AppTest
    raw = io.open(os.path.join(HERE, "fixtures", fx), encoding="utf-8").read()
    at = AppTest.from_file(os.path.join(HERE, "app.py"), default_timeout=300)
    at.session_state["raw"] = raw
    at.run()
    assert not at.exception, at.exception
    assert any(m.value for m in at.metric)


def test_stale_module_guard_fires_before_any_helper_import():
    """The guard must run BEFORE `import rating` / `import rs_parser`.  A stale
    Streamlit Cloud module raises ImportError on the import line itself, and the
    redacted Cloud traceback that follows says nothing useful.  Verified by
    source order, because the failure only reproduces against a cached module."""
    src = io.open(os.path.join(HERE, "app.py"), encoding="utf-8").read()
    guard = src.index("_REQUIRED = {")
    stop = src.index("st.stop()", guard)
    for line in src[:guard].splitlines():
        assert not line.startswith(("import rating", "import rs_parser",
                                    "from rs_parser", "from rating")), line
    assert src.index("import rating", stop) > stop
    assert "except Exception" in src[guard:stop], \
        "the guard must survive a module that fails to import at all"
