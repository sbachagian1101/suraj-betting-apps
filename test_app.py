"""Regression tests for the parser, the rating and the Streamlit script.

Run with:  python -m pytest -q
"""

from __future__ import annotations

import io
import os

import pytest

import rating
import rs_parser

HERE = os.path.dirname(os.path.abspath(__file__))


def load(name: str) -> rs_parser.Race:
    path = os.path.join(HERE, "fixtures", name)
    return rs_parser.parse(io.open(path, encoding="utf-8").read())


BULLI = "bulli_r6_2026-09-01.txt"
QLAKE = "qlakeside_r7_2026-09-01.txt"
HSHM = "horsham_r8_2026-09-01.txt"


# --- header -------------------------------------------------------------------

@pytest.mark.parametrize("fixture,track,rno,dist,surface,going,grade", [
    (BULLI, "Bulli", 6, 340, "AW", "GOOD", "5th Grade"),
    (QLAKE, "Q Lakeside", 7, 457, "AW", "GOOD", "5th Grade"),
    (HSHM, "Horsham", 8, 410, "AW", "GOOD", "Free For All"),
])
def test_header(fixture, track, rno, dist, surface, going, grade):
    r = load(fixture)
    assert (r.track, r.race_no, r.dist_m) == (track, rno, dist)
    assert (r.surface, r.going, r.grade) == (surface, going, grade)
    assert r.race_date is not None and r.race_date.isoformat() == "2026-09-01"


@pytest.mark.parametrize("fixture,code", [
    (BULLI, "BULI"), (QLAKE, "QLAK"), (HSHM, "HSHM")])
def test_track_code_inferred_from_run_tables(fixture, code):
    assert load(fixture).track_code() == code


# --- field composition --------------------------------------------------------

@pytest.mark.parametrize("fixture,live,scratched", [
    (BULLI, 8, 2), (QLAKE, 5, 4), (HSHM, 6, 1)])
def test_field_sizes(fixture, live, scratched):
    r = load(fixture)
    assert len(r.field_) == live
    assert sum(1 for x in r.runners if x.scratched) == scratched


def test_scratched_runners_are_identified_by_box():
    r = load(QLAKE)
    scr = {x.box for x in r.runners if x.scratched}
    assert scr == {2, 3, 7, 9}
    assert [x.box for x in r.field_] == [1, 4, 5, 6, 8]


def test_name_is_the_dog_not_the_trainer():
    """The dog's name precedes the tag block; the trainer follows the box.
    Box 1 at Bulli is EXPLORE, *trained by* Frank Micallef."""
    r = load(BULLI)
    one = next(x for x in r.field_ if x.box == 1)
    assert one.name == "EXPLORE"
    assert one.trainer == "FRANK MICALLEF"


def test_names_with_punctuation_and_country_suffix():
    assert any(x.name == "WHERE'S HARRY" for x in load(BULLI).runners)
    assert any(x.name == "ADOBE RAVE (NZ)" for x in load(QLAKE).runners)


def test_every_live_runner_has_box_odds_and_ten_runs():
    for fx in (BULLI, QLAKE, HSHM):
        for x in load(fx).field_:
            assert x.box is not None, f"{fx}: {x.name} has no box"
            assert x.odds and x.odds > 1.0, f"{fx}: {x.name} has no price"
            assert len(x.runs) == 10, f"{fx}: {x.name} has {len(x.runs)} runs"


# --- record panel (glued labels) ---------------------------------------------

def test_glued_record_labels_are_peeled():
    r = load(HSHM)
    meeki = next(x for x in r.field_ if x.box == 1)
    assert meeki.record("Career") == (23, 8, 3, 2)
    assert meeki.record("Course") == (6, 4, 0, 0)
    assert meeki.record("Dist") == (14, 7, 2, 1)
    assert (meeki.win_pct, meeki.place_pct) == (35.0, 57.0)


def test_zero_records_are_kept_not_dropped():
    """`Course0: 0 0 0` must survive - a blank-cell filter loses it."""
    ace = next(x for x in load(BULLI).field_ if x.box == 2)
    assert ace.record("Course") == (0, 0, 0, 0)
    assert ace.record("Turf") == (7, 3, 2, 1)


# --- run table ----------------------------------------------------------------

def test_run_row_columns_are_read_by_header():
    first = next(x for x in load(BULLI).field_ if x.box == 1).runs[0]
    assert (first.pos, first.field_size) == (7, 7)
    assert first.margin == pytest.approx(7.9)
    assert first.track == "RIST"
    assert (first.dist_m, first.surface, first.going) == (324, "T", "G")
    assert (first.box, first.sp, first.sectional) == (8, 41.0, 7.27)
    assert first.beat_or_beaten_by == "ZIPPING POPYRIN"
    assert first.days_ago == 5


def test_empty_sectional_does_not_shift_columns():
    """`...$23\\t\\tFLASH BOOM BANG` - the blank Sec.Time must stay a blank cell."""
    run = next(x for x in load(BULLI).field_ if x.box == 1).runs[1]
    assert run.sectional is None
    assert run.sp == 23.0
    assert run.beat_or_beaten_by == "FLASH BOOM BANG"
    assert run.dist_m == 340


def test_a_win_is_stored_as_a_negative_margin():
    run = next(x for x in load(BULLI).field_ if x.box == 1).runs[-1]
    assert run.pos == 1
    assert run.margin == pytest.approx(-2.0)


def test_impossible_field_size_is_flagged_not_believed():
    r = load(HSHM)
    bad = [run for x in r.field_ for run in x.runs if run.field_size_suspect]
    assert len(bad) == 5
    assert all(run.field_size is None for run in bad)
    assert any("impossible" in w for w in r.warnings)


def test_spell_marker_lines_are_not_parsed_as_runs():
    """'2 months 10 days' sits inside the run table and must be skipped."""
    one = next(x for x in load(BULLI).field_ if x.box == 1)
    assert all(run.run_date is not None for run in one.runs)


def test_straight_and_foreign_tracks_are_classified():
    bulli = load(BULLI)
    dyna = next(x for x in bulli.field_ if x.box == 6)
    assert all(run.track_kind == "straight" for run in dyna.runs)  # all RIST
    adobe = next(x for x in load(QLAKE).field_ if x.box == 1)
    assert all(run.track_kind == "foreign" for run in adobe.runs)  # NZD prizemoney


# --- robustness ---------------------------------------------------------------

def test_garbage_input_does_not_raise():
    r = rs_parser.parse("hello world\nnothing to see here")
    assert r.field_ == []
    assert r.warnings


def test_parser_is_insensitive_to_line_endings():
    raw = io.open(os.path.join(HERE, "fixtures", BULLI), encoding="utf-8").read()
    a = rs_parser.parse(raw)
    b = rs_parser.parse(raw.replace("\n", "\r\n"))
    assert [x.name for x in a.field_] == [x.name for x in b.field_]
    assert len(a.field_[0].runs) == len(b.field_[0].runs)


# --- rating -------------------------------------------------------------------

@pytest.mark.parametrize("fixture", [BULLI, QLAKE, HSHM])
def test_probabilities_are_a_distribution(fixture):
    rated, _ = rating.rate(load(fixture))
    assert len(rated) == len(load(fixture).field_)
    assert sum(r.p_final for r in rated) == pytest.approx(1.0)
    assert sum(r.p_model for r in rated) == pytest.approx(1.0)
    assert sum(r.p_market for r in rated) == pytest.approx(1.0)
    assert all(0.0 < r.p_final < 1.0 for r in rated)
    assert rated == sorted(rated, key=lambda r: -r.p_final)


@pytest.mark.parametrize("fixture", [BULLI, QLAKE, HSHM])
def test_place_probabilities_are_ordered_and_bounded(fixture):
    rated, _ = rating.rate(load(fixture))
    for r in rated:
        assert r.p_final <= r.p_top2 + 1e-9 <= r.p_top3 + 1e-9 <= 1.0 + 1e-9
    n = len(rated)
    assert sum(r.p_top2 for r in rated) == pytest.approx(2.0)
    assert sum(r.p_top3 for r in rated) == pytest.approx(min(3.0, float(n)))


def test_selections_match_the_written_analysis():
    """The three races this model was developed on. Bulli and Horsham were
    correct; Q Lakeside picked the horse that ran 3rd. All three are pinned so a
    change to the parser or the rating is visible, not silent."""
    assert rating.rate(load(BULLI))[0][0].name == "LIZZIE LONG LEGS"
    assert rating.rate(load(QLAKE))[0][0].name == "DAWN SURE CAN"
    assert rating.rate(load(HSHM))[0][0].name == "PAW PALMER"


def test_market_weight_of_one_ignores_the_market():
    p = rating.Params(market_weight=1.0)
    rated, _ = rating.rate(load(HSHM), p)
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
    weak = rating.rate(race, rating.Params(prior_starts=7.0))[0]
    strong = rating.rate(race, rating.Params(prior_starts=15.0))[0]
    pen_weak = next(r for r in weak if r.box == 4).terms["distance"]
    pen_strong = next(r for r in strong if r.box == 4).terms["distance"]
    assert pen_weak < pen_strong < 0
    assert abs(pen_strong) < 0.5 * abs(pen_weak)


def test_vacant_box_term_is_off_by_default_and_wired_when_enabled():
    """Q Lakeside runs boxes 1,4,5,6,8 with 2,3,7 empty."""
    race = load(QLAKE)
    base = {r.box: r.terms["vacant box"] for r in rating.rate(race)[0]}
    assert set(base.values()) == {0.0}
    on = {r.box: r.terms["vacant box"] for r in rating.rate(
        race, rating.Params(k_gap=0.5))[0]}
    assert on[4] == pytest.approx(1.0)   # boxes 2 and 3 vacant inside box 4
    assert on[8] == pytest.approx(0.5)   # box 7 vacant inside box 8
    assert on[5] == pytest.approx(0.0)   # box 4 is occupied, immediately inside
    assert on[6] == pytest.approx(0.0)   # box 5 is occupied, immediately inside
    assert on[1] == pytest.approx(0.0)   # railer


def test_vacant_box_gap_counts_only_to_the_next_occupied_box():
    """Horsham runs boxes 1,3,4,6,7,8 - box 6 has box 5 empty inside it, but
    box 4 is occupied beyond that, so the gap is 1 and not 2."""
    on = {r.box: r.terms["vacant box"] for r in rating.rate(
        load(HSHM), rating.Params(k_gap=0.5))[0]}
    assert on[3] == pytest.approx(0.5)   # box 2 vacant (scratched) inside box 3
    assert on[6] == pytest.approx(0.5)   # box 5 vacant inside box 6
    assert on[4] == pytest.approx(0.0)
    assert on[7] == pytest.approx(0.0)


# --- streamlit script ---------------------------------------------------------

def test_streamlit_app_runs_clean():
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(os.path.join(HERE, "app.py"), default_timeout=120)
    at.run()
    assert not at.exception, at.exception


def test_streamlit_app_renders_a_prediction_for_a_pasted_race():
    from streamlit.testing.v1 import AppTest
    raw = io.open(os.path.join(HERE, "fixtures", HSHM), encoding="utf-8").read()
    at = AppTest.from_file(os.path.join(HERE, "app.py"), default_timeout=180)
    at.session_state["raw"] = raw
    at.run()
    assert not at.exception, at.exception
    body = " ".join(str(m.value) for m in at.markdown) + " ".join(
        str(m.value) for m in at.subheader)
    assert "Horsham" in body or any(
        "PAW PALMER" in str(m.value) for m in at.metric)
