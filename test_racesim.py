"""Parser, model, simulation, renderer and app smoke tests on the Canberra R5 fixture."""
from __future__ import annotations

import io
import os
import re

import numpy as np
import pytest
from PIL import Image

import horse_parser as hp
import race_anim as ra
import race_model as rm
import race_sim as rs

HERE = os.path.dirname(os.path.abspath(__file__))


@pytest.fixture(scope="module")
def form_text():
    return open(os.path.join(HERE, "fixture_canberra_r5.txt"), encoding="utf-8").read()


@pytest.fixture(scope="module")
def speed_text():
    return open(os.path.join(HERE, "fixture_canberra_r5_speedmap.txt"), encoding="utf-8").read()


@pytest.fixture(scope="module")
def parsed(form_text):
    return hp.parse(form_text)


@pytest.fixture(scope="module")
def rated(parsed, speed_text):
    header, runners, _ = parsed
    return rm.rate_field(header, runners, hp.parse_speed_map(speed_text))


# --------------------------------------------------------------------------
# parser
# --------------------------------------------------------------------------
def test_header(parsed):
    h, _, _ = parsed
    assert h["track"] == "Canberra" and h["race_no"] == 5
    assert h["distance_m"] == 1400 and h["surface"] == "TURF"
    assert h["going"] == "SOFT" and h["going_rating"] == 6


def test_field_and_scratchings(parsed):
    _, runners, warnings = parsed
    assert len(runners) == 20
    active = [r for r in runners if not r["scratched"]]
    assert len(active) == 15
    assert {r["tab"] for r in runners if r["scratched"]} == {7, 8, 12, 16, 17}
    assert not any("missing" in w for w in warnings)


def test_runner_details(parsed):
    _, runners, _ = parsed
    by = {r["tab"]: r for r in runners}
    d = by[1]
    assert d["wt"] == 63.0 and d["bp"] == 5 and d["claim"] == 2.0
    assert d["jockey"] == "DALE COLE" and d["trainer"] == "NELL FOLEY"
    assert d["jrat"] == 4.1 and d["trat"] == 2.4
    assert d["bf_odds"] == 11.0
    assert d["ohr"] == 59 and d["dslr"] == 20 and d["runup"] == 3
    assert d["win_dists"] == [1000, 1200]
    runs = d["recent_runs"]
    assert runs[0]["finish"] == 1 and runs[0]["ohr"] == 59
    assert runs[0]["race_time_s"] == pytest.approx(72.27) and runs[0]["sec600_s"] == pytest.approx(36.44)
    assert runs[0]["runup_tag"] == "2U" and runs[0]["settle_pos"] == 3
    assert runs[0]["direction"] == "Clockwise"
    assert runs[0]["prize"] == 16000
    # glued filters panel: Soft record and AW/Turf split
    assert (by[9]["AW_wins"], by[9]["AW_starts"]) == (1, 4)
    assert (by[9]["Turf_wins"], by[9]["Turf_starts"]) == (0, 4)
    assert by[9]["Soft_starts"] == 0
    # slow beginner flag needs two of the last three runs
    assert by[19]["recent_runs"][1]["slow_begin"] and by[19]["recent_runs"][2]["slow_begin"]
    # first-up horse with no run-up tag
    assert by[18]["dslr"] == 118 and by[18].get("runup") is None


def test_speed_map(speed_text):
    sm = hp.parse_speed_map(speed_text)
    assert len(sm) == 15
    assert sm[1] == {"aes": 17.0, "afs": 17.6, "bp": 5.0, "jr": 4.1}
    assert sm[20]["afs"] == 18.0 and sm[3]["bp"] == 13.0


def test_speed_map_empty():
    assert hp.parse_speed_map("") == {}
    assert hp.parse_speed_map("some unrelated text\n1\tfoo\n") == {}


def test_track_direction(parsed):
    h, runners, _ = parsed
    assert hp.track_direction(runners, h["track"]) == "clockwise"
    assert hp.track_direction([], "") == "clockwise"


# --------------------------------------------------------------------------
# model
# --------------------------------------------------------------------------
def test_class_points():
    assert rm.class_points("BM58") == 58
    assert rm.class_points("MDN") == 46
    assert rm.class_points("3U CL3") == 58
    assert rm.class_points("4U BM50") == 50
    assert rm.class_points("3 MDN", 100000) > rm.class_points("MDN", 27000)
    assert rm.class_points("34 OPEN", 100000) > 72


def test_rating_output(rated):
    rr = rated["runners"]
    assert len(rr) == 15
    assert abs(sum(r["win_prob"] for r in rr) - 1.0) < 1e-6
    assert abs(sum(r["place_prob"] for r in rr) - 3.0) < 1e-6
    assert [r["rank"] for r in rr] == list(range(1, 16))
    assert rr[0]["exp_margin"] == 0.0 and all(r["exp_margin"] >= 0 for r in rr)
    assert max(r["win_prob"] for r in rr) < 0.45          # over-confidence guard
    assert set(rr[0]["components"]) == set(rated["meta"]["components"])
    assert 80 < rated["meta"]["pred_time_s"] < 90        # 1400 m Soft 6 ~ 1:24
    # every runner has a finite rating even with thin form (Ikon Park: 2 runs)
    assert all(np.isfinite(r["rating"]) for r in rr)


def test_market_weight_is_capped(parsed, speed_text):
    header, runners, _ = parsed
    sm = hp.parse_speed_map(speed_text)
    r0 = rm.rate_field(header, runners, sm, market_weight=0.0)
    r1 = rm.rate_field(header, runners, sm, market_weight=0.9)      # asks for 90 %, gets 10 %
    assert r0["meta"]["market_weight"] == 0.0
    assert r1["meta"]["market_weight"] == pytest.approx(rm.MARKET_WEIGHT_CAP)
    fav = next(r for r in r1["runners"] if r["tab"] == 4)           # Alabama Hussy, $4.9 fav
    fav0 = next(r for r in r0["runners"] if r["tab"] == 4)
    out = next(r for r in r1["runners"] if r["tab"] == 1)           # Deekaygeebee, $11, model top
    out0 = next(r for r in r0["runners"] if r["tab"] == 1)
    # the market nudges the favourite up *relative to* the outsider the model prefers ...
    assert fav["rating"] - out["rating"] > fav0["rating"] - out0["rating"]
    # ... but the model's own ordering still dominates: the top pick is unchanged
    assert r0["runners"][0]["tab"] == r1["runners"][0]["tab"]
    # the ranking with no market is not the market ranking
    mk = sorted(r0["runners"], key=lambda r: r["market_prob"], reverse=True)
    assert [r["tab"] for r in mk] != [r["tab"] for r in r0["runners"]]


def test_without_speed_map(parsed):
    header, runners, _ = parsed
    r = rm.rate_field(header, runners, None)
    assert len(r["runners"]) == 15 and not r["meta"]["has_speed_map"]
    assert all(x["aes"] is None for x in r["runners"])


def test_barrier_penalty_grows(parsed, speed_text):
    header, runners, _ = parsed
    sm = hp.parse_speed_map(speed_text)
    rr = rm.rate_field(header, runners, sm)["runners"]
    by = {r["tab"]: r for r in rr}
    assert by[15]["components"]["barrier"] == 0.0                   # gate 1
    assert by[13]["components"]["barrier"] < by[4]["components"]["barrier"] < 0   # gate 15 < gate 8


# --------------------------------------------------------------------------
# simulation
# --------------------------------------------------------------------------
def test_simulation_shape_and_monotone(rated):
    sim = rs.simulate(rated, "expected")
    assert sim.n == 15 and sim.distance == 1400
    assert sim.gap.shape == (15, 1401)
    assert np.all(sim.gap >= -1e-9)
    assert np.allclose(sim.gap.min(axis=0), 0.0)        # someone leads at every point
    # leader distance is monotone in time and spans the race
    us = np.linspace(0, 1, 200)
    s = np.array([sim.leader_distance(u) for u in us])
    assert np.all(np.diff(s) > 0) and s[0] == 0 and abs(s[-1] - 1400) < 1e-6
    # expected mode finishes in rating order with the model's margins
    top = rated["runners"][0]
    assert sim.tabs[sim.finish_order[0]] == top["tab"]
    assert sim.finish_margin[sim.finish_order[0]] == 0.0
    # the early leader has the strongest early speed
    order_settled = sim.order_at(0.32 * 1400)
    es = sim.extras["early_speed"]
    assert order_settled[0][0] == int(np.argmax(es))


def test_random_mode_differs_but_is_plausible(rated):
    a = rs.simulate(rated, "random", seed=1)
    b = rs.simulate(rated, "random", seed=2)
    assert a.finish_order != b.finish_order or not np.allclose(a.finish_margin, b.finish_margin)
    assert a.finish_margin.max() <= 15.0
    calls = rs.running_calls(a)
    assert [c["checkpoint"] for c in calls][0] == "Jump" and calls[-1]["to_go"] == 0


def test_phase_labels(rated):
    sim = rs.simulate(rated)
    assert sim.phase(50) == "The jump"
    assert sim.phase(300) == "Settling down"
    assert sim.phase(700) == "Mid-race"
    assert sim.phase(900) == "On the turn"
    assert sim.phase(1200) == "Home straight"
    assert sim.phase(1400) == "Finish"


# --------------------------------------------------------------------------
# renderer
# --------------------------------------------------------------------------
def test_gif_duration_frames_and_size(rated, parsed):
    header = parsed[0]
    sim = rs.simulate(rated)
    gif = ra.render_gif(sim, header, clockwise=True, duration_s=12, fps=8, hold_s=2.0)
    im = Image.open(io.BytesIO(gif))
    assert im.format == "GIF" and im.is_animated
    assert im.n_frames == 10 * 8
    total_ms = 0
    for k in range(im.n_frames):
        im.seek(k)
        total_ms += im.info["duration"]
    assert total_ms <= ra.MAX_DURATION_S * 1000
    assert len(gif) < 4_000_000


def test_gif_never_exceeds_30s(rated, parsed):
    sim = rs.simulate(rated)
    gif = ra.render_gif(sim, parsed[0], duration_s=90, fps=8)        # asks for 90 s
    im = Image.open(io.BytesIO(gif))
    total_ms = 0
    for k in range(im.n_frames):
        im.seek(k)
        total_ms += im.info["duration"]
    assert total_ms <= 30_000


def test_snapshots_and_direction(rated, parsed):
    sim = rs.simulate(rated)
    snaps = ra.snapshots(sim, parsed[0], clockwise=True)
    assert [s[0] for s in snaps] == ["Just after the start", "Mid-race", "Turning for home", "The finish"]
    assert all(img.size == (ra.W, ra.H) for _, img in snaps)
    # every horse is drawn inside the canvas throughout, both directions
    for cw in (True, False):
        tr = ra.Track(ra.course_length(1400), clockwise=cw)
        for s in np.linspace(0, 1400, 57):
            x, y = tr.point(s, 60)
            assert 0 <= x <= ra.PANEL_X and 0 <= y <= ra.H
    # clockwise: finish sits at the left end of the home (bottom) straight and
    # the approach comes from the right; anticlockwise mirrors it
    cw = ra.Track(ra.course_length(1400), clockwise=True)
    acw = ra.Track(ra.course_length(1400), clockwise=False)
    assert cw.point(0)[0] < cw.point(150)[0]
    assert acw.point(0)[0] > acw.point(150)[0]


def test_pretty_name():
    assert ra.pretty_name("FREDDY'S SHOCK") == "Freddy's Shock"
    assert ra.pretty_name("SHE'SADARE") == "She'sadare"


# --------------------------------------------------------------------------
# app
# --------------------------------------------------------------------------
def test_app_guard_runs_before_helper_imports():
    src = open(os.path.join(HERE, "app.py"), encoding="utf-8").read()
    guard = src.index("_REQUIRED = {")
    for mod in ("horse_parser", "race_model", "race_sim", "race_anim"):
        assert not re.search(rf"^(import|from)\s+{mod}\b", src[:guard], re.M)


def test_app_smoke(form_text, speed_text):
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(os.path.join(HERE, "app.py"), default_timeout=240)
    at.run()
    assert not at.exception
    at.text_area(key="form_text").set_value(form_text)
    at.text_area(key="speed_text").set_value(speed_text)
    at.run()
    assert not at.exception, at.exception
    texts = " ".join(m.value for m in at.markdown) + " ".join(m.value for m in at.metric)
    assert "Model pick" in texts
    assert any("Canberra" in m.value for m in at.metric)
    assert len(at.dataframe) >= 1
