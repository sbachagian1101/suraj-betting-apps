"""Offline tests on captured feed snapshots plus an AppTest smoke run.
Nothing here touches the network."""
from __future__ import annotations

import io
import json
import os
import re
from datetime import date

import pytest

import common
import model as M
import pipeline
import rs_paste
from sources import hkjc, hrn, ladbrokes, pmu, sportinglife, winningform

HERE = os.path.dirname(os.path.abspath(__file__))
FX = os.path.join(HERE, "fixtures")


def read(name: str, enc: str = "utf-8") -> str:
    return io.open(os.path.join(FX, name), encoding=enc, errors="replace").read()


# --------------------------------------------------------------------------- #
# paste
# --------------------------------------------------------------------------- #
def test_identity_from_markdown_header_paste():
    idn = rs_paste.identity(read("rs_grafton_r2_2026-09-07_header.txt"))
    assert idn.country == "AUS" and idn.venue == "Grafton" and idn.race_no == 2
    assert idn.race_date == date(2026, 9, 7) and idn.start_time == "13:55"
    assert idn.dist_m == 1200 and idn.surface == "TURF" and idn.going == "GOOD" and idn.going_rating == "4"
    assert idn.race_type == "MDN" and idn.prize == 27000.0 and idn.currency == "AUD"
    assert idn.race_name.startswith("KOSCIUSZKO")
    assert idn.complete


def test_header_only_paste_is_not_a_parse_failure():
    idn, card = rs_paste.parse_paste(read("rs_grafton_r2_2026-09-07_header.txt"))
    assert card.entries == [] and card.warnings == []
    assert any("header only" in n for n in card.notes)


def test_full_fields_paste_yields_runners_and_runs():
    idn, card = rs_paste.parse_paste(read("rs_deauville_r1_2026-09-01.txt"))
    assert idn.venue == "Deauville" and idn.race_no == 1 and idn.dist_m == 1600
    assert len(card.field_) == 15
    calas = card.find(1, "Calas")
    assert calas.barrier == 10 and calas.weight == 61.5 and calas.jockey == "Maxime Guyon"
    assert calas.career == (25, 5, 2, 5) and calas.sex == "G"
    assert len(calas.runs) == 10 and calas.runs[0].margin_l == 1.7 and calas.runs[0].dist_m == 1600
    assert calas.runs[2].pos == 1 and calas.runs[2].margin_l == -0.4
    assert calas.odds == 4.2 and calas.odds_open == pytest.approx(4.2 / 1.04)


@pytest.mark.parametrize("line,expect", [
    ("1200m TURF GOOD 4", (1200, "TURF", "GOOD", "4")),
    ("2750m SAND STANDARD", (2750, "SAND", "STANDARD", "")),
    ("1m1f207y TURF GOOD", (2000, "TURF", "GOOD", "")),
    ("1600m ALL WEATHER STANDARD", (1600, "AW", "STANDARD", "")),
    ("340m ALL WEATHER GOOD", (340, "AW", "GOOD", "")),
])
def test_distance_line_is_parsed_not_matched(line, expect):
    assert rs_paste.parse_distance_line(line) == expect


def test_clean_markdown_links_bullets_and_pipe_tables():
    raw = "* [Race 2](https://x/y/R2)\n| Tab | Horse |\n|---|---|\n| 1 | CALAS |\n"
    out = rs_paste.clean_markdown(raw)
    assert out.splitlines()[0] == "Race 2"
    assert "Tab\tHorse" in out and "1\tCALAS" in out and "---" not in out


# --------------------------------------------------------------------------- #
# sources on snapshots
# --------------------------------------------------------------------------- #
def test_ladbrokes_runner_fields():
    ev = json.load(io.open(os.path.join(FX, "lb_grafton_r2.json"), encoding="utf-8"))
    card = ladbrokes.card_from_event(ev["data"])
    assert card.venue == "Grafton" and card.race_no == 2 and card.dist_m == 1200
    assert len(card.field_) == 9
    e = card.find(1, "Sarmat")
    assert e.barrier == 8 and e.weight == 59.0 and e.jockey == "Stephen Cummins"
    assert e.odds == 6.5 and e.odds_open == 4.0 and len(e.flucs) == 3 and e.flucs[0].when is not None
    assert e.speed_label == "Backmarker" and e.settling == 1.0
    assert e.jockey_stats == (50, 2, 10) and e.trainer_stats == (65, 4, 16)
    assert e.move_pct == pytest.approx(0.625)


def test_sportinglife_rides_and_previous_results():
    rj = json.load(io.open(os.path.join(FX, "sl_race.json"), encoding="utf-8"))
    card = sportinglife.card_from_race(rj, None, date(2026, 9, 7), "UK")
    assert card.dist_m == 1218 and card.race_class == "Class 5" and card.prize == 4397.0
    e = card.find(1, "Distant Rumble")
    assert e.official_rating == 68 and e.weight == pytest.approx(62.6, abs=0.1) and e.sex == "G"
    assert e.odds == 17.0 and e.last_run_days == 46 and len(e.runs) == 6
    assert e.runs[0].pos == 7 and e.runs[0].field_size == 8 and e.runs[0].margin_l > 0


def test_sportinglife_estimated_margin_monotone():
    assert sportinglife.estimated_margin(1, 10) < 0
    assert sportinglife.estimated_margin(2, 10) < sportinglife.estimated_margin(5, 10)


def test_hkjc_racecard_horse_and_sectional():
    card = hkjc.parse_racecard(read("hk_racecard_hv1.html"), date(2026, 9, 9))
    assert card.venue == "Happy Valley" and card.race_no == 1 and card.dist_m == 1200
    assert card.race_class == "Class 5" and card.prize == 875000.0 and card.start_time == "19:05"
    e = card.find(1, "Flash Star")
    assert e.barrier == 11 and e.official_rating == 40 and e.last_run_days == 101 and e.gear == "TT"
    # card shows "6/9/9/11/5/10" oldest -> latest; the model wants latest first, 10+ as 0
    assert e.weight == pytest.approx(61.2, abs=0.1) and e.form_string == "050996"
    runs, career, extras = hkjc.parse_horse(read("hk_horse_real.html"), date(2026, 9, 9))
    assert career == (40, 2, 6, 3) and len(runs) == 20
    assert runs[0].pos == 1 and runs[0].dist_m == 1800 and runs[0].race_name.isdigit()
    sec = hkjc.parse_sectional(read("hk_sectional_real.html"))
    assert sec["BEAUTY MISSILE"]["last_400"] == 23.04 and sec["BEAUTY MISSILE"]["positions"] == [13, 13, 12, 9, 1]


def test_winningform_race_page():
    card = winningform.parse_race(read("wf_fairview_r1.html", "cp1252"), date(2026, 9, 9))
    assert card.race_no == 1 and card.dist_m == 1300 and card.surface == "AW"
    assert len(card.field_) == 6
    e = card.find(1, "Madra Rua")
    assert e.official_rating == 73 and e.weight == 61.0 and e.barrier == 4
    assert e.jockey_stats == (30, 4, 10) and e.career == (8, 0, 2, 1) and len(e.runs) == 8
    assert e.runs[0].run_date == date(2026, 7, 25) and e.runs[0].pos == 4 and e.runs[0].margin_l == 4.05
    first = card.find(2, "Twice A Queen")
    assert first.extras.get("first_run") and first.barrier == 1 and first.weight == 57.0


def test_winningform_cover_parsing():
    html = ('<span class="rb2">Fairview (ECP)<br>Wednesday, 9 September 2026</span>'
            '<table><tr><td><a href="GAU/2P260909_1.htm">Race 1</a></td>'
            '<td><a href="GAU/2P260909_2.htm">Race 2</a></td></tr></table>')
    mts = winningform.parse_cover(html)
    assert mts[0]["venue"] == "Fairview" and mts[0]["date"] == date(2026, 9, 9)
    assert mts[0]["races"] == {1: "GAU/2P260909_1.htm", 2: "GAU/2P260909_2.htm"}


def test_hrn_entries_table():
    card = hrn.parse_track_page(read("hrn_track.html"), 1, date(2026, 9, 7))
    assert card.race_no == 1 and card.start_time.startswith("1:30") and card.surface == "DIRT"
    assert card.dist_m == 366 and card.prize == 20800.0
    e = card.field_[0]
    assert e.number == 1 and e.barrier == 1 and e.trainer == "Melvin O. Cordova" and e.jockey == "Gerardo Vera"
    assert e.extras["morning_line"] == 9.0


def test_pmu_gap_and_musique():
    assert pmu.gap_lengths({"knownValue": "DEUX_LONGUEURS", "rawValue": "2 L"}) == 2.0
    assert pmu.gap_lengths({"rawValue": "1/2 L"}) == 0.5
    assert pmu.gap_lengths({"knownValue": "TETE"}) == 0.2
    assert pmu.parse_musique("4a0a4a1a1a0m(25)6m") == "404110-6"


def test_pmu_past_runs_rebuild_margins():
    perf = json.load(io.open(os.path.join(FX, "pmu_plat_perf.json"), encoding="utf-8"))
    runs = pmu._past_runs(perf["participants"][0], date(2026, 9, 7))
    assert runs and runs[0].pos is not None and runs[0].field_size == 7
    assert runs[0].margin_l is not None and runs[0].prize == 31100


# --------------------------------------------------------------------------- #
# common helpers
# --------------------------------------------------------------------------- #
def test_helpers():
    assert common.frac_to_decimal("7/2") == 4.5 and common.frac_to_decimal("EVS") == 2.0
    assert common.frac_to_decimal("8/10F") == 1.8 and common.frac_to_decimal("SP") is None
    assert common.imperial_to_m("6f 12y") == 1218 and common.imperial_to_m("1m 2f") == 2012
    assert common.lengths_from_text("2 1/4") == 2.25 and common.lengths_from_text("nk") == 0.3
    assert common.venue_matches("Grafton", "Grafton (NSW)") and common.venue_matches("Saint-Cloud", "SAINT-CLOUD")
    assert common.venue_matches("Wolverhampton (AW)", "Wolverhampton") and not common.venue_matches("Perth", "Windsor")


def test_merge_joins_by_number_then_name():
    a = common.RaceCard(entries=[common.Entry(number=1, name="Calas"), common.Entry(number=2, name="Purple Lion (Ire)")])
    b = common.RaceCard(sources=["X"], entries=[common.Entry(number=1, name="CALAS", odds=4.0, jockey="J"),
                                                 common.Entry(number=None, name="Purple Lion", barrier=5)])
    common.merge_cards(a, b)
    assert a.entries[0].odds == 4.0 and a.entries[0].jockey == "J" and a.entries[1].barrier == 5
    assert a.warnings == []


def test_merge_prefer_fields_overrides_market_only():
    a = common.RaceCard(entries=[common.Entry(number=1, name="A", odds=5.0, jockey="Paste J")])
    b = common.RaceCard(sources=["LB"], entries=[common.Entry(number=1, name="A", odds=4.0, jockey="LB J")])
    common.merge_cards(a, b, prefer_fields=common.MARKET_FIELDS)
    assert a.entries[0].odds == 4.0 and a.entries[0].jockey == "Paste J"


# --------------------------------------------------------------------------- #
# model
# --------------------------------------------------------------------------- #
def _card_with_prices(prices, runs=True):
    card = common.RaceCard(country="AUS", dist_m=1200, surface="TURF", going="GOOD", currency="AUD", prize=27000)
    for i, px in enumerate(prices, start=1):
        e = common.Entry(number=i, name=f"H{i}", barrier=i, weight=56.0, odds=px, career=(10, 2, 2, 1))
        if runs:
            e.runs = [common.PastRun(run_date=date(2026, 8, 1), days_ago=37, dist_m=1200, surface="TURF",
                                     pos=i, field_size=8, margin_l=(-0.5 if i == 1 else 1.5 * i), currency="AUD",
                                     prize=27000)]
        card.entries.append(e)
    return card


def test_devig_sums_to_one_and_keeps_order():
    p = M.devig([2.0, 4.0, 8.0, 16.0])
    assert sum(p) == pytest.approx(1.0) and p[0] > p[1] > p[2] > p[3]


def test_no_form_means_market_exactly():
    card = _card_with_prices([2.0, 4.0, 6.0, 10.0], runs=False)
    for e in card.entries:
        e.career = (0, 0, 0, 0)
        e.barrier = None
        e.weight = None
    rows, _ = M.rate(card, M.Params(market_weight=0.4))
    for r in rows:
        assert r.p_final == pytest.approx(r.p_market, abs=1e-9)


def test_form_moves_probabilities_and_top3_is_coherent():
    card = _card_with_prices([3.0, 3.0, 3.0, 3.0, 3.0, 3.0])
    rows, _ = M.rate(card, M.Params())
    assert rows[0].name == "H1" and rows[0].p_final > rows[-1].p_final
    assert sum(r.p_final for r in rows) == pytest.approx(1.0)
    assert sum(r.p_top3 for r in rows) == pytest.approx(3.0, abs=1e-6)
    assert all(0 <= r.p_top3 <= 1 for r in rows)


def test_market_move_rewards_steamers():
    card = _card_with_prices([4.0, 4.0, 4.0, 4.0], runs=False)
    card.entries[0].odds_open = 8.0        # steamed 8 -> 4
    card.entries[1].odds_open = 2.0        # drifted 2 -> 4
    rows, _ = M.rate(card, M.Params(k_move=1.0))
    by = {r.name: r for r in rows}
    assert by["H1"].terms["market move"] > 0 > by["H2"].terms["market move"]
    assert by["H1"].move_label == "steamer" and by["H2"].move_label == "drifter"


def test_simulation_matches_probabilities():
    card = _card_with_prices([2.0, 4.0, 6.0, 10.0, 20.0])
    rows, _ = M.rate(card, M.Params(sims=40000))
    sim = M.simulate(rows, M.Params(sims=40000))
    for i, r in enumerate(rows):
        assert sim.position_matrix[i, 0] == pytest.approx(r.p_final, abs=0.02)
    assert sim.position_matrix.sum(axis=1) == pytest.approx([1.0] * 5, abs=1e-9)
    assert sim.exactas and sim.exactas[0][1] > sim.exactas[-1][1]


def test_speed_score_from_label_and_settling():
    assert M.speed_score(common.Entry(speed_label="Leader")) == 1.0
    assert M.speed_score(common.Entry(speed_label="Backmarker")) == 0.05
    assert M.speed_score(common.Entry(settling=2.0)) == pytest.approx(0.8)
    assert M.speed_score(common.Entry()) is None


def test_country_defaults_and_book_percentage():
    assert M.defaults_for("HK").k_rating == 0.14 and M.defaults_for("XX").spread == 3.2
    assert M.book_percentage([2.0, 4.0, 4.0]) == pytest.approx(100.0)


# --------------------------------------------------------------------------- #
# pipeline (offline: sources monkeypatched)
# --------------------------------------------------------------------------- #
def test_pipeline_merges_paste_header_with_feed(monkeypatch):
    ev = json.load(io.open(os.path.join(FX, "lb_grafton_r2.json"), encoding="utf-8"))

    def fake_lb(day, venue, race_no, country="AUS"):
        c = ladbrokes.card_from_event(ev["data"])
        c.race_date = day
        return c

    def fail(*a, **k):
        raise common.SourceError("offline")
    monkeypatch.setattr(pipeline.ladbrokes, "fetch", fake_lb)
    monkeypatch.setattr(pipeline.betfair, "fetch", fail)
    A = pipeline.analyse(read("rs_grafton_r2_2026-09-07_header.txt"))
    assert A.card.venue == "Grafton" and A.card.race_no == 2 and A.card.dist_m == 1200
    assert A.card.race_class == "MDN" and A.card.prize == 27000.0 and A.card.currency == "AUD"
    assert len(A.rows) == 9 and A.sim is not None
    assert any(s.name == "Ladbrokes AU" and s.ok for s in A.status)
    assert any(s.name == "Betfair Exchange" and not s.ok for s in A.status)
    assert sum(r.p_final for r in A.rows) == pytest.approx(1.0)
    assert A.card.warnings == []


def test_pipeline_requires_identity():
    with pytest.raises(ValueError):
        pipeline.analyse("nothing useful here")


# --------------------------------------------------------------------------- #
# app smoke + stale-guard ordering
# --------------------------------------------------------------------------- #
def test_guard_precedes_helper_imports():
    src = io.open(os.path.join(HERE, "app.py"), encoding="utf-8").read()
    guard = src.index("_REQUIRED")
    for mod in ("import charts", "import model", "import pipeline", "import rs_paste", "import common"):
        assert src.index(mod) > guard


def test_app_renders_landing_page():
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(os.path.join(HERE, "app.py"), default_timeout=60)
    at.run()
    assert not at.exception
    assert any("Horse Race Predictor" in t.value for t in at.title)
