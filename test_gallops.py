from datetime import date
from pathlib import Path

import pytest

import gallops as g

HERE = Path(__file__).resolve().parent
SAMPLE = (HERE / "data" / "sample_06sep2026.txt").read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def races():
    return g.parse_races(SAMPLE)


def test_parses_two_races_with_results(races):
    assert [r.race_no for r in races] == [1, 2]
    assert all(r.race_date == date(2026, 9, 6) for r in races)
    assert races[0].results == ["Lomu", "Indigenous", "Lava county"]
    assert [r.name for r in races[1].runners] == [
        "WARBIRD", "TSUNAMI WARNING", "IKO IKO", "IBHELE", "ZACATOO", "BOOM TOWN"]


def test_gallop_lines(races):
    lava = races[0].runners[0]
    assert len(lava.gallops) == 4
    first, bt = lava.gallops[0], lava.gallops[-1]
    assert (first.t600, first.t200, first.equipment) == (33.79, 12.05, "Acc")
    assert first.first400 == pytest.approx(21.74)
    assert bt.barrier_trial and bt.equipment == "Lib (BT)" and bt.t600 == 45.67
    # Comment keywords
    assert lava.gallops[1].company == "We Have Touchdown"
    rosalie = races[0].runners[4]
    assert rosalie.gallops[0].beaten_by == "La Cadiere D'Azur"
    # Missing jockey and missing 600 m clock
    warbird = races[1].runners[0]
    assert warbird.gallops[2].jockey == "" and warbird.gallops[2].company == "Siriano"
    tsunami = races[1].runners[1]
    assert tsunami.gallops[2].t600 is None and tsunami.gallops[2].t200 == 12.16


def test_features(races):
    df = g.race_frame(races[0])
    lomu = df.loc[4]
    assert lomu["horse"] == "LOMU"
    assert lomu["n_gallops"] == 4 and lomu["gallops_14d"] == 3 and lomu["days_since"] == 4
    assert lomu["taper"] == pytest.approx(36.42 - 34.55)
    assert lomu["finish"] == 1
    assert df.loc[1, "barrier_trial"] == 1
    assert df.loc[1, "best_adj600"] == pytest.approx(33.79 + 0.5)   # Acc penalty
    assert df.loc[5, "beaten_n"] == 1
    assert df.loc[3, "finish"] == 2 and df.loc[2, "finish"] == 0


def test_freshness_penalty():
    r = g.Runner(1, "X", [g.Gallop(date(2026, 8, 20), "", 34.0, 23.0, 12.0, "Lib", False, "")])
    assert g.runner_features(r, date(2026, 9, 6))["fresh_penalty"] == pytest.approx(4.0)
    assert g.runner_features(r, date(2026, 8, 24))["fresh_penalty"] == 0.0
    assert g.runner_features(r, date(2026, 8, 21))["fresh_penalty"] == 2.0


def test_scores_sum_to_100_and_rank(races):
    for race in races:
        df = g.score_race(race)
        assert df["win_pct"].sum() == pytest.approx(100.0)
        assert list(df["rank"]) == list(range(1, len(df) + 1))
    exp = g.explain(races[0])
    assert set(g.DEFAULT_WEIGHTS) <= set(exp.columns)


def test_default_weights_on_sample(races):
    rep = g.evaluate(races)
    assert rep["races"] == 2
    assert rep["winner_hit"] == 1.0          # both winners top-rated (in-sample, see README)


def test_fit_refuses_small_samples(races):
    w, info = g.fit_weights(races)
    assert info["fitted"] is False and w == g.DEFAULT_WEIGHTS


def test_store_roundtrip(tmp_path, races):
    path = tmp_path / "races.jsonl"
    assert g.save_races(races, path) == 2
    assert g.save_races(races[:1], path) == 2          # replaced, not duplicated
    back = g.load_races(path)
    assert [r.race_no for r in back] == [1, 2]
    assert g.race_frame(back[0]).equals(g.race_frame(races[0]))


def test_parse_without_results():
    text = SAMPLE.split("Actual results")[0]
    race = g.parse_races(text)[0]
    assert race.results == [] and "finish" not in g.race_frame(race)
