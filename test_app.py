"""Run with:  python -m pytest -q  (from the bulgaria-xg folder)."""
import numpy as np
import pandas as pd
import pytest

import model as M


@pytest.fixture(scope="module")
def leagues():
    return M.discover_leagues([M.DATA_DIR])


@pytest.fixture(scope="module")
def matches(leagues):
    return M.load_matches(leagues["bulgaria-second-league"])


@pytest.fixture(scope="module")
def kaz(leagues):
    return M.load_matches(leagues["kazakhstan-first-division"])


def test_filename_parsing():
    info = M.parse_filename("kazakhstan-first-division-teams-2025-to-2025-stats.csv")
    assert info == {"league": "kazakhstan-first-division", "kind": "teams", "season": "2025"}
    info = M.parse_filename(r"C:\x\bulgaria-second-league-matches-2025-to-2026-stats.csv")
    assert info["season"] == "2025-26" and info["kind"] == "matches"
    assert M.parse_filename("random.csv") is None
    assert M.league_label("kazakhstan-first-division") == "Kazakhstan First Division"
    assert M.league_label("uzbekistan-uzbekistan-super-league") == "Uzbekistan Super League"


def test_discover_leagues(leagues):
    assert set(leagues) >= {"bulgaria-second-league", "kazakhstan-first-division"}
    for files in leagues.values():
        assert all("matches" in p.name for p in files)     # teams files never feed the model


def test_teams_file_is_rejected():
    teams_file = next(M.DATA_DIR.glob("*teams*.csv"))
    with pytest.raises(ValueError, match="not a FootyStats matches export"):
        M.load_matches([teams_file])


def test_load_shapes(matches):
    assert len(M.completed(matches)) >= 300
    assert {"CSKA Sofia II", "Spartak Pleven"} <= set(matches.home)
    assert set(matches.season) == {"2025-26", "2026-27"}
    # post-match season PPG columns are never loaded (they leak the result)
    assert "home_ppg" not in matches.columns


def test_kazakhstan_loads_separately(kaz, matches):
    assert set(kaz.season) == {"2025", "2026"}        # calendar-year seasons from the file name
    assert not (set(kaz.home) & set(matches.home))    # no team-name overlap between leagues
    assert 0.5 < M.completed(kaz).has_xg.mean() < 0.9  # partial xG coverage is handled
    c = M.completed(kaz)
    assert (c.hxg > 0).sum() + (c.hxg == 0).sum() == len(c)


def test_no_result_leak_in_fixtures(matches):
    f = M.fixtures(matches)
    assert (f.result == -1).all()


def test_market_probs():
    p = M.market_probs(1.54, 3.65, 5.70)
    assert abs(p.sum() - 1) < 1e-9 and p[0] > p[1] > p[2]
    assert M.market_probs(0, 0, 0) is None
    assert abs(M.overround(2.0, 2.0, 2.0) - 1.5) < 1e-9


def test_fit_and_predict(matches):
    f = M.fit(matches)
    pr = M.predict(f, "CSKA Sofia II", "Spartak Pleven")
    assert abs(pr["probs"].sum() - 1) < 1e-6
    assert pr["probs"][0] > 0.5           # clear home favourite in Sept 2026
    assert 0 < pr["over25"] < 1 and 0 < pr["btts"] < 1
    assert abs(pr["matrix"].sum() - 1) < 1e-9
    with pytest.raises(KeyError):
        M.predict(f, "Nobody FC", "Spartak Pleven")


def test_fit_kazakhstan(kaz):
    f = M.fit(kaz)
    pr = M.predict(f, "Shakhter Karagandy", "Aktobe Jas")
    assert pr["probs"][0] > 0.6           # runaway leader at home to a struggler


def test_fit_is_causal(matches):
    """A fit 'as of' a date must ignore matches on or after that date."""
    c = M.completed(matches)
    cutoff = c.kickoff.iloc[100]
    f = M.fit(matches, as_of=cutoff)
    assert f.n_matches == (c.kickoff < cutoff).sum()


def test_ratings_table(matches):
    t = M.ratings_table(M.fit(matches))
    assert t.rating.is_monotonic_decreasing
    assert (t.attack > 0).all() and (t.defence > 0).all()


def test_walk_forward_and_summary(matches):
    c = M.completed(matches)
    bt = M.walk_forward(matches, start=len(c) - 25)
    assert len(bt) == 25
    s = M.summarise(bt)
    assert set(s.index) >= {"Bookmaker (fair)", "Model", "League base rates"}
    assert (s["log loss"] > 0).all()
    cal = M.calibration(bt)
    assert cal.n.sum() == 3 * bt.k_h.notna().sum()
    roi = M.flat_stake_roi(bt)
    assert roi.iloc[-1].selection == "All"


def test_kelly():
    assert M.kelly(0.5, 2.0) == 0
    assert M.kelly(0.6, 2.0) == pytest.approx(0.2)
    assert M.kelly(0.9, 1.0) == 0


def test_form_and_h2h(matches):
    f = M.recent_form(matches, "CSKA Sofia II", 5)
    assert len(f) == 5 and set(f.res) <= {"W", "D", "L"}
    h = M.head_to_head(matches, "CSKA Sofia II", "Spartak Pleven")
    assert len(h) >= 2
