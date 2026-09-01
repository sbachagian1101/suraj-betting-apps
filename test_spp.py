"""Tests for SoccerPredictorPro.

The ones that matter are the consistency tests: every market is read off one
distribution, so if the HT/FT joint stops agreeing with the 1X2 numbers the app
is quietly lying on one of its tabs.
"""
from __future__ import annotations

import ast
import glob
import os

import numpy as np
import pandas as pd
import pytest

import spp_data as spd
import spp_model as spm

HERE = os.path.dirname(os.path.abspath(__file__))
TEAM_FILES = sorted(glob.glob(os.path.join(HERE, "sample_data", "*-teams-*.csv")))[-2:]
MATCH_FILES = sorted(glob.glob(os.path.join(HERE, "sample_data", "*-matches-*.csv")))[-2:]


@pytest.fixture(scope="module")
def team_ratings():
    kind, frame, _ = spd.load(TEAM_FILES)
    assert kind == spd.KIND_TEAM
    return spm.build(kind, frame), frame


@pytest.fixture(scope="module")
def match_ratings():
    kind, frame, _ = spd.load(MATCH_FILES)
    assert kind == spd.KIND_MATCH
    return spm.build(kind, frame), frame


# --- detection ---------------------------------------------------------------

def test_sniff_distinguishes_the_two_shapes():
    assert spd.sniff(pd.read_csv(TEAM_FILES[0])) == spd.KIND_TEAM
    assert spd.sniff(pd.read_csv(MATCH_FILES[0])) == spd.KIND_MATCH
    assert spd.sniff(pd.DataFrame({"a": [1]})) is None


def test_duplicate_upload_is_dropped_not_double_counted():
    """A '... (1).csv' second copy would otherwise double every rate."""
    kind, frame, notes = spd.load(TEAM_FILES + [TEAM_FILES[0]])
    assert kind == spd.KIND_TEAM
    _, plain, _ = spd.load(TEAM_FILES)
    assert len(frame) == len(plain)
    assert any("duplicate" in n.lower() for n in notes)


def test_mixed_upload_picks_one_shape_and_says_so():
    kind, _, notes = spd.load(TEAM_FILES + MATCH_FILES[:1])
    assert kind == spd.KIND_TEAM
    assert any("ignored" in n.lower() for n in notes)


# --- the consistency guarantees ----------------------------------------------

@pytest.mark.parametrize("fixture_name", ["team_ratings", "match_ratings"])
def test_htft_marginals_match_the_other_tabs(fixture_name, request):
    ratings, _ = request.getfixturevalue(fixture_name)
    home, away = ratings.teams[0], ratings.teams[1]
    p = spm.predict(ratings, home, away)
    j = p["htft"]
    ft = [p["ft"]["home"], p["ft"]["draw"], p["ft"]["away"]]
    ht = [p["ht"]["home"], p["ht"]["draw"], p["ht"]["away"]]
    np.testing.assert_allclose(j.sum(0), ft, atol=1e-6)
    np.testing.assert_allclose(j.sum(1), ht, atol=1e-6)
    assert abs(j.sum() - 1) < 1e-9
    assert (j >= 0).all()


def test_raking_is_actually_needed():
    """Guards the fix: the unraked convolution really does drift off the 1X2.

    If this ever stops failing, the raking step in `htft` has become a no-op and
    the test above would pass for the wrong reason.
    """
    lh, la, share = 1.61, 0.92, 0.45
    ft = spm.result_probs(spm.score_matrix(lh, la, spm.DEFAULT_RHO))
    ht_m, sh_m = spm.half_matrices(lh, la, share)
    raw = spm.htft(ht_m, sh_m)                       # no targets -> unraked
    drift = abs(raw.sum(0) - np.array([ft["home"], ft["draw"], ft["away"]])).max()
    assert drift > 1e-3, "unraked joint no longer drifts; raking may be pointless"


@pytest.mark.parametrize("fixture_name", ["team_ratings", "match_ratings"])
def test_probabilities_are_well_formed(fixture_name, request):
    ratings, _ = request.getfixturevalue(fixture_name)
    p = spm.predict(ratings, ratings.teams[0], ratings.teams[1])
    for key in ("ft", "ht"):
        s = sum(p[key].values())
        assert abs(s - 1) < 1e-6, f"{key} sums to {s}"
        assert all(0 <= v <= 1 for v in p[key].values())
    assert abs(p["ft_matrix"].sum() - 1) < 1e-9


# --- halves ------------------------------------------------------------------

def test_second_half_is_never_negative_and_carries_more_goals(team_ratings):
    """The failure that motivated splitting rather than rating halves directly.

    Rating half time from a ratio of two team rates gave Esteghlal-Persepolis
    half-time lambdas of 1.27/0.80 against a league average of 0.41/0.38 --
    implying a *quieter* second half, which is backwards.
    """
    ratings, _ = team_ratings
    assert 0.25 <= ratings.ht_share <= 0.60
    for home in ratings.teams[:6]:
        for away in ratings.teams[:6]:
            if home == away:
                continue
            lh, la = spm.expected_goals(ratings, home, away)
            first = (lh + la) * ratings.ht_share
            second = (lh + la) * (1 - ratings.ht_share)
            assert second > 0
            assert second > first, "second half should carry more goals than the first"


def test_half_time_draw_is_the_common_half_time_state(team_ratings):
    ratings, _ = team_ratings
    p = spm.predict(ratings, ratings.teams[0], ratings.teams[1])
    assert p["ht"]["draw"] > p["ft"]["draw"], "levelness should decay through the match"


# --- Asian handicap ----------------------------------------------------------

def test_handicap_rows_are_a_partition(team_ratings):
    ratings, _ = team_ratings
    p = spm.predict(ratings, ratings.teams[0], ratings.teams[1])
    ah = p["asian"]
    np.testing.assert_allclose(ah["home_win"] + ah["push"] + ah["away_win"], 1.0, atol=1e-8)
    assert ah["home_win"].is_monotonic_increasing, "giving the home side more start " \
                                                   "must never lower its cover chance"


def test_quarter_lines_can_half_push():
    """A -0.25 line splits across 0.0 and -0.5, so a draw half-loses."""
    m = spm.score_matrix(1.3, 1.1, 0.0)
    ah = spm.asian_handicap(m, [-0.25, 0.0, -0.5])
    q = ah[ah["line"] == -0.25].iloc[0]
    assert 0 < q["push"] < ah[ah["line"] == 0.0].iloc[0]["push"]
    assert ah[ah["line"] == -0.5].iloc[0]["push"] == pytest.approx(0.0, abs=1e-12)


def test_level_fixture_has_a_fair_line_near_zero():
    m = spm.score_matrix(1.2, 1.2, 0.0)
    assert abs(spm.fair_line(spm.asian_handicap(m))) <= 0.25


def test_handicap_confidence_is_not_high_for_every_fixture(team_ratings):
    """Regression: measuring separation off the best line on the ladder read the
    +2.00 handicap, which is ~99% for any fixture, so this band said High always.
    """
    ratings, _ = team_ratings
    bands = set()
    for home in ratings.teams[:6]:
        for away in ratings.teams[:6]:
            if home != away:
                bands.add(spm.predict(ratings, home, away)["conf_ah"]["band"])
    assert len(bands) > 1, f"handicap confidence never varies: {bands}"


def test_handicap_confidence_tracks_the_level_line(team_ratings):
    ratings, _ = team_ratings
    p = spm.predict(ratings, ratings.teams[0], ratings.teams[1])
    ah = p["asian"]
    expected = float(ah.loc[ah["line"] == 0.0, "home_no_push"].iloc[0])
    assert p["p_level"] == pytest.approx(expected)
    assert 0.0 <= p["p_level"] <= 1.0


# --- confidence --------------------------------------------------------------

def test_confidence_is_capped_by_the_weaker_of_separation_and_sample():
    strong = spm.confidence([0.70, 0.20, 0.10], n_matches=100)
    assert strong["band"] == spm.CONF_HIGH
    thin = spm.confidence([0.70, 0.20, 0.10], n_matches=5)
    assert thin["band"] == spm.CONF_LOW, "a tiny sample must not read as High"
    assert "data" in thin["why"]
    flat = spm.confidence([0.35, 0.34, 0.31], n_matches=100)
    assert flat["band"] == spm.CONF_LOW


# --- markets that must stay absent -------------------------------------------

def test_no_goals_markets_are_exposed(team_ratings):
    """BTTS and Over/Under scored worse than the base rate; they must not reappear."""
    ratings, _ = team_ratings
    p = spm.predict(ratings, ratings.teams[0], ratings.teams[1])
    banned = {"btts", "over", "under", "over_25", "btts_yes"}
    assert not any(k.lower() in banned or k.lower().startswith(("over_", "under_", "btts"))
                   for k in p), f"a goals market leaked into predict(): {sorted(p)}"


def test_app_does_not_render_goals_markets():
    src = open(os.path.join(HERE, "app.py"), encoding="utf-8").read()
    body = src.split('"""', 2)[-1]          # skip the module docstring
    for bad in ("markets(", "btts_yes", "over_25"):
        assert bad not in body, f"app.py references {bad}"


# --- the stale-module guard --------------------------------------------------

def test_guard_runs_before_any_helper_import():
    """The guard is useless below the imports it is meant to protect.

    A cached module raises ImportError on the `import spp_model` line, and
    Streamlit Cloud redacts the message, so the page shows an opaque error
    instead of the actionable one.
    """
    tree = ast.parse(open(os.path.join(HERE, "app.py"), encoding="utf-8").read())
    guard_line = None
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name) and t.id == "_REQUIRED":
                    guard_line = node.lineno
    assert guard_line is not None, "_REQUIRED guard has gone missing"
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            names = [a.name for a in node.names] + ([node.module] if
                                                    isinstance(node, ast.ImportFrom) else [])
            if any(n and n.startswith(("spp_", "soccer_")) for n in names):
                assert node.lineno > guard_line, \
                    f"line {node.lineno} imports a helper above the guard"


def test_guard_lists_every_symbol_the_app_uses():
    src = open(os.path.join(HERE, "app.py"), encoding="utf-8").read()
    tree = ast.parse(src)
    required = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id == "_REQUIRED" for t in node.targets):
            required = ast.literal_eval(node.value)
    used = {"spp_data": set(), "spp_model": set()}
    alias = {"spd": "spp_data", "spm": "spp_model"}
    for node in ast.walk(tree):
        if (isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name)
                and node.value.id in alias):
            used[alias[node.value.id]].add(node.attr)
    for mod, names in used.items():
        missing = names - set(required.get(mod, ()))
        assert not missing, f"{mod}: guard does not list {sorted(missing)}"


# --- season recency ----------------------------------------------------------

def test_recent_seasons_outweigh_older_ones():
    """Without decay, a 30-match season three years back ties last season.

    Two synthetic teams with identical current form but opposite histories must
    not come out with the same rating.
    """
    def row(team, season, scored_home, mp=30):
        return {"common_name": team, "season": season, "matches_played": mp,
                "matches_played_home": mp // 2, "matches_played_away": mp // 2,
                "goals_scored_per_match_home": scored_home,
                "goals_scored_per_match_away": 1.0,
                "goals_conceded_per_match_home": 1.0,
                "goals_conceded_per_match_away": 1.0,
                "goals_scored_per_match_half_time_home": 0.4,
                "goals_scored_per_match_half_time_away": 0.4}

    frame = pd.DataFrame([
        row("Rising", "2024/2025", 0.5), row("Rising", "2025/2026", 2.5),
        row("Fading", "2024/2025", 2.5), row("Fading", "2025/2026", 0.5),
    ])
    frame, _ = spd._clean_team(frame, [])
    r = spm.from_team_files(frame)
    assert r.atk_home["Rising"] > r.atk_home["Fading"], \
        "the team scoring more *recently* must rate higher"


def test_season_decay_is_actually_applied():
    assert 0 < spm.SEASON_DECAY < 1


def test_distinct_seasons_are_all_kept():
    """Regression: three team files of a 16-team league export 16 rows each with
    identical headers, so a shape-based duplicate check silently binned a whole
    season. The app then ran on 2024/25 + 2026/27 and skipped 2025/26 entirely.
    """
    files = sorted(glob.glob(os.path.join(HERE, "sample_data", "*-teams-*.csv")))
    assert len(files) >= 3, "need three sample seasons to exercise this"
    kind, frame, notes = spd.load(files)
    assert kind == spd.KIND_TEAM
    assert frame["season_label"].nunique() == len(files), \
        f"expected {len(files)} seasons, kept {sorted(frame['season_label'].unique())}"
    assert not any("duplicate" in n.lower() for n in notes)
    expected = sum(len(pd.read_csv(f)) for f in files)
    assert len(frame) == expected


def test_a_real_duplicate_is_still_caught():
    files = sorted(glob.glob(os.path.join(HERE, "sample_data", "*-teams-*.csv")))
    kind, frame, notes = spd.load(files + [files[0]])
    _, plain, _ = spd.load(files)
    assert len(frame) == len(plain)
    assert any("duplicate" in n.lower() for n in notes)
