"""Tests for the Soccerway predictor.

Unit tests run offline on captured feed snippets. The two tests marked
``live`` hit Soccerway and are skipped automatically when offline.
"""
from datetime import datetime, timezone

import pytest
import requests

import model as M
import soccerway as SW

F = SW.FIELD_SEP
K = SW.KV_SEP


def rec(**kv):
    return F.join(f"{k}{K}{v}" for k, v in kv.items())


# --------------------------------------------------------------------------- #
# URL parsing and feed parsing
# --------------------------------------------------------------------------- #
def test_parse_team_url_variants():
    assert SW.parse_team_url("https://us.soccerway.com/team/groningen/MBUGcjb9/") == ("groningen", "MBUGcjb9")
    assert SW.parse_team_url(" www.soccerway.com/team/twente/dhOKTHGA ") == ("twente", "dhOKTHGA")
    assert SW.parse_team_url("https://us.soccerway.com/team/twente/dhOKTHGA/results/") == ("twente", "dhOKTHGA")


def test_parse_team_url_accepts_flashscore_links():
    assert SW.parse_team_url("https://www.flashscore.com/team/groningen/MBUGcjb9/") == ("groningen", "MBUGcjb9")
    assert SW.parse_team_url("https://www.flashscore.co.uk/team/twente/dhOKTHGA/") == ("twente", "dhOKTHGA")
    assert SW.parse_team_url("https://www.flashscore.com.au/team/twente/dhOKTHGA/results/") == ("twente", "dhOKTHGA")


def test_parse_team_url_rejects_junk():
    with pytest.raises(SW.SoccerwayError):
        SW.parse_team_url("https://us.soccerway.com/game/groningen-MBUGcjb9/psv-M9UEHJWi/")


def test_clean_name_strips_country_tag():
    assert SW.clean_name("Twente (Ned)") == "Twente"
    assert SW.clean_name("Qarabag (Aze)") == "Qarabag"
    assert SW.clean_name("G.A. Eagles") == "G.A. Eagles"
    assert SW.clean_name("Sittard") == "Sittard"


def test_parse_feed_splits_records_and_fields():
    text = "~" + rec(ZA="NETHERLANDS: Eredivisie", ZB="139") + "~" + rec(AA="YZz3PZW7", AG="2", AH="3")
    recs = SW.parse_feed(text)
    assert recs[0]["ZA"] == "NETHERLANDS: Eredivisie"
    assert recs[1] == {"AA": "YZz3PZW7", "AG": "2", "AH": "3"}


def _results_html():
    header = rec(ZA="NETHERLANDS: Eredivisie")
    m1 = rec(AA="YZz3PZW7", AD="1787940000", AB="3", AE="Groningen", PX="MBUGcjb9",
             AF="Sittard", PY="YH8HX5iP", AG="2", AH="3")
    m2 = rec(AA="hdZnTHXr", AD="1787488200", AB="3", AE="PSV", PX="M9UEHJWi",
             AF="Groningen", PY="MBUGcjb9", AG="5", AH="1")
    other = rec(AA="zzzzzzzz", AD="1787488200", AB="3", AE="Ajax", PX="aaaaaaaa",
                AF="Feyenoord", PY="bbbbbbbb", AG="1", AH="1")
    fixture = rec(AA="nHJfzC2N", AD="1788689700", AB="1", AE="Groningen", PX="MBUGcjb9",
                  AF="Twente", PY="dhOKTHGA")
    return "<html>~" + "~".join([header, m1, m2, other, fixture, m1]) + "~</html>"


def test_matches_from_html_filters_and_dedupes():
    ms = SW._parse_matches(_results_html(), "MBUGcjb9")
    ids = [m.id for m in ms]
    assert ids == ["YZz3PZW7", "hdZnTHXr", "nHJfzC2N"]       # other team dropped, duplicate dropped
    assert all(m.competition == "NETHERLANDS: Eredivisie" for m in ms)
    m = ms[0]
    assert m.finished and m.home_score == 2 and m.away_score == 3
    assert m.kickoff == datetime(2026, 8, 28, 18, 0, tzinfo=timezone.utc)
    assert m.side_of("MBUGcjb9") == "H" and ms[1].side_of("MBUGcjb9") == "A"
    fx = ms[2]
    assert not fx.finished and fx.home_score is None


def test_stats_parsing_uses_full_time_section_only(monkeypatch):
    text = "~".join([
        rec(SE="Game"), rec(SF="Top stats"),
        rec(SD="432", SG="Expected goals (xG)", SH="1.92", SI="0.87"),
        rec(SD="12", SG="Ball possession", SH="68%", SI="32%"),
        rec(SE="1st Half"),
        rec(SD="432", SG="Expected goals (xG)", SH="0.50", SI="0.10"),
    ])
    monkeypatch.setattr(SW, "_get", lambda url, params=None: type("R", (), {"text": text})())
    stats = SW.match_stats("x")
    assert SW.xg_from_stats(stats) == (1.92, 0.87)
    assert stats["Ball possession"] == (68.0, 32.0)


def _daily_feed():
    return "~".join([
        rec(ZA="ENGLAND: Premier League"),
        rec(AA="QsyJgS7m", AD="1788699600", AB="3", AE="Everton", AF="Manchester Utd",
            PX="KluSTr9s", PY="ppjDR086", WU="everton", WV="manchester-united", AG="2", AH="2"),
        rec(AA="Glagw7N6", AD="1788708600", AB="1", AE="Arsenal", AF="Chelsea",
            PX="hA1Zm19f", PY="4fGZN2oK", WU="arsenal", WV="chelsea"),
        rec(ZA="ENGLAND: Championship"),
        rec(AA="cccccccc", AD="1788708600", AB="1", AE="Leeds", AF="Hull",
            PX="11111111", PY="22222222", WU="leeds", WV="hull"),
        rec(ZA="FRANCE: Ligue 1"),
        rec(AA="ffffffff", AD="1788718600", AB="2", AE="Marseille", AF="Paris FC",
            PX="33333333", PY="44444444", WU="marseille", WV="paris-fc", AG="1", AH="0"),
    ])


def test_daily_feed_parses_all_leagues_with_slugs():
    ms = SW._parse_matches(_daily_feed())
    assert [m.id for m in ms] == ["QsyJgS7m", "Glagw7N6", "cccccccc", "ffffffff"]
    assert ms[0].competition == "ENGLAND: Premier League" and ms[3].competition == "FRANCE: Ligue 1"
    assert ms[0].home_slug == "everton" and ms[0].away_slug == "manchester-united"
    assert [m.status for m in ms] == ["Finished", "Scheduled", "Scheduled", "Live"]


def test_select_leagues_matches_loosely_and_reports_misses():
    ms = SW._parse_matches(_daily_feed())
    found, missing = SW.select_leagues(
        ["England Premier League", "Framce: Ligue 1", "GERMANY: Bundesliga", "premier league", ""], ms)
    assert found["England Premier League"] == "ENGLAND: Premier League"
    assert found["Framce: Ligue 1"] == "FRANCE: Ligue 1"           # one misspelt token tolerated
    assert found["premier league"] == "ENGLAND: Premier League"    # fewest extra tokens wins
    assert missing == ["GERMANY: Bundesliga"]


def test_history_before_excludes_fixture_and_later_matches():
    ms = SW._parse_matches(_results_html(), "MBUGcjb9")
    fixture = next(m for m in ms if m.id == "nHJfzC2N")
    hist = SW.history_before(ms, fixture)
    assert [m.id for m in hist] == ["YZz3PZW7", "hdZnTHXr"]
    # a finished match must not count as its own history
    own = next(m for m in ms if m.id == "YZz3PZW7")
    assert [m.id for m in SW.history_before(ms, own)] == ["hdZnTHXr"]


def test_pipeline_cache_returns_same_object_within_ttl():
    import pipeline as P
    P.clear_cache()
    calls = []
    v1 = P._cached(("k",), 60, lambda: calls.append(1) or {"x": 1})
    v2 = P._cached(("k",), 60, lambda: calls.append(1) or {"x": 2})
    assert v1 is v2 and calls == [1]
    P.clear_cache()


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
def _match(i, home_id, away_id, hs, as_):
    return SW.Match(id=f"m{i}", kickoff=datetime(2026, 8, i + 1, tzinfo=timezone.utc),
                    competition="X", home_id=home_id, home_name="H" + home_id,
                    away_id=away_id, away_name="A" + away_id, home_score=hs, away_score=as_,
                    stage="3")


def _player(pid, rating, starter=True):
    return SW.Player(id=pid, name=pid.title(), number="1", role="Midfielder", rating=rating, starter=starter)


def _records(team, xg_for, xg_against, ratings_by_match):
    out = []
    for i, ((xf, xa), ratings) in enumerate(zip(zip(xg_for, xg_against), ratings_by_match)):
        m = _match(i, team, "opp", 2, 1)
        starters = [_player(pid, r) for pid, r in ratings.items()]
        out.append(SW.MatchRecord(match=m, side="H", goals_for=2, goals_against=1,
                                  xg_for=xf, xg_against=xa,
                                  team_rating=sum(ratings.values()) / len(ratings),
                                  formation="4-3-3", starters=starters, opponent_rating=6.5))
    return out


def test_profile_blends_and_shrinks():
    recs = _records("t", [2.0, 2.0, 2.0], [1.0, 1.0, 1.0], [{"a": 7.0}] * 3)
    p = M.profile("T", recs)
    assert p.form == "WWW" and p.points == 9
    n_eff = 1 + M.DECAY + M.DECAY ** 2
    assert p.n_eff == pytest.approx(n_eff)
    blended_att = M.XG_WEIGHT * 2.0 + (1 - M.XG_WEIGHT) * 2.0
    expected = (blended_att * n_eff + M.LEAGUE_AVG * M.PRIOR_GAMES) / (n_eff + M.PRIOR_GAMES)
    assert p.attack == pytest.approx(expected)
    assert p.xg_games == 3


def test_profile_without_xg_falls_back_to_goals():
    recs = _records("t", [None, None], [None, None], [{"a": 6.0}] * 2)
    p = M.profile("T", recs)
    assert p.xg_for is None and p.xg_games == 0
    n_eff = 1 + M.DECAY
    assert p.attack == pytest.approx((2.0 * n_eff + M.LEAGUE_AVG * M.PRIOR_GAMES) / (n_eff + M.PRIOR_GAMES))


def test_recent_matches_weigh_more():
    # newest match xG 3.0, older ones 1.0: weighted mean must sit above the plain mean
    recs = _records("t", [3.0, 1.0, 1.0, 1.0], [1.0] * 4, [{"a": 6.5}] * 4)
    p = M.profile("T", recs)
    plain = (3.0 + 1.0 + 1.0 + 1.0) / 4
    assert p.xg_for > plain
    w = M.recency_weights(4)
    assert p.xg_for == pytest.approx((3.0 * w[0] + sum(w[1:])) / sum(w))
    # reversed order (big game oldest) must come out below the plain mean
    p2 = M.profile("T", _records("t", [1.0, 1.0, 1.0, 3.0], [1.0] * 4, [{"a": 6.5}] * 4))
    assert p2.xg_for < plain


def test_separate_rates_and_lineup_windows():
    ids = [f"p{i}" for i in range(11)]
    recent = {pid: 7.5 for pid in ids}
    old = {pid: 6.0 for pid in ids}
    recs = _records("t", [1.5] * 6, [1.0] * 6, [recent, recent, old, old, old, old])
    p = M.profile("T", recs, n_rates=6, n_lineup=2)
    assert p.n == 6 and p.n_lineup == 2
    assert p.avg_rating == pytest.approx(7.5)        # only the two recent matches
    la = M.assess_lineup([_player(pid, None) for pid in ids], "today", "4-3-3", p)
    assert la.rows[0]["Starts (last N)"] == 2
    assert la.lineup_rating == pytest.approx(7.5)
    p_all = M.profile("T", recs, n_rates=6, n_lineup=6)
    assert p_all.avg_rating == pytest.approx(6.5)
    # asking for more than exists just uses what there is
    p_short = M.profile("T", recs, n_rates=15, n_lineup=10)
    assert p_short.n == 6 and p_short.n_lineup == 6


def test_probabilities_sum_to_one_and_home_edge():
    a = M.profile("A", _records("a", [1.5] * 3, [1.2] * 3, [{"p": 6.8}] * 3))
    b = M.profile("B", _records("b", [1.5] * 3, [1.2] * 3, [{"p": 6.8}] * 3))
    pred = M.predict(a, b)
    assert pred.p_home + pred.p_draw + pred.p_away == pytest.approx(1.0, abs=1e-9)
    assert pred.p_home > pred.p_away                    # identical teams: home advantage shows
    assert sum(map(sum, pred.grid)) == pytest.approx(1.0)


def test_lineup_gap_moves_expected_goals():
    ids = [f"p{i}" for i in range(11)]
    strong = {pid: 7.5 for pid in ids}
    weak = {pid: 6.0 for pid in ids}
    recs = _records("a", [1.5] * 3, [1.2] * 3, [strong, strong, weak])
    prof = M.profile("A", recs)
    # today's XI = the same players -> gap 0
    same = M.assess_lineup([_player(pid, None) for pid in ids], "today", "4-3-3", prof)
    assert same.gap == pytest.approx(0.0, abs=1e-9)
    assert same.rated == 11 and not same.missing
    # today's XI = five regulars replaced by unknowns -> five newcomers, five regulars missing
    mixed = [_player(pid, None) for pid in ids[:6]] + [_player(f"new{i}", None) for i in range(5)]
    mixed_la = M.assess_lineup(mixed, "today", "4-3-3", prof)
    assert len(mixed_la.newcomers) == 5 and len(mixed_la.missing) == 5
    base = M.predict(prof, prof)
    up = M.assess_lineup([_player(pid, None) for pid in ids], "today", "4-3-3", prof)
    up.gap = 0.5
    boosted = M.predict(prof, prof, up, None)
    assert boosted.exp_home > base.exp_home and boosted.exp_away < base.exp_away


def test_implied_removes_overround():
    imp = M.implied({"home": 3.6, "draw": 4.0, "away": 1.85})
    assert imp["home"] + imp["draw"] + imp["away"] == pytest.approx(1.0)
    assert imp["overround"] > 0
    assert M.implied({"home": 3.6, "draw": None, "away": 1.85}) is None


def test_insights_mention_form_and_book():
    a = M.profile("Alpha", _records("a", [2.5] * 3, [0.8] * 3, [{"p": 7.0}] * 3))
    b = M.profile("Beta", _records("b", [0.9] * 3, [1.9] * 3, [{"p": 6.2}] * 3))
    pred = M.predict(a, b)
    text = "\n".join(M.insights(a, b, pred, None, None, {"home": 2.0, "draw": 3.5, "away": 3.8,
                                                           "home_open": 2.2, "draw_open": 3.5, "away_open": 3.4}))
    assert "Alpha" in text and "form WWW" in text
    assert "Book (bet365)" in text
    assert "Market movement" in text


# --------------------------------------------------------------------------- #
# Live (skipped offline)
# --------------------------------------------------------------------------- #
def _online() -> bool:
    try:
        requests.head(SW.SITE, timeout=5)
        return True
    except requests.RequestException:
        return False


live = pytest.mark.skipif(not _online(), reason="no network")


@live
def test_live_results_and_stats():
    ms = SW.team_results("groningen", "MBUGcjb9")
    assert ms and ms[0].finished
    stats = SW.match_stats("YZz3PZW7")             # Groningen 2-3 Sittard, 28 Aug 2026
    assert SW.xg_from_stats(stats) == (1.92, 0.87)


@live
def test_app_end_to_end_without_matplotlib(monkeypatch):
    """The score grid shading must not need matplotlib (absent on Streamlit Cloud).

    Hide matplotlib and purge pandas' Styler modules so its has_mpl flag is
    re-evaluated, then run the whole app with Analyse pressed.
    """
    import sys
    from streamlit.testing.v1 import AppTest

    class _Block:
        def find_spec(self, name, path=None, target=None):
            if name == "matplotlib" or name.startswith("matplotlib."):
                raise ImportError("matplotlib hidden for test")
            return None

    for mod in list(sys.modules):
        if mod == "matplotlib" or mod.startswith("matplotlib.") or mod.startswith("pandas.io.formats.style"):
            monkeypatch.delitem(sys.modules, mod, raising=False)
    monkeypatch.setattr(sys, "meta_path", [_Block()] + sys.meta_path)

    at = AppTest.from_file("app.py", default_timeout=120)
    at.run()
    at.button[0].click().run()
    assert not at.exception, at.exception
    # whichever side hosts the next meeting, both clubs must be in the title
    assert "Groningen" in at.title[0].value and "Twente" in at.title[0].value
    assert any("Insights" in h.value for h in at.subheader)
    assert len(at.metric) >= 5


@live
def test_league_day_page_runs_for_premier_league():
    """Second page: pick one league for today, run it, expect a table and expanders."""
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file("app.py", default_timeout=600)
    at.run()
    at.switch_page("views/league_day.py")
    at.run()
    at.text_area[0].set_value("ENGLAND: Premier League").run()
    at.button(key="run_all_btn").click().run()
    assert not at.exception, at.exception
    body = "\n".join(t.value for t in at.markdown) + "\n".join(c.value for c in at.caption)
    if "No matches found" in body:
        pytest.skip("no Premier League matches today")
    assert at.dataframe, "summary table missing"
    assert at.expander, "per-match expanders missing"
    assert any("matches predicted" in c.value for c in at.caption)


def test_day_index_splits_country_and_league():
    import pipeline as P
    ms = SW._parse_matches(_daily_feed())
    idx = P.DayIndex(ms)
    assert idx.countries() == ["England", "France"]
    assert idx.leagues("England") == ["Championship", "Premier League"]
    fx = idx.fixtures("England", "Premier League")
    assert [m.id for m in fx] == ["QsyJgS7m", "Glagw7N6"]        # sorted by kick-off
    assert idx.fixtures("France", "Premier League") == []
    assert P.DayIndex.split("Friendly") == ("Other", "Friendly")


@live
def test_pick_a_match_flow_predicts_first_fixture():
    """Fetch Today Matches -> country -> league -> match -> Predict."""
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file("app.py", default_timeout=600)
    at.run()
    at.switch_page("views/league_day.py")
    at.run()
    at.button(key="fetch_btn").click().run()
    assert not at.exception, at.exception
    countries = at.selectbox(key="country_sel").options
    assert countries, "no countries loaded from the day feed"
    # pick the first country that has a league with at least one fixture
    at.selectbox(key="country_sel").select(countries[0]).run()
    leagues = at.selectbox(key="league_sel").options
    assert leagues
    at.selectbox(key="league_sel").select(leagues[0]).run()
    first_match = at.selectbox(key="match_sel").options[0]       # formatted label
    at.selectbox(key="match_sel").select(first_match).run()
    at.button(key="predict_btn").click().run()
    assert not at.exception, at.exception
    assert at.title, "prediction panel missing"
    assert " v " in at.title[0].value
    assert len(at.metric) >= 5


@live
def test_live_lineups_have_eleven_rated_starters():
    lu = SW.match_lineups("YZz3PZW7")
    assert set(lu) == {"HOME", "AWAY"}
    home = lu["HOME"]
    assert home.formation == "4-2-3-1" and len(home.starters) == 11
    assert all(p.rating is not None for p in home.starters)
