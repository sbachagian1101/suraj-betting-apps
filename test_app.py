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


def test_mauritius_time_is_utc_plus_four():
    kick = datetime(2026, 9, 7, 17, 0, tzinfo=timezone.utc)
    assert SW.to_mu(kick).strftime("%H:%M") == "21:00"
    assert SW.fmt_mu(kick) == "21:00"
    assert SW.fmt_mu(kick, with_date=True) == "Mon 07 Sep 21:00"
    # a late UTC kick-off rolls into the next Mauritian day
    late = datetime(2026, 9, 7, 21, 30, tzinfo=timezone.utc)
    assert SW.fmt_mu(late, with_date=True) == "Tue 08 Sep 01:30"


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


def test_select_leagues_strict_needs_exact_name():
    ms = SW._parse_matches(_daily_feed())
    found, missing = SW.select_leagues(
        ["ENGLAND: Premier League", "England premier league", "FRANCE: Ligue 2", "premier league"],
        ms, strict=True)
    assert found == {"ENGLAND: Premier League": "ENGLAND: Premier League",
                     "England premier league": "ENGLAND: Premier League"}   # case/punctuation only
    assert missing == ["FRANCE: Ligue 2", "premier league"]                 # no fuzz, no partials


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


def _odds_with_gap(pred, side, gap):
    """Bookmaker odds whose margin-free implied probability on ``side`` is the
    model's probability minus ``gap``, the rest split evenly."""
    p = {"home": pred.p_home, "draw": pred.p_draw, "away": pred.p_away}
    imp = {side: p[side] - gap}
    others = [k for k in p if k != side]
    rest = sum(p[k] for k in others)
    for k in others:                      # remaining mass in the model's proportions,
        imp[k] = (1 - imp[side]) * p[k] / rest   # so their gaps are negative and ``side`` is biggest
    return {k: 1 / v for k, v in imp.items()}


def test_bet_signal_needs_majority_and_band():
    strong = M.profile("A", _records("a", [2.6] * 3, [0.6] * 3, [{"p": 7.2}] * 3))
    weak = M.profile("B", _records("b", [0.7] * 3, [2.0] * 3, [{"p": 6.3}] * 3))
    pred = M.predict(strong, weak)
    assert pred.p_home > M.BET_MIN_PROB                 # home favourite well above 50%
    assert M.bet_signal(pred, None) is None
    sig = M.bet_signal(pred, _odds_with_gap(pred, "home", 0.10))
    assert sig["side"] == "home" and sig["bet"] and sig["label"] == "BET HOME TEAM"
    assert sig["reasons"] == []
    # gap outside the band -> NO BET even with the majority
    sig = M.bet_signal(pred, _odds_with_gap(pred, "home", 0.30))
    assert not sig["bet"] and any("above" in r for r in sig["reasons"])
    sig = M.bet_signal(pred, _odds_with_gap(pred, "home", 0.02))
    assert not sig["bet"] and any("below" in r for r in sig["reasons"])
    # the away side has a gap in band but the model gives it well under 50% -> NO BET
    sig = M.bet_signal(pred, _odds_with_gap(pred, "away", 0.10))
    assert sig["side"] == "away" and not sig["bet"] and sig["label"] == "NO BET"
    assert any("not above 50%" in r for r in sig["reasons"])
    # an even match: nobody above 50%, so never a bet whatever the gap
    even = M.predict(M.profile("A", _records("a", [1.5] * 3, [1.2] * 3, [{"p": 6.8}] * 3)),
                     M.profile("B", _records("b", [1.5] * 3, [1.2] * 3, [{"p": 6.8}] * 3)))
    assert max(even.p_home, even.p_draw, even.p_away) < M.BET_MIN_PROB
    assert not M.bet_signal(even, _odds_with_gap(even, "home", 0.10))["bet"]


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


def test_on_day_trims_the_feed_padding():
    import pipeline as P
    from datetime import date
    ms = SW._parse_matches(_daily_feed())
    # feed kick-offs: 1788699600 = Sun 6 Sep 2026 13:00 UTC, 1788708600 = 15:30, 1788718600 = 18:16
    assert ms[0].kickoff == datetime(2026, 9, 6, 13, 0, tzinfo=timezone.utc)
    assert len(P.on_day(ms, date(2026, 9, 6), 4)) == 4
    assert P.on_day(ms, date(2026, 9, 7), 4) == []
    # at UTC-14 the 13:00 UTC match is still 5 Sep locally, the later ones are 6 Sep
    assert [m.id for m in P.on_day(ms, date(2026, 9, 5), -14)] == ["QsyJgS7m"]
    assert len(P.on_day(ms, date(2026, 9, 6), -14)) == 3


def test_refresh_tiers():
    import board as B
    now = datetime(2026, 9, 7, 12, 0, tzinfo=timezone.utc)
    from datetime import timedelta as td
    assert B.refresh_minutes(now + td(minutes=20), now) == 5
    assert B.refresh_minutes(now + td(minutes=30), now) == 5
    assert B.refresh_minutes(now + td(minutes=45), now) == 15
    assert B.refresh_minutes(now + td(minutes=60), now) == 15
    assert B.refresh_minutes(now + td(hours=2), now) == 30
    assert B.refresh_minutes(now + td(hours=3), now) == 30
    assert B.refresh_minutes(now + td(hours=5), now) == 60
    assert B.refresh_minutes(now - td(minutes=1), now) is None


def test_board_computes_due_rows_with_fake_compute(monkeypatch):
    import time as _t
    import board as B
    import pipeline as P
    calls = []

    def fake_compute(m, n_rates, n_lineup):
        calls.append(m.id)
        a = M.profile(m.home_name, _records("h", [1.5] * 3, [1.0] * 3, [{"p": 7.0}] * 3))
        b = M.profile(m.away_name, _records("a", [1.2] * 3, [1.3] * 3, [{"p": 6.8}] * 3))
        return P.Analysis(m, P.TeamData("h", m.home_id, m.home_name, a.records),
                          P.TeamData("a", m.away_id, m.away_name, b.records),
                          a, b, None, None, M.predict(a, b), None, n_rates, n_lineup)

    monkeypatch.setattr(B, "daily_cached", lambda *a, **k: [])      # no network for status refresh
    monkeypatch.setattr(B, "LOOP_SLEEP", 0.05)
    ms = SW._parse_matches(_daily_feed())
    scheduled = [m for m in ms if m.stage == "1"]
    bd = B.Board(compute=fake_compute)
    bd.configure(scheduled, 8, 4, 0, 4)
    deadline = _t.time() + 10
    while _t.time() < deadline and bd.status()["computed"] < len(scheduled):
        _t.sleep(0.1)
    bd.stop()
    S = bd.status()
    assert S["computed"] == len(scheduled) == 2 and S["errors"] == 0
    assert sorted(calls) == sorted(m.id for m in scheduled)
    rows = bd.snapshot()
    assert all(r.analysis is not None and r.next_due is None for r in rows)   # kick-offs are in the past -> frozen
    # reconfiguring with the same windows keeps the computed predictions
    bd.configure(scheduled, 8, 4, 0, 4)
    bd.stop()
    assert bd.status()["computed"] == 2
    # a finished match is predicted once (from pre-match history) and then left alone
    fin = [m for m in ms if m.stage == "3"]
    bd2 = B.Board(compute=fake_compute)
    bd2.configure(fin, 8, 4, 0, 4)
    deadline = _t.time() + 5
    while _t.time() < deadline and bd2.status()["computed"] < len(fin):
        _t.sleep(0.1)
    _t.sleep(0.3)
    bd2.stop()
    assert bd2.status()["computed"] == len(fin) == 1
    assert all(r.next_due is None for r in bd2.snapshot())


def test_time_window_uses_mauritius_hours():
    import board_ui as BU
    m = _match(0, "h", "a", None, None)
    m.kickoff = datetime(2026, 9, 7, 17, 0, tzinfo=timezone.utc)     # 21:00 Mauritius
    assert BU.in_window(m, 10, 22) and BU.in_window(m, 21, 24) and BU.in_window(m, 0, 21)
    assert not BU.in_window(m, 10, 15) and not BU.in_window(m, 22, 24)
    m.kickoff = datetime(2026, 9, 7, 6, 30, tzinfo=timezone.utc)     # 10:30 Mauritius
    assert BU.in_window(m, 10, 15) and not BU.in_window(m, 11, 15)
    assert BU.HOURS[0] == "00:00" and BU.HOURS[-1] == "24:00" and len(BU.HOURS) == 25


def test_board_keeps_started_matches_and_predicts_finished_ones_once(monkeypatch):
    import time as _t
    import board as B
    import board_ui as BU
    import pipeline as P
    from datetime import timedelta as td
    now = datetime.now(timezone.utc)
    ms = SW._parse_matches(_daily_feed())
    for m in ms:                                   # make kick-offs relative to now
        m.kickoff = now + td(hours=1)
    fin, sched = ms[0], ms[1]                      # Everton (finished) and Arsenal (scheduled), same league
    fin.kickoff = now - td(hours=2)
    live = ms[3]; live.stage = "2"; live.kickoff = now - td(minutes=30)
    # selection: live and finished always, scheduled only inside the horizon
    picked = BU.select_board_matches(ms, {"ENGLAND: Premier League", "FRANCE: Ligue 1"}, 0.5, now)
    assert [m.id for m in picked] == [fin.id, live.id]
    picked = BU.select_board_matches(ms, {"ENGLAND: Premier League"}, 2, now)
    assert [m.id for m in picked] == [fin.id, sched.id]

    calls = []

    def fake_compute(m, n_rates, n_lineup):
        calls.append(m.id)
        a = M.profile(m.home_name, _records("h", [1.5] * 3, [1.0] * 3, [{"p": 7.0}] * 3))
        b = M.profile(m.away_name, _records("a", [1.2] * 3, [1.3] * 3, [{"p": 6.8}] * 3))
        return P.Analysis(m, P.TeamData("h", m.home_id, m.home_name, a.records),
                          P.TeamData("a", m.away_id, m.away_name, b.records),
                          a, b, None, None, M.predict(a, b), None, n_rates, n_lineup)

    monkeypatch.setattr(B, "daily_cached", lambda *a, **k: [])
    monkeypatch.setattr(B, "LOOP_SLEEP", 0.05)
    bd = B.Board(compute=fake_compute)
    bd.configure([fin, sched], 8, 4, 0, 4)
    deadline = _t.time() + 10
    while _t.time() < deadline and bd.status()["computed"] < 2:
        _t.sleep(0.1)
    assert bd.status()["computed"] == 2 and calls.count(fin.id) == 1     # finished match predicted once
    # the scheduled match kicks off: an update with only the *new* scheduled list keeps it
    sched.stage = "2"
    bd.configure([fin], 8, 4, 0, 4)              # same league, sched not in the list any more
    assert sched.id in {r.match.id for r in bd.snapshot()}
    bd.stop()
    _t.sleep(0.2)
    assert calls.count(fin.id) == 1               # never refreshed after computing


def test_charts_build_from_an_analysis():
    import charts as C
    import pipeline as P
    m = _match(0, "h", "a", 1, 0)
    a = M.profile("Home", _records("h", [1.5] * 3, [1.0] * 3, [{"p": 7.0}] * 3))
    b = M.profile("Away", _records("a", [1.2] * 3, [1.3] * 3, [{"p": 6.8}] * 3))
    pred = M.predict(a, b)
    odds = {"home": 2.1, "draw": 3.4, "away": 3.6, "home_open": 2.1, "draw_open": 3.4, "away_open": 3.6}
    A = P.Analysis(m, P.TeamData("h", "h", "Home", a.records), P.TeamData("a", "a", "Away", b.records),
                   a, b, None, None, pred, odds, 3, 3)
    for fig in (C.model_vs_book(A), C.outcome_pie(A), C.score_heatmap(A), C.xg_history(A)):
        assert fig.to_dict()["data"]
    assert C.rating_bars(A) is None


def test_alerts_imminent_started_and_beep():
    import alerts as AL
    from datetime import timedelta as td
    now = datetime(2026, 9, 7, 12, 0, tzinfo=timezone.utc)
    soon = _match(1, "h", "a", None, None); soon.stage = "1"; soon.kickoff = now + td(minutes=4)
    later = _match(2, "h", "a", None, None); later.stage = "1"; later.kickoff = now + td(minutes=6)
    gone = _match(3, "h", "a", 0, 0); gone.stage = "1"; gone.kickoff = now - td(minutes=1)
    live = _match(4, "h", "a", 1, 0); live.stage = "2"; live.kickoff = now - td(minutes=20)
    assert AL.imminent(soon, now) and not AL.imminent(later, now)
    assert not AL.imminent(gone, now) and not AL.imminent(live, now)
    assert AL.started(live, now) and AL.started(gone, now) and not AL.started(soon, now)
    # announce once: second call with the id recorded returns nothing
    seen: set[str] = set()
    first = AL.new_alerts([soon, later, live], now, seen)
    assert [m.id for m in first] == [soon.id]
    seen.update(m.id for m in first)
    assert AL.new_alerts([soon, later, live], now, seen) == []
    wav = AL.beep_wav()
    assert wav[:4] == b"RIFF" and wav[8:12] == b"WAVE" and len(wav) > 10000
    assert AL.odds_colour("1.85").startswith("background-color: " + AL.ODDS_GREEN)
    assert AL.odds_colour("2.50").startswith("background-color: " + AL.ODDS_YELLOW)
    assert AL.odds_colour("3.60").startswith("background-color: " + AL.ODDS_BLUE)
    assert AL.odds_colour("") == "" and AL.odds_colour(None) == ""


def test_board_frame_and_styler_layer_colours():
    import alerts as AL
    import board as B
    import board_ui as BU
    import pipeline as P
    from datetime import timedelta as td
    now = datetime(2026, 9, 7, 12, 0, tzinfo=timezone.utc)
    m1 = _match(1, "h", "a", None, None); m1.stage = "1"; m1.kickoff = now + td(minutes=3)
    m2 = _match(2, "h2", "a2", 1, 0); m2.stage = "2"; m2.kickoff = now - td(minutes=30)
    a = M.profile("H", _records("h", [1.5] * 3, [1.0] * 3, [{"p": 7.0}] * 3))
    b = M.profile("A", _records("a", [1.2] * 3, [1.3] * 3, [{"p": 6.8}] * 3))
    odds = {"home": 1.85, "draw": 3.4, "away": 4.2, "home_open": 1.9, "draw_open": 3.4, "away_open": 4.0}
    A = P.Analysis(m1, P.TeamData("h", "h", "H", a.records), P.TeamData("a", "a", "A", b.records),
                   a, b, None, None, M.predict(a, b), odds, 3, 3)
    rows = [B.Row(match=m1, analysis=A, computed_at=now), B.Row(match=m2)]
    df = BU.board_frame(rows, now)
    assert list(df["Odds H"]) == ["1.85", ""] and df.iloc[0]["Home %"] != ""
    assert BU.starts_in(m1, now).endswith("s") and BU.starts_in(m2, now) == "LIVE"
    sty = BU.style_board(df, blink_ids={m1.id}, blink_on=True, started_ids={m2.id})
    html = sty.to_html()
    assert AL.PURPLE in html and AL.ROSE in html          # row colours present
    assert AL.ODDS_GREEN in html and AL.ODDS_BLUE in html  # odds cell colours present
    # on the blinking row, the odds cell's own colour is declared after the row colour, so it wins
    computed = sty._compute()
    odds_col = list(df.columns).index("Odds H")
    bg = [v for k, v in computed.ctx[(0, odds_col)] if k.strip() == "background-color"]
    assert bg and bg[-1].strip() == AL.ODDS_GREEN
    # blink off: the imminent row has no row colour, the started row keeps rose
    off = BU.style_board(df, blink_ids={m1.id}, blink_on=False, started_ids={m2.id}).to_html()
    assert AL.PURPLE not in off and AL.ROSE in off


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
