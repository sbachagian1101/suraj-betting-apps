"""Feature engineering for the three racing disciplines.

Features are computed only from information available before the target race.
Current market odds are deliberately excluded; they are used later for EV.
"""
from __future__ import annotations

from collections import defaultdict
from datetime import date
import math
import re
from typing import Any

import numpy as np
import pandas as pd

from .parser import ParsedRace


COMPONENT_COLUMNS = [
    "recent_finish_score",
    "recent_win_signal",
    "recent_place_signal",
    "margin_signal",
    "speed_signal",
    "rating_signal",
    "suitability_signal",
    "connection_signal",
    "fitness_signal",
    "setup_signal",
    "reliability_signal",
    "trial_signal",
]

MODEL_FEATURE_COLUMNS = COMPONENT_COLUMNS + [
    "history_count",
    "career_starts",
    "career_win_rate",
    "career_place_rate",
    "days_since_run",
    "data_quality",
]


def _safe_float(value: Any, default: float = np.nan) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _weighted_mean(values: list[float], weights: list[float]) -> float:
    if not values:
        return np.nan
    arr = np.asarray(values, dtype=float)
    w = np.asarray(weights, dtype=float)
    mask = np.isfinite(arr) & np.isfinite(w) & (w > 0)
    if not mask.any():
        return np.nan
    return float(np.average(arr[mask], weights=w[mask]))


def _recency_weights(n: int, decay: float = 0.78) -> list[float]:
    return [decay**i for i in range(n)]


def _rate(wins: Any, starts: Any, prior_rate: float, prior_n: float = 8.0) -> float:
    w = _safe_float(wins, 0.0)
    s = _safe_float(starts, 0.0)
    return float((w + prior_rate * prior_n) / (s + prior_n))


def _place_rate(places: Any, starts: Any, prior_rate: float, prior_n: float = 8.0) -> float:
    p = _safe_float(places, 0.0)
    s = _safe_float(starts, 0.0)
    return float((p + prior_rate * prior_n) / (s + prior_n))


def _going_key(going: str | None) -> str | None:
    g = str(going or "").upper()
    if "HEAVY" in g or re.fullmatch(r"H\s*\d*", g):
        return "heavy"
    if "SOFT" in g or g == "S":
        return "soft"
    if "FIRM" in g or g in {"F", "GF"}:
        return "firm"
    if "GOOD" in g or g in {"G", "GS", "Y", "YIELDING"}:
        return "good"
    if "STANDARD" in g or "ALL WEATHER" in g or g == "N":
        return "aw"
    return None


def _fitness_score(days: float, discipline: str) -> float:
    if not np.isfinite(days):
        return -0.1
    if discipline == "greyhound":
        if 4 <= days <= 14:
            return 1.0
        if 2 <= days < 4 or 15 <= days <= 28:
            return 0.4
        if days < 2:
            return -0.6
        return max(-1.0, 0.3 - (days - 28) / 70)
    if discipline == "harness":
        if 5 <= days <= 28:
            return 1.0
        if 2 <= days < 5 or 29 <= days <= 60:
            return 0.4
        if days < 2:
            return -0.6
        return max(-1.0, 0.3 - (days - 60) / 120)
    # Thoroughbreds typically race less often.
    if 12 <= days <= 42:
        return 1.0
    if 7 <= days < 12 or 43 <= days <= 80:
        return 0.5
    if days < 7:
        return -0.3
    return max(-1.0, 0.2 - (days - 80) / 180)


def _finish_percentile(finish: Any, field_size: Any) -> float | None:
    f = _safe_float(finish)
    n = _safe_float(field_size)
    if not np.isfinite(f) or not np.isfinite(n) or n <= 1:
        return None
    return float(1.0 - (f - 1.0) / (n - 1.0))


def _sectional_delta(sectionals: str | None) -> float | None:
    """Mean horse-minus-race sectional delta from R:/H: pairs; lower is better."""
    if not sectionals:
        return None
    pairs = re.findall(r"R:\s*([0-9.]+).*?H:\s*([0-9.]+)", sectionals, re.I)
    if not pairs:
        return None
    diffs = [float(h) - float(r) for r, h in pairs]
    return float(np.mean(diffs))


def _data_quality(runner: dict[str, Any], histories: list[dict[str, Any]], discipline: str) -> float:
    history_points = min(len(histories), 5) / 5.0
    complete = 0
    possible = 0
    checks = [
        runner.get("trainer_win_rate"),
        runner.get("jockey_win_rate") if discipline == "thoroughbred" else runner.get("driver_win_rate"),
        runner.get("car_starts"),
        runner.get("dist_starts"),
        runner.get("dls"),
    ]
    if discipline == "greyhound":
        checks += [runner.get("current_box_starts"), runner.get("track_distance_best_s")]
    for value in checks:
        possible += 1
        if value is not None and value != "":
            complete += 1
    field_completeness = complete / max(possible, 1)
    q = 0.20 + 0.55 * history_points + 0.25 * field_completeness
    if runner.get("making_debut"):
        q = max(q, 0.30 + 0.08 * min(_safe_float(runner.get("trial_count"), 0.0), 3.0))
    return float(np.clip(q, 0.20, 1.0))


def _runner_features(card: ParsedRace, runner: dict[str, Any], histories: list[dict[str, Any]], trials: list[dict[str, Any]]) -> dict[str, Any]:
    discipline = card.discipline
    h = sorted(histories, key=lambda x: x.get("date") or "", reverse=True)[:8]
    weights = _recency_weights(len(h))

    finish_scores: list[float] = []
    margin_scores: list[float] = []
    win_flags: list[float] = []
    place_flags: list[float] = []
    speed_values: list[float] = []
    ratings: list[float] = []
    clean_flags: list[float] = []

    target_dist = _safe_float(card.race.get("distance_m"))
    target_track = str(card.race.get("track") or "").upper()
    target_going = _going_key(card.race.get("going"))
    similar_distance_results: list[float] = []
    same_track_results: list[float] = []
    same_going_results: list[float] = []

    for run in h:
        pct = _finish_percentile(run.get("finish"), run.get("field_size"))
        if pct is not None:
            finish_scores.append(pct)
            win_flags.append(1.0 if run.get("finish") == 1 else 0.0)
            place_cut = min(3, max(2, int(_safe_float(run.get("field_size"), 8) // 4)))
            place_flags.append(1.0 if run.get("finish") and run["finish"] <= place_cut else 0.0)
        margin = _safe_float(run.get("margin"))
        if np.isfinite(margin):
            # A logarithm prevents one disaster from swamping all recent form.
            margin_scores.append(-math.log1p(max(margin, 0.0)))
        clean_flags.append(0.0 if run.get("disqualified") else 1.0)

        dist = _safe_float(run.get("distance_m"))
        if np.isfinite(dist) and np.isfinite(target_dist) and target_dist > 0:
            similarity = abs(dist - target_dist) / target_dist
            if similarity <= 0.12 and pct is not None:
                similar_distance_results.append(pct)
        if str(run.get("track") or "").upper() == target_track and pct is not None:
            same_track_results.append(pct)
        if _going_key(run.get("going")) == target_going and pct is not None:
            same_going_results.append(pct)

        if discipline == "greyhound":
            delta = _safe_float(run.get("time_behind_s"))
            if np.isfinite(delta):
                speed_values.append(-delta)
            elif np.isfinite(dist) and _safe_float(run.get("runner_time_s")) > 0:
                speed_values.append(dist / _safe_float(run.get("runner_time_s")))
        elif discipline == "harness":
            adj = _safe_float(run.get("mile_rate_adj"))
            if np.isfinite(adj):
                speed_values.append(-adj)  # more negative adjustment means faster
            else:
                mr = _safe_float(run.get("mile_rate_s"))
                if np.isfinite(mr):
                    speed_values.append(-mr)
        else:
            rating = _safe_float(run.get("rating"))
            if np.isfinite(rating):
                ratings.append(rating)
            delta = _sectional_delta(run.get("sectionals"))
            if delta is not None:
                speed_values.append(-delta)
            elif np.isfinite(dist) and _safe_float(run.get("race_time_s")) > 0:
                # Approximate speed; within-race z-scoring limits scale problems.
                speed_values.append(dist / _safe_float(run.get("race_time_s")))

    wf = weights[: len(finish_scores)]
    wm = weights[: len(margin_scores)]
    ws = weights[: len(speed_values)]
    wr = weights[: len(ratings)]
    recent_finish = _weighted_mean(finish_scores, wf)
    recent_win = _weighted_mean(win_flags, weights[: len(win_flags)])
    recent_place = _weighted_mean(place_flags, weights[: len(place_flags)])
    margin_signal = _weighted_mean(margin_scores, wm)
    speed_signal = _weighted_mean(speed_values, ws)
    rating_signal = _weighted_mean(ratings, wr)

    car_starts = _safe_float(runner.get("car_starts"), 0.0)
    car_wins = _safe_float(runner.get("car_wins"), 0.0)
    car_places = _safe_float(runner.get("car_places"), 0.0)
    base_win = 1 / max(_safe_float(card.race.get("field_size"), 10.0), 2.0)
    base_place = min(0.45, 3 * base_win)
    career_win = _rate(car_wins, car_starts, base_win, prior_n=10)
    career_place = _place_rate(car_places, car_starts, base_place, prior_n=10)

    dist_rate = _rate(runner.get("dist_wins"), runner.get("dist_starts"), base_win)
    course_dist_rate = _rate(runner.get("crs_and_dist_wins"), runner.get("crs_and_dist_starts"), base_win)
    track_rate = _rate(runner.get("crs_wins"), runner.get("crs_starts"), base_win)
    going_rate = base_win
    if target_going:
        going_rate = _rate(runner.get(f"{target_going}_wins"), runner.get(f"{target_going}_starts"), base_win)

    empirical_candidates = [
        np.mean(similar_distance_results) if similar_distance_results else np.nan,
        np.mean(same_track_results) if same_track_results else np.nan,
        np.mean(same_going_results) if same_going_results else np.nan,
    ]
    empirical_finite = [float(x) for x in empirical_candidates if np.isfinite(x)]
    empirical_suitability = float(np.mean(empirical_finite)) if empirical_finite else np.nan
    record_suitability = np.mean([dist_rate, course_dist_rate, track_rate, going_rate])
    suitability = record_suitability if not np.isfinite(empirical_suitability) else 0.55 * record_suitability + 0.45 * empirical_suitability

    connection_rates = [_safe_float(runner.get("trainer_win_rate"))]
    if discipline == "thoroughbred":
        connection_rates.append(_safe_float(runner.get("jockey_win_rate")))
    elif discipline == "harness":
        connection_rates.append(_safe_float(runner.get("driver_win_rate")))
    connection = float(np.nanmean(connection_rates)) if np.isfinite(connection_rates).any() else np.nan

    dls = _safe_float(runner.get("dls"))
    if not np.isfinite(dls) and h:
        race_date = card.race.get("date")
        last_date = h[0].get("date")
        if race_date and last_date:
            try:
                dls = (date.fromisoformat(race_date) - date.fromisoformat(last_date)).days
            except ValueError:
                pass
    fitness = _fitness_score(dls, discipline)

    setup = 0.0
    if discipline == "greyhound":
        starts = _safe_float(runner.get("current_box_starts"), 0.0)
        wins = _safe_float(runner.get("current_box_wins"), 0.0)
        setup = _rate(wins, starts, base_win, prior_n=8)
        td_best = _safe_float(runner.get("track_distance_best_s"))
        if np.isfinite(td_best):
            setup += -0.02 * td_best
    elif discipline == "harness":
        setup = -_safe_float(runner.get("handicap_metres"), 0.0) / 25.0
        hcp = str(runner.get("hcp") or "")
        if hcp.lower().startswith("fr"):
            setup += 0.35
        elif hcp.lower().startswith("sr"):
            setup -= 0.20
    else:
        # Current carried weight is only a small context term; in handicaps it also
        # reflects ability, so the direction should not dominate the form model.
        setup = -0.03 * _safe_float(runner.get("weight_kg"), 0.0)

    clean_rate = float(np.mean(clean_flags)) if clean_flags else 0.75
    reliability = clean_rate
    if discipline == "harness":
        dq_rate = 1.0 - clean_rate
        reliability -= 1.5 * dq_rate

    trial_signal = np.nan
    if trials:
        places = [_safe_float(t.get("trial_place")) for t in trials]
        places = [p for p in places if np.isfinite(p)]
        if places:
            trial_signal = 1.0 / min(places)

    quality = _data_quality(runner, h, discipline)
    row = {
        "tab": runner.get("tab"),
        "runner": runner.get("runner"),
        "discipline": discipline,
        "market_odds": runner.get("market_odds"),
        "scratched": bool(runner.get("scratched")),
        "history_count": len(h),
        "career_starts": car_starts,
        "career_win_rate": career_win,
        "career_place_rate": career_place,
        "days_since_run": dls,
        "recent_finish_score": recent_finish,
        "recent_win_signal": recent_win,
        "recent_place_signal": recent_place,
        "margin_signal": margin_signal,
        "speed_signal": speed_signal,
        "rating_signal": rating_signal,
        "suitability_signal": suitability,
        "connection_signal": connection,
        "fitness_signal": fitness,
        "setup_signal": setup,
        "reliability_signal": reliability,
        "trial_signal": trial_signal,
        "data_quality": quality,
    }
    return row


def build_feature_frame(card: ParsedRace, include_scratched: bool = False) -> pd.DataFrame:
    by_runner: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for run in card.histories:
        by_runner[str(run.get("runner"))].append(run)
    trials_by_runner: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for trial in card.trials:
        trials_by_runner[str(trial.get("runner"))].append(trial)

    rows = []
    for runner in card.runners:
        if runner.get("scratched") and not include_scratched:
            continue
        rows.append(_runner_features(card, runner, by_runner[runner["runner"]], trials_by_runner[runner["runner"]]))
    return pd.DataFrame(rows)
