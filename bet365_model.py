"""Bet365 multi-meeting horse-race prediction and continuous learning.

The model keeps four visible specialist opinions:
1. Bet365 Analyst Source
2. Independent Recent Form & Class
3. Distance / Going / Fitness
4. Pace / Barrier / Weight

A regularised pairwise learning layer is trained only on actual results entered
after the pre-race prediction.  Current and historical prices are excluded.
"""
from __future__ import annotations

from datetime import datetime
from hashlib import sha256
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
import math
import random
import statistics

from results_parser import RaceResult

MODEL_VERSION = "bet365-horse-v1.0"
RAW_FEATURE_NAMES = [
    "suggested_pick", "source_mention_strength", "overview_sentiment", "runner_sentiment",
    "form_score", "form_trend", "current_prep_runs", "recent_form", "latest_form",
    "previous_form", "latest_margin_quality", "previous_margin_quality", "last_win",
    "last_top3", "class_advantage", "class_text_adjustment", "distance_suitability",
    "latest_distance_suitability", "going_suitability", "going_positive", "going_negative",
    "fitness", "weight_advantage", "barrier_advantage", "expected_leader", "on_pace_history",
    "closer_history", "factual_comment_score", "consistency", "history_coverage",
    "first_start", "age_profile", "jockey_present", "trainer_present", "jump_relevance",
]
FEATURE_NAMES = [
    "source_logp", "form_logp", "suitability_logp", "pace_logp",
    "source_rank_strength", "form_rank_strength", "suitability_rank_strength", "pace_rank_strength",
    "baseline_score", *RAW_FEATURE_NAMES,
]
FEATURE_LABELS = {
    "suggested_pick": "Bet365 suggested-play inclusion",
    "source_mention_strength": "priority in Bet365 race overview",
    "overview_sentiment": "Bet365 overview assessment",
    "runner_sentiment": "Bet365 runner assessment",
    "form_score": "recent finishing sequence",
    "form_trend": "recent finishing trajectory",
    "current_prep_runs": "current preparation depth",
    "recent_form": "recency-weighted factual form",
    "latest_form": "latest-run competitiveness",
    "previous_form": "previous-run competitiveness",
    "latest_margin_quality": "latest beaten/winning margin",
    "previous_margin_quality": "previous beaten/winning margin",
    "last_win": "last-start win",
    "last_top3": "last-start top-three finish",
    "class_advantage": "previous-class strength versus today",
    "class_text_adjustment": "explicit class rise/drop context",
    "distance_suitability": "historical distance fit",
    "latest_distance_suitability": "latest-run distance relevance",
    "going_suitability": "going compatibility",
    "going_positive": "explicit positive condition signal",
    "going_negative": "explicit negative condition signal",
    "fitness": "days-since-run and preparation fitness",
    "weight_advantage": "relative carried weight",
    "barrier_advantage": "draw suitability",
    "expected_leader": "expected race-map advantage",
    "on_pace_history": "historical on-pace ability",
    "closer_history": "historical closing ability",
    "factual_comment_score": "factual positive/negative run comments",
    "consistency": "recent performance consistency",
    "history_coverage": "amount of usable form history",
    "first_start": "first-start uncertainty",
    "age_profile": "age and development profile",
    "jockey_present": "known jockey booking",
    "trainer_present": "known trainer",
    "jump_relevance": "relevant obstacle-race evidence",
    "source_logp": "Bet365 Analyst Source model",
    "form_logp": "Independent Recent Form model",
    "suitability_logp": "Distance/Going/Fitness model",
    "pace_logp": "Pace/Barrier/Weight model",
    "baseline_score": "untrained four-model consensus",
}


def default_model_state() -> Dict[str, Any]:
    now = datetime.now().isoformat(timespec="seconds")
    return {
        "version": MODEL_VERSION,
        "created_at": now,
        "updated_at": now,
        "feature_names": list(FEATURE_NAMES),
        "weights": [0.0] * len(FEATURE_NAMES),
        "training_races": 0,
        "training_pairs": 0,
        "learned_influence": 0.0,
        "training_metrics": {},
        "notes": (
            "Price-free Bet365 form and analyst model. The source overview and suggested play are retained as one visible specialist opinion; "
            "current and historical odds are excluded from all features."
        ),
    }


def _finite(value: Any) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _clip(value: float, lo: float = -4.0, hi: float = 4.0) -> float:
    return max(lo, min(hi, float(value)))


def _mean_std(values: Iterable[Optional[float]]) -> Tuple[float, float]:
    clean = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    if not clean:
        return 0.0, 1.0
    mean = statistics.fmean(clean)
    sd = statistics.pstdev(clean) if len(clean) > 1 else 1.0
    return mean, sd if sd > 1e-9 else 1.0


def _z(value: Optional[float], stats: Tuple[float, float], missing: float = 0.0) -> float:
    if value is None:
        return missing
    mean, sd = stats
    return _clip((float(value) - mean) / sd)


def _softmax(scores: Dict[int, float], temperature: float = 1.0) -> Dict[int, float]:
    if not scores:
        return {}
    temp = max(0.15, float(temperature))
    maximum = max(scores.values())
    exps = {number: math.exp(max(-35.0, min(35.0, (score - maximum) / temp))) for number, score in scores.items()}
    total = sum(exps.values()) or 1.0
    return {number: value / total for number, value in exps.items()}


def _rank_map(scores: Dict[int, float]) -> Dict[int, int]:
    return {number: rank for rank, (number, _) in enumerate(sorted(scores.items(), key=lambda item: (-item[1], item[0])), start=1)}


def _rank_strength(rank: int, field_size: int) -> float:
    if field_size <= 1:
        return 1.0
    return 1.0 - (rank - 1.0) / (field_size - 1.0)


def _finish_percentile(run: Optional[Dict[str, Any]]) -> Optional[float]:
    if not run:
        return None
    finish = _finite(run.get("finish"))
    field = _finite(run.get("field_size"))
    if finish is None or not run.get("completed", True):
        return None
    if field is not None and field > 1:
        return max(0.0, min(1.0, 1.0 - (finish - 1.0) / (field - 1.0)))
    return max(0.0, 1.0 - (finish - 1.0) / 9.0)


def _margin_quality(run: Optional[Dict[str, Any]]) -> Optional[float]:
    if not run:
        return None
    if run.get("won"):
        margin = abs(_finite(run.get("margin")) or 0.0)
        return min(1.35, 0.95 + 0.07 * margin)
    margin = _finite(run.get("margin"))
    if margin is None:
        return 0.45
    return max(0.0, math.exp(-max(0.0, margin) / 7.0))


def _distance_similarity(run_distance: Optional[float], target_distance: Optional[float]) -> Optional[float]:
    if run_distance is None or target_distance is None or target_distance <= 0:
        return None
    scale = max(180.0, 0.22 * target_distance)
    return math.exp(-abs(float(run_distance) - float(target_distance)) / scale)


def _going_similarity(run_going: str, target_going: str) -> Optional[float]:
    if not run_going or run_going == "Unknown" or not target_going or target_going == "Unknown":
        return None
    if run_going == target_going:
        return 1.0
    wet = {"Soft", "Heavy"}
    if run_going in wet and target_going in wet:
        return 0.72
    if "Synthetic" in (run_going, target_going):
        return 0.28
    return 0.45


def _fitness(days_since: Optional[float], stage: str, first_start: bool) -> float:
    if first_start:
        return 0.50
    if days_since is None:
        base = 0.48
    else:
        days = max(0.0, float(days_since))
        # Broad optimum because flat and jumps programs differ substantially.
        centre = 20.0
        spread = 32.0
        base = math.exp(-((days - centre) / spread) ** 2)
        if days > 100:
            base = max(base, 0.30 * math.exp(-(days - 100.0) / 220.0))
    stage_low = str(stage or "").lower()
    if "third-up" in stage_low or "fourth-up" in stage_low:
        base += 0.12
    elif "second-up" in stage_low:
        base += 0.06
    elif "resum" in stage_low or "first-up" in stage_low or "back from" in stage_low:
        base -= 0.03
    return max(0.05, min(1.10, base))


def _age_profile(age: Optional[float], discipline: str) -> Optional[float]:
    if age is None:
        return None
    centre = 7.0 if discipline in {"Hurdle", "Steeplechase"} else 4.5
    spread = 4.0 if discipline in {"Hurdle", "Steeplechase"} else 3.5
    return math.exp(-((float(age) - centre) / spread) ** 2)


def _weighted_mean(items: Sequence[Tuple[float, float]]) -> Optional[float]:
    total = sum(max(0.0, weight) for weight, _ in items)
    if total <= 1e-12:
        return None
    return sum(max(0.0, weight) * value for weight, value in items) / total


def _recent_form_features(runner: Dict[str, Any], race: Dict[str, Any]) -> Dict[str, float]:
    runs = list(runner.get("historical_runs") or [])
    recent_items: List[Tuple[float, float]] = []
    distance_items: List[Tuple[float, float]] = []
    going_items: List[Tuple[float, float]] = []
    jump_items: List[Tuple[float, float]] = []
    values: List[float] = []
    for idx, run in enumerate(runs):
        percentile = _finish_percentile(run)
        margin = _margin_quality(run)
        if percentile is None and margin is None:
            continue
        quality = 0.65 * (percentile if percentile is not None else 0.45) + 0.35 * (margin if margin is not None else 0.45)
        days = _finite(run.get("days_ago"))
        recency = math.exp(-(days if days is not None else 45.0) / 85.0)
        class_delta = (_finite(run.get("class_score")) or race.get("class_score", 56.0)) - float(race.get("class_score") or 56.0)
        adjusted = quality + 0.010 * class_delta
        recent_items.append((recency, adjusted))
        values.append(adjusted)
        dsim = _distance_similarity(_finite(run.get("distance_m")), _finite(race.get("distance_m")))
        if dsim is not None:
            distance_items.append((recency, dsim * quality))
        gsim = _going_similarity(str(run.get("going") or ""), str(race.get("going") or ""))
        if gsim is not None:
            going_items.append((recency, gsim * quality))
        raw = str(run.get("raw_text") or "").lower()
        relevant_jump = 1.0 if (race.get("discipline") == "Hurdle" and ("hdle" in raw or "hurdle" in raw)) or (race.get("discipline") == "Steeplechase" and ("chase" in raw or "steeple" in raw)) else 0.0
        if race.get("discipline") != "Flat":
            jump_items.append((recency, relevant_jump * quality))
    latest = runs[0] if runs else None
    previous = runs[1] if len(runs) > 1 else None
    latest_q = _finish_percentile(latest)
    previous_q = _finish_percentile(previous)
    latest_class = (_finite(latest.get("class_score")) if latest else None)
    current_class = _finite(race.get("class_score")) or 56.0
    class_advantage = ((latest_class - current_class) / 15.0) if latest_class is not None else 0.0
    latest_days = _finite(latest.get("days_ago")) if latest else None
    latest_dist = _distance_similarity(_finite(latest.get("distance_m")) if latest else None, _finite(race.get("distance_m")))
    consistency = 1.0 - min(1.0, statistics.pstdev(values) if len(values) > 1 else 0.35)
    return {
        "recent_form": _weighted_mean(recent_items) if recent_items else 0.45,
        "latest_form": latest_q if latest_q is not None else 0.45,
        "previous_form": previous_q if previous_q is not None else 0.45,
        "latest_margin_quality": _margin_quality(latest) if latest else 0.45,
        "previous_margin_quality": _margin_quality(previous) if previous else 0.45,
        "last_win": 1.0 if latest and latest.get("won") else 0.0,
        "last_top3": 1.0 if latest_q is not None and (_finite(latest.get("finish")) or 99) <= 3 else 0.0,
        "class_advantage": _clip(class_advantage, -2.0, 2.0),
        "distance_suitability": _weighted_mean(distance_items) if distance_items else 0.45,
        "latest_distance_suitability": latest_dist if latest_dist is not None else 0.45,
        "going_suitability": _weighted_mean(going_items) if going_items else 0.50,
        "fitness": _fitness(latest_days, runner.get("fitness_stage", ""), bool(runner.get("first_start"))),
        "consistency": max(0.0, min(1.0, consistency)),
        "history_coverage": min(1.0, len(runs) / 3.0),
        "jump_relevance": _weighted_mean(jump_items) if jump_items else (0.50 if race.get("discipline") == "Flat" else 0.25),
    }


def _model_influence(training_races: int) -> float:
    if training_races <= 0:
        return 0.0
    return min(0.52, 0.08 + 0.48 * (1.0 - math.exp(-training_races / 110.0)))


def _specialist_weights(race: Dict[str, Any]) -> Dict[str, float]:
    active = [r for r in race.get("runners", []) if r.get("status") == "ACTIVE"]
    first_share = sum(bool(r.get("first_start")) for r in active) / max(1, len(active))
    if race.get("discipline") in {"Hurdle", "Steeplechase"}:
        weights = {"source": 0.27, "form": 0.34, "suitability": 0.29, "pace": 0.10}
    else:
        weights = {"source": 0.30, "form": 0.34, "suitability": 0.22, "pace": 0.14}
    if first_share >= 0.30:
        weights["source"] += 0.06
        weights["form"] -= 0.04
        weights["suitability"] -= 0.02
    total = sum(weights.values())
    return {key: value / total for key, value in weights.items()}


def build_feature_context(race: Dict[str, Any]) -> Dict[int, Dict[str, Any]]:
    active = [runner for runner in race.get("runners", []) if runner.get("status") == "ACTIVE"]
    if not active:
        return {}
    field_size = len(active)
    weights = [_finite(r.get("weight")) for r in active]
    barriers = [_finite(r.get("barrier")) for r in active]
    weight_stats = _mean_std(weights)
    barrier_max = max([b for b in barriers if b is not None] or [float(field_size)])
    pace_pressure = sum(bool(r.get("pace_mentioned")) or bool(r.get("on_pace_history")) for r in active)
    raw: Dict[int, Dict[str, float]] = {}
    for runner in active:
        number = int(runner["number"])
        recent = _recent_form_features(runner, race)
        mention_rank = runner.get("source_mention_rank")
        mention_strength = _rank_strength(int(mention_rank), field_size) if mention_rank else 0.22
        weight_value = _finite(runner.get("weight"))
        weight_advantage = -_z(weight_value, weight_stats) if weight_value is not None else 0.0
        barrier = _finite(runner.get("barrier"))
        sprint_factor = max(0.0, min(1.0, (1800.0 - float(race.get("distance_m") or 1800)) / 1000.0)) if race.get("discipline") == "Flat" else 0.05
        barrier_advantage = (1.0 - (barrier - 1.0) / max(1.0, barrier_max - 1.0)) * sprint_factor if barrier is not None else 0.50 * sprint_factor
        expected_leader = 1.0 if runner.get("pace_mentioned") else 0.0
        if pace_pressure >= 4:
            expected_leader *= 0.72
        raw[number] = {
            "suggested_pick": 1.0 if runner.get("suggested_pick") else 0.0,
            "source_mention_strength": mention_strength,
            "overview_sentiment": float(runner.get("overview_sentiment") or 0.0),
            "runner_sentiment": float(runner.get("runner_sentiment") or 0.0),
            "form_score": float(runner.get("form_score") or 0.50),
            "form_trend": float(runner.get("form_trend") or 0.0),
            "current_prep_runs": min(1.0, float(runner.get("current_prep_runs") or 0) / 5.0),
            **recent,
            "class_text_adjustment": float(runner.get("class_text_adjustment") or 0.0),
            "going_positive": float(runner.get("going_positive") or 0.0),
            "going_negative": float(runner.get("going_negative") or 0.0),
            "weight_advantage": weight_advantage,
            "barrier_advantage": barrier_advantage,
            "expected_leader": expected_leader,
            "on_pace_history": float(runner.get("on_pace_history") or 0.0),
            "closer_history": float(runner.get("closer_history") or 0.0),
            "factual_comment_score": float(runner.get("factual_comment_score") or 0.0),
            "first_start": 1.0 if runner.get("first_start") else 0.0,
            "age_profile": _age_profile(_finite(runner.get("age")), str(race.get("discipline"))) or 0.50,
            "jockey_present": 1.0 if str(runner.get("jockey") or "").strip() else 0.0,
            "trainer_present": 1.0 if str(runner.get("trainer") or "").strip() else 0.0,
        }
    # Field standardisation for specialist scores.
    stats = {name: _mean_std(values.get(name) for values in raw.values()) for name in RAW_FEATURE_NAMES}
    source_scores: Dict[int, float] = {}
    form_scores: Dict[int, float] = {}
    suitability_scores: Dict[int, float] = {}
    pace_scores: Dict[int, float] = {}
    for number, values in raw.items():
        z = {name: _z(values.get(name), stats[name]) for name in RAW_FEATURE_NAMES}
        source_scores[number] = (
            1.35*z["suggested_pick"] + 1.00*z["source_mention_strength"] + 0.65*z["overview_sentiment"] + 0.35*z["runner_sentiment"]
        )
        form_scores[number] = (
            0.70*z["recent_form"] + 0.50*z["latest_form"] + 0.26*z["previous_form"] + 0.40*z["form_score"] +
            0.24*z["form_trend"] + 0.30*z["latest_margin_quality"] + 0.14*z["previous_margin_quality"] +
            0.34*z["class_advantage"] + 0.24*z["last_win"] + 0.16*z["last_top3"] + 0.18*z["consistency"] +
            0.16*z["factual_comment_score"] + 0.16*z["jump_relevance"]
        )
        suitability_scores[number] = (
            0.56*z["distance_suitability"] + 0.32*z["latest_distance_suitability"] + 0.38*z["going_suitability"] +
            0.30*z["fitness"] + 0.22*z["weight_advantage"] + 0.20*z["class_text_adjustment"] +
            0.18*z["going_positive"] - 0.18*z["going_negative"] + 0.14*z["age_profile"] - 0.10*z["first_start"]
        )
        pace_scores[number] = (
            0.62*z["expected_leader"] + 0.42*z["on_pace_history"] + 0.28*z["closer_history"] +
            0.34*z["barrier_advantage"] + 0.18*z["weight_advantage"] + 0.12*z["current_prep_runs"]
        )
    source_p = _softmax(source_scores, 1.05)
    form_p = _softmax(form_scores, 1.15)
    suitability_p = _softmax(suitability_scores, 1.10)
    pace_p = _softmax(pace_scores, 1.20)
    specialist_weights = _specialist_weights(race)
    baseline_log: Dict[int, float] = {}
    for number in raw:
        baseline_log[number] = (
            specialist_weights["source"] * math.log(max(source_p[number], 1e-12)) +
            specialist_weights["form"] * math.log(max(form_p[number], 1e-12)) +
            specialist_weights["suitability"] * math.log(max(suitability_p[number], 1e-12)) +
            specialist_weights["pace"] * math.log(max(pace_p[number], 1e-12))
        )
    baseline_stats = _mean_std(baseline_log.values())
    source_ranks = _rank_map(source_scores); form_ranks = _rank_map(form_scores)
    suitability_ranks = _rank_map(suitability_scores); pace_ranks = _rank_map(pace_scores)
    context: Dict[int, Dict[str, Any]] = {}
    for runner in active:
        number = int(runner["number"])
        values = raw[number]
        vector = [
            math.log(max(source_p[number], 1e-12)), math.log(max(form_p[number], 1e-12)),
            math.log(max(suitability_p[number], 1e-12)), math.log(max(pace_p[number], 1e-12)),
            _rank_strength(source_ranks[number], field_size), _rank_strength(form_ranks[number], field_size),
            _rank_strength(suitability_ranks[number], field_size), _rank_strength(pace_ranks[number], field_size),
            _z(baseline_log[number], baseline_stats),
            *[_z(values.get(name), stats[name]) for name in RAW_FEATURE_NAMES],
        ]
        context[number] = {
            "runner": runner,
            "raw": values,
            "feature_vector": vector,
            "source_score": source_scores[number], "form_score_model": form_scores[number],
            "suitability_score": suitability_scores[number], "pace_score": pace_scores[number],
            "source_prob": source_p[number], "form_prob": form_p[number],
            "suitability_prob": suitability_p[number], "pace_prob": pace_p[number],
            "source_rank": source_ranks[number], "form_rank": form_ranks[number],
            "suitability_rank": suitability_ranks[number], "pace_rank": pace_ranks[number],
            "baseline_log": baseline_log[number], "baseline_z": _z(baseline_log[number], baseline_stats),
            "specialist_weights": specialist_weights,
        }
    return context


def _learned_scores(context: Dict[int, Dict[str, Any]], state: Dict[str, Any]) -> Dict[int, float]:
    weights = list(state.get("weights") or [])
    if len(weights) != len(FEATURE_NAMES):
        weights = [0.0] * len(FEATURE_NAMES)
    raw = {number: sum(float(w) * float(x) for w, x in zip(weights, row["feature_vector"])) for number, row in context.items()}
    stats = _mean_std(raw.values())
    return {number: _z(value, stats) for number, value in raw.items()}


def _plackett_luce_positions(probabilities: Dict[int, float], simulations: int, seed: int) -> Tuple[Dict[int, float], Dict[int, float]]:
    numbers = sorted(probabilities)
    top3 = {number: 0 for number in numbers}
    rank_sum = {number: 0.0 for number in numbers}
    rng = random.Random(seed)
    simulations = max(400, min(50000, int(simulations)))
    for _ in range(simulations):
        remaining = list(numbers)
        order: List[int] = []
        while remaining:
            total = sum(max(1e-12, probabilities[n]) for n in remaining)
            target = rng.random() * total
            cumulative = 0.0
            selected = remaining[-1]
            for number in remaining:
                cumulative += max(1e-12, probabilities[number])
                if cumulative >= target:
                    selected = number; break
            order.append(selected); remaining.remove(selected)
        for rank, number in enumerate(order, start=1):
            rank_sum[number] += rank
            if rank <= 3:
                top3[number] += 1
    return ({n: top3[n] / simulations for n in numbers}, {n: rank_sum[n] / simulations for n in numbers})


def predict_race(race: Dict[str, Any], model_state: Optional[Dict[str, Any]] = None, simulations: int = 6000) -> Dict[str, Any]:
    state = model_state or default_model_state()
    context = build_feature_context(race)
    if not context:
        return {"race_id": race.get("race_id"), "rows": [], "order": [], "overall_confidence": 0.0}
    learned = _learned_scores(context, state)
    influence = _model_influence(int(state.get("training_races") or 0))
    final_scores = {number: (1.0 - influence) * row["baseline_z"] + influence * learned.get(number, 0.0) for number, row in context.items()}
    win_probs = _softmax(final_scores, 1.0)
    ranks = _rank_map(final_scores)
    seed = int(sha256(str(race.get("race_id") or race.get("title")).encode()).hexdigest()[:12], 16)
    top3_probs, expected_ranks = _plackett_luce_positions(win_probs, simulations, seed)
    specialist_rank_maps = {
        "source": {n: row["source_rank"] for n, row in context.items()},
        "form": {n: row["form_rank"] for n, row in context.items()},
        "suitability": {n: row["suitability_rank"] for n, row in context.items()},
        "pace": {n: row["pace_rank"] for n, row in context.items()},
    }
    rows: List[Dict[str, Any]] = []
    sorted_probs = sorted(win_probs.values(), reverse=True)
    top_gap = sorted_probs[0] - sorted_probs[1] if len(sorted_probs) > 1 else sorted_probs[0]
    for number, row in context.items():
        rank_values = [mapping[number] for mapping in specialist_rank_maps.values()]
        spread = max(rank_values) - min(rank_values)
        agreement = max(0.0, 1.0 - spread / max(2.0, len(context) - 1.0))
        raw = row["raw"]
        coverage = 0.55 * raw.get("history_coverage", 0.0) + 0.20 * (1.0 - raw.get("first_start", 0.0)) + 0.25 * (1.0 if row["runner"].get("source_mention_rank") else 0.45)
        local_sep = max(0.0, win_probs[number] - statistics.median(win_probs.values()))
        confidence = 9.0 * max(0.0, min(1.0, 0.34*agreement + 0.34*coverage + 0.18*min(1.0, local_sep*len(context)*2.5) + 0.14*min(1.0, top_gap*len(context)*2.0)))
        if raw.get("first_start"):
            confidence *= 0.78
        contributions = sorted(zip(FEATURE_NAMES, row["feature_vector"]), key=lambda item: abs(item[1]), reverse=True)
        positive = [(FEATURE_LABELS.get(name, name), value) for name, value in contributions if value > 0.35][:5]
        negative = [(FEATURE_LABELS.get(name, name), value) for name, value in contributions if value < -0.35][:5]
        rows.append({
            "rank": ranks[number], "number": number, "horse": row["runner"].get("horse", ""),
            "win_probability": win_probs[number], "top3_probability": top3_probs[number],
            "expected_rank": expected_ranks[number], "confidence": confidence,
            "fair_odds": 1.0 / max(win_probs[number], 1e-12),
            "source_rank": row["source_rank"], "form_rank": row["form_rank"],
            "suitability_rank": row["suitability_rank"], "pace_rank": row["pace_rank"],
            "source_probability": row["source_prob"], "form_probability": row["form_prob"],
            "suitability_probability": row["suitability_prob"], "pace_probability": row["pace_prob"],
            "baseline_score": row["baseline_z"], "learned_score": learned.get(number, 0.0),
            "model_agreement": agreement, "history_coverage": raw.get("history_coverage", 0.0),
            "first_start": bool(row["runner"].get("first_start")), "barrier": row["runner"].get("barrier"),
            "weight": row["runner"].get("weight"), "jockey": row["runner"].get("jockey", ""),
            "trainer": row["runner"].get("trainer", ""), "positive_signals": positive,
            "negative_signals": negative, "runner": row["runner"],
        })
    rows.sort(key=lambda item: item["rank"])
    for item in rows:
        if item["rank"] == 1: item["classification"] = "FINAL WIN PICK"
        elif item["rank"] == 2: item["classification"] = "MAIN DANGER"
        elif item["rank"] <= 3: item["classification"] = "TOP-3 CONTENDER"
        elif item["rank"] <= 5: item["classification"] = "FIRST-FOUR/EXOTICS"
        else: item["classification"] = "OUTSIDER"
        item["explanation"] = explain_runner(race, item, influence)
    overall = statistics.fmean([row["confidence"] for row in rows[:min(3, len(rows))]]) if rows else 0.0
    return {
        "race_id": race.get("race_id"), "meeting": race.get("meeting"), "race_no": race.get("race_no"),
        "title": race.get("title"), "distance_m": race.get("distance_m"), "discipline": race.get("discipline"),
        "order": [row["number"] for row in rows], "order_text": "-".join(str(row["number"]) for row in rows),
        "rows": rows, "overall_confidence": overall, "learned_influence": influence,
        "training_races": int(state.get("training_races") or 0),
        "specialist_weights": next(iter(context.values()))["specialist_weights"],
        "odds_influence": 0.0,
    }


def explain_runner(race: Dict[str, Any], item: Dict[str, Any], influence: float) -> str:
    runner = item["runner"]
    lines = [
        f"{item['rank']}. No. {item['number']} {item['horse']}",
        f"Final win probability: {100*item['win_probability']:.1f}% | Top-three: {100*item['top3_probability']:.1f}% | Confidence: {item['confidence']:.1f}/9.",
        f"Specialist ranks — Bet365 Analyst {item['source_rank']}, Independent Form {item['form_rank']}, Suitability/Fitness {item['suitability_rank']}, Pace/Draw {item['pace_rank']}.",
        f"The continuously learned layer currently contributes {100*influence:.1f}% of the final score; the transparent four-model consensus contributes the balance.",
    ]
    if runner.get("suggested_pick"):
        lines.append("Bet365's pre-race suggested play explicitly included this horse.")
    elif runner.get("source_mention_rank"):
        lines.append(f"Bet365 discussed this horse at source priority position {runner['source_mention_rank']}.")
    else:
        lines.append("The Bet365 overview did not place this horse among its principal named chances.")
    if runner.get("first_start"):
        lines.append("This is a first starter or very low-data runner, so race-speed uncertainty reduces confidence.")
    if runner.get("fitness_stage"):
        lines.append(f"Preparation-stage signal: {runner['fitness_stage']}.")
    lines.append(f"Current race context: {race.get('distance_m')}m, {race.get('going')}, {race.get('discipline')}; weight {runner.get('weight') or 'not supplied'}kg, barrier {runner.get('barrier') or 'not supplied'}.")
    lines.append("")
    lines.append("Strongest positive factors:")
    if item.get("positive_signals"):
        lines.extend(f"  - {label}: +{value:.2f}" for label, value in item["positive_signals"])
    else:
        lines.append("  - No single positive factor clearly separated this horse from the field.")
    lines.append("Main negative or uncertainty factors:")
    if item.get("negative_signals"):
        lines.extend(f"  - {label}: {value:.2f}" for label, value in item["negative_signals"])
    else:
        lines.append("  - No major model-specific negative was detected; normal race uncertainty remains.")
    if runner.get("narrative"):
        lines.extend(["", "Parsed Bet365 runner summary:", runner["narrative"]])
    lines.extend(["", "Current and historical betting prices are excluded. Actual results are used only after a frozen pre-race prediction has been saved."])
    return "\n".join(lines)


def pairwise_pairs_for_result(field: Sequence[int], result: RaceResult) -> List[Tuple[int, int]]:
    active = [int(x) for x in field]
    active_set = set(active)
    groups = [[int(n) for n in group if int(n) in active_set] for group in result]
    groups = [group for group in groups if group]
    listed = {number for group in groups for number in group}
    pairs: List[Tuple[int, int]] = []
    for index, group in enumerate(groups):
        lower_known = [number for later in groups[index + 1:] for number in later]
        unknown = [number for number in active if number not in listed]
        for better in group:
            for worse in lower_known + unknown:
                if better != worse:
                    pairs.append((better, worse))
    return pairs


def training_record_from_race(race: Dict[str, Any], result: RaceResult) -> Dict[str, Any]:
    context = build_feature_context(race)
    field = sorted(context)
    return {
        "schema_version": 1,
        "record_id": str(race.get("race_id") or sha256(f"{race.get('date')}|{race.get('meeting')}|{race.get('race_no')}".encode()).hexdigest()[:20]),
        "date": race.get("date", ""), "meeting": race.get("meeting", ""), "race_no": race.get("race_no"),
        "title": race.get("title", ""), "distance_m": race.get("distance_m"), "discipline": race.get("discipline", ""),
        "field": field, "horses": {str(number): context[number]["runner"].get("horse", "") for number in field},
        "feature_names": list(FEATURE_NAMES),
        "feature_vectors": {str(number): context[number]["feature_vector"] for number in field},
        "baseline_scores": {str(number): context[number]["baseline_z"] for number in field},
        "result": [[int(x) for x in group] for group in result],
        "saved_at": datetime.now().isoformat(timespec="seconds"),
    }


def _training_rows(records: Sequence[Dict[str, Any]]) -> List[List[float]]:
    rows: List[List[float]] = []
    for record in records:
        if record.get("feature_names") != FEATURE_NAMES:
            continue
        vectors = record.get("feature_vectors") or {}
        field = [int(x) for x in record.get("field") or []]
        for better, worse in pairwise_pairs_for_result(field, record.get("result") or []):
            a, b = vectors.get(str(better)), vectors.get(str(worse))
            if not a or not b or len(a) != len(FEATURE_NAMES) or len(b) != len(FEATURE_NAMES):
                continue
            rows.append([float(x) - float(y) for x, y in zip(a, b)])
    return rows


def _sigmoid(value: float) -> float:
    value = max(-35.0, min(35.0, value))
    return 1.0 / (1.0 + math.exp(-value))


def train_model(records: Sequence[Dict[str, Any]], epochs: int = 520, learning_rate: float = 0.28, l2: float = 0.55) -> Dict[str, Any]:
    state = default_model_state()
    pairs = _training_rows(records)
    if not pairs:
        return state
    n = len(pairs)
    try:
        import numpy as np
        matrix = np.asarray(pairs, dtype=float)
        weights_np = np.zeros(len(FEATURE_NAMES), dtype=float)
        for epoch in range(max(1, int(epochs))):
            logits = np.clip(matrix @ weights_np, -35.0, 35.0)
            probs = 1.0 / (1.0 + np.exp(-logits))
            gradient = matrix.T @ (probs - 1.0) + l2 * weights_np
            step = learning_rate / (1.0 + epoch / 180.0)
            weights_np -= step * gradient / n
            weights_np = np.clip(weights_np, -2.3, 2.3)
        weights = [float(x) for x in weights_np.tolist()]
    except Exception:
        weights = [0.0] * len(FEATURE_NAMES)
        for epoch in range(max(1, int(epochs))):
            gradient = [l2 * w for w in weights]
            for diff in pairs:
                p = _sigmoid(sum(w*x for w, x in zip(weights, diff)))
                coeff = p - 1.0
                for index, value in enumerate(diff):
                    gradient[index] += coeff * value
            step = learning_rate / (1.0 + epoch / 180.0)
            for index in range(len(weights)):
                weights[index] = _clip(weights[index] - step * gradient[index] / n, -2.3, 2.3)
    state["weights"] = weights
    state["training_races"] = len(records)
    state["training_pairs"] = n
    state["learned_influence"] = _model_influence(len(records))
    state["updated_at"] = datetime.now().isoformat(timespec="seconds")
    state["training_metrics"] = evaluate_records(records, state)
    return state


def _record_prediction(record: Dict[str, Any], state: Dict[str, Any]) -> Dict[int, float]:
    vectors = record.get("feature_vectors") or {}
    baseline = record.get("baseline_scores") or {}
    weights = list(state.get("weights") or [])
    if len(weights) != len(FEATURE_NAMES):
        weights = [0.0] * len(FEATURE_NAMES)
    raw = {int(number): sum(w * float(x) for w, x in zip(weights, vector)) for number, vector in vectors.items() if len(vector) == len(FEATURE_NAMES)}
    stats = _mean_std(raw.values())
    learned = {number: _z(value, stats) for number, value in raw.items()}
    influence = _model_influence(int(state.get("training_races") or 0))
    return {number: (1.0-influence)*float(baseline.get(str(number), 0.0)) + influence*learned.get(number, 0.0) for number in raw}


def evaluate_records(records: Sequence[Dict[str, Any]], state: Dict[str, Any]) -> Dict[str, Any]:
    winner_top1 = winner_top3 = 0
    winner_ranks: List[int] = []
    pair_correct = pair_total = 0
    top4_overlap: List[float] = []
    for record in records:
        scores = _record_prediction(record, state)
        if not scores:
            continue
        ranks = _rank_map(scores)
        result: RaceResult = record.get("result") or []
        if result and result[0]:
            winners = [int(number) for number in result[0] if int(number) in ranks]
            if winners:
                best = min(ranks[number] for number in winners)
                winner_ranks.append(best); winner_top1 += int(best == 1); winner_top3 += int(best <= 3)
        for better, worse in pairwise_pairs_for_result(record.get("field") or [], result):
            if better in scores and worse in scores:
                pair_total += 1; pair_correct += int(scores[better] > scores[worse])
        known = [number for group in result[:4] for number in group]
        if known:
            predicted = {number for number, rank in ranks.items() if rank <= min(4, len(ranks))}
            top4_overlap.append(len(predicted.intersection(known)) / min(4, len(known)))
    races = len(winner_ranks)
    return {
        "evaluated_races": races,
        "winner_top1_pct": 100.0*winner_top1/races if races else 0.0,
        "winner_top3_pct": 100.0*winner_top3/races if races else 0.0,
        "mean_winner_rank": statistics.fmean(winner_ranks) if winner_ranks else None,
        "pairwise_accuracy_pct": 100.0*pair_correct/pair_total if pair_total else 0.0,
        "mean_known_top4_overlap_pct": 100.0*statistics.fmean(top4_overlap) if top4_overlap else 0.0,
        "metric_scope": "Training-fit diagnostics; leave-one-meeting-out and future chronological testing are required for honest validation.",
    }


def record_identifier(race: Dict[str, Any]) -> str:
    return str(race.get("race_id") or sha256(f"{race.get('date')}|{race.get('meeting')}|{race.get('race_no')}".encode()).hexdigest()[:20])
