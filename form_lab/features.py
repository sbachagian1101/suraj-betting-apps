"""One feature path for historical replay and live cards; no market columns."""
from __future__ import annotations

from collections import defaultdict
import math

import numpy as np
import pandas as pd

from .schema import instant, training_issue

NUMERIC = [
    "field_size", "distance_m", "age", "draw_fraction", "draw_sprint",
    "carried_weight_kg", "official_rating", "history_starts", "history_wins",
    "log_days_since_run", "recent_form", "recent_beaten_lengths", "win_rate",
    "top3_rate", "completion_rate", "distance_form", "course_form", "going_form",
    "surface_form", "jockey_starts", "jockey_win_rate", "trainer_starts", "trainer_win_rate",
]
CATEGORICAL = ["course_id", "surface", "going", "race_class", "sex", "gear", "jockey_id", "trainer_id"]


def _mean(values, weights=None):
    return float(np.average(values, weights=weights)) if len(values) else np.nan


def _quality(run):
    if run["status"] != "finished":
        return 0.0
    return 1.0 - (run["finish_position"] - 1) / max(1, run["field_size"] - 1)


def build_features(records, target_ids=None):
    """Replay results by publication time, before constructing each snapshot.

Only completed full-field records update connection statistics. They include
earlier holdout outcomes when known: this is rolling evaluation with frozen
model parameters, not a frozen-history simulation. No label from the target
race is used to construct its features.
"""
    records = sorted(records, key=lambda r: (instant(r["prediction_at"]), r["race_id"]))
    events = sorted((r for r in records if training_issue(r) is None),
                    key=lambda r: instant(r["result"]["available_at"]))
    observed, connection = defaultdict(dict), defaultdict(lambda: [0, 0, 0.0])
    cursor, rows = 0, []
    for race in records:
        at = instant(race["prediction_at"])
        while cursor < len(events) and instant(events[cursor]["result"]["available_at"]) <= at:
            past = events[cursor]
            outcomes = {e["runner_id"]: e for e in past["result"]["runners"]}
            size = sum(e["status"] == "active" for e in past["runners"])
            for e in past["runners"]:
                if e["status"] != "active":
                    continue
                o = outcomes[e["runner_id"]]
                won = int(o.get("finish_position") == 1)
                h = {k: past.get(k, "") for k in ("race_id", "country", "discipline", "course_id",
                                                  "distance_m", "surface", "going")}
                h.update(off_at=past["scheduled_off"], available_at=past["result"]["available_at"],
                         field_size=size, finish_position=o.get("finish_position"), status=o["status"],
                         beaten_lengths=None)
                observed[e["horse_id"]][past["race_id"]] = h
                for role in ("jockey_id", "trainer_id"):
                    if e.get(role):
                        v = connection[(past["country"], past["discipline"], role, e[role])]
                        v[0] += 1
                        v[1] += won
                        v[2] += 1 / size
            cursor += 1
        if target_ids is not None and race["race_id"] not in target_ids:
            continue
        active = sorted((e for e in race["runners"] if e["status"] == "active"),
                        key=lambda e: e["runner_id"])
        outcomes = {e["runner_id"]: e for e in (race.get("result") or {}).get("runners", [])}
        winners = sum(e.get("finish_position") == 1 for e in outcomes.values())
        n = len(active)
        for e in active:
            history = dict(observed[e["horse_id"]])
            # Attached histories carry additional source fields, such as margins.
            # Conflicting results are refused rather than silently overwritten.
            for h in e.get("history", []):
                previous = history.get(h["race_id"])
                if previous and any(previous[k] != h[k] for k in ("status", "finish_position", "field_size")):
                    raise ValueError(f"Conflicting history for {e['horse_id']} / {h['race_id']}")
                history[h["race_id"]] = h
            history = sorted((h for h in history.values()
                              if instant(h["available_at"]) <= at
                              and h["discipline"] == race["discipline"]
                              and h["status"] != "non_runner"),
                             key=lambda h: instant(h["off_at"]))
            recent = history[-8:]
            days = [(at - instant(h["off_at"])).total_seconds() / 86400 for h in recent]
            weights = [math.exp(-d / 90) for d in days]
            # A fixed smoothing strength is a baseline hyperparameter, not a fitted Bayesian posterior.
            starts = len(history)
            expected = sum(1 / h["field_size"] for h in history)
            wins = sum(h.get("finish_position") == 1 for h in history)
            top3 = sum(h["status"] == "finished" and h["finish_position"] <= 3 for h in history)
            result = outcomes.get(e["runner_id"], {})
            row = {k: race[k] for k in ("race_id", "meeting_id", "country", "discipline", "prediction_at", "scheduled_off")}
            row.update(runner_id=e["runner_id"], horse_id=e["horse_id"], horse_name=e["horse_name"],
                       label_known_at=(race.get("result") or {}).get("available_at"),
                       y=(float(result.get("finish_position") == 1) / winners if winners else np.nan),
                       finish_position=result.get("finish_position"), outcome_status=result.get("status"),
                       eligible=training_issue(race) is None,
                       field_size=n, distance_m=race["distance_m"], age=e.get("age"),
                       draw_fraction=(e["draw"] - 1) / max(1, n - 1) if e.get("draw") else np.nan,
                       carried_weight_kg=e.get("carried_weight_kg"), official_rating=e.get("official_rating"),
                       history_starts=starts, history_wins=wins,
                       log_days_since_run=math.log1p(days[-1]) if days else np.nan,
                       recent_form=_mean([_quality(h) for h in recent], weights),
                       recent_beaten_lengths=_mean([min(h["beaten_lengths"], 30) for h in recent
                                                   if h.get("beaten_lengths") is not None]),
                       win_rate=(wins + 10 * (expected / starts if starts else 1/n)) / (starts + 10),
                       top3_rate=(top3 + 10 * min(3/n, 1)) / (starts + 10),
                       completion_rate=(sum(h["status"] == "finished" for h in history) + 9) / (starts + 10))
            row["draw_sprint"] = row["draw_fraction"] * 1200 / race["distance_m"]
            for feature, selector in {
                "distance_form": lambda h: abs(h["distance_m"] / race["distance_m"] - 1) <= .15,
                "course_form": lambda h: bool(race["course_id"]) and h.get("course_id") == race["course_id"],
                "going_form": lambda h: bool(race.get("going")) and h.get("going") == race.get("going"),
                "surface_form": lambda h: bool(race.get("surface")) and h.get("surface") == race.get("surface"),
            }.items():
                matched = [_quality(h) for h in history if selector(h)]
                row[feature] = (sum(matched) + 5 * (_mean([_quality(h) for h in history]) if history else .5)) / (len(matched) + 5)
            for role in ("jockey", "trainer"):
                s, w, exp = connection[(race["country"], race["discipline"], role + "_id", e.get(role + "_id", ""))]
                row[role + "_starts"] = s
                row[role + "_win_rate"] = (w + 20 * (exp / s if s else 1/n)) / (s + 20)
            for col in CATEGORICAL:
                row[col] = str((race if col in {"course_id", "surface", "going", "race_class"} else e).get(col) or "unknown")
            rows.append(row)
    return pd.DataFrame(rows)


class Transform:
    """Fit imputation/scaling on training rows only; persist the exact feature list."""
    def fit(self, frame):
        x = frame[NUMERIC].astype(float)
        self.medians = x.median().fillna(0).to_dict()
        filled = x.fillna(self.medians)
        self.means = filled.mean().to_dict()
        self.scales = filled.std(ddof=0).replace(0, 1).to_dict()
        return self

    def apply(self, frame):
        raw = frame[NUMERIC].astype(float)
        filled = raw.fillna(self.medians)
        z = (filled - pd.Series(self.means)) / pd.Series(self.scales)
        missing = raw.isna().astype(float).add_suffix("__missing")
        groups = frame.race_id
        mean = filled.groupby(groups).transform("mean")
        sd = filled.groupby(groups).transform("std").fillna(0).replace(0, 1)
        relative = ((filled - mean) / sd).add_suffix("__race_z")
        return pd.concat([z, missing, relative], axis=1).fillna(0)

    def to_dict(self):
        return {"numeric_features": NUMERIC, "medians": self.medians, "means": self.means, "scales": self.scales}

    @classmethod
    def from_dict(cls, data):
        if data["numeric_features"] != NUMERIC:
            raise ValueError("Feature schema differs from the fitted model")
        t = cls()
        t.medians, t.means, t.scales = (data[k] for k in ("medians", "means", "scales"))
        return t
