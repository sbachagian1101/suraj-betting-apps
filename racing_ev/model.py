"""Market-independent race probability model and optional trained-model blend."""
from __future__ import annotations

from dataclasses import dataclass
import io
import math
from typing import Any

import joblib
import numpy as np
import pandas as pd

from .features import COMPONENT_COLUMNS, MODEL_FEATURE_COLUMNS


WEIGHTS: dict[str, dict[str, float]] = {
    "thoroughbred": {
        "recent_finish_score": 0.27,
        "recent_win_signal": 0.12,
        "recent_place_signal": 0.08,
        "margin_signal": 0.15,
        "speed_signal": 0.11,
        "rating_signal": 0.10,
        "suitability_signal": 0.08,
        "connection_signal": 0.06,
        "fitness_signal": 0.04,
        "setup_signal": 0.02,
        "reliability_signal": 0.03,
        "trial_signal": 0.07,
    },
    "harness": {
        "recent_finish_score": 0.24,
        "recent_win_signal": 0.10,
        "recent_place_signal": 0.07,
        "margin_signal": 0.10,
        "speed_signal": 0.21,
        "rating_signal": 0.05,
        "suitability_signal": 0.06,
        "connection_signal": 0.07,
        "fitness_signal": 0.04,
        "setup_signal": 0.10,
        "reliability_signal": 0.10,
        "trial_signal": 0.03,
    },
    "greyhound": {
        "recent_finish_score": 0.22,
        "recent_win_signal": 0.10,
        "recent_place_signal": 0.06,
        "margin_signal": 0.08,
        "speed_signal": 0.26,
        "rating_signal": 0.02,
        "suitability_signal": 0.06,
        "connection_signal": 0.05,
        "fitness_signal": 0.04,
        "setup_signal": 0.14,
        "reliability_signal": 0.05,
        "trial_signal": 0.01,
    },
}

TEMPERATURE = {"thoroughbred": 1.35, "harness": 1.30, "greyhound": 1.20}


@dataclass
class ProbabilityResult:
    table: pd.DataFrame
    model_name: str
    warning: str | None = None


def _robust_z(series: pd.Series) -> pd.Series:
    x = pd.to_numeric(series, errors="coerce")
    if x.notna().sum() < 2:
        return pd.Series(np.zeros(len(x)), index=x.index, dtype=float)
    median = float(x.median())
    mad = float((x - median).abs().median())
    if mad > 1e-9:
        z = (x - median) / (1.4826 * mad)
    else:
        sd = float(x.std(ddof=0))
        z = (x - median) / (sd if sd > 1e-9 else 1.0)
    return z.fillna(0.0).clip(-3.0, 3.0)


def _softmax(scores: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    z = np.asarray(scores, dtype=float) / max(float(temperature), 1e-6)
    z = z - np.nanmax(z)
    exp = np.exp(np.clip(z, -50, 50))
    total = exp.sum()
    return exp / total if total > 0 else np.repeat(1.0 / len(z), len(z))


def _baseline_probabilities(features: pd.DataFrame, discipline: str) -> tuple[pd.DataFrame, np.ndarray]:
    df = features.copy().reset_index(drop=True)
    weights = WEIGHTS.get(discipline, WEIGHTS["thoroughbred"])
    score = np.zeros(len(df), dtype=float)
    for feature, weight in weights.items():
        z = _robust_z(df.get(feature, pd.Series(np.zeros(len(df)))))
        df[f"z_{feature}"] = z
        df[f"contrib_{feature}"] = z * weight
        score += df[f"contrib_{feature}"].to_numpy(float)
    df["form_score"] = score
    raw = _softmax(score, TEMPERATURE.get(discipline, 1.35))

    # Runner-specific shrinkage protects debutants and sparse profiles from false
    # precision. Variable-confidence probabilities are renormalised within race.
    uniform = 1.0 / max(len(df), 1)
    quality = pd.to_numeric(df["data_quality"], errors="coerce").fillna(0.35).clip(0.2, 1.0).to_numpy()
    confidence = 0.32 + 0.58 * quality
    adjusted = confidence * raw + (1.0 - confidence) * uniform
    adjusted = adjusted / adjusted.sum()
    return df, adjusted


def load_artifact(data: bytes | io.BytesIO | str) -> dict[str, Any]:
    if isinstance(data, bytes):
        return joblib.load(io.BytesIO(data))
    return joblib.load(data)


def _trained_probabilities(features: pd.DataFrame, artifact: dict[str, Any]) -> np.ndarray:
    columns = artifact.get("feature_columns", MODEL_FEATURE_COLUMNS)
    X = features.reindex(columns=columns)
    preds = []
    for key in ("logistic_model", "tree_model"):
        model = artifact.get(key)
        if model is not None:
            preds.append(np.asarray(model.predict_proba(X)[:, 1], dtype=float))
    if not preds:
        raise ValueError("The uploaded model artifact has no supported estimators.")
    raw = np.mean(preds, axis=0)
    raw = np.clip(raw, 1e-8, None)
    temperature = float(artifact.get("temperature", 1.0))
    logits = np.log(raw) / max(temperature, 1e-6)
    return _softmax(logits, 1.0)


def score_race(
    features: pd.DataFrame,
    discipline: str,
    artifact: dict[str, Any] | None = None,
    trained_weight: float = 0.70,
) -> ProbabilityResult:
    if features.empty:
        return ProbabilityResult(features.copy(), "No model", "No active runners to score.")
    df, baseline = _baseline_probabilities(features, discipline)
    model_name = "Transparent form baseline"
    warning = (
        "Probabilities are a shrinkage-controlled baseline until a labelled, out-of-time "
        "trained model is supplied. Treat them as research estimates, not guarantees."
    )
    final = baseline
    if artifact is not None:
        try:
            artifact_discipline = str(artifact.get("discipline", "")).strip().lower()
            if artifact_discipline and artifact_discipline != discipline:
                raise ValueError(
                    f"model was trained for {artifact_discipline}, but this race is {discipline}"
                )
            trained = _trained_probabilities(features, artifact)
            w = float(np.clip(trained_weight, 0.0, 1.0))
            final = w * trained + (1.0 - w) * baseline
            final = final / final.sum()
            df["trained_probability"] = trained
            model_name = f"Trained ensemble + form baseline ({w:.0%}/{1-w:.0%})"
            warning = None
        except Exception as exc:  # keep the app usable when an incompatible artifact is uploaded
            warning = f"Uploaded model could not be used; baseline shown instead. Reason: {exc}"
    df["win_probability"] = final
    df["fair_odds"] = np.where(final > 0, 1.0 / final, np.nan)
    df["probability_uncertainty_pp"] = (1.0 - df["data_quality"].clip(0, 1)) * 8.0 + 2.0
    df["model_rank"] = df["win_probability"].rank(method="first", ascending=False).astype(int)
    return ProbabilityResult(df.sort_values("model_rank").reset_index(drop=True), model_name, warning)


def explain_runner(row: pd.Series, top_n: int = 4) -> list[tuple[str, float]]:
    parts = []
    for feature in COMPONENT_COLUMNS:
        key = f"contrib_{feature}"
        if key in row and pd.notna(row[key]):
            parts.append((feature, float(row[key])))
    return sorted(parts, key=lambda x: abs(x[1]), reverse=True)[:top_n]
