"""Out-of-time training utilities for labelled runner-race feature rows."""
from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .features import MODEL_FEATURE_COLUMNS


@dataclass
class TrainingResult:
    artifact: dict[str, Any]
    metrics: dict[str, float]
    validation_predictions: pd.DataFrame


def _normalise_by_race(values: np.ndarray, race_ids: pd.Series, temperature: float = 1.0) -> np.ndarray:
    values = np.clip(np.asarray(values, dtype=float), 1e-12, 1.0)
    out = np.zeros_like(values)
    ids = race_ids.astype(str).to_numpy()
    for race in pd.unique(ids):
        idx = np.flatnonzero(ids == race)
        logits = np.log(values[idx]) / max(temperature, 1e-6)
        logits -= logits.max()
        p = np.exp(logits)
        out[idx] = p / p.sum()
    return out


def _race_log_loss(y: np.ndarray, p: np.ndarray) -> float:
    return float(-np.mean(np.log(np.clip(p[np.asarray(y) == 1], 1e-12, 1.0))))


def train_models(frame: pd.DataFrame) -> TrainingResult:
    required = {"race_id", "race_date", "won"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError("Training CSV is missing: " + ", ".join(sorted(missing)))
    df = frame.copy()
    if "discipline" not in df.columns:
        raise ValueError("Training CSV must include a discipline column so horse, harness and greyhound models are not mixed.")
    disciplines = sorted({str(v).strip().lower() for v in df["discipline"].dropna() if str(v).strip()})
    if len(disciplines) != 1:
        raise ValueError(
            "Train one discipline at a time. The uploaded CSV contains: "
            + (", ".join(disciplines) if disciplines else "no valid discipline")
        )
    discipline = disciplines[0]
    if discipline not in {"thoroughbred", "harness", "greyhound"}:
        raise ValueError(f"Unsupported discipline in training data: {discipline}")
    df["discipline"] = discipline
    df["race_date"] = pd.to_datetime(df["race_date"], errors="coerce")
    df["won"] = pd.to_numeric(df["won"], errors="coerce").fillna(0).astype(int)
    df = df.dropna(subset=["race_date", "race_id"]).sort_values(["race_date", "race_id"])
    race_dates = df.groupby("race_id")["race_date"].min().sort_values()
    if len(race_dates) < 80:
        raise ValueError("At least 80 labelled races are required for a minimally credible time split; 300+ is preferred.")
    cut = max(1, int(len(race_dates) * 0.80))
    train_ids = set(race_dates.index[:cut])
    valid_ids = set(race_dates.index[cut:])
    train = df[df["race_id"].isin(train_ids)].copy()
    valid = df[df["race_id"].isin(valid_ids)].copy()
    if valid["won"].sum() < 10:
        raise ValueError("Validation period has too few winners.")

    columns = [c for c in MODEL_FEATURE_COLUMNS if c in df.columns]
    if len(columns) < 8:
        raise ValueError("Training data contains too few engineered feature columns.")
    X_train, y_train = train[columns], train["won"]
    X_valid, y_valid = valid[columns], valid["won"]

    logistic = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=3000, class_weight="balanced", C=0.35)),
    ])
    tree = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", HistGradientBoostingClassifier(
            learning_rate=0.045,
            max_iter=260,
            max_leaf_nodes=15,
            min_samples_leaf=25,
            l2_regularization=2.0,
            random_state=42,
        )),
    ])
    field_size = train.groupby("race_id")["won"].transform("size").to_numpy(float)
    sample_weight = np.where(y_train.to_numpy() == 1, np.maximum(field_size - 1, 1), 1.0)
    logistic.fit(X_train, y_train, model__sample_weight=sample_weight)
    tree.fit(X_train, y_train, model__sample_weight=sample_weight)

    raw = 0.5 * logistic.predict_proba(X_valid)[:, 1] + 0.5 * tree.predict_proba(X_valid)[:, 1]
    best_t, best_loss = 1.0, float("inf")
    for t in np.linspace(0.55, 3.0, 100):
        p = _normalise_by_race(raw, valid["race_id"], float(t))
        loss = _race_log_loss(y_valid.to_numpy(), p)
        if loss < best_loss:
            best_t, best_loss = float(t), loss
    p = _normalise_by_race(raw, valid["race_id"], best_t)
    scored = valid[["race_id", "race_date", "won"]].copy()
    scored["predicted_probability"] = p
    top_idx = scored.groupby("race_id")["predicted_probability"].idxmax()
    top1 = float(scored.loc[top_idx, "won"].mean())
    brier = float(np.mean((p - y_valid.to_numpy()) ** 2))

    artifact = {
        "version": 1,
        "discipline": discipline,
        "feature_columns": columns,
        "logistic_model": logistic,
        "tree_model": tree,
        "temperature": best_t,
        "trained_races": int(len(train_ids)),
        "validation_races": int(len(valid_ids)),
        "validation_log_loss": best_loss,
        "validation_top1": top1,
        "validation_brier": brier,
    }
    return TrainingResult(
        artifact=artifact,
        metrics={
            "training_races": float(len(train_ids)),
            "validation_races": float(len(valid_ids)),
            "validation_log_loss": best_loss,
            "validation_top1": top1,
            "validation_brier": brier,
            "temperature": best_t,
        },
        validation_predictions=scored,
    )


def artifact_bytes(artifact: dict[str, Any]) -> bytes:
    buffer = BytesIO()
    joblib.dump(artifact, buffer)
    return buffer.getvalue()
