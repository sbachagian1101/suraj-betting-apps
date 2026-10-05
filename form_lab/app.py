"""Separate Streamlit entry point; reads only the new engine's files."""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import streamlit as st

from form_lab.schema import instant, read_jsonl
from form_lab.training import Bundle

st.set_page_config(page_title="Form Lab", page_icon="🏇", layout="wide")
st.title("Form Lab")
st.caption("Form-only research predictions · Australia · South Africa · Great Britain · France")

model_root = Path(os.environ.get("FORM_LAB_MODELS", "form_lab/models"))
cards_path = Path(os.environ.get("FORM_LAB_CARDS", "form_lab/data/cards.jsonl"))
history_path = Path(os.environ.get("FORM_LAB_HISTORY", "form_lab/data/history.jsonl"))
manifests = sorted(model_root.glob("*/manifest.json"))
if not manifests or not cards_path.exists() or not history_path.exists():
    st.info("Awaiting an audited race archive and a trained model. Predictions will appear once these are connected.")
    st.write("The first build includes a conditional-logit baseline, CatBoost and chronological validation. "
             "The dynamic Bayesian model is a later addition.")
    st.stop()

try:
    options = {}
    for p in manifests:
        m = json.loads(p.read_text())
        options[f"{m['country']} · {m['discipline']} · {m['model_version']}"] = p.parent
    selected = st.sidebar.selectbox("Model", list(options))
    bundle = Bundle(options[selected])
    m = bundle.manifest
    if m["data_kind"] == "synthetic":
        st.warning("Synthetic test model: these are pipeline demonstration probabilities, not trained racing forecasts.")
    cards = [r for r in read_jsonl(cards_path) if (r["country"], r["discipline"]) == (m["country"], m["discipline"])]
    history = read_jsonl(history_path) if history_path.exists() else []
    st.sidebar.caption(f"Information cutoff: {m['knowledge_cutoff']}")
    st.sidebar.caption(f"Baseline/CatBoost weights: {m['weights'][0]:.2f} / {m['weights'][1]:.2f}")
    if not cards:
        st.info("No cards match this model's country and discipline.")
        st.stop()
    courses = sorted({r["course_id"] for r in cards})
    for course, venue_tab in zip(courses, st.tabs(courses)):
        with venue_tab:
            races = sorted((r for r in cards if r["course_id"] == course), key=lambda r: instant(r["scheduled_off"]))
            for race, race_tab in zip(races, st.tabs([f"R{r.get('race_no') or i+1}" for i, r in enumerate(races)])):
                with race_tab:
                    st.caption(f"{race['distance_m']} m · {race.get('surface', '')} · {race.get('going', '')} · Start {race['scheduled_off']}")
                    if instant(race["scheduled_off"]) <= datetime.now(timezone.utc):
                        st.info("The scheduled start has passed. New live predictions are disabled for this card.")
                        continue
                    prediction = bundle.predict(race, history)
                    st.caption(f"Race snapshot: {race['prediction_at']}")
                    table = pd.DataFrame(prediction["runners"])
                    shown = table[["horse_name", "baseline_win", "catboost_win", "win_probability", "top3_probability_pl",
                                   "model_disagreement_pp", "history_starts"]].copy()
                    shown.columns = ["Horse", "Baseline win %", "CatBoost win %", "Combined win %", "Top-three %", "Disagreement pp", "Past starts"]
                    for col in ("Baseline win %", "CatBoost win %", "Combined win %", "Top-three %"):
                        shown[col] = (shown[col] * 100).round(1)
                    st.dataframe(shown, hide_index=True, width="stretch")
                    st.caption(prediction["top3_note"])
                    if table.history_starts.eq(0).any():
                        st.warning("Some runners have no verified past starts in the archive; their form evidence is limited.")
                    with st.expander("Data quality and baseline explanations"):
                        st.write("Contributions explain the linear baseline relative to this field; they do not explain the entire ensemble.")
                        st.json([{k: r[k] for k in ("horse_name", "missing_numeric_fields", "baseline_contributions")} for r in prediction["runners"]])
except (ValueError, KeyError, OSError) as exc:
    st.error(f"Prediction unavailable: {exc}")
