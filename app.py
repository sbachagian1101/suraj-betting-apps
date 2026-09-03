from __future__ import annotations

import json
import re
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st

from racing_ev.features import build_feature_frame
from racing_ev.model import explain_runner, load_artifact, score_race
from racing_ev.odds import add_value_columns
from racing_ev.parser import ParsedRace, parse_race
from racing_ev.training import artifact_bytes, train_models


st.set_page_config(
    page_title="Racing EV Lab",
    page_icon="🏁",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
.block-container {padding-top: 1.4rem; padding-bottom: 3rem;}
[data-testid="stMetricValue"] {font-size: 1.55rem;}
.small-note {font-size: 0.88rem; opacity: 0.82;}
.ev-positive {padding: .65rem .8rem; border-radius: .5rem; background: rgba(30,160,90,.10);}
</style>
""",
    unsafe_allow_html=True,
)


def _race_id(card: ParsedRace) -> str:
    race = card.race
    bits = [
        str(race.get("discipline", "race")),
        str(race.get("date", "unknown-date")),
        str(race.get("track", "unknown-track")),
        f"R{race.get('race_no', 'x')}",
    ]
    return "_".join(re.sub(r"[^A-Za-z0-9-]+", "-", x).strip("-").lower() for x in bits)


def _json_safe(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if np.isnan(value) else float(value)
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    return value


def _serialisable_runner_frame(card: ParsedRace) -> pd.DataFrame:
    rows = []
    for runner in card.runners:
        row = runner.copy()
        if isinstance(row.get("box_stats"), dict):
            row["box_stats"] = json.dumps(row["box_stats"], sort_keys=True)
        rows.append(row)
    return pd.DataFrame(rows)


def _recalculate(card: ParsedRace, features: pd.DataFrame, odds_frame: pd.DataFrame, artifact: dict | None, devig: str, commission: float) -> tuple[pd.DataFrame, str, str | None, float]:
    scored = score_race(features, card.discipline, artifact=artifact)
    table = scored.table.copy()
    edited = odds_frame.set_index("runner")["market_odds"]
    table["market_odds"] = table["runner"].map(edited)
    value, overround = add_value_columns(table, method=devig, commission=commission)
    return value, scored.model_name, scored.warning, overround


def _display_prediction_table(df: pd.DataFrame) -> None:
    show = df[[
        "model_rank", "tab", "runner", "market_odds", "win_probability", "fair_odds",
        "market_probability_fair", "probability_edge", "ev_pct", "data_quality", "value_grade",
    ]].copy()
    show["win_probability"] *= 100.0
    show["market_probability_fair"] *= 100.0
    show["probability_edge"] *= 100.0
    show["data_quality"] *= 100.0
    show.columns = [
        "Rank", "No.", "Runner", "Odds", "Model win %", "Model fair odds",
        "No-vig market %", "Probability edge", "EV %", "Data quality %", "Assessment",
    ]
    st.dataframe(
        show,
        hide_index=True,
        use_container_width=True,
        column_config={
            "Odds": st.column_config.NumberColumn(format="%.2f"),
            "Model win %": st.column_config.NumberColumn(format="%.1f%%"),
            "Model fair odds": st.column_config.NumberColumn(format="%.2f"),
            "No-vig market %": st.column_config.NumberColumn(format="%.1f%%"),
            "Probability edge": st.column_config.NumberColumn(format="%+.1f%%"),
            "EV %": st.column_config.NumberColumn(format="%+.1f%%"),
            "Data quality %": st.column_config.ProgressColumn(min_value=0.0, max_value=100.0, format="%.0f%%"),
        },
    )


def _make_training_rows(card: ParsedRace, features: pd.DataFrame, winner: str | None = None) -> pd.DataFrame:
    out = features.copy()
    # Feature engineering already carries some race metadata (notably discipline).
    # Assign metadata first, then reorder it, rather than blindly inserting a
    # duplicate column and crashing the whole Streamlit page.
    metadata = {
        "race_id": _race_id(card),
        "race_date": card.race.get("date"),
        "track": card.race.get("track"),
        "race_no": card.race.get("race_no"),
        "race_name": card.race.get("race_name"),
        "discipline": card.discipline,
    }
    for column, value in metadata.items():
        out[column] = value
    out["won"] = (out["runner"] == winner).astype(int) if winner else 0
    front = ["race_id", "race_date", "track", "race_no", "race_name", "discipline"]
    remaining = [column for column in out.columns if column not in front]
    return out[front + remaining]


for key, default in {
    "card": None,
    "features": None,
    "prediction": None,
    "model_name": None,
    "model_warning": None,
    "overround": None,
    "artifact": None,
    "artifact_bytes": None,
    "training_metrics": None,
}.items():
    st.session_state.setdefault(key, default)


st.title("🏁 Racing EV Lab")
st.caption("One parser and probability/EV workflow for thoroughbred, harness and greyhound Racing & Sports Enhanced Form text.")

with st.sidebar:
    st.header("Calculation settings")
    discipline_choice = st.selectbox("Input discipline", ["Auto-detect", "Thoroughbred", "Harness", "Greyhound"])
    devig_method = st.selectbox("Remove bookmaker margin using", ["Power", "Proportional"], index=0)
    exchange_commission_pct = st.number_input("Exchange commission on winnings (%)", min_value=0.0, max_value=15.0, value=0.0, step=0.5)
    st.divider()
    st.subheader("Optional trained model")
    model_file = st.file_uploader("Upload a Racing EV Lab .joblib model", type=["joblib", "pkl"], key="model_upload")
    if model_file is not None:
        try:
            st.session_state.artifact = load_artifact(model_file.getvalue())
            st.success("Trained model loaded.")
        except Exception as exc:
            st.error(f"Model could not be loaded: {exc}")
    if st.session_state.artifact:
        artifact = st.session_state.artifact
        st.caption(
            f"Discipline: {str(artifact.get('discipline', 'legacy/unspecified')).title()}  ·  "
            f"Training races: {artifact.get('trained_races', '—')}  ·  "
            f"Validation races: {artifact.get('validation_races', '—')}"
        )
    st.divider()
    st.caption("Current odds never enter the independent form score. They are applied only after the probability calculation to estimate market edge and EV.")


tabs = st.tabs([
    "1 · Input",
    "2 · Prediction & EV",
    "3 · Parsed data",
    "4 · Data builder",
    "5 · Model lab",
    "6 · Method",
])

with tabs[0]:
    st.subheader("Paste or upload one Enhanced Form race")
    col1, col2 = st.columns([1, 1])
    with col1:
        uploaded = st.file_uploader("Upload .md or .txt", type=["md", "txt"], key="race_upload")
    with col2:
        st.info("Use Select All → Copy on the Enhanced Form page, or save the copied content as Markdown/text. Odds can be corrected in the next tab.")
    pasted = st.text_area("Paste complete race text", height=330, placeholder="Paste the full Enhanced Form page here…")
    if st.button("Parse race and calculate", type="primary", use_container_width=True):
        raw = uploaded.getvalue().decode("utf-8", errors="replace") if uploaded is not None else pasted
        chosen = discipline_choice.lower() if discipline_choice != "Auto-detect" else "auto"
        with st.spinner("Parsing race, runner profiles and past starts…"):
            card = parse_race(raw, chosen)
            features = build_feature_frame(card) if card.runners else pd.DataFrame()
        st.session_state.card = card
        st.session_state.features = features
        if features.empty:
            st.error("No active runners could be scored. Check the parser warnings below.")
        else:
            odds = features[["tab", "runner", "market_odds"]].copy()
            try:
                pred, model_name, warning, overround = _recalculate(
                    card, features, odds, st.session_state.artifact,
                    devig_method.lower(), exchange_commission_pct / 100.0,
                )
                st.session_state.prediction = pred
                st.session_state.model_name = model_name
                st.session_state.model_warning = warning
                st.session_state.overround = overround
                st.success(f"Parsed {card.race.get('field_size', len(features))} active runners and {len(card.histories)} historical starts.")
            except Exception as exc:
                st.session_state.prediction = None
                st.warning(f"Race parsed, but EV is unavailable until every active runner has valid decimal odds: {exc}")
        for warning in card.warnings:
            st.warning(warning)

    card = st.session_state.card
    if card:
        r = card.race
        st.markdown("### Current parsed race")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Discipline", str(r.get("discipline", "—")).title())
        m2.metric("Track / race", f"{r.get('track', '—')} R{r.get('race_no', '—')}")
        m3.metric("Distance", r.get("distance_text", "—"))
        m4.metric("Active field", r.get("field_size", "—"))
        st.write(f"**{r.get('race_name', 'Unnamed race')}** · {r.get('date_text', '')} · {r.get('surface', '')} {r.get('going', '')}")

with tabs[1]:
    card = st.session_state.card
    features = st.session_state.features
    if card is None or features is None or features.empty:
        st.info("Parse a race in the Input tab first.")
    else:
        st.subheader("Independent win probabilities versus current odds")
        odds_default = features[["tab", "runner", "market_odds"]].copy()
        st.caption("Verify the detected odds and replace them with the same bookmaker/exchange snapshot for every runner. Mixed timestamps or mixed bookmakers distort EV.")
        odds_edited = st.data_editor(
            odds_default,
            hide_index=True,
            use_container_width=True,
            disabled=["tab", "runner"],
            column_config={
                "tab": st.column_config.NumberColumn("No.", format="%d"),
                "runner": st.column_config.TextColumn("Runner"),
                "market_odds": st.column_config.NumberColumn("Current decimal odds", min_value=1.01, step=0.05, format="%.2f"),
            },
            key=f"odds_{_race_id(card)}",
        )
        if st.button("Recalculate EV with edited odds", type="primary"):
            try:
                pred, model_name, warning, overround = _recalculate(
                    card, features, odds_edited, st.session_state.artifact,
                    devig_method.lower(), exchange_commission_pct / 100.0,
                )
                st.session_state.prediction = pred
                st.session_state.model_name = model_name
                st.session_state.model_warning = warning
                st.session_state.overround = overround
            except Exception as exc:
                st.error(str(exc))

        prediction = st.session_state.prediction
        if prediction is not None:
            top = prediction.sort_values("win_probability", ascending=False).iloc[0]
            best_ev = prediction.sort_values("ev_per_unit", ascending=False).iloc[0]
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Model top pick", top["runner"])
            c2.metric("Top-pick win chance", f"{top['win_probability']:.1%}")
            c3.metric("Detected market overround", f"{100 * float(st.session_state.overround or 0):.1f}%")
            c4.metric("Highest model EV", f"{best_ev['runner']} · {best_ev['ev_pct']:+.1f}%")
            st.caption(f"Model: {st.session_state.model_name}")
            if st.session_state.model_warning:
                st.warning(st.session_state.model_warning)

            _display_prediction_table(prediction)

            chart = prediction.sort_values("win_probability", ascending=False).set_index("runner")[["win_probability", "market_probability_fair"]]
            chart.columns = ["Model win probability", "No-vig market probability"]
            st.bar_chart(chart)

            candidates = prediction[(prediction["ev_per_unit"] > 0) & (prediction["probability_edge"] > 0)].copy()
            st.markdown("### Positive-EV candidates under the entered odds")
            if candidates.empty:
                st.info("The model does not identify a positive-EV runner at these prices.")
            else:
                st.dataframe(
                    candidates[["runner", "market_odds", "win_probability", "fair_odds", "probability_edge", "ev_pct", "value_grade"]],
                    hide_index=True,
                    use_container_width=True,
                )
            st.caption("EV per unit = model probability × net decimal odds − 1. A positive estimate can still lose; it is not proof of a profitable edge.")

with tabs[2]:
    card = st.session_state.card
    if card is None:
        st.info("Parse a race first.")
    else:
        st.subheader("What the parser extracted")
        race_df = pd.DataFrame([card.race])
        runner_df = _serialisable_runner_frame(card)
        history_df = pd.DataFrame(card.histories)
        trial_df = pd.DataFrame(card.trials)
        t1, t2, t3, t4 = st.tabs(["Race", "Runners", "Historical starts", "Trials"])
        with t1:
            st.dataframe(race_df, hide_index=True, use_container_width=True)
        with t2:
            st.dataframe(runner_df, hide_index=True, use_container_width=True)
        with t3:
            st.dataframe(history_df, hide_index=True, use_container_width=True)
        with t4:
            st.dataframe(trial_df, hide_index=True, use_container_width=True)

        payload = {
            "race": _json_safe(card.race),
            "runners": _json_safe(card.runners),
            "histories": _json_safe(card.histories),
            "trials": _json_safe(card.trials),
            "warnings": card.warnings,
        }
        d1, d2, d3, d4 = st.columns(4)
        d1.download_button("Race CSV", race_df.to_csv(index=False), f"{_race_id(card)}_race.csv", "text/csv")
        d2.download_button("Runners CSV", runner_df.to_csv(index=False), f"{_race_id(card)}_runners.csv", "text/csv")
        d3.download_button("Starts CSV", history_df.to_csv(index=False), f"{_race_id(card)}_starts.csv", "text/csv")
        d4.download_button("Complete JSON", json.dumps(payload, indent=2), f"{_race_id(card)}.json", "application/json")

        st.markdown("### Runner explanations")
        pred = st.session_state.prediction
        if pred is not None:
            for _, row in pred.sort_values("model_rank").iterrows():
                with st.expander(f"#{int(row['model_rank'])} · {row['runner']} · {row['win_probability']:.1%}"):
                    st.write(f"Data quality: **{row['data_quality']:.0%}** · History used: **{int(row['history_count'])} starts**")
                    parts = explain_runner(row, 6)
                    explanation_rows = []
                    for feature, contribution in parts:
                        explanation_rows.append({
                            "Factor": feature.replace("_", " ").title(),
                            "Direction": "Supports" if contribution > 0 else "Opposes",
                            "Contribution": contribution,
                        })
                    st.dataframe(pd.DataFrame(explanation_rows), hide_index=True, use_container_width=True)

with tabs[3]:
    st.subheader("Batch parser and canonical dataset builder")
    st.write("Upload several saved Enhanced Form pages. The app will build consistent race, runner-feature and historical-start tables across all three disciplines. Train separate probability models for thoroughbred, harness and greyhound racing.")
    batch_files = st.file_uploader("Upload multiple .md/.txt race pages", type=["md", "txt"], accept_multiple_files=True, key="batch_upload")
    if batch_files and st.button("Build combined dataset"):
        race_rows: list[dict] = []
        feature_frames: list[pd.DataFrame] = []
        history_rows: list[dict] = []
        warnings: list[str] = []
        for file in batch_files:
            raw = file.getvalue().decode("utf-8", errors="replace")
            card_i = parse_race(raw)
            rid = _race_id(card_i)
            race_row = {"race_id": rid, **card_i.race, "source_file": file.name}
            race_rows.append(race_row)
            if card_i.runners:
                f = build_feature_frame(card_i)
                f.insert(0, "race_id", rid)
                f.insert(1, "race_date", card_i.race.get("date"))
                f["source_file"] = file.name
                feature_frames.append(f)
            for run in card_i.histories:
                history_rows.append({"race_id_context": rid, "source_file": file.name, **run})
            warnings.extend([f"{file.name}: {w}" for w in card_i.warnings])
        races = pd.DataFrame(race_rows)
        features_all = pd.concat(feature_frames, ignore_index=True) if feature_frames else pd.DataFrame()
        histories = pd.DataFrame(history_rows)
        st.success(f"Processed {len(races)} race files, {len(features_all)} active runners and {len(histories)} historical starts.")
        st.dataframe(features_all.head(200), hide_index=True, use_container_width=True)
        b1, b2, b3 = st.columns(3)
        b1.download_button("Download races.csv", races.to_csv(index=False), "races.csv", "text/csv")
        b2.download_button("Download runner_features.csv", features_all.to_csv(index=False), "runner_features.csv", "text/csv")
        b3.download_button("Download historical_starts.csv", histories.to_csv(index=False), "historical_starts.csv", "text/csv")
        for warning in warnings:
            st.warning(warning)

with tabs[4]:
    st.subheader("Build and calibrate a trained model")
    st.warning("Train one discipline at a time. Do not train on current bookmaker odds, current neural ratings, or any information published after the race. Those create leakage and make backtests look unrealistically strong.")
    card = st.session_state.card
    features = st.session_state.features
    if card is not None and features is not None and not features.empty:
        winner = st.selectbox("After the race, select the actual winner to create labelled rows", ["— not known yet —"] + features["runner"].tolist())
        chosen_winner = None if winner.startswith("—") else winner
        rows = _make_training_rows(card, features, chosen_winner)
        st.download_button(
            "Download this race's labelled feature rows",
            rows.to_csv(index=False),
            f"{_race_id(card)}_training_rows.csv",
            "text/csv",
            disabled=chosen_winner is None,
        )
        template = _make_training_rows(card, features, None).head(0)
        st.download_button("Download training schema", template.to_csv(index=False), "racing_ev_training_schema.csv", "text/csv")

    st.markdown("#### Train from accumulated races")
    training_file = st.file_uploader("Upload combined labelled runner-feature CSV", type=["csv"], key="training_csv")
    if training_file is not None and st.button("Train with chronological holdout", type="primary"):
        try:
            data = pd.read_csv(training_file)
            with st.spinner("Fitting logistic + gradient-boosting ensemble and selecting calibration temperature…"):
                trained = train_models(data)
            st.session_state.artifact = trained.artifact
            st.session_state.artifact_bytes = artifact_bytes(trained.artifact)
            st.session_state.training_metrics = trained.metrics
            st.success("Model trained and loaded for the current session.")
        except Exception as exc:
            st.error(str(exc))
    if st.session_state.training_metrics:
        metrics = st.session_state.training_metrics
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Training races", int(metrics["training_races"]))
        m2.metric("Validation races", int(metrics["validation_races"]))
        m3.metric("Validation top-1", f"{metrics['validation_top1']:.1%}")
        m4.metric("Race log loss", f"{metrics['validation_log_loss']:.3f}")
        st.download_button(
            "Download trained .joblib model",
            st.session_state.artifact_bytes,
            "racing_ev_model.joblib",
            "application/octet-stream",
        )

with tabs[5]:
    st.subheader("How the system should be used")
    st.markdown(
        r"""
### Canonical data layers

1. **Race context:** discipline, country, date/time, track, race number/type/class, distance, surface/going, prize and field size.
2. **Current runner snapshot:** runner number, weight, barrier/box/handicap, rider/driver, trainer, strike rates, career/course/distance/going records, days since run, current gear and scratch status.
3. **Historical starts:** date, track, distance, class, surface/going, finish, field size, beaten margin, speed/time/sectional measures, draw/box, carried weight, rider/driver, trouble notes and historical SP.
4. **Odds snapshots:** bookmaker/exchange, timestamp, decimal odds and commission. Keep them outside the independent form model.
5. **Outcome labels:** official finishing position, winner flag and scratch/non-starter status.

### Probability workflow

The built-in baseline creates discipline-specific form components, standardises them **within the current race**, combines them into a latent performance score and applies a softmax so all active runners sum to 100%. Sparse profiles are shrunk toward an equal-chance prior.

For a credible production model, accumulate many completed races and train chronologically. Keep every runner from the same race in the same fold. Calibrate on races later than the training period, then test on a still-later untouched period.

### Market and EV

For decimal odds \(O_i\), raw implied probability is \(q_i=1/O_i\). The app removes the market margin across the complete field, calculates independent model probability \(p_i\), then reports:

- **Probability edge:** \(p_i-p_{market,i}\)
- **Model fair odds:** \(1/p_i\)
- **Expected value per unit:** \(EV_i=p_i\times O_{net,i}-1\)

A positive EV estimate is meaningful only when probabilities are calibrated out of time and the odds snapshot is realistic and timestamped. Evaluate log loss, Brier score, calibration, top-pick strike rate, return on turnover and drawdown—not strike rate alone.
"""
    )
    st.error("Research tool only. Racing outcomes are uncertain, odds move, and model error can turn apparent value into a loss. Never treat estimated EV as guaranteed profit.")
