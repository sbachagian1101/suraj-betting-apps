"""Mauritius Gallops -- score a Champ de Mars race from its training sheet.

Pages: **Predict** (paste the GALLOPS text, score, explain, save),
**History** (stored races, model report, fit weights) and **Method**.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd
import streamlit as st

import gallops as g

st.set_page_config(page_title="Mauritius Gallops", page_icon=":material/directions_run:", layout="wide")

HERE = Path(__file__).resolve().parent
DATA = Path(os.environ.get("GALLOPS_DATA", HERE / "data"))
DATA.mkdir(parents=True, exist_ok=True)
STORE = DATA / "races.jsonl"
WEIGHTS_FILE = DATA / "weights.json"
SAMPLE = HERE / "data" / "sample_06sep2026.txt"

SHOW = ["horse", "rank", "win_pct", "score", "n_gallops", "gallops_14d", "days_since",
        "workload", "best_adj600", "last_adj600", "taper", "std600", "trend600",
        "finish_ratio", "lib_share", "beat_n", "beaten_n", "missing_600"]


def load_weights() -> dict:
    if WEIGHTS_FILE.exists():
        try:
            return json.loads(WEIGHTS_FILE.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            pass
    return dict(g.DEFAULT_WEIGHTS)


def style_finish(df: pd.DataFrame):
    if "finish" not in df:
        return df
    colours = {1: "background-color:#a9e2be;color:#0b3d22;font-weight:700",
               2: "background-color:#d5efdd", 3: "background-color:#eef7f1"}

    def row_style(row):
        s = colours.get(int(row.get("finish", 0)), "")
        return [s] * len(row)
    return df.style.apply(row_style, axis=1).format(precision=2)


page = st.sidebar.radio("Page", ["Predict", "History", "Method"])
weights = load_weights()

if page == "Predict":
    st.title("Mauritius Gallops")
    st.caption("Paste one or more `GALLOPS for Race N [...]` blocks. Add an "
               "`Actual results: A - B - C` line under a race to store it for fitting.")
    default = SAMPLE.read_text(encoding="utf-8") if SAMPLE.exists() else ""
    text = st.text_area("Gallop sheet", value=st.session_state.get("text", default), height=320)
    st.session_state["text"] = text
    c1, c2, c3 = st.columns(3)
    run = c1.button("Score races", type="primary")
    save = c2.button("Save races with results")
    temp = c3.slider("Win % temperature", 0.5, 3.0, 1.0, 0.1,
                     help="Higher spreads the win % more evenly across the field.")

    races = []
    if text.strip():
        try:
            races = g.parse_races(text)
        except ValueError as exc:
            st.error(str(exc))
    if save:
        with_results = [r for r in races if r.results]
        if not with_results:
            st.warning("No race in the paste has an `Actual results:` line.")
        else:
            n = g.save_races(with_results, STORE)
            st.success(f"Stored {len(with_results)} race(s); {n} in the history file.")
    if run or races:
        if not races:
            st.warning("No `GALLOPS for Race N` header found.")
        for race in races:
            st.subheader(f"Race {race.race_no} - {race.race_date:%A %d %B %Y}")
            df = g.score_race(race, weights, temperature=temp)
            cols = [c for c in SHOW if c in df] + (["finish"] if "finish" in df else [])
            st.dataframe(style_finish(df[cols].round(2)), width="stretch", hide_index=True)
            top = df.iloc[0]
            st.markdown(f"**Top pick:** {top['horse']} ({top['win_pct']:.0f}%). "
                        f"Place zone: {', '.join(df['horse'].iloc[:3])}.")
            with st.expander("Why - contribution of each feature (weight x within-race z-score)"):
                st.dataframe(g.explain(race, weights).round(2), width="stretch", hide_index=True)
            with st.expander("Parsed gallops"):
                rows = [{"horse": r.name, "date": gl.day, "jockey": gl.jockey, "600m": gl.t600,
                         "200m": gl.t200, "first 400": gl.first400, "finish ratio": gl.finish_ratio,
                         "equip": gl.equipment, "comment": gl.comment}
                        for r in race.runners for gl in r.gallops]
                st.dataframe(pd.DataFrame(rows).round(2), width="stretch", hide_index=True)

elif page == "History":
    st.title("Stored races")
    st.caption("On Streamlit Cloud the disk is wiped on every restart, so download the history "
               "file after saving races and upload it again next session.")
    up = st.file_uploader("Upload a saved history file (races.jsonl)", type=["jsonl", "json", "txt"])
    if up is not None:
        try:
            uploaded = [g.Race.from_json(json.loads(line)) for line in
                        up.getvalue().decode("utf-8").splitlines() if line.strip()]
            n = g.save_races(uploaded, STORE)
            st.success(f"Restored {len(uploaded)} race(s); {n} in the history file.")
        except (ValueError, KeyError) as exc:
            st.error(f"Could not read that file: {exc}")
    races = g.load_races(STORE)
    st.write(f"{len(races)} race(s) with results in `{STORE.name}`.")
    if STORE.exists():
        st.download_button("Download history file", STORE.read_bytes(), file_name="races.jsonl",
                           mime="application/jsonl")
    if races:
        rep = g.evaluate(races, weights)
        c = st.columns(4)
        c[0].metric("Races", rep["races"])
        c[1].metric("Winner log-loss", f"{rep['log_loss']:.3f}")
        c[2].metric("Top pick won", f"{100 * rep['winner_hit']:.0f}%")
        c[3].metric("Top pick placed", f"{100 * rep['top_pick_placed']:.0f}%")
        base = g.evaluate(races, {})  # equal scores -> uniform win %
        st.caption(f"A no-skill model (equal chances) scores log-loss {base['log_loss']:.3f} on the same races. "
                   "Lower is better.")
        rows = []
        for race in races:
            df = g.score_race(race, weights)
            w = df[df["finish"] == 1]
            rows.append({"date": race.race_date, "race": race.race_no, "runners": len(df),
                         "top pick": df.iloc[0]["horse"], "top pick finish": int(df.iloc[0]["finish"]),
                         "winner": w.iloc[0]["horse"] if len(w) else "?",
                         "winner rank": int(w.iloc[0]["rank"]) if len(w) else None})
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

        st.subheader("Weights")
        st.json(weights)
        c1, c2 = st.columns(2)
        if c1.button("Fit weights on stored races"):
            new_w, info = g.fit_weights(races, weights)
            if info["fitted"]:
                WEIGHTS_FILE.write_text(json.dumps(new_w, indent=2), encoding="utf-8")
                st.success(f"Fitted on {info['races']} races, log-loss {info['log_loss']:.3f}. Saved.")
                st.download_button("Download fitted weights", json.dumps(new_w, indent=2),
                                   file_name="weights.json", mime="application/json")
            else:
                st.warning(f"Not fitted: {info['reason']} (have {info['races']}).")
        if c2.button("Reset to default weights"):
            if WEIGHTS_FILE.exists():
                WEIGHTS_FILE.unlink()
            st.rerun()

else:
    st.title("Method")
    st.markdown((HERE / "README.md").read_text(encoding="utf-8"))
