"""Bet365 Predictor — the desktop engine on the web."""
from __future__ import annotations

import copy
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import streamlit as st

st.set_page_config(page_title="Bet365 Predictor",
                   page_icon=":material/trophy:", layout="wide")

# Streamlit Cloud pulls new files but keeps imported modules in sys.modules, so
# a deploy can run new app code against stale helpers. Fail loudly with the fix.
try:
    import bet365_model as MD
    import bet365_parser as PS
    import results_parser as RP
    import storage as ST
    for _m, _a in [(PS, "parse_bet365_text"), (MD, "predict_race"),
                   (MD, "train_model"), (MD, "training_record_from_race"),
                   (MD, "record_identifier"), (MD, "evaluate_records"),
                   (RP, "parse_multi_meeting_results"),
                   (RP, "validate_result_for_race"), (ST, "load_model_state")]:
        if not hasattr(_m, _a):
            raise ImportError(f"{_m.__name__} is missing {_a}")
except ImportError as exc:
    st.error(f"**Stale modules** ({exc}).\n\nOn Streamlit Cloud: "
             "**Manage app → ⋮ → Reboot app**. A rerun is not enough.",
             icon=":material/error:")
    st.stop()

DATA = Path("data")
# Two bundled cards. The first is the developer's original five meetings, and
# it is the one `seed_results.txt` belongs to — so the full loop, card in,
# results in, retrain, works out of the box. The second is a newer four-meeting
# card with no matching results.
SAMPLE_WITH_RESULTS = Path("samples/Bet365_predictions_sample.txt")
SAMPLE_RECENT = Path("sample_data/races_bet365.txt")
SAMPLE_RESULTS = Path("seed_results.txt")

CSS = """
<style>
.b3-row{display:flex;gap:12px;margin:4px 0 14px 0;flex-wrap:wrap}
.b3-card{flex:1 1 150px;border-radius:14px;padding:13px 15px;color:#fff;
  box-shadow:0 5px 16px rgba(0,0,0,.15)}
.b3-card .lab{font-size:.70rem;letter-spacing:.10em;text-transform:uppercase;
  opacity:.92}
.b3-card .val{font-size:1.75rem;font-weight:700;line-height:1.15;margin-top:2px}
.b3-card .sub{font-size:.75rem;opacity:.92}
.b3-a{background:linear-gradient(135deg,#14532d,#2f9e44)}
.b3-b{background:linear-gradient(135deg,#1e3a5f,#3b82c4)}
.b3-c{background:linear-gradient(135deg,#5b3a1e,#c2822f)}
.b3-d{background:linear-gradient(135deg,#3f2a56,#7c5bab)}
</style>
"""


def cards(items):
    html = "<div class='b3-row'>"
    for lab, val, sub, cls in items:
        html += (f"<div class='b3-card {cls}'><div class='lab'>{lab}</div>"
                 f"<div class='val'>{val}</div><div class='sub'>{sub}</div></div>")
    st.markdown(html + "</div>", unsafe_allow_html=True)



def _signal_text(signals, limit: int = 6) -> str:
    """Render the model's (label, weight) signal pairs.

    They arrive as tuples, so joining them as if they were strings raises
    `sequence item 0: expected str instance, tuple found`.
    """
    out = []
    for item in (signals or [])[:limit]:
        if isinstance(item, (list, tuple)) and len(item) >= 2:
            label, weight = item[0], item[1]
            try:
                out.append(f"{label} ({float(weight):+.2f})")
            except (TypeError, ValueError):
                out.append(str(label))
        else:
            out.append(str(item))
    return ", ".join(out)


@st.cache_data(show_spinner=False)
def parse_text(text: str):
    races = PS.parse_bet365_text(text, "pasted_text")
    warn = []
    for r in races:
        warn.extend(r.get("warnings", []) or [])
    return races, warn


@st.cache_resource
def baseline_state():
    """The trained state shipped with the repo."""
    return ST.load_model_state(DATA), ST.load_training_store(DATA), \
        ST.load_training_history(DATA)


def current_state():
    """Session state wins over the shipped baseline once retraining happens."""
    if "b3_state" not in st.session_state:
        s, store, hist = baseline_state()
        st.session_state["b3_state"] = copy.deepcopy(s)
        st.session_state["b3_store"] = copy.deepcopy(store)
        st.session_state["b3_hist"] = copy.deepcopy(hist)
        st.session_state["b3_trained_here"] = False
    return (st.session_state["b3_state"], st.session_state["b3_store"],
            st.session_state["b3_hist"])


st.markdown(CSS, unsafe_allow_html=True)
state, store, history = current_state()

# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.title("Bet365 Predictor")
    st.caption("Paste a Bet365 race-card export covering any number of "
               "meetings. Prices are never used as a predictive feature.")
    st.divider()
    sims = st.select_slider("Finishing-position simulations",
                            [1000, 3000, 6000, 12000, 25000], value=6000)
    st.divider()
    st.metric("Training races", state.get("training_races", 0))
    st.metric("Pairwise relationships", f"{state.get('training_pairs', 0):,}")
    st.metric("Learned-layer influence",
              f"{100 * state.get('learned_influence', 0.0):.1f}%")
    st.caption("Odds influence is **0%** by design — the odds firewall keeps "
               "current and historical prices out of every feature.")
    st.divider()
    st.markdown("**Trained state**")
    st.download_button(
        "Download trained state (JSON)",
        json.dumps({"model_state": state, "training_store": store,
                    "training_history": history}, indent=1),
        file_name=f"bet365_state_{datetime.now(timezone.utc):%Y%m%d_%H%M}.json",
        mime="application/json", width="stretch")
    up_state = st.file_uploader("Restore a saved state", type=["json"],
                                key="b3_restore")
    if up_state is not None and st.button("Restore", width="stretch"):
        try:
            blob = json.loads(up_state.getvalue().decode("utf-8"))
            st.session_state["b3_state"] = blob["model_state"]
            st.session_state["b3_store"] = blob.get("training_store", [])
            st.session_state["b3_hist"] = blob.get("training_history", [])
            st.session_state["b3_trained_here"] = True
            st.success("State restored. Re-run the predictions.")
            st.rerun()
        except Exception as exc:                              # noqa: BLE001
            st.error(f"Could not read that file — {type(exc).__name__}: {exc}")

input_tab, parsed_tab, pred_tab, results_tab, diag_tab = st.tabs(
    ["1 · Input", "2 · Parsed meetings", "3 · Race predictions",
     "4 · Results & training", "5 · Model & diagnostics"])

# -------------------------------------------------------------------- input
with input_tab:
    st.warning(
        "**Retraining lives in this browser session only.** Streamlit Cloud "
        "gives every app a fresh, empty disk on each restart, so the desktop "
        "version's automatic `data/` persistence cannot work here. The trained "
        "state shipped with the app is the starting point every time; anything "
        "you teach it afterwards is kept in the session and is lost on reload "
        "unless you **download the trained state** from the sidebar and "
        "restore it next time.",
        icon=":material/cloud_off:")

    c1, c2, c3 = st.columns(3)
    if c1.button("Load 5-meeting card (has results)", width="stretch"):
        st.session_state["b3_raw"] = SAMPLE_WITH_RESULTS.read_text(
            encoding="utf-8")
        st.session_state.pop("b3_pred", None)
        st.session_state.pop("b3_res", None)
    if c2.button("Load 4-meeting card", width="stretch"):
        st.session_state["b3_raw"] = SAMPLE_RECENT.read_text(encoding="utf-8")
        st.session_state.pop("b3_pred", None)
        st.session_state.pop("b3_res", None)
    if c3.button("Clear", width="stretch"):
        st.session_state["b3_raw"] = ""
        st.session_state.pop("b3_pred", None)
        st.session_state.pop("b3_res", None)

    up = st.file_uploader("…or load a .txt race card", type=["txt"])
    if up is not None:
        st.session_state["b3_raw"] = up.getvalue().decode("utf-8", "ignore")
        st.session_state.pop("b3_pred", None)

    st.text_area("Bet365 race-card text — any number of meetings",
                 key="b3_raw", height=300,
                 placeholder="Paste the complete Bet365 text export here.")
    st.caption("The parser reads meeting, date, race title, distance, going, "
               "rail, discipline, runners, scratchings, form sequence, "
               "trainer, jockey, age, sex, weight, barrier, the Bet365 "
               "overview and suggested play, and each runner's recent "
               "factual run descriptions.")

raw = st.session_state.get("b3_raw", "")
races: list = []
if raw.strip():
    try:
        races, parse_warnings = parse_text(raw)
    except Exception as exc:                                  # noqa: BLE001
        with input_tab:
            st.error(f"The parser failed — {type(exc).__name__}: {exc}")
        races, parse_warnings = [], []
else:
    parse_warnings = []

meetings: dict = {}
for r in races:
    meetings.setdefault(r.get("meeting", "Unknown"), []).append(r)

with input_tab:
    if raw.strip():
        if races:
            st.success(
                f"Parsed **{len(races)} races** across "
                f"**{len(meetings)} meetings** — "
                + ", ".join(f"{k} ({len(v)})" for k, v in meetings.items()),
                icon=":material/check_circle:")
            if st.button("Predict all races", type="primary", width="stretch"):
                out = {}
                bar = st.progress(0.0, "Scoring…")
                for i, race in enumerate(races, 1):
                    try:
                        out[MD.record_identifier(race)] = MD.predict_race(
                            race, state, simulations=int(sims))
                    except Exception as exc:                  # noqa: BLE001
                        out[MD.record_identifier(race)] = {"error": str(exc)}
                    bar.progress(i / len(races), f"Scoring… {i}/{len(races)}")
                bar.empty()
                st.session_state["b3_pred"] = out
                st.success(f"{len(out)} races scored — open the "
                           "**Race predictions** tab.",
                           icon=":material/check_circle:")
        else:
            st.error("No races were read. The text should be a Bet365 "
                     "race-card export with meeting headers and runner rows.",
                     icon=":material/error:")

if not races:
    for t in (parsed_tab, pred_tab, results_tab):
        with t:
            st.info("Load or paste a race card on the first tab to begin.",
                    icon=":material/content_paste:")

# ----------------------------------------------------------- parsed meetings
if races:
    with parsed_tab:
        st.subheader(f"{len(races)} races · {len(meetings)} meetings")
        if parse_warnings:
            with st.expander(f"{len(parse_warnings)} parser warning(s)"):
                for w in parse_warnings[:200]:
                    st.write("• ", w)
        mt = st.tabs(list(meetings))
        for tab, (name, rs) in zip(mt, meetings.items()):
            with tab:
                st.dataframe(pd.DataFrame([{
                    "Race": r.get("race_no"), "Title": r.get("title"),
                    "Distance": r.get("distance_m"), "Going": r.get("going"),
                    "Rail": r.get("rail"), "Discipline": r.get("discipline"),
                    "Active": r.get("active_field_size"),
                    "Declared": r.get("declared_field_size"),
                    "Scratched": sum(1 for x in r.get("runners", [])
                                     if x.get("status") == "SCRATCHED"),
                } for r in rs]), width="stretch", hide_index=True)

                labels = [f"R{r.get('race_no')} · {r.get('title', '')[:40]}"
                          for r in rs]
                choice = st.selectbox("Race", labels, key=f"parse_{name}")
                race = rs[labels.index(choice)]
                st.dataframe(pd.DataFrame([{
                    "No": ru.get("number"), "Horse": ru.get("horse"),
                    "Form": ru.get("form"), "Barrier": ru.get("barrier"),
                    "Weight": ru.get("weight"), "Age": ru.get("age"),
                    "Sex": ru.get("sex"), "Trainer": ru.get("trainer"),
                    "Jockey": ru.get("jockey"),
                    "Suggested play": ru.get("suggested_play"),
                    "Status": ru.get("status"),
                    "Past runs": len(ru.get("historical_runs", []) or []),
                } for ru in race.get("runners", [])]),
                    width="stretch", hide_index=True)
                if race.get("overview"):
                    with st.expander("Bet365 overview"):
                        st.write(race["overview"])

# --------------------------------------------------------- race predictions
preds = st.session_state.get("b3_pred") or {}
if races:
    with pred_tab:
        if not preds:
            st.info("Press **Predict all races** on the Input tab.",
                    icon=":material/insights:")
        else:
            mt = st.tabs(list(meetings))
            for tab, (name, rs) in zip(mt, meetings.items()):
                with tab:
                    labels = [f"R{r.get('race_no')} · {r.get('title', '')[:40]}"
                              for r in rs]
                    choice = st.selectbox("Race", labels, key=f"pred_{name}")
                    race = rs[labels.index(choice)]
                    p = preds.get(MD.record_identifier(race))
                    if not p:
                        st.info("Not scored yet.")
                        continue
                    if "error" in p:
                        st.error(f"This race could not be scored — {p['error']}")
                        continue
                    rows = p["rows"]
                    top = rows[0]
                    cards([
                        ("Predicted order", p["order_text"][:26],
                         f"{len(rows)} active runners", "b3-b"),
                        ("Top selection", f"#{top['number']} {top['horse'][:16]}",
                         f"{100*top['win_probability']:.1f}% win", "b3-a"),
                        ("Confidence", f"{p['overall_confidence']:.1f} / 9",
                         "race-level", "b3-c"),
                        ("Learned layer",
                         f"{100*p['learned_influence']:.1f}%",
                         f"odds {100*p['odds_influence']:.0f}%", "b3-d"),
                    ])
                    st.caption(
                        f"{p['meeting']} R{p['race_no']} · {p['title']} · "
                        f"{p['distance_m']}m · {p['discipline']} — specialist "
                        "weights " + ", ".join(
                            f"{k} {100*v:.0f}%"
                            for k, v in p["specialist_weights"].items()))

                    view = st.radio(
                        "View", ["Final trained", "Model agreement",
                                 "Specialist ranks", "Horse explanations"],
                        horizontal=True, key=f"view_{name}")

                    if view == "Final trained":
                        st.dataframe(pd.DataFrame([{
                            "Pos": r["rank"], "No": r["number"],
                            "Horse": r["horse"],
                            "Win %": 100 * r["win_probability"],
                            "Top-3 %": 100 * r["top3_probability"],
                            "Exp. rank": r["expected_rank"],
                            "Conf": r["confidence"],
                            "Fair odds": r["fair_odds"],
                            "Class": r["classification"],
                            "Jockey": r.get("jockey"),
                        } for r in rows]), width="stretch", hide_index=True,
                            column_config={
                                "Win %": st.column_config.ProgressColumn(
                                    "Win %", format="%.1f%%", min_value=0.0,
                                    max_value=float(max(
                                        100 * r["win_probability"]
                                        for r in rows))),
                                "Top-3 %": st.column_config.NumberColumn(
                                    format="%.1f%%"),
                                "Exp. rank": st.column_config.NumberColumn(
                                    format="%.2f"),
                                "Fair odds": st.column_config.NumberColumn(
                                    format="$%.2f")})
                    elif view == "Model agreement":
                        st.dataframe(pd.DataFrame([{
                            "No": r["number"], "Horse": r["horse"],
                            "Agreement %": 100 * r["model_agreement"],
                            "Bet365 analyst": r["source_rank"],
                            "Independent form": r["form_rank"],
                            "Suitability": r["suitability_rank"],
                            "Pace / draw / weight": r["pace_rank"],
                            "Final": r["rank"],
                        } for r in rows]), width="stretch", hide_index=True,
                            column_config={"Agreement %":
                                           st.column_config.NumberColumn(
                                               format="%.0f%%")})
                        st.caption("Lower is better in every rank column. "
                                   "Agreement is how closely the four "
                                   "specialists concur about that runner.")
                    elif view == "Specialist ranks":
                        st.dataframe(pd.DataFrame([{
                            "No": r["number"], "Horse": r["horse"],
                            "Analyst %": 100 * r["source_probability"],
                            "Form %": 100 * r["form_probability"],
                            "Suitability %": 100 * r["suitability_probability"],
                            "Pace %": 100 * r["pace_probability"],
                            "Baseline": r["baseline_score"],
                            "Learned": r["learned_score"],
                        } for r in rows]), width="stretch", hide_index=True,
                            column_config={c: st.column_config.NumberColumn(
                                c, format="%.1f%%") for c in
                                ("Analyst %", "Form %", "Suitability %",
                                 "Pace %")})
                    else:
                        for r in rows:
                            with st.expander(
                                    f"{r['rank']}. #{r['number']} {r['horse']} "
                                    f"— {100*r['win_probability']:.1f}% win"):
                                st.write(r["explanation"])
                                # each signal is a (label, weight) pair, not
                                # a string — joining them raw raises
                                # "expected str instance, tuple found"
                                pos = _signal_text(r.get("positive_signals"))
                                neg = _signal_text(r.get("negative_signals"))
                                if pos:
                                    st.success("Positive — " + pos)
                                if neg:
                                    st.warning("Negative — " + neg)

# ------------------------------------------------------ results & training
if races:
    with results_tab:
        flash = st.session_state.pop("b3_flash", None)
        if flash:
            st.success(flash, icon=":material/model_training:")
        st.subheader("Enter actual results and retrain")
        st.caption(
            "Accepts the raw Bet365 result style (`1-5-4-3Fixed OddsQuaddie`) "
            "and the simplified style (`R1: 1-5-4-3`). Partial results are "
            "valid, and a dead heat is written with a comma — `R7: 6-3-4,5`.")
        # The bundled seed_results.txt belongs to the developer's original
        # five meetings, which are not the meetings in this race card - loading
        # it would map to nothing. A template built from what is actually
        # parsed always matches.
        rc1, rc2 = st.columns(2)
        if rc1.button("Load the bundled results", width="stretch",
                      help="These belong to the 5-meeting card. They will map "
                           "to nothing if a different card is loaded."):
            st.session_state["b3_res"] = SAMPLE_RESULTS.read_text(
                encoding="utf-8")
        if rc2.button("Insert a template for the meetings loaded",
                      width="stretch"):
            lines = []
            for mname, rs in meetings.items():
                lines.append(mname)
                for r in rs:
                    lines.append(f"R{r.get('race_no')}: ")
                lines.append("")
            st.session_state["b3_res"] = "\n".join(lines)

        st.text_area("Results text", key="b3_res", height=220,
                     placeholder="Meeting name, then one result line per race.")

        res_raw = st.session_state.get("b3_res", "")
        if res_raw.strip():
            try:
                mapping = RP.parse_multi_meeting_results(res_raw,
                                                         list(meetings))
            except Exception as exc:                          # noqa: BLE001
                st.error(f"Could not read those results — "
                         f"{type(exc).__name__}: {exc}")
                mapping = {}

            rows, valid = [], []
            for mname, per_race in (mapping or {}).items():
                for rno, result in per_race.items():
                    race = next((r for r in meetings.get(mname, [])
                                 if r.get("race_no") == rno), None)
                    if race is None:
                        rows.append({"Meeting": mname, "Race": rno,
                                     "Result": RP.result_display(result),
                                     "Status": "no matching race parsed"})
                        continue
                    ok, msg = RP.validate_result_for_race(race, result)
                    rows.append({"Meeting": mname, "Race": rno,
                                 "Result": RP.result_display(result),
                                 "Status": "ok" if ok else msg})
                    if ok:
                        valid.append((race, result))
            if rows:
                st.dataframe(pd.DataFrame(rows), width="stretch",
                             hide_index=True)
            st.caption(f"{len(valid)} of {len(rows)} result line(s) map cleanly "
                       "onto a parsed race.")

            if valid and st.button(f"Save {len(valid)} result(s) and retrain",
                                   type="primary", width="stretch"):
                new_store = list(store)
                index = {r.get("race_id"): i
                         for i, r in enumerate(new_store) if r.get("race_id")}
                added = replaced = 0
                for race, result in valid:
                    rec = MD.training_record_from_race(race, result)
                    rid = rec.get("race_id") or MD.record_identifier(race)
                    rec["race_id"] = rid
                    if rid in index:
                        new_store[index[rid]] = rec
                        replaced += 1
                    else:
                        index[rid] = len(new_store)
                        new_store.append(rec)
                        added += 1
                with st.spinner("Retraining…"):
                    new_state = MD.train_model(new_store)
                    metrics = MD.evaluate_records(new_store, new_state)
                st.session_state["b3_state"] = new_state
                st.session_state["b3_store"] = new_store
                st.session_state["b3_hist"] = list(history) + [{
                    "at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    "races": len(new_store), "added": added,
                    "replaced": replaced, "metrics": metrics}]
                st.session_state["b3_trained_here"] = True
                st.session_state.pop("b3_pred", None)
                # The sidebar metrics render near the top of the script, before
                # this block runs, so on this pass they would still show the
                # pre-training counts. Rerun so every figure on the page comes
                # from the new state, and carry the message across.
                st.session_state["b3_flash"] = (
                    f"Retrained on {len(new_store)} races "
                    f"({added} added, {replaced} replaced). Predictions were "
                    "cleared — press **Predict all races** again. Download the "
                    "trained state from the sidebar to keep it.")
                st.rerun()

# --------------------------------------------------------------- diagnostics
with diag_tab:
    st.subheader("Model state")
    if st.session_state.get("b3_trained_here"):
        st.info("This state was retrained in **this session**. It is not "
                "saved anywhere — download it from the sidebar.",
                icon=":material/cloud_off:")
    c = st.columns(4)
    c[0].metric("Training races", state.get("training_races", 0))
    c[1].metric("Pairwise pairs", f"{state.get('training_pairs', 0):,}")
    c[2].metric("Learned influence",
                f"{100 * state.get('learned_influence', 0.0):.1f}%")
    c[3].metric("Schema", state.get("version", "—"))

    tm = state.get("training_metrics") or {}
    if tm:
        st.markdown("#### Training metrics")
        # training_metrics holds ints, floats and a string. Formatting only
        # the floats leaves one object column of mixed types, which Arrow
        # cannot type — and Streamlit ships dataframes as Arrow, so the table
        # fails to render in the browser. Everything becomes a string.
        st.dataframe(pd.DataFrame([{
            "Metric": k.replace("_", " ").capitalize(),
            "Value": (f"{v:.4f}" if isinstance(v, float) else str(v))}
            for k, v in tm.items()]), width="stretch", hide_index=True)

    names = state.get("feature_names") or []
    weights = state.get("weights") or []
    if names and len(names) == len(weights):
        st.markdown("#### Learned coefficients")
        w = pd.DataFrame({"Feature": names, "Weight": weights})
        w["abs"] = w.Weight.abs()
        st.dataframe(w.sort_values("abs", ascending=False)
                     .drop(columns="abs"), width="stretch", hide_index=True,
                     column_config={"Weight": st.column_config.NumberColumn(
                         format="%.4f")})

    if history:
        st.markdown("#### Training history")
        st.dataframe(pd.DataFrame(history), width="stretch", hide_index=True)

    st.markdown("""
#### What the bundled numbers mean

The state shipped with the app was trained on the developer's original five
meetings — 35 labelled races, 1,358 pairwise finishing relationships. Its
leave-one-meeting-out diagnostic reported the winner ranked first **25.7%** of
the time, top three **48.6%**, mean actual-winner rank **4.37**, pairwise
finishing accuracy **62.0%**.

Those five meetings were all on the same date, so that is a cross-meeting
diagnostic and not a forward-looking one. The honest test is a meeting
predicted before its results are known — which is what the Results tab is for.

#### The odds firewall

Current and historical prices are excluded from every predictive feature, and
the app calculates no EV or stake sizing. Bet365's written analyst view is kept
as one separately visible source model, not as a price. Odds influence reads
**0%** because it is structurally zero, not because it happens to be small.
""")
