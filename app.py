"""FormPredict — form score and win probability from Racing & Sports meeting files.

Upload one or more `<date>-<TRACK>-T.xlsx` exports and every race on the card is
scored on form alone. There is no market anywhere in this app, because that
export has no price column in it.
"""
from __future__ import annotations

import pandas as pd
import streamlit as st

import form_model as fm
import meeting_parser as mp

st.set_page_config(page_title="FormPredict", page_icon=":horse_racing:",
                   layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
.block-container {padding-top: 1.15rem; padding-bottom: 3rem;}
div[data-testid="stMetric"] {border:1px solid rgba(128,128,128,.22);
  border-radius:.7rem; padding:.55rem .8rem;}
.pick {border:1px solid rgba(46,204,113,.55); background:rgba(46,204,113,.10);
  border-radius:.8rem; padding:.75rem 1rem; margin:.4rem 0 .8rem;}
.muted {opacity:.65; font-size:.78rem; letter-spacing:.06em;}
</style>
""", unsafe_allow_html=True)

st.title("FormPredict")
st.caption("Form score and win probability for every horse on the card — "
           "from Racing & Sports `<date>-<TRACK>-T.xlsx` meeting files.")

# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("Settings")
    shrink = st.slider(
        "Place probability shrink", 0.0, 0.5, float(fm.DEFAULT_PLACE_SHRINK), 0.05,
        help="Pulls place probabilities toward the field's base rate. Shipped at "
             "0.00: measured at six strengths the calibration error moved "
             "non-monotonically across half a point, which is noise, so no "
             "correction is applied by default.")
    sims = st.select_slider("Place simulations", [2000, 6000, 12000, 25000],
                            value=fm.DEFAULT_SIMS,
                            help="More simulations give steadier place "
                                 "probabilities and take slightly longer.")
    show_all = st.checkbox("Show every runner", value=True,
                           help="Off shows only the top six on form.")

    st.divider()
    st.subheader("Red alert")
    st.caption("Marks a shortlisted horse meeting all four conditions.")
    red_on = st.checkbox("Enable the red alert", value=True)
    red_runs = st.number_input("More than N career starts", 0, 30,
                               fm.RED_MIN_CAREER_RUNS, 1, disabled=not red_on)
    red_last = st.number_input("Finished at least this well last start", 1, 6,
                               fm.RED_MAX_LAST_FINISH, 1, disabled=not red_on)
    red_jky = st.number_input("Jockey inside the race's top N by rating", 1, 10,
                              fm.RED_TOP_JOCKEYS, 1, disabled=not red_on,
                              help="Ties share the best rank, so where several "
                                   "jockeys are rated alike they all count — "
                                   "the rating genuinely does not separate "
                                   "them in that race.")
    red_top = st.number_input("Only apply within the top N on form", 1, 10,
                              fm.RED_WITHIN_TOP, 1, disabled=not red_on)
    st.divider()
    st.markdown(
        f"**Measured on {fm.MEASURED['races']} races**\n\n"
        f"Top pick won **{100*fm.MEASURED['top1_win']:.1f}%** "
        f"(dart throw {100*fm.BASELINE['win']:.1f}%).\n\n"
        f"Its top three placed **{100*fm.MEASURED['top3_place']:.1f}%** "
        f"(dart throw {100*fm.BASELINE['place']:.1f}%).")
    st.caption("Strike rates, not profit. No odds are used anywhere in this app.")

upload_tab, card_tab, method_tab = st.tabs(
    ["1 · Upload", "2 · Whole Card", "Method"])

st.session_state.setdefault("card", None)

# ---------------------------------------------------------------- upload
with upload_tab:
    st.subheader("Upload meeting files")
    ups = st.file_uploader(
        "Racing & Sports meeting exports", type=["xlsx"],
        accept_multiple_files=True,
        help="Files named like 2026-08-27-SEYMOUR-T.xlsx. Upload as many "
             "meetings as you like — they are kept separate.")

    if ups:
        rows, problems = [], []
        for up in ups:
            try:
                raw = pd.read_excel(up, header=None)
                rows += mp.parse_grid(raw, mp.track_name(up.name),
                                      mp.meeting_date(up.name))
            except Exception as exc:                       # noqa: BLE001
                problems.append(f"**{up.name}** — {type(exc).__name__}: {exc}")
        for msg in problems:
            st.error(msg)
        card = mp.to_frame(rows)
        st.session_state["card"] = None if card.empty else card
        if card.empty and not problems:
            st.error("No runners could be read. The expected layout is stacked "
                     "race blocks, each headed by a row starting `Tab`.")

    card = st.session_state["card"]
    if card is None:
        st.info("Upload a meeting file to begin.")
    else:
        c1, c2, c3 = st.columns(3)
        c1.metric("Meetings", card["track"].nunique())
        c2.metric("Races", card["race_id"].nunique())
        c3.metric("Runners", len(card))
        for w in mp.warnings_for(card):
            st.warning(w)

        ids = mp.races(card)
        labels = {}
        for rid in ids:
            g = card[card.race_id == rid]
            labels[rid] = "{} R{} — {} runners".format(
                g["track"].iloc[0], int(g["race"].iloc[0]), len(g))
        pick = st.selectbox("Race", ids, format_func=lambda r: labels[r])
        g = card[card.race_id == pick]

        auto = fm.places_paid(len(g))
        c1, c2 = st.columns([1, 3])
        places = c1.selectbox("Places paid", [1, 2, 3, 4],
                              index=[1, 2, 3, 4].index(auto),
                              help="Standard terms: 3 for 8+ runners, 2 for "
                                   "5–7, win only under 5.")
        meta = g.iloc[0]
        dist_txt = "" if pd.isna(meta["dist"]) else f"{int(meta['dist'])}m"
        c2.markdown("**{}**  \n{} · {} {} · {} runners · {} place(s) paid".format(
            meta["race_name"] or "", dist_txt, meta["surface"] or "?",
            meta["going"] or "", len(g), places))

        red_kw = dict(min_runs=int(red_runs), max_last=int(red_last),
                      top_jockeys=int(red_jky),
                      within_top=int(red_top) if red_on else 0)
        table = fm.rate_race(g, places=int(places), shrink=float(shrink),
                             sims=int(sims), **red_kw)
        top = table.iloc[0]
        st.markdown(
            '<div class="pick"><div class="muted">TOP RATED ON FORM</div>'
            '<h2 style="margin:.15rem 0">#{} · {}</h2>'
            '<b>Form score {:.0f}</b> · Win {:.1f}% · Place {:.1f}% · '
            'Fair ${:.2f}</div>'.format(
                int(top["Tab"]), top["Horse"], top["Form score"],
                top["Win%"], top["Place%"], top["Fair win $"]),
            unsafe_allow_html=True)

        top3 = " · ".join("#{} {}".format(int(r["Tab"]), r["Horse"])
                          for _, r in table.head(3).iterrows())
        st.info(
            "**Top three on form: {}** — read these as a group, not an order. "
            "Over the {} races this was measured on, the 1st, 2nd and 3rd rated "
            "horses won {:.0f}%, {:.0f}% and {:.0f}% — the same to within the "
            "margin of error — while rank 4 fell to {:.0f}%.".format(
                top3, fm.MEASURED["races"],
                100 * fm.RANK_STATS[1]["win"], 100 * fm.RANK_STATS[2]["win"],
                100 * fm.RANK_STATS[3]["win"], 100 * fm.RANK_STATS[4]["win"]))

        if red_on:
            if (table["Alert"] != "").any():
                st.error(fm.red_summary(table))
            else:
                st.caption(fm.red_summary(table))

        shown = table if show_all else table.head(6)
        st.dataframe(fm.style(shown), width="stretch", hide_index=True)

        if red_on:
            with st.expander("Red alert — condition by condition"):
                st.dataframe(
                    fm.red_detail(table, min_runs=int(red_runs),
                                  top_jockeys=int(red_jky)),
                    width="stretch", hide_index=True)
                st.caption(
                    f"All four must hold, and only the top {int(red_top)} on "
                    "form are eligible — a horse outside the shortlist is not a "
                    "selection, so flagging it would be noise.")
        st.caption(
            "**Form score** is relative to today's field — 100 means top rated "
            "here, not good in absolute terms. **Win%** is calibrated: across "
            "six probability bands the mean gap between predicted and actual "
            "was 1.5 points. **Place%** runs a few points generous above ~55%. "
            "**Fair $** is 1 ÷ probability — only bet when the real market pays "
            "more than that.")

        with st.expander("Why each runner rates where it does"):
            st.dataframe(
                table[["Rank", "Tab", "Horse", "Form score", "Why"]].style.format(
                    {"Form score": "{:.0f}"}),
                width="stretch", hide_index=True)

# ---------------------------------------------------------------- whole card
with card_tab:
    card = st.session_state["card"]
    if card is None:
        st.info("Upload a meeting file in **1 · Upload** first.")
    else:
        st.subheader("Top three in every race")
        depth = st.slider("Runners shown per race", 1, 6, 3)
        ids = mp.races(card)
        rows = []
        for rid in ids:
            g = card[card.race_id == rid]
            t = fm.rate_race(g, shrink=float(shrink), sims=int(sims),
                             min_runs=int(red_runs), max_last=int(red_last),
                             top_jockeys=int(red_jky),
                             within_top=int(red_top) if red_on else 0)
            for _, r in t.head(depth).iterrows():
                rows.append({
                    "Meeting": g["track"].iloc[0],
                    "Race": int(g["race"].iloc[0]),
                    "Rank": int(r["Rank"]), "Tab": int(r["Tab"]),
                    "Horse": r["Horse"], "Alert": r["Alert"],
                    "Form score": r["Form score"],
                    "Win%": r["Win%"], "Place%": r["Place%"],
                    "Fair win $": r["Fair win $"]})
        out = pd.DataFrame(rows)
        if red_on:
            n_alert = int((out["Alert"] != "").sum())
            if n_alert and st.checkbox(
                    f"Show only the {n_alert} red-alert runner(s)", value=False):
                out = out[out["Alert"] != ""]
        st.dataframe(
            out.style.format({"Form score": "{:.0f}", "Win%": "{:.1f}%",
                              "Place%": "{:.1f}%", "Fair win $": "${:.2f}"})
               .apply(lambda r: ["background-color: rgba(231,76,60,.30)"] * len(r)
                      if r.get("Alert") else [""] * len(r), axis=1),
            width="stretch", hide_index=True, height=560)
        st.download_button(
            "Download this card as CSV", out.to_csv(index=False).encode("utf-8"),
            file_name="formpredict-card.csv", mime="text/csv")
        st.caption(f"{out['Race'].count()} rows across "
                   f"{card.race_id.nunique()} races.")

# ---------------------------------------------------------------- method
with method_tab:
    st.subheader("How this works, and what it is worth")
    rr, rl, rj, rt = (fm.RED_MIN_CAREER_RUNS, fm.RED_MAX_LAST_FINISH,
                      fm.RED_TOP_JOCKEYS, fm.RED_WITHIN_TOP)
    R = fm.RED_MEASURED
    ra, ro = R["alerts"], R["others"]
    raw, rap = 100 * R["alert_win"], 100 * R["alert_place"]
    row, rop = 100 * R["other_win"], 100 * R["other_place"]
    rwp, rpp = R["win_p"], R["place_p"]
    rsh, rfr, rof = 100 * R["share_of_shortlist"], R["fired_in_races"], R["of_races"]
    st.markdown(f"""
### The one fact that shapes everything

**The meeting export contains no market prices.** Not a blank column — the
format has no odds column at all. So this is necessarily a form-only model, and
nothing here is a claim about value or profit.

### What it scored

**{fm.MEASURED['races']} races** from 27 August 2026 — nine meetings across
Australia, Britain and Ireland, {fm.MEASURED['runners']} runners, results taken
from the published board.

| | top pick wins | its top 3 place | winner in top 3 |
|---|---|---|---|
| dart throw | {100*fm.BASELINE['win']:.1f}% | {100*fm.BASELINE['place']:.1f}% | {100*fm.BASELINE['top3_has_winner']:.1f}% |
| **this model** | **{100*fm.MEASURED['top1_win']:.1f}%** | **{100*fm.MEASURED['top3_place']:.1f}%** | **{100*fm.MEASURED['top3_has_winner']:.1f}%** |

Win probabilities are calibrated — across six bands the mean gap between
predicted and actual was **{fm.MEASURED['calibration_error']:.3f}**, and the most
confident band predicted 31.8% against 33.3% actual.

### Read the top three as a group

| form rank | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|
| win% | {100*fm.RANK_STATS[1]['win']:.1f} | {100*fm.RANK_STATS[2]['win']:.1f} | {100*fm.RANK_STATS[3]['win']:.1f} | {100*fm.RANK_STATS[4]['win']:.1f} | {100*fm.RANK_STATS[5]['win']:.1f} |
| place% | {100*fm.RANK_STATS[1]['place']:.1f} | {100*fm.RANK_STATS[2]['place']:.1f} | {100*fm.RANK_STATS[3]['place']:.1f} | {100*fm.RANK_STATS[4]['place']:.1f} | {100*fm.RANK_STATS[5]['place']:.1f} |

Ranks 1–3 are indistinguishable from each other; rank 4 falls away. The score
resolves a **group of three**, not an order within it.

### The red alert

A separate, hand-specified filter that sits **on top of** the ranking. It marks
a shortlisted horse that is experienced, arriving in form, already race-fit and
well ridden — all four must hold:

1. more than **{rr}** career starts
2. **placed** (finished {rl} or better) at its last start
3. has **already had a run this preparation** — not resuming from a spell
4. its jockey is inside the race's **top {rj}** by jockey rating

Only the top **{rt}** on form are eligible; a horse outside the shortlist is not
a selection, so flagging it would be noise. All four thresholds are adjustable
in the sidebar.

*How "already had a run this preparation" is read:* Racing & Sports form strings
run oldest to newest with `x` marking a spell, so a trailing `x` means the horse
resumes today. Checked two ways on this card — the rightmost digit matched the
independent last-start column for **726 of 726** runners, and every one of the
105 horses whose form ends in `x` had been off more than 60 days (median 124)
against none of the 626 that had not (median 17).

#### What it did on the 65 races with results

| within the top three | n | won | placed |
|---|---|---|---|
| **with the alert** | {ra} | **{raw:.1f}%** | **{rap:.1f}%** |
| without it | {ro} | {row:.1f}% | {rop:.1f}% |

Fisher exact two-sided **p = {rwp:.3f}** on wins, **p = {rpp:.3f}** on places.
It fired on about {rsh:.0f}% of shortlisted runners, in {rfr} of {rof} races.

**Read that as encouraging, not established.** Three reasons it is not proof:

- **n = {ra}.** One race either way moves the win rate about four points.
- **All 65 races are a single day.** Nothing here separates the rule from
  "that Thursday suited experienced, in-form horses".
- **The four conditions all correlate with simply being a good horse**, so part
  of the gap is the rule re-finding what the form score already knew rather than
  adding something to it.

The honest use is to log results against it. If it holds up over a few hundred
races, it means something.

### Why the weights are fixed, not fitted

Fitting was tried first and lost. A 28-feature conditional logit scored
**16.9%** on top pick — *worse than half a dozen single columns* — and gave
negative weights to "better record at this distance" and "better record on this
surface". One cause for both: 65 races cannot identify 28 free parameters, so
the fit chases noise into signs that are physically backwards.

The fixed weights cost less than they look. Against **200 random weightings** of
the same 14 columns they sit *inside* the spread on every measure (top-1 median
0.200, range 0.169–0.246; these weights 0.215). Removing the highest-weighted
column entirely moves top-1 from 0.215 to 0.200. **The choice of columns does the
work, not the numbers on them.**

### What the score is built from

Within-race z-scores, so every figure is relative to today's opponents:

| weight | column |
|---|---|
""" + "\n".join(
        f"| {w:.1f} | {fm.LABELS.get(c, c)} |"
        for c, w in sorted(fm.WEIGHTS.items(), key=lambda kv: -kv[1])) + f"""

### What {fm.MEASURED['races']} races cannot tell you

Every rate above carries a 95% interval roughly **±10 points** wide. The model
clearly beats a dart throw. Whether it beats its own best single column —
last-start beaten margin alone, 24.6% — is **not resolvable** at this sample
size; three races separate them. It ships because it wins clearly on
placegetters and top-three coverage.

**No place-probability correction is applied.** Plackett–Luce overstates the
place chance at the top, and does here (0.62 predicted against 0.57 actual). A
shrink toward the base rate was measured at six strengths and the calibration
error moved non-monotonically across half a point — noise. Picking the minimum
would be tuning on the same races it is scored against, so the figures ship
uncorrected and the optimism is stated instead. The slider is in the sidebar if
you want it.

---
*Prediction is probabilistic decision support, not a guaranteed outcome. Gamble
responsibly — Gambling Help 1800 858 858,
[gamblinghelponline.org.au](https://www.gamblinghelponline.org.au).*
""")
