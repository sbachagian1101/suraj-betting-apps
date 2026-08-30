"""FT Score Predictor — a full-time score matrix from two FootyStats panels."""
from __future__ import annotations

import glob

import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="FT Score Predictor",
                   page_icon=":material/scoreboard:", layout="wide")

# Streamlit Cloud pulls new files but keeps imported modules in sys.modules, so
# a deploy can run new app code against stale helpers. Fail loudly with the fix.
try:
    import backtest as BT
    import model as MD
    import panel as PN
    for _m, _a in [(PN, "parse_panels"), (PN, "split_three"),
                   (MD, "predict"), (MD, "score_matrix"), (BT, "run")]:
        if not hasattr(_m, _a):
            raise ImportError(f"{_m.__name__} is missing {_a}")
except ImportError as exc:
    st.error(f"**Stale modules** ({exc}).\n\nOn Streamlit Cloud: "
             "**Manage app → ⋮ → Reboot app**. A rerun is not enough.",
             icon=":material/error:")
    st.stop()

CSS = """
<style>
.fs-row{display:flex;gap:12px;margin:6px 0 16px 0;flex-wrap:wrap}
.fs-card{flex:1 1 160px;border-radius:16px;padding:14px 16px;color:#fff;
  box-shadow:0 6px 18px rgba(0,0,0,.16)}
.fs-card .lab{font-size:.72rem;letter-spacing:.10em;text-transform:uppercase;
  opacity:.92}
.fs-card .val{font-size:2.0rem;font-weight:700;line-height:1.15;margin-top:2px}
.fs-card .sub{font-size:.78rem;opacity:.92}
.fs-a{background:linear-gradient(135deg,#0f4c5c,#2a9d8f)}
.fs-b{background:linear-gradient(135deg,#3d348b,#7678ed)}
.fs-c{background:linear-gradient(135deg,#9d2226,#d1495b)}
.fs-d{background:linear-gradient(135deg,#1b3a4b,#3f7cac)}
.fs-bar{display:flex;height:32px;border-radius:10px;overflow:hidden;
  font-size:.8rem;font-weight:600;color:#fff;margin:2px 0 10px 0}
.fs-bar div{display:flex;align-items:center;justify-content:center}
.fs-meter{background:rgba(128,128,128,.20);border-radius:9px;height:30px;
  position:relative;overflow:hidden;margin-bottom:5px}
.fs-meter .fill{height:100%;border-radius:9px;
  background:linear-gradient(90deg,#20707f,#4fb3a5)}
.fs-meter .txt{position:absolute;inset:0;display:flex;align-items:center;
  justify-content:space-between;padding:0 12px;font-weight:600;font-size:.84rem}
</style>
"""

# The colour scale is interpolated here rather than taken from
# Styler.background_gradient, which imports matplotlib - not a Streamlit
# dependency, so a shaded grid renders locally and dies on Cloud.
_SCALE = [(255, 255, 217), (199, 233, 180), (127, 205, 187),
          (65, 182, 196), (44, 127, 184), (37, 52, 148)]


def shade(v, vmax):
    if not np.isfinite(v) or vmax <= 0:
        return ""
    t = float(np.clip(v / vmax, 0.0, 1.0)) ** 0.6
    pos = t * (len(_SCALE) - 1)
    i = int(np.floor(pos))
    j = min(i + 1, len(_SCALE) - 1)
    f = pos - i
    r, g, b = (round(_SCALE[i][k] + f * (_SCALE[j][k] - _SCALE[i][k]))
               for k in range(3))
    fg = "#f5f7fa" if (0.299 * r + 0.587 * g + 0.114 * b) < 140 else "#10243b"
    return f"background-color: rgb({r},{g},{b}); color: {fg}"


def cards(items):
    html = "<div class='fs-row'>"
    for lab, val, sub, cls in items:
        html += (f"<div class='fs-card {cls}'><div class='lab'>{lab}</div>"
                 f"<div class='val'>{val}</div><div class='sub'>{sub}</div></div>")
    st.markdown(html + "</div>", unsafe_allow_html=True)


def meter(label, p, right=""):
    st.markdown(
        f"<div class='fs-meter'><div class='fill' style='width:{p*100:.1f}%'>"
        f"</div><div class='txt'><span>{label}</span>"
        f"<span>{p*100:.1f}%{right}</span></div></div>",
        unsafe_allow_html=True)


@st.cache_data(show_spinner=False)
def parse_both(home_text: str, away_text: str):
    """Panels from the two boxes, tolerant of both being pasted into one.

    On FootyStats the two team panels sit side by side, so a single copy brings
    back both in order. If the away box is empty and the home box holds two
    panels, that is what happened: the first is the home side.
    """
    ph = PN.parse_panels(home_text) if home_text.strip() else []
    pa = PN.parse_panels(away_text) if away_text.strip() else []
    if not pa and len(ph) >= 2:
        return ph[0], ph[1], "both panels came from the home box, in order"
    if not ph and len(pa) >= 2:
        return pa[0], pa[1], "both panels came from the away box, in order"
    return (ph[0] if ph else None), (pa[0] if pa else None), ""


st.markdown(CSS, unsafe_allow_html=True)

with st.sidebar:
    st.title("FT Score Predictor")
    st.caption("Two FootyStats team panels in, a full-time score matrix out.")
    st.divider()
    swap = st.toggle("Swap home and away", value=False)
    venue_w = st.slider(
        "Weight on the venue column", 0.0, 1.0, 0.65, 0.05,
        help="1.0 uses the Home/Away column alone; 0.0 uses Overall alone. "
             "Half a season is enough to be worth using and not enough to "
             "trust outright.")
    xg_w = st.slider(
        "Weight on xG over goals", 0.0, 1.0, 0.50, 0.05,
        help="xG is the steadier signal; actual goals carry the finishing.")
    with st.expander("Grid parameters"):
        rho = st.slider("Dixon–Coles ρ", -0.20, 0.10, -0.05, 0.01)
        disp = st.slider("Over-dispersion r (0 = Poisson)", 0.0, 30.0, 0.0, 1.0)
    st.divider()
    st.caption("No home-advantage multiplier is applied: the Home and Away "
               "columns already contain it, and multiplying again would count "
               "the same effect twice.")

paste_tab, data_tab, score_tab, acc_tab, method_tab = st.tabs(
    ["1 · Paste", "2 · Parsed", "3 · FT score", "Accuracy", "Method"])

with paste_tab:
    samples = sorted(glob.glob("sample_data/*.txt"))
    c0, c1 = st.columns([3, 1])
    pick = c0.selectbox("Bundled example", ["—"] + [s.rsplit("/", 1)[-1].rsplit("\\", 1)[-1]
                                                    for s in samples])
    if c1.button("Load", width="stretch") and pick != "—":
        path = next(s for s in samples if s.endswith(pick))
        txt = open(path, encoding="utf-8").read()
        parts = PN.parse_panels(txt)
        st.session_state["ft_home"] = txt
        st.session_state["ft_away"] = ""
        st.session_state.pop("ft_pred", None)

    a, b = st.columns(2)
    with a:
        st.markdown("##### :material/home: Home team panel")
        st.text_area("Home panel", key="ft_home", height=340,
                     label_visibility="collapsed",
                     placeholder="Paste the home team's FootyStats panel — the "
                                 "block with Form/Results/PPG and the "
                                 "Stats · Overall · Home · Away table.")
    with b:
        st.markdown("##### :material/flight_takeoff: Away team panel")
        st.text_area("Away panel", key="ft_away", height=340,
                     label_visibility="collapsed",
                     placeholder="Paste the away team's panel here. You can "
                                 "also leave this empty and paste both panels "
                                 "into the box on the left.")
    st.caption(
        "The two panels sit side by side on FootyStats, so one copy usually "
        "brings back both — paste that into the left box and leave the right "
        "one empty; the first panel is taken as the home side. Use **Swap "
        "home and away** in the sidebar if they come back the other way round.")

home_text = st.session_state.get("ft_home", "")
away_text = st.session_state.get("ft_away", "")
if not home_text.strip() and not away_text.strip():
    for t in (data_tab, score_tab):
        with t:
            st.info("Paste a team panel on the first tab to begin.",
                    icon=":material/content_paste:")
    H = A = None
else:
    H, A, note = parse_both(home_text, away_text)
    if swap:
        H, A = A, H
    if H is None or A is None:
        with data_tab:
            st.error("Two team panels are needed. Each must include the "
                     "`Stats Overall Home Away` header and the rows beneath "
                     "it.", icon=":material/error:")
        with score_tab:
            st.info("Nothing to score yet.", icon=":material/scoreboard:")
        H = A = None

if H is not None and A is not None:
    with data_tab:
        st.subheader(f"{H['team']}  (home)   vs   {A['team']}  (away)")
        if note:
            st.caption(note)
        miss = {p["team"]: PN.missing(p) for p in (H, A)}
        for team, mlist in miss.items():
            if mlist:
                st.warning(f"{team}: could not read {', '.join(mlist)} — "
                           "those rows will fall back to what is available.",
                           icon=":material/report:")
        tbl = PN.table([H, A])
        show = tbl.set_index("Team").T
        st.dataframe(show.map(lambda v: "—" if v is None or
                              (isinstance(v, float) and not np.isfinite(v))
                              else (f"{v:,.2f}" if isinstance(v, (int, float,
                                                                 np.floating))
                                    and not isinstance(v, bool) else str(v))),
                     width="stretch", height=520)
        st.caption(
            "Every figure is read straight from the panel. The numbers arrive "
            "concatenated — `AVG2.942.713.10` — and are split using the fact "
            "that **Overall must lie between Home and Away**, since it is "
            "their weighted average. That one constraint resolves splits a "
            "string alone cannot.")

    r = MD.predict(H, A, venue_weight=venue_w, xg_weight=xg_w,
                   rho=rho, dispersion=(disp or None))
    m = r["matrix"]

    with score_tab:
        hg, ag = r["pick"]
        st.subheader(f"{H['team']}  vs  {A['team']}")
        cards([
            ("Most likely score", f"{hg}–{ag}", f"{100*r['pick_prob']:.1f}%",
             "fs-d"),
            (H["team"][:18], f"{100*r['home']:.1f}%",
             f"expected {r['lh']:.2f} goals", "fs-a"),
            ("Draw", f"{100*r['draw']:.1f}%", "&nbsp;", "fs-b"),
            (A["team"][:18], f"{100*r['away']:.1f}%",
             f"expected {r['la']:.2f} goals", "fs-c"),
        ])
        st.markdown(
            f"<div class='fs-bar'>"
            f"<div style='width:{r['home']*100:.2f}%;background:#2a9d8f'>"
            f"{r['home']*100:.0f}%</div>"
            f"<div style='width:{r['draw']*100:.2f}%;background:#7678ed'>"
            f"{r['draw']*100:.0f}%</div>"
            f"<div style='width:{r['away']*100:.2f}%;background:#d1495b'>"
            f"{r['away']*100:.0f}%</div></div>", unsafe_allow_html=True)

        st.markdown("#### The score matrix")
        mat = pd.DataFrame(
            100 * m,
            index=[f"{H['team'][:12]} {i}" for i in range(m.shape[0])],
            columns=[f"{A['team'][:12]} {j}" for j in range(m.shape[1])])
        vmax = float(np.nanmax(mat.to_numpy()))
        st.dataframe(mat.style.map(lambda v: shade(v, vmax)).format("{:.2f}"),
                     width="stretch", height=430)
        st.caption("Each cell is the probability of that exact full-time "
                   "score, in percent. Rows are the home team's goals, "
                   "columns the away team's.")

        st.markdown("#### Most likely scorelines")
        tops = MD.top_scores(m, 10)
        st.dataframe(
            pd.DataFrame([{"Score": f"{i}–{j}", "Probability %": 100 * p,
                           "Odds": (1 / p if p > 0 else np.inf)}
                          for (i, j), p in tops]),
            width="stretch", hide_index=True,
            column_config={
                "Probability %": st.column_config.ProgressColumn(
                    "Probability %", format="%.2f%%", min_value=0.0,
                    max_value=float(100 * tops[0][1])),
                "Odds": st.column_config.NumberColumn("Fair odds",
                                                      format="%.1f")})

        st.markdown("#### Markets")
        m1, m2 = st.columns(2)
        with m1:
            meter("Both teams to score", r["btts"])
            meter("Over 1.5 goals", r["over15"])
            meter("Over 2.5 goals", r["over25"])
        with m2:
            meter("Over 3.5 goals", r["over35"])
            meter(f"{H['team'][:16]} clean sheet", float(m[:, 0].sum()))
            meter(f"{A['team'][:16]} clean sheet", float(m[0, :].sum()))
        st.caption(f"Expected total goals **{r['exp_goals']:.2f}** — "
                   f"{r['lh']:.2f} from {H['team']}, {r['la']:.2f} from "
                   f"{A['team']}.")

        idx = np.arange(m.shape[0])
        tot = idx[:, None] + idx[None, :]
        dist = np.array([float(m[tot == k].sum()) for k in range(9)])
        st.markdown("#### Total goals")
        st.bar_chart(pd.DataFrame({"probability %": 100 * dist},
                                  index=[str(k) for k in range(9)]),
                     height=220, color="#2a9d8f")

with acc_tab:
    st.subheader("Tested against the five matches with known results")
    try:
        bt = BT.run(venue_weight=0.65, xg_weight=0.50)
    except Exception as exc:                                   # noqa: BLE001
        st.error(f"The backtest could not run — {type(exc).__name__}: {exc}")
        bt = pd.DataFrame()
    if not bt.empty:
        n = len(bt)
        show = bt[["match", "lh", "la", "exp", "pick", "pick_p", "actual",
                   "p_actual", "rank_actual"]].rename(columns={
            "match": "Match", "lh": "λ home", "la": "λ away",
            "exp": "Expected goals", "pick": "Most likely",
            "pick_p": "its %", "actual": "Actual",
            "p_actual": "% it gave the actual", "rank_actual": "Rank of actual"})
        st.dataframe(show, width="stretch", hide_index=True,
                     column_config={
                         "λ home": st.column_config.NumberColumn(format="%.2f"),
                         "λ away": st.column_config.NumberColumn(format="%.2f"),
                         "Expected goals": st.column_config.NumberColumn(format="%.2f"),
                         "its %": st.column_config.NumberColumn(format="%.1f%%"),
                         "% it gave the actual": st.column_config.NumberColumn(format="%.2f%%")})
        c = st.columns(4)
        c[0].metric("Exact scoreline", f"{int(bt.exact.sum())}/{n}")
        c[1].metric("1X2 correct", f"{int(bt.res_ok.sum())}/{n}")
        c[2].metric("BTTS correct", f"{int(bt.btts_ok.sum())}/{n}")
        c[3].metric("Over/under 2.5", f"{int(bt.o25_ok.sum())}/{n}")
        st.markdown(f"""
**What these five say, and what they do not.**

The exact scoreline came up **{int(bt.exact.sum())} times in {n}**. That looks
extraordinary and mostly is not: the most likely scoreline in a football match
carries roughly an 11% chance, and hitting two or more from five happens about
**10%** of the time by luck alone. It is not evidence of skill at this sample
size.

The model expected **{bt.exp.mean():.2f}** goals a game and the five produced
**{bt.actual_goals.mean():.2f}**. The gap is real but small next to the noise:
the standard error of a five-match mean at that rate is about 0.80 goals, so
this is around **1.5 standard errors** — the kind of gap five matches produce
routinely.

Scoreline log-loss was **{bt.logloss.mean():.3f}** against **{BT.baseline_logloss():.3f}**
for a league-average Poisson that knows nothing about either team. Better, but
by a margin five matches cannot certify.

Worth noticing about the sample itself: the home side won all five, both teams
scored in all five, and all five went over 2.5 goals. A set that one-sided is a
reminder that these are five matches someone chose, not five drawn at random.
""")

with method_tab:
    st.markdown("""
### What is read

The FootyStats team-comparison panel: league and position, form and points per
game for Overall / Home / Away, and the stats block — Win %, AVG, Scored,
Conceded, BTTS, CS, FTS, xG and xGA, each across the same three columns.

The numbers come through **concatenated with no separator**: `AVG2.942.713.10`
must become 2.94, 2.71, 3.10. As a string that is ambiguous — `1.542.231.2`
could split 1.54/2.23/1.2 or 1.5/42.2/31.2, and FootyStats drops trailing
zeros so field widths are not fixed. The disambiguator is arithmetic:
**Overall is a weighted average of Home and Away, so it must lie between
them.** Every tokenisation is enumerated and that constraint keeps the right
one.

### The rates

    λ_home = mean(home scored at home,  away conceded away)
    λ_away = mean(away scored away,     home conceded at home)

and the same pairing on xG/xGA, the two blended. **No home-advantage
multiplier is applied.** The Home and Away columns already contain it — a home
scoring rate *is* a home scoring rate — so a further home factor would count
the same effect twice. That is the specification error this shape of input
invites, and it is the main reason these panels are worth more than five
pasted match pages: the venue split comes from half a season rather than two
or three games.

Venue figures are shrunk toward the team's overall figure, because eight or
nine matches is enough to be worth using and not enough to take at face value.

The grid is Poisson with the **Dixon–Coles** low-score correction, optionally
negative-binomial to fatten the high-scoring corner.

### What it cannot do

The score matrix is a distribution, not a prediction. Even a well-specified
model puts only about **11%** on its most likely scoreline, because football
scorelines genuinely are that uncertain — the single most likely outcome is
wrong roughly nine times in ten. Read the grid as a shape, and the markets
(1X2, BTTS, over/under) as the parts that carry real information.

There is **no market price** in this input and the measured record is **five
matches**, which is enough to catch a badly broken model and nowhere near
enough to establish a good one.

---

*Probabilistic decision support, not a guaranteed outcome. Gamble responsibly —
Gambling Help 1800 858 858,
[gamblinghelponline.org.au](https://www.gamblinghelponline.org.au).*
""")
