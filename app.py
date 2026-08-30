"""Match Insight — paste two teams' last five matches, get a prediction."""
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Match Insight", page_icon=":material/stadium:",
                   layout="wide")

# Streamlit Cloud pulls new files but keeps imported modules in sys.modules, so
# a deploy can run new app code against stale helpers. Fail loudly with the fix.
try:
    import methods as X
    import metrics as M
    import parser as P
    for _m, _a in [(P, "parse"), (M, "team_profile"), (X, "run_all"),
                   (X, "ensemble"), (X, "corners_markets")]:
        if not hasattr(_m, _a):
            raise ImportError(f"{_m.__name__} is missing {_a}")
except ImportError as exc:
    st.error(f"**Stale modules** ({exc}).\n\nOn Streamlit Cloud: "
             "**Manage app → ⋮ → Reboot app**. A rerun is not enough.",
             icon=":material/error:")
    st.stop()

SAMPLE = "sample_data/sturm_graz_ii_vs_rapid_wien_ii.txt"

CSS = """
<style>
.mi-row{display:flex;gap:14px;margin:6px 0 18px 0;flex-wrap:wrap}
.mi-card{flex:1 1 190px;border-radius:16px;padding:16px 18px;color:#fff;
  box-shadow:0 6px 18px rgba(0,0,0,.16)}
.mi-card .lab{font-size:.78rem;letter-spacing:.10em;text-transform:uppercase;
  opacity:.92}
.mi-card .val{font-size:2.1rem;font-weight:700;line-height:1.15;margin-top:2px}
.mi-card .sub{font-size:.8rem;opacity:.92}
.mi-h{background:linear-gradient(135deg,#1e6f5c,#2fa37f)}
.mi-d{background:linear-gradient(135deg,#5b5f97,#8186c0)}
.mi-a{background:linear-gradient(135deg,#96341f,#d1613c)}
.mi-n{background:linear-gradient(135deg,#26415e,#3f6d99)}
.mi-bar{display:flex;height:34px;border-radius:10px;overflow:hidden;
  font-size:.82rem;font-weight:600;color:#fff;margin:2px 0 6px 0}
.mi-bar div{display:flex;align-items:center;justify-content:center;
  min-width:0;white-space:nowrap}
.mi-meter{background:rgba(128,128,128,.20);border-radius:9px;height:30px;
  position:relative;overflow:hidden;margin-bottom:4px}
.mi-meter .fill{height:100%;border-radius:9px;
  background:linear-gradient(90deg,#2b7a78,#4fb3a5)}
.mi-meter .txt{position:absolute;inset:0;display:flex;align-items:center;
  justify-content:space-between;padding:0 12px;font-weight:600;font-size:.85rem}
</style>
"""


# Styler.background_gradient goes through matplotlib, which is not a Streamlit
# dependency: the gradient rendered locally, where matplotlib happens to be
# installed, and the deployed app died with "background_gradient requires
# matplotlib" the moment the Prediction tab scrolled to the score grid. Pulling
# in matplotlib for one gradient costs ~50MB on every cold start, so the scale
# is interpolated here instead. Same family as the openpyxl and scikit-learn
# traps: a library that is present locally and absent on Cloud.
_YLGNBU = [(255, 255, 217), (199, 233, 180), (127, 205, 187),
           (65, 182, 196), (44, 127, 184), (37, 52, 148)]


def _shade(v, vmax):
    if not np.isfinite(v) or vmax <= 0:
        return ""
    t = float(np.clip(v / vmax, 0.0, 1.0)) ** 0.65
    pos = t * (len(_YLGNBU) - 1)
    i = int(np.floor(pos))
    j = min(i + 1, len(_YLGNBU) - 1)
    f = pos - i
    r, g, b = (round(_YLGNBU[i][k] + f * (_YLGNBU[j][k] - _YLGNBU[i][k]))
               for k in range(3))
    fg = "#f5f7fa" if (0.299 * r + 0.587 * g + 0.114 * b) < 140 else "#10243b"
    return f"background-color: rgb({r},{g},{b}); color: {fg}"


@st.cache_data(show_spinner=False)
def do_parse(text: str):
    df, dropped = P.parse(text, return_dropped=True)
    return df, P.teams(df), dropped


def cards(items):
    html = "<div class='mi-row'>"
    for lab, val, sub, cls in items:
        html += (f"<div class='mi-card {cls}'><div class='lab'>{lab}</div>"
                 f"<div class='val'>{val}</div><div class='sub'>{sub}</div></div>")
    st.markdown(html + "</div>", unsafe_allow_html=True)


def tri_bar(ph, pd_, pa, lh, la):
    seg = [(ph, "#2fa37f", lh), (pd_, "#8186c0", "Draw"), (pa, "#d1613c", la)]
    html = "<div class='mi-bar'>"
    for p, c, lab in seg:
        html += (f"<div style='width:{max(p*100,0):.2f}%;background:{c}'>"
                 f"{p*100:.0f}%</div>")
    st.markdown(html + "</div>", unsafe_allow_html=True)


def meter(label, p, right=""):
    st.markdown(
        f"<div class='mi-meter'><div class='fill' style='width:{p*100:.1f}%'>"
        f"</div><div class='txt'><span>{label}</span>"
        f"<span>{p*100:.1f}%{right}</span></div></div>",
        unsafe_allow_html=True)


st.markdown(CSS, unsafe_allow_html=True)

# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.title("Match Insight")
    st.caption("Paste the last five matches for two teams. Everything is "
               "derived from what you paste — no stored ratings.")
    st.divider()
    st.subheader("Match weighting")
    half_life = st.slider("Recency half-life (days)", 10, 240,
                          int(M.HALF_LIFE_DAYS), 5,
                          help="A match this many days older counts half.")
    w_league = st.slider("League match", 0.0, 1.0, 1.00, 0.05)
    w_cup = st.slider("Cup match", 0.0, 1.0, 0.90, 0.05)
    w_friendly = st.slider("Friendly", 0.0, 1.0, 0.35, 0.05,
                           help="A friendly is evidence about a squad, not a team.")
    kind_weight = {"League": w_league, "Cup": w_cup, "Friendly": w_friendly,
                   "Unknown": 0.7}
    st.divider()
    with st.expander("Model parameters"):
        rho = st.slider("Dixon–Coles ρ", -0.20, 0.10, -0.05, 0.01)
        cov = st.slider("Bivariate covariance", 0.0, 0.35, 0.10, 0.01)
        nb_r = st.slider("Negative-binomial r", 2.0, 30.0, 8.0, 1.0)
        rate_sd = st.slider("Rate uncertainty (log sd)", 0.0, 0.60, 0.28, 0.02)
        corner_line = st.select_slider("Corners line",
                                       [7.5, 8.5, 9.5, 10.5, 11.5], 8.5)

paste_tab, data_tab, pred_tab, method_tab = st.tabs(
    ["1 · Paste data", "2 · Parsed data", "3 · Prediction", "Method"])

# ------------------------------------------------------------------- paste
with paste_tab:
    c1, c2 = st.columns([1, 1])
    if c1.button("Load the bundled example", width="stretch"):
        st.session_state["raw"] = open(SAMPLE, encoding="utf-8").read()
    if c2.button("Clear", width="stretch"):
        st.session_state["raw"] = ""
        st.session_state.pop("pred", None)
    raw = st.text_area(
        "Paste the FootyStats match pages — five for each team",
        value=st.session_state.get("raw", ""), height=340,
        placeholder="Copy each match page (Ctrl+A, Ctrl+C on the FootyStats "
                    "match page) and paste them one after another here.")
    st.session_state["raw"] = raw
    st.caption("The parser needs four things from each page: the date line, "
               "the `TeamA vs TeamB` line, `Final Results` with the score, and "
               "the `Data` block. Everything else on the page is ignored. "
               "Pages pasted twice — the head-to-head appears in both teams' "
               "sets — are de-duplicated.")

raw = st.session_state.get("raw", "")
if not raw.strip():
    with data_tab:
        st.info("Paste some match pages on the first tab to begin.",
                icon=":material/content_paste:")
    with pred_tab:
        st.info("Nothing parsed yet.", icon=":material/insights:")
    st.stop()

df, ts, dropped = do_parse(raw)
if df.empty or len(ts) < 2:
    with data_tab:
        st.error("No complete matches were found. Each page needs its date "
                 "line, the `vs` line, `Final Results` and the `Data` block.",
                 icon=":material/error:")
    st.stop()

# ------------------------------------------------------------- parsed data
with data_tab:
    pages = len(df) + len(dropped)
    st.subheader(f"{pages} pages read · {len(df)} distinct matches")
    if len(dropped):
        rows = ", ".join(
            f"{r.home} {int(r.hg)}-{int(r.ag)} {r.away} on {r.date:%d %b %Y}"
            for _, r in dropped.iterrows())
        st.info(
            f"**{len(dropped)} page(s) were the same match pasted twice** and "
            f"counted once: {rows}. The head-to-head between the two teams "
            "you are predicting appears in both teams' sets of five — counting "
            "it twice would double-weight that fixture.",
            icon=":material/content_copy:")
    warn = P.warnings_for(df, ts)
    for w in warn:
        st.warning(w, icon=":material/report:")

    c1, c2, c3 = st.columns([2, 2, 1])
    home = c1.selectbox("Home team", ts, index=min(1, len(ts) - 1))
    away = c2.selectbox("Away team", [t for t in ts if t != home], index=0)
    c3.metric("Teams seen", len(ts))

    base = M.sample_baselines(df)
    ph = M.team_profile(df, home, base, half_life, kind_weight)
    pa = M.team_profile(df, away, base, half_life, kind_weight)

    st.markdown("#### Every match read from the text")
    show = df[["date", "competition", "kind", "home", "hg", "ag", "away",
               "h_xg", "a_xg", "h_corners", "a_corners", "h_shots", "a_shots",
               "h_sot", "a_sot", "h_cards", "a_cards", "h_possession"]].copy()
    show["date"] = show["date"].dt.date
    st.dataframe(show, width="stretch", hide_index=True,
                 column_config={
                     "kind": st.column_config.TextColumn("Type", width="small"),
                     "hg": st.column_config.NumberColumn("H", width="small"),
                     "ag": st.column_config.NumberColumn("A", width="small"),
                 })

    st.markdown("#### Match importance and weight")
    st.caption("Recency and match type combine into the weight each match "
               "carries. Adjust either in the sidebar.")
    wc1, wc2 = st.columns(2)
    for col, prof in ((wc1, ph), (wc2, pa)):
        with col:
            st.markdown(f"**{prof['team']}** — form {prof.get('form','')} "
                        f"(oldest → newest)")
            mt = prof["matches"][["date", "opponent", "venue", "gf", "ga",
                                  "xg_for", "xg_against", "corners_for",
                                  "corners_against", "kind", "weight"]].copy()
            mt["date"] = mt["date"].dt.date
            st.dataframe(mt, width="stretch", hide_index=True,
                         column_config={
                             "weight": st.column_config.ProgressColumn(
                                 "Weight", format="%.2f", min_value=0.0,
                                 max_value=1.0)})

    st.markdown("#### Derived indices")
    st.dataframe(M.profile_display([ph, pa]), width="stretch")
    st.caption(
        "**Attack strength** is chances created against the sample average; "
        "**defence strength** is chances allowed, inverted, so above 1 is good "
        "for both. **Attack weakness** is the share of xG not converted, and "
        "**defence weakness** the share conceded above xGA — a team can be "
        "strong and wasteful at the same time. Shrunk figures pull each rate "
        "toward the sample mean, because five matches on their own forecast "
        "worse than the average does.")

    st.divider()
    if st.button("⚽  Predict match", type="primary", width="stretch"):
        ctx = X.build_context(df, ph, pa, base, rho=rho, cov=cov, nb_r=nb_r,
                              rate_sd=rate_sd)
        res = X.run_all(ctx)
        ens = X.ensemble(res)
        ch, ca = M.expected_corners(ph, pa, base)
        st.session_state["pred"] = {
            "home": home, "away": away, "res": res, "ens": ens,
            "ctx": {k: ctx[k] for k in ("lh", "la", "lh_xg", "la_xg",
                                        "lh_goals", "la_goals")},
            "corners": X.corners_markets(ch, ca, corner_line),
        }
        st.success("Prediction ready — open the **Prediction** tab.",
                   icon=":material/check_circle:")

# ------------------------------------------------------------- prediction
with pred_tab:
    pr = st.session_state.get("pred")
    if not pr:
        st.info("Choose the two teams on the **Parsed data** tab and press "
                "**Predict match**.", icon=":material/insights:")
    else:
        e, ctxv = pr["ens"], pr["ctx"]
        H, A = pr["home"], pr["away"]
        st.subheader(f"{H}  vs  {A}")

        pick = max((("home", e["home"], H), ("draw", e["draw"], "Draw"),
                    ("away", e["away"], A)), key=lambda x: x[1])
        cards([
            (H, f"{100*e['home']:.1f}%", f"expected goals {ctxv['lh']:.2f}", "mi-h"),
            ("Draw", f"{100*e['draw']:.1f}%", "&nbsp;", "mi-d"),
            (A, f"{100*e['away']:.1f}%", f"expected goals {ctxv['la']:.2f}", "mi-a"),
            ("Most likely", pick[2], f"{100*pick[1]:.1f}% — "
             f"{e['n_methods']} methods", "mi-n"),
        ])
        tri_bar(e["home"], e["draw"], e["away"], H, A)

        st.markdown("#### Markets")
        m1, m2 = st.columns(2)
        with m1:
            meter("Both teams to score — yes", e["btts"])
            meter("Both teams to score — no", 1 - e["btts"])
            meter("Over 2.5 goals", e["over25"])
            meter("Under 2.5 goals", 1 - e["over25"])
        with m2:
            c = pr["corners"]
            meter(f"Corners over {c['line']}", c["over"],
                  f"  ·  {c['expected']:.1f} expected")
            meter(f"Corners under {c['line']}", c["under"])
            meter("Over 1.5 goals", e["over15"])
            meter("Over 3.5 goals", e["over35"])

        st.markdown("#### Correct score")
        m = np.asarray(e["matrix"])
        top = X.top_scores(m, 6)
        cards([(f"{s}", f"{100*p:.1f}%", "&nbsp;", "mi-n") for s, p in top[:4]])
        mat = pd.DataFrame(100 * m,
                           index=[f"{H[:14]} {i}" for i in range(m.shape[0])],
                           columns=[f"{A[:14]} {j}" for j in range(m.shape[1])])
        vmax = float(np.nanmax(mat.to_numpy())) if mat.size else 1.0
        st.dataframe(
            mat.style.map(lambda v: _shade(v, vmax)).format("{:.2f}"),
            width="stretch", height=340)
        st.caption("Every cell is the probability of that exact full-time "
                   "score, in percent. Rows are the home team's goals.")

        st.markdown("#### Goal totals")
        idx = np.arange(m.shape[0])
        tot = idx[:, None] + idx[None, :]
        dist = np.array([float(m[tot == k].sum()) for k in range(9)])
        st.bar_chart(pd.DataFrame({"probability %": 100 * dist},
                                  index=[str(k) for k in range(9)]),
                     height=220, color="#2fa37f")

        st.markdown("#### Corners distribution")
        cc = pr["corners"]
        st.bar_chart(pd.DataFrame({"probability %": 100 * cc["dist"][:21]},
                                  index=[str(k) for k in cc["ks"][:21]]),
                     height=200, color="#5b5f97")

        st.markdown("#### What each method said")
        rows = []
        for r in pr["res"]:
            if "error" in r:
                rows.append({"Method": r["method"], "Note": r["error"]})
                continue
            rows.append({
                "Method": r["method"], "1": 100 * r["home"],
                "X": 100 * r["draw"], "2": 100 * r["away"],
                "BTTS": 100 * r.get("btts", np.nan),
                "O2.5": 100 * r.get("over25", np.nan),
                "λ home": r.get("lh"), "λ away": r.get("la"),
                "Note": r.get("note", "")})
        mt = pd.DataFrame(rows)
        st.dataframe(mt, width="stretch", hide_index=True,
                     column_config={c: st.column_config.NumberColumn(
                         c, format="%.1f%%") for c in ("1", "X", "2",
                                                       "BTTS", "O2.5")})
        st.bar_chart(mt.set_index("Method")[["1", "X", "2"]], height=300,
                     color=["#2fa37f", "#8186c0", "#d1613c"])
        st.warning(
            f"The methods disagree by **{100*e['spread']:.1f} points** on the "
            "home win (standard deviation). That is *not* a confidence "
            "interval. Most of these methods are the same two scoring rates "
            "under different count distributions, so their agreement is close "
            "to guaranteed and says nothing about whether the rates are right.",
            icon=":material/warning:")

# ----------------------------------------------------------------- method
with method_tab:
    st.markdown("""
### What the app does

It reads pasted FootyStats match pages, keeps the four blocks that matter,
builds each team's rates from them, and runs thirteen prediction methods over
those rates.

### The methods

| | method | what makes it different |
|---|---|---|
| 1 | Poisson (goals) | independent Poisson on goal rates |
| 2 | Poisson (xG) | the same, driven by xG |
| 3 | Dixon–Coles | Poisson plus the low-score dependence correction |
| 4 | Bivariate Poisson | a shared component makes the two scores correlated |
| 5 | Negative binomial | allows scores more spread out than Poisson |
| 6 | Skellam | exact distribution of the goal difference |
| 7 | Monte Carlo | simulates with the *rates themselves* uncertain |
| 8 | Shots × conversion | goals from shot volume and finishing rate |
| 9 | Empirical resample | resamples the scorelines actually produced |
| 10 | Bradley–Terry | strength fitted to results, Davidson draw term |
| 11 | Elo | sequential ratings over the parsed matches |
| 12 | Form logistic | ordered logit on the points-per-game gap |
| 13 | Weakness-adjusted | rates nudged by the finishing and keeping gaps |

### How the weighting works

Each match carries `recency × importance`. Recency is exponential decay on
days before the most recent match, so a fixture from the previous season is
worth a fraction of last weekend's. Importance is read from the competition
named on the page — a **friendly** is evidence about a squad rather than a
team and is heavily discounted, a cup tie sits between the two. Both are
adjustable, and every per-match weight is shown on the Parsed data tab rather
than applied silently.

### Shrinkage, and why the numbers look tame

Five matches is about ten numbers. An unshrunk five-match scoring rate is a
**worse** forecast than simply using the average, because almost all of its
movement is noise — one 4-2 shifts it by 0.4 goals a game. Every rate here is
pulled toward the sample mean by `n/(n+4)`, so a team's own record carries a
little over half the weight. That is why the probabilities sit closer to the
middle than a five-match record might suggest. It is the correct response to a
small sample, not timidity.

### What this cannot do

**Thirteen methods is not thirteen opinions.** Ten of them are fed by the same
two scoring rates and differ only in the count distribution wrapped around
them. They will agree, and their agreement is not evidence. The real
uncertainty is in the rates, and on five matches a side it is large — wide
enough that the gap between a 32% and a 43% outcome here is not something to
lean on.

There is also **no market price** in this data and **no measured track record**:
nothing in the app has been validated against out-of-sample results, because
the input is a single fixture pasted by hand. Treat the output as a structured
reading of the form you pasted, not as a tested forecast.

---

*Probabilistic decision support, not a guaranteed outcome. Gamble responsibly —
Gambling Help 1800 858 858,
[gamblinghelponline.org.au](https://www.gamblinghelponline.org.au).*
""")
