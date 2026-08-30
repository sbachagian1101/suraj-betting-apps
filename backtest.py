"""Score the model against the matches whose results are known.

Five matches. That is enough to catch a model that is badly wrong and nowhere
near enough to establish that one is right, so the numbers below are reported
with the sample size attached and nothing is tuned on them.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import poisson

import model as MD
import panel as PN

# file -> (home goals, away goals). Read off the Final Results panels.
RESULTS = {
    "aik_vs_hammarby.txt": (3, 2),
    "honefoss_vs_follo.txt": (2, 1),
    "bodo_vs_rosenborg.txt": (4, 2),
    "stabaek_vs_staal.txt": (2, 1),
    "dinamo_minsk_ii_vs_smorgon.txt": (3, 2),
}


def run(**kw) -> pd.DataFrame:
    rows = []
    for f, (hg, ag) in RESULTS.items():
        ps = PN.parse_panels(open(f"sample_data/{f}", encoding="utf-8").read())
        if len(ps) < 2:
            continue
        home, away = ps[0], ps[1]
        r = MD.predict(home, away, **kw)
        m = r["matrix"]
        tops = MD.top_scores(m, 10)
        order = [s for s, _ in tops]
        actual = (hg, ag)
        res = "H" if hg > ag else ("D" if hg == ag else "A")
        pick_res = ("H" if r["home"] >= max(r["draw"], r["away"])
                    else ("D" if r["draw"] >= r["away"] else "A"))
        rows.append({
            "match": f"{home['team'][:18]} v {away['team'][:18]}",
            "lh": r["lh"], "la": r["la"], "exp": r["exp_goals"],
            "pick": f"{r['pick'][0]}-{r['pick'][1]}",
            "pick_p": 100 * r["pick_prob"],
            "actual": f"{hg}-{ag}", "actual_goals": hg + ag,
            "p_actual": 100 * MD.score_prob(m, hg, ag),
            "rank_actual": (order.index(actual) + 1) if actual in order else None,
            "exact": r["pick"] == actual,
            "in_top3": actual in order[:3],
            "in_top5": actual in order[:5],
            "res_ok": pick_res == res,
            "btts_ok": (r["btts"] >= 0.5) == (hg > 0 and ag > 0),
            "o25_ok": (r["over25"] >= 0.5) == (hg + ag > 2.5),
            "logloss": -np.log(max(MD.score_prob(m, hg, ag), 1e-12)),
        })
    return pd.DataFrame(rows)


def baseline_logloss(mean_goals: float = 1.35) -> float:
    """A league-average Poisson that knows nothing about either team."""
    m = MD.score_matrix(mean_goals, mean_goals, rho=0.0)
    tot = []
    for f, (hg, ag) in RESULTS.items():
        tot.append(-np.log(max(MD.score_prob(m, hg, ag), 1e-12)))
    return float(np.mean(tot))


def report(df: pd.DataFrame, label: str = "") -> None:
    n = len(df)
    print(f"\n===== {label or 'model'} — {n} matches =====")
    print(df[["match", "lh", "la", "exp", "pick", "pick_p", "actual",
              "p_actual", "rank_actual", "res_ok"]].to_string(
        index=False, float_format=lambda x: f"{x:.2f}"))
    print(f"\n  exact scoreline      {df.exact.sum()}/{n}")
    print(f"  actual in top 3      {df.in_top3.sum()}/{n}")
    print(f"  actual in top 5      {df.in_top5.sum()}/{n}")
    print(f"  1X2 correct          {df.res_ok.sum()}/{n}")
    print(f"  BTTS side correct    {df.btts_ok.sum()}/{n}")
    print(f"  Over/Under 2.5 right {df.o25_ok.sum()}/{n}")
    print(f"  mean expected goals  {df.exp.mean():.2f}   "
          f"actual {df.actual_goals.mean():.2f}")
    print(f"  scoreline log-loss   {df.logloss.mean():.3f}   "
          f"(league-average Poisson {baseline_logloss():.3f})")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "sweep":
        print("xg_weight sweep — nothing here is tuned on, five matches "
              "cannot choose a parameter")
        for xw in (0.0, 0.25, 0.5, 0.75, 1.0):
            d = run(xg_weight=xw)
            print(f"  xg_weight {xw:.2f}  exp {d.exp.mean():.2f}  "
                  f"logloss {d.logloss.mean():.3f}  "
                  f"top3 {d.in_top3.sum()}/{len(d)}  "
                  f"1X2 {d.res_ok.sum()}/{len(d)}")
    else:
        report(run(), "default (venue 0.65, xG 0.50, Dixon-Coles)")
