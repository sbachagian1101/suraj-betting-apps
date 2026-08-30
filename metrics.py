"""Turn parsed matches into team rates, strength and weakness indices.

Five matches per team is about ten numbers. Left alone, rates built from that
swing wildly - a 4-2 win moves a five-match attack rate by 0.4 goals a game -
so every rate here is **shrunk toward the sample mean**, by the standard
`n/(n+k)` credibility weight. With an effective sample of five and k=4 the
team's own record carries a little over half the weight and the sample average
carries the rest. That is not conservatism for its own sake: an unshrunk
five-match rate is a worse forecast than the league average, and shrinkage is
the arithmetic that says by how much.

Match weight combines two things the user asked to be separated:

* **recency** - exponential decay on days before the most recent match. A
  fixture from the previous season should not count the same as last weekend.
* **importance** - a friendly is evidence about a squad, not about a team; it
  is downweighted, and a cup tie sits between the two.

Both are exposed so they can be overridden per match rather than taken on
trust.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import parser as P

# how much a match of each kind counts
KIND_WEIGHT = {"League": 1.00, "Cup": 0.90, "Friendly": 0.35, "Unknown": 0.70}
HALF_LIFE_DAYS = 60.0
SHRINK_K = 4.0                  # prior strength, in matches
HOME_ADV_PRIOR = 1.10           # goal-rate multiplier, shrunk toward
MAX_GOALS = 8


def match_weights(tm: pd.DataFrame, half_life: float = HALF_LIFE_DAYS,
                  kind_weight: dict | None = None) -> pd.Series:
    if tm.empty:
        return pd.Series(dtype=float)
    kw = {**KIND_WEIGHT, **(kind_weight or {})}
    newest = tm["date"].max()
    age = (newest - tm["date"]).dt.days.clip(lower=0).astype(float)
    recency = 0.5 ** (age / max(half_life, 1e-9))
    imp = tm["kind"].map(lambda k: kw.get(k, 0.7)).astype(float)
    return (recency * imp).rename("weight")


def _wmean(values, weights, default=np.nan) -> float:
    v = np.asarray(values, dtype=float)
    w = np.asarray(weights, dtype=float)
    ok = np.isfinite(v) & np.isfinite(w) & (w > 0)
    if not ok.any():
        return default
    return float(np.sum(v[ok] * w[ok]) / np.sum(w[ok]))


def _shrink(rate: float, prior: float, n_eff: float, k: float = SHRINK_K) -> float:
    if not np.isfinite(rate):
        return prior
    a = n_eff / (n_eff + k)
    return a * rate + (1 - a) * prior


def sample_baselines(df: pd.DataFrame) -> dict:
    """League-ish averages taken from every match on the page."""
    if df.empty:
        return {"goals": 1.35, "xg": 1.35, "corners": 4.8, "home_adv": HOME_ADV_PRIOR}
    goals = float(np.nanmean(np.r_[df.hg.to_numpy(float), df.ag.to_numpy(float)]))
    xg = goals
    if "h_xg" in df and df.h_xg.notna().any():
        xg = float(np.nanmean(np.r_[df.h_xg.to_numpy(float),
                                    df.a_xg.to_numpy(float)]))
    corners = np.nan
    if "h_corners" in df and df.h_corners.notna().any():
        corners = float(np.nanmean(np.r_[df.h_corners.to_numpy(float),
                                         df.a_corners.to_numpy(float)]))
    hg, ag = float(df.hg.mean()), float(df.ag.mean())
    raw_adv = (hg / ag) ** 0.5 if ag > 0.05 else HOME_ADV_PRIOR
    # nine matches cannot measure home advantage; meet the prior half way
    adv = float(np.clip(0.5 * raw_adv + 0.5 * HOME_ADV_PRIOR, 0.85, 1.45))
    return {"goals": max(goals, 0.2), "xg": max(xg, 0.2),
            "corners": corners if np.isfinite(corners) else 4.8,
            "home_adv": adv, "raw_home_adv": raw_adv}


def team_profile(df: pd.DataFrame, team: str, base: dict,
                 half_life: float = HALF_LIFE_DAYS,
                 kind_weight: dict | None = None) -> dict:
    tm = P.team_matches(df, team)
    if tm.empty:
        return {"team": team, "n": 0}
    w = match_weights(tm, half_life, kind_weight)
    n_eff = float(w.sum())

    gf = _wmean(tm.gf, w)
    ga = _wmean(tm.ga, w)
    xgf = _wmean(tm.xg_for, w, gf)
    xga = _wmean(tm.xg_against, w, ga)
    cf = _wmean(tm.corners_for, w, base["corners"])
    ca = _wmean(tm.corners_against, w, base["corners"])

    gf_s = _shrink(gf, base["goals"], n_eff)
    ga_s = _shrink(ga, base["goals"], n_eff)
    xgf_s = _shrink(xgf, base["xg"], n_eff)
    xga_s = _shrink(xga, base["xg"], n_eff)
    cf_s = _shrink(cf, base["corners"], n_eff)
    ca_s = _shrink(ca, base["corners"], n_eff)

    # Strength is about chances created and allowed; weakness is the gap
    # between chances and what actually went in at either end. A team can be
    # strong and wasteful at once, and the two indices say so separately.
    atk_strength = xgf_s / base["xg"]
    def_strength = base["xg"] / max(xga_s, 1e-6)
    atk_weak = (xgf - gf) / xgf if (np.isfinite(xgf) and xgf > 1e-6) else 0.0
    def_weak = (ga - xga) / xga if (np.isfinite(xga) and xga > 1e-6) else 0.0

    pts = tm.result.map({"W": 3, "D": 1, "L": 0}).astype(float)
    return {
        "team": team, "n": int(len(tm)), "n_eff": n_eff,
        "gf": gf, "ga": ga, "xgf": xgf, "xga": xga,
        "gf_s": gf_s, "ga_s": ga_s, "xgf_s": xgf_s, "xga_s": xga_s,
        "corners_for": cf, "corners_against": ca,
        "corners_for_s": cf_s, "corners_against_s": ca_s,
        "atk_strength": atk_strength, "def_strength": def_strength,
        "atk_weakness": atk_weak, "def_weakness": def_weak,
        "ppg": _wmean(pts, w),
        "btts_rate": _wmean(((tm.gf > 0) & (tm.ga > 0)).astype(float), w),
        "over25_rate": _wmean(((tm.gf + tm.ga) > 2.5).astype(float), w),
        "cs_rate": _wmean((tm.ga == 0).astype(float), w),
        "fts_rate": _wmean((tm.gf == 0).astype(float), w),
        "shots_for": _wmean(tm.shots_for, w),
        "sot_for": _wmean(tm.sot_for, w),
        "shots_against": _wmean(tm.shots_against, w),
        "sot_against": _wmean(tm.sot_against, w),
        "possession": _wmean(tm.possession_for, w),
        "cards_for": _wmean(tm.cards_for, w),
        "form": "".join(tm.result.tolist()[::-1]),
        "matches": tm.assign(weight=w),
    }


def expected_goals(home: dict, away: dict, base: dict,
                   use_xg: bool = True) -> tuple[float, float]:
    """Rates for the fixture, from attack against defence plus home ground."""
    mu = base["xg"] if use_xg else base["goals"]
    if use_xg:
        ha, hd = home["xgf_s"] / mu, home["xga_s"] / mu
        aa, ad = away["xgf_s"] / mu, away["xga_s"] / mu
    else:
        ha, hd = home["gf_s"] / mu, home["ga_s"] / mu
        aa, ad = away["gf_s"] / mu, away["ga_s"] / mu
    adv = base["home_adv"]
    lh = mu * ha * ad * adv
    la = mu * aa * hd / adv
    return float(np.clip(lh, 0.15, 6.0)), float(np.clip(la, 0.15, 6.0))


def expected_corners(home: dict, away: dict, base: dict) -> tuple[float, float]:
    mu = max(base["corners"], 0.5)
    ch = mu * (home["corners_for_s"] / mu) * (away["corners_against_s"] / mu)
    ca = mu * (away["corners_for_s"] / mu) * (home["corners_against_s"] / mu)
    return float(np.clip(ch, 0.5, 12.0)), float(np.clip(ca, 0.5, 12.0))


def profile_table(profiles: list[dict]) -> pd.DataFrame:
    rows = []
    for p in profiles:
        rows.append({
            "Team": p["team"], "Matches": p["n"],
            "Effective n": round(p.get("n_eff", 0), 2),
            "Form (old→new)": p.get("form", ""),
            "PPG": p.get("ppg"), "Goals for": p.get("gf"),
            "Goals against": p.get("ga"),
            "xG": p.get("xgf"), "xGA": p.get("xga"),
            "xG (shrunk)": p.get("xgf_s"), "xGA (shrunk)": p.get("xga_s"),
            "Attack strength": p.get("atk_strength"),
            "Defence strength": p.get("def_strength"),
            "Attack weakness": p.get("atk_weakness"),
            "Defence weakness": p.get("def_weakness"),
            "Corners for": p.get("corners_for"),
            "Corners against": p.get("corners_against"),
            "Shots": p.get("shots_for"), "On target": p.get("sot_for"),
            "Possession %": p.get("possession"),
            "BTTS rate": p.get("btts_rate"), "Over 2.5 rate": p.get("over25_rate"),
            "Clean sheets": p.get("cs_rate"), "Failed to score": p.get("fts_rate"),
        })
    return pd.DataFrame(rows)


def profile_display(profiles: list[dict]) -> pd.DataFrame:
    """The profile table transposed for reading, as strings.

    Transposing puts a team per column, which mixes the text row (form) with
    the numeric ones. Arrow cannot type a column holding both, and Streamlit
    ships dataframes to the browser as Arrow, so the table fails to render at
    all. Formatting to strings first is what makes the transpose safe.
    """
    t = profile_table(profiles)
    out = t.set_index("Team").T
    return out.map(lambda v: "—" if v is None or (isinstance(v, float) and not
                                                  np.isfinite(v))
                   else (f"{v:,.2f}" if isinstance(v, (int, float, np.floating))
                         and not isinstance(v, bool) else str(v)))


if __name__ == "__main__":
    d = P.parse(open("sample_data/sturm_graz_ii_vs_rapid_wien_ii.txt",
                     encoding="utf-8").read())
    base = sample_baselines(d)
    print("baselines:", {k: round(v, 3) for k, v in base.items()})
    ts = P.teams(d)[:2]
    home, away = ts[1], ts[0]          # Sturm Graz II at home
    ph = team_profile(d, home, base)
    pa = team_profile(d, away, base)
    t = profile_table([ph, pa])
    pd.set_option("display.width", 240)
    print(t.T.to_string())
    lh, la = expected_goals(ph, pa, base)
    print(f"\nexpected goals  {home} {lh:.2f}  |  {away} {la:.2f}")
    ch, ca = expected_corners(ph, pa, base)
    print(f"expected corners {ch:.2f} / {ca:.2f}  total {ch+ca:.2f}")
    print("\nmatch weights for", home)
    print(ph["matches"][["date", "opponent", "venue", "gf", "ga", "kind",
                         "weight"]].to_string(index=False))
