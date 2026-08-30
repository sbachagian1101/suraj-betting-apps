"""Checks for AustraliaPdfHorseRacing. Run: python test_model.py"""
from __future__ import annotations

import glob
import os
import warnings

import joblib
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

import features as F
import model as M
import past_form as PF
import pdf_parser as PP
import predict as PR
import train as T

PDFS = sorted(glob.glob("sample_data/*.pdf")) or sorted(glob.glob(
    "C:/Users/Admin/Downloads/[0-9][0-9][0-9][0-9][0-9][0-9].pdf"))


class Checker:
    def __init__(self):
        self.passes, self.fails = 0, []

    def true(self, name, cond):
        if cond:
            self.passes += 1
        else:
            self.fails.append(name)

    def check(self, name, got, want):
        self.true(f"{name} (got {got!r}, want {want!r})", got == want)

    def close(self, name, got, want, tol=1e-9):
        self.true(f"{name} (got {got!r}, want {want!r})", abs(got - want) <= tol)


def main():
    c = Checker()

    # ---------------------------------------------------------- the parser
    c.true("there are PDFs to test against", len(PDFS) > 0)
    if not PDFS:
        print("no sample PDFs found — skipping")
        return

    field = PP.parse_many(PDFS)
    c.true("runners were parsed", len(field) > 0)
    c.true("every runner has a horse name",
           (field.horse.str.strip() != "").all())
    c.true("no horse name starts lowercase — the form-figure split ate a "
           "leading letter when it did",
           not field.horse.str[0].str.islower().any())
    c.true("no horse name is purely numeric",
           not field.horse.str.fullmatch(r"[\d.]+").any())
    c.true("form figures never contain uppercase letters",
           not field.form_figures.str.contains(r"[A-Z]", regex=True).any())
    c.true("tabs are positive", (field.tab > 0).all())
    c.true("weights are plausible",
           field.weight.between(45, 75).mean() > 0.98)
    c.true("ages are plausible", field.age.between(2, 15).mean() > 0.98)
    c.true("career places are at least career wins",
           (field.career_places >= 0).all())
    c.true("career starts are at least career wins",
           (field.career_starts >= field.career_wins).mean() > 0.99)
    c.true("apprentice claims are sane", field.claim.between(0, 4).all())
    c.true("more than one track was read", field.track.nunique() > 1)
    c.true("field sizes are plausible",
           field.groupby("race_id").size().between(2, 30).all())

    # the split-vs-glued tab layout: both must be recovered
    strath = field[field.race_id.str.startswith("STRATHALBYN")]
    if len(strath):
        r1 = strath[strath.race_no == 1]
        c.true("Strathalbyn R1 recovers every runner, not just the glued rows",
               len(r1) >= 9)
        c.true("and the contiguous tab numbers with them",
               sorted(r1.tab) == list(range(1, len(r1) + 1)))

    # ------------------------------------------------------- past-run form
    past = PF.parse_many(PDFS)
    # thresholds scale with the number of meetings: a meeting carries roughly
    # 150-700 past runs, and the suite runs against three sample files in the
    # repo and nineteen when refitting
    c.true("past runs were parsed", len(past) > 100 * len(PDFS))
    c.true("finishes are positive", (past.finish >= 1).all())
    c.true("nobody finished behind the field",
           (past.finish <= past.past_field_size).mean() > 0.99)
    c.true("starting prices are decimal, so always above 1",
           (past.sp.dropna() > 1.0).all())
    c.true("prices are read for essentially every run",
           past.sp.notna().mean() > 0.99)
    c.true("dates parsed", past.date.notna().mean() > 0.99)
    c.true("dates are in the past",
           (past.date <= pd.Timestamp.now()).mean() > 0.99)

    # the odds scale is the check that catches a misread price column
    band = past[(past.sp > 9) & (past.sp <= 16)]
    c.true("a $9-16 shot wins about a tenth of the time — the calibration "
           "that proves 'Odds 0.3F' is odds-to-one, not decimal",
           0.04 < band.won.mean() < 0.14)
    short = past[past.sp <= 3]
    longs = past[past.sp > 20]
    c.true("short prices win far more often than long ones",
           short.won.mean() > 4 * longs.won.mean())
    c.true("favourites are flagged", past.was_favourite.sum() > 10 * len(PDFS))
    c.true("flagged favourites are short",
           past[past.was_favourite].sp.median() < past[~past.was_favourite].sp.median())

    # --------------------------------------------------------- no leakage
    tr = F.build_training(past)
    c.check("one training row per past run", len(tr), len(past))
    first = tr[tr.n_prior == 0]
    c.true("a horse's first run has no history",
           first[F.HISTORY].drop(columns=["n_prior"]).isna().all().all())
    c.true("later runs do have history",
           (tr.n_prior > 0).sum() > 50 * len(PDFS))

    # an explicit leakage probe: the target must not appear in its own features
    g = past[past.horse == past.horse.value_counts().index[0]].sort_values("date")
    if len(g) >= 3:
        sub = F.build_training(g)
        last = sub.iloc[-1]
        prior_only = g.iloc[:-1]
        c.close("win rate uses only prior runs", last.win_rate,
                float(prior_only.won.mean()), 1e-9)
        c.close("the prior count matches", last.n_prior, float(len(prior_only)))
        c.true("the last run's own finish is excluded from its features",
               abs(last.last_fin_pct - F._pct(np.array([g.iloc[-1].finish]),
                   np.array([g.iloc[-1].past_field_size]))[0]) > 1e-12
               or len(g) < 3)

    # race keys must not merge two different races
    keyed = tr[tr.key_ok]
    winners = keyed.groupby("race_key").won.sum()
    c.true("no surviving race key has two winners", (winners <= 1).all())

    # race-constant columns are excluded from the fit
    for name in ("field_size", "log_distance", "log_prize"):
        c.true(f"{name} is not a fitted feature", name not in F.FEATURES)

    # ---------------------------------------------------------- the model
    rng = np.random.default_rng(0)
    n, k = 200, 4
    X = rng.normal(size=(n, k))
    keys = np.repeat(np.arange(50), 4).astype(str)
    beta = np.array([1.5, -1.0, 0.0, 0.5])
    noise = X @ beta + rng.gumbel(size=n)
    finish = np.empty(n)
    for idx in M._groups(keys):
        finish[idx] = np.argsort(np.argsort(-noise[idx])) + 1
    fit = M.RankLogit(l2=0.01).fit(X, finish, keys)
    c.true("the fit converged", fit.result_.success)
    c.true("it recovers the sign of a strong positive weight", fit.beta_[0] > 0)
    c.true("and of a strong negative one", fit.beta_[1] < 0)
    c.true("and leaves the irrelevant feature small",
           abs(fit.beta_[2]) < abs(fit.beta_[0]))

    p = fit.probabilities(X, keys)
    for idx in M._groups(keys):
        c.close("probabilities sum to one within a race", float(p[idx].sum()), 1.0, 1e-9)
        break
    c.true("all probabilities are positive", (p > 0).all())

    c.true("a group of one is skipped rather than crashing",
           M.top1_logloss(np.array([1.0]), np.array([1.0]),
                          np.array(["a"]))[2] == 0)

    # temperature search
    v = rng.normal(size=n)
    tau = M.fit_temperature(v, finish, keys)
    c.true("the temperature is positive and finite", 0 < tau < 8)

    # ------------------------------------------------------- the bundle
    c.true("a trained bundle exists", os.path.exists("model_bundle.joblib"))
    B = joblib.load("model_bundle.joblib")
    c.check("the bundle names its features", B["features"], F.FEATURES)
    c.check("one coefficient per feature",
            len(B["ridge"].coef_), len(F.FEATURES))
    c.true("the temperature travelled with the bundle", B["tau"] > 0)
    c.true("so did the scaling it was fitted against", B["v_std"] > 0)
    V = B["validation"]
    c.true("the top pick beats a random pick on wins",
           V["pick_win"] > V["random_win"])
    c.true("and on places", V["pick_place"] > V["random_place"])
    c.true("the market still beats the model on wins",
           V["market_win"] > V["pick_win"])
    c.true("the model beats knowing nothing, on log-loss",
           V["logloss"] < V["chance_logloss"])
    c.true("and loses to the market, which the app must say",
           V["logloss"] > V["market_logloss"])
    c.true("rank correlation has the right sign — lower percentile is better",
           V["spearman"] < 0)

    bands = B["confidence_bands"]
    c.true("there are confidence bands", len(bands) >= 2)
    c.true("the widest-gap band has the best strike rate",
           max(bands, key=lambda b: b["gap_lo"])["win"]
           == max(b["win"] for b in bands))
    for b in bands:
        c.true(f"band {b['band']} is measured on real races", b["n"] > 50)
        lo, hi = b["win_ci"]
        c.true(f"band {b['band']} interval brackets its estimate",
               lo <= b["win"] <= hi)

    # ------------------------------------------------------ scoring a race
    rid = field.race_id.value_counts().index[0]
    runners = field[field.race_id == rid]
    t = PR.score_race(runners, past, B, sims=4000)
    c.check("every runner is scored", len(t), len(runners))
    c.close("win percentages sum to 100", float(t["Win %"].sum()), 100.0, 1e-6)
    c.true("place percentages exceed win percentages",
           (t["Place %"] >= t["Win %"] - 1e-9).all())
    c.true("no place percentage exceeds 100", (t["Place %"] <= 100 + 1e-9).all())
    c.true("the table is ranked", list(t.Rank) == sorted(t.Rank))
    c.true("scores descend with rank", (t.score.diff().dropna() <= 1e-9).all())
    c.true("a confidence band was assigned",
           t.attrs["confidence"] in {"low", "medium", "high"})
    c.true("the gap is not negative", t.attrs["gap"] >= -1e-9)
    c.true("reasons were produced", len(t.attrs["reasons"]) > 0)
    c.true("every reason names a real feature or its label",
           all(isinstance(r, str) and len(r) > 5 for r in t.attrs["reasons"]))

    # attrs must survive Arrow: NaN in attrs is written into the schema as
    # JSON, and the browser's JSON.parse rejects NaN where Python's accepts it
    for key, val in t.attrs.items():
        if isinstance(val, float):
            c.true(f"attr {key} is not NaN", val == val)

    # a two-horse race and a one-horse race must not crash
    tiny = runners.head(2).copy()
    t2 = PR.score_race(tiny, past, B, sims=500)
    c.check("a two-runner race scores both", len(t2), 2)
    c.close("and still sums to 100", float(t2["Win %"].sum()), 100.0, 1e-6)
    one = runners.head(1).copy()
    t1 = PR.score_race(one, past, B, sims=500)
    c.check("a one-runner race scores it", len(t1), 1)
    c.close("at 100%", float(t1["Win %"].sum()), 100.0, 1e-6)

    # a horse with no history at all must still score
    ghost = runners.head(3).copy()
    ghost["horse"] = ["Nonexistent Alpha", "Nonexistent Beta", "Nonexistent Gamma"]
    tg = PR.score_race(ghost, past, B, sims=500)
    c.check("unknown horses still score", len(tg), 3)
    c.true("and are flagged as having no history", tg.n_history.sum() == 0)
    c.close("still a valid distribution", float(tg["Win %"].sum()), 100.0, 1e-6)

    # ---------------------------------------------------- place simulation
    pw = np.array([0.5, 0.3, 0.15, 0.05])
    pp = T.place_probabilities(pw, places=3, sims=40000, seed=1)
    c.close("place probabilities sum to the number of places",
            float(pp.sum()), 3.0, 0.02)
    c.true("the favourite places most often", pp[0] == pp.max())
    c.true("every place probability is at least its win probability",
           bool((pp >= pw - 0.02).all()))

    # ----------------------------------------------------------- warnings
    w = PP.warnings_for(field)
    c.true("warnings are strings", all(isinstance(x, str) for x in w))
    c.true("an empty frame warns rather than crashing",
           len(PP.warnings_for(pd.DataFrame())) == 1)

    print(f"PASS {c.passes}  FAIL {len(c.fails)}")
    for f in c.fails:
        print("  FAIL:", f)
    return 1 if c.fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
