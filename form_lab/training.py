"""Four chronological blocks; frozen base models through final evaluation."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import SCHEMA_VERSION, VERSION
from .features import CATEGORICAL, NUMERIC, Transform, build_features
from .models import ConditionalLogit, blend, fit_blend, fit_temperature, groups, metrics, pl_topk, race_softmax
from .schema import DataError, audit, instant, iso, require, training_issue, validate


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def cat_frame(frame, transform):
    return pd.concat([transform.apply(frame), frame[CATEGORICAL]], axis=1)


def split_frame(frame, blend_start, calibration_start, test_start, min_races=20):
    b, c, t = [instant(x) for x in (blend_start, calibration_start, test_start)]
    require(b < c < t, "Split boundaries must be increasing")
    at = pd.to_datetime(frame.prediction_at, utc=True)
    mask = {"train": at < b, "blend": (at >= b) & (at < c),
            "calibration": (at >= c) & (at < t), "test": at >= t}
    # Keep a meeting and a UTC prediction date entirely within one block.
    allocation = pd.DataFrame({"block": np.select(list(mask.values()), list(mask), default="invalid"),
                               "meeting": frame.meeting_id, "date": at.dt.date})
    for key in ("meeting", "date"):
        require(allocation.groupby(key).block.nunique().max() == 1, f"A {key} crosses a split boundary")
    blocks = {}
    for name, index in mask.items():
        part = frame.loc[index].reset_index(drop=True)
        require(part.race_id.nunique() >= min_races, f"{name} needs at least {min_races} eligible races")
        end = {"train": b, "blend": c, "calibration": t}.get(name)
        if end:
            require((pd.to_datetime(part.label_known_at, utc=True) < end).all(),
                    f"{name}: some results were unavailable before the next block")
        blocks[name] = part
    return blocks


def _paired_interval(combined_rows, baseline_rows, reps=1000):
    # One resampling unit per meeting; dependent races stay together.
    a = pd.DataFrame(combined_rows).set_index("race_id")
    b = pd.DataFrame(baseline_rows).set_index("race_id")
    a["delta"] = a.log_loss - b.log_loss
    sums = a.groupby("meeting_id").delta.agg(["sum", "count"])
    if len(sums) < 10:
        return {"mean_delta_log_loss": float(a.delta.mean()), "interval": None,
                "reason": "Fewer than 10 meetings; interval withheld"}
    rng = np.random.default_rng(741)
    values = sums.to_numpy()
    draws = []
    for _ in range(reps):
        selected = values[rng.integers(0, len(values), size=len(values))]
        draws.append(selected[:, 0].sum() / selected[:, 1].sum())
    return {"mean_delta_log_loss": float(a.delta.mean()),
            "interval": [float(x) for x in np.quantile(draws, [.025, .975])], "resampling_unit": "meeting"}


def train(records, output, *, country, discipline, blend_start, calibration_start, test_start,
          iterations=400, depth=5, min_races=20):
    """No final all-data refit: that would change the tested ensemble."""
    from catboost import CatBoostRanker, Pool
    for r in records:
        validate(r)
    require(len({r['race_id'] for r in records}) == len(records), "Duplicate race snapshots")
    scoped = [r for r in records if r["country"] == country and r["discipline"] == discipline]
    require(bool(scoped), "No races for selected country/discipline")
    frame = build_features(records)
    frame = frame.loc[(frame.country == country) & (frame.discipline == discipline) & frame.eligible].reset_index(drop=True)
    blocks = split_frame(frame, blend_start, calibration_start, test_start, min_races)
    transform = Transform().fit(blocks["train"])
    tr = blocks["train"]
    linear = ConditionalLogit().fit(transform.apply(tr), tr.y.to_numpy(), groups(tr.race_id))
    cb = CatBoostRanker(loss_function="QuerySoftMax", iterations=iterations, depth=depth,
                        learning_rate=.04, l2_leaf_reg=8., random_seed=741, thread_count=2,
                        verbose=False, allow_writing_files=False)
    cb.fit(Pool(cat_frame(tr, transform), label=tr.y, group_id=tr.race_id,
                cat_features=CATEGORICAL))
    probabilities = {}
    for name in ("blend", "calibration", "test"):
        part = blocks[name]
        sl = groups(part.race_id)
        probabilities[name] = [linear.predict(transform.apply(part), sl),
                                race_softmax(cb.predict(cat_frame(part, transform)), sl)]
    weights = fit_blend(probabilities["blend"], blocks["blend"].y.to_numpy(), groups(blocks["blend"].race_id))
    cal = blocks["calibration"]
    cal_p = blend(probabilities["calibration"], weights, groups(cal.race_id))
    temperature = fit_temperature(cal_p, cal.y.to_numpy(), groups(cal.race_id))
    test = blocks["test"]
    sl = groups(test.race_id)
    predictions = {"equal_chance": 1 / test.field_size.to_numpy(),
                   "conditional_logit": probabilities["test"][0], "catboost": probabilities["test"][1],
                   "ensemble": blend(probabilities["test"], weights, sl, temperature)}
    data_kind = "synthetic" if all(r["source"] == "synthetic-test" for r in scoped) else "collector_data_provenance_requires_audit"
    report = {"status": "synthetic_pipeline_check" if data_kind == "synthetic" else "historical_holdout_only", "data_audit": audit(scoped),
              "test_metrics": {}, "test_race_metrics": {}, "subgroups": {}}
    for name, p in predictions.items():
        report["test_metrics"][name], report["test_race_metrics"][name] = metrics(p, test)
    report["ensemble_minus_logit"] = _paired_interval(report["test_race_metrics"]["ensemble"],
                                                      report["test_race_metrics"]["conditional_logit"])
    for col in ("surface", "course_id"):
        report["subgroups"][col] = {}
        for value in test[col].unique():
            subset = test[col].eq(value).to_numpy()
            report["subgroups"][col][value] = metrics(predictions["ensemble"][subset], test.loc[subset].reset_index(drop=True))[0]
    cutoff = max(instant(d) for name in ("train", "blend", "calibration") for d in blocks[name].label_known_at)
    manifest = {
        "engine_version": VERSION, "schema_version": SCHEMA_VERSION, "country": country, "discipline": discipline,
        "data_kind": data_kind,
        "created_at": iso(datetime.now(timezone.utc)), "knowledge_cutoff": iso(cutoff),
        "data_sha256": digest(records), "split_boundaries": {"blend": blend_start, "calibration": calibration_start, "test": test_start},
        "blocks": {name: {"races": d.race_id.nunique(), "runners": len(d),
                          "first_prediction": d.prediction_at.iloc[0], "last_prediction": d.prediction_at.iloc[-1]}
                   for name, d in blocks.items()},
        "models": ["conditional_logit_baseline", "catboost_querysoftmax"], "weights": weights,
        "temperature": temperature, "parameters": {"iterations": iterations, "depth": depth, "l2_logit": linear.l2},
        "categorical_features": CATEGORICAL, "transform": transform.to_dict(), "linear_beta": linear.beta.tolist(),
        "packages": {p: importlib.metadata.version(p) for p in ("numpy", "pandas", "scipy", "catboost")},
        "limitations": ["Dynamic hierarchical Bayesian model not yet implemented", "No real-data performance claim without audited source provenance",
                        "PL top-three assumes a strict complete finishing order", "Holdout is historical, not prospective",
                        "Fixed baseline hyperparameters; repeat evaluation on new periods, not repeated selection on this test"]}
    # Version includes settings, inputs and learned weights, excluding wall-clock time.
    version_body = {k: v for k, v in manifest.items() if k != "created_at"}
    manifest["model_version"] = f"{country}-{discipline}-{digest(version_body)[:16]}"
    output = Path(output)
    require(not output.exists(), "Output directory already exists; use a new model version directory")
    output.mkdir(parents=True)
    cb.save_model(str(output / "catboost.cbm"))
    manifest["catboost_sha256"] = hashlib.sha256((output / "catboost.cbm").read_bytes()).hexdigest()
    (output / "evaluation.json").write_text(json.dumps(report, indent=2, allow_nan=False))
    test_output = test[["race_id", "runner_id", "horse_name", "y", "prediction_at"]].copy()
    for name, p in predictions.items():
        test_output[name] = p
    test_output.to_csv(output / "holdout_predictions.csv", index=False)
    # Publish the manifest last; an interrupted write is not a loadable model.
    temporary_manifest = output / "manifest.pending"
    temporary_manifest.write_text(json.dumps(manifest, indent=2, allow_nan=False))
    temporary_manifest.rename(output / "manifest.json")
    return manifest, report


class Bundle:
    def __init__(self, directory):
        from catboost import CatBoostRanker
        path = Path(directory)
        self.manifest = m = json.loads((path / "manifest.json").read_text())
        require(m["schema_version"] == SCHEMA_VERSION and m["engine_version"] == VERSION, "Model/engine version mismatch")
        version_body = {k: v for k, v in m.items() if k not in {"created_at", "model_version", "catboost_sha256"}}
        expected_version = f"{m['country']}-{m['discipline']}-{digest(version_body)[:16]}"
        require(m["model_version"] == expected_version, "Model manifest checksum mismatch")
        require(m["categorical_features"] == CATEGORICAL, "Categorical schema mismatch")
        require(hashlib.sha256((path / "catboost.cbm").read_bytes()).hexdigest() == m["catboost_sha256"], "Model file checksum mismatch")
        self.transform = Transform.from_dict(m["transform"])
        self.linear = ConditionalLogit()
        self.linear.beta = np.array(m["linear_beta"])
        self.catboost = CatBoostRanker().load_model(str(path / "catboost.cbm"))

    def predict(self, race, history=()):
        validate(race)
        m = self.manifest
        now = datetime.now(timezone.utc)
        require(instant(race["prediction_at"]) <= now, "Prediction snapshot is future-dated")
        require((race["country"], race["discipline"]) == (m["country"], m["discipline"]), "Wrong country/discipline model")
        require(instant(race["prediction_at"]) > instant(m["knowledge_cutoff"]), "Model knows outcomes after this prediction time")
        require(race.get("result") is None, "Live prediction input must not contain the target result")
        require(bool(history), "Canonical historical archive is required to rebuild trained form features")
        for old in history:
            validate(old)
        require(race["race_id"] not in {h["race_id"] for h in history}, "Target race is already in history")
        require(len({h["race_id"] for h in history}) == len(history), "Duplicate history races")
        frame = build_features(list(history) + [race], target_ids={race["race_id"]})
        frame = frame.loc[frame.race_id == race["race_id"]].reset_index(drop=True)
        sl = groups(frame.race_id)
        x = self.transform.apply(frame)
        linear = self.linear.predict(x, sl)
        boosted = race_softmax(self.catboost.predict(cat_frame(frame, self.transform)), sl)
        p = blend([linear, boosted], m["weights"], sl, m["temperature"])
        top3 = pl_topk(p, 3)
        features_digest = digest({"numeric": x.to_dict("list"), "categorical": frame[CATEGORICAL].to_dict("list")})
        rows = []
        for i, row in frame.iterrows():
            contribution = (x.iloc[i] - x.mean()) * self.linear.beta
            indices = contribution.abs().sort_values(ascending=False).index[:3]
            rows.append({"runner_id": row.runner_id, "horse_name": row.horse_name,
                         "baseline_win": float(linear[i]), "catboost_win": float(boosted[i]),
                         "win_probability": float(p[i]), "top3_probability_pl": float(top3[i]),
                         "model_disagreement_pp": float(abs(linear[i]-boosted[i])*100),
                         "history_starts": int(row.history_starts),
                         "missing_numeric_fields": int(frame.loc[i, NUMERIC].isna().sum()),
                         "baseline_contributions": {key: float(contribution[key]) for key in indices}})
        rows.sort(key=lambda r: (-r["win_probability"], r["runner_id"]))
        return {"model_version": m["model_version"], "race_id": race["race_id"],
                "prediction_at": race["prediction_at"], "knowledge_cutoff": m["knowledge_cutoff"],
                "snapshot_sha256": digest(race), "features_sha256": features_digest,
                "created_at": iso(now), "runners": rows,
                "mode": "prospective" if now < instant(race["scheduled_off"]) else "historical_replay",
                "data_kind": m["data_kind"],
                "top3_note": "Plackett-Luce estimate; no DNF/tie adjustment; not bookmaker place terms"}
