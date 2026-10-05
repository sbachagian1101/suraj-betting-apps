from copy import deepcopy
import json
import sqlite3

import numpy as np
import pandas as pd
import pytest

from form_lab.bridge import from_legacy_card
from form_lab.features import CATEGORICAL, NUMERIC, Transform, build_features
from form_lab.models import ConditionalLogit, groups, pl_topk, race_softmax
from form_lab.schema import DataError, audit, training_issue, validate
from form_lab.storage import append
from form_lab.training import Bundle, split_frame, train
from form_lab.tests.helpers import synthetic_records


@pytest.mark.parametrize("mutation", [
    lambda r: r.update(prediction_at=r["scheduled_off"]),
    lambda r: r.update(captured_at=r["result"]["available_at"]),
    lambda r: r.update(field_complete=False),
    lambda r: r["runners"][0].update(odds=2.5),
    lambda r: r["runners"][0].update(horse_id=r["runners"][1]["horse_id"]),
    lambda r: r["result"]["runners"].pop(),
    lambda r: r.update(prediction_at="2025-01-01T12:00:00"),
    lambda r: r["runners"][0].update(carried_weight_kg=float("nan")),
])
def test_bad_data_refused(mutation):
    r = synthetic_records(1)[0]
    mutation(r)
    with pytest.raises(DataError):
        validate(r)


def test_late_scratch_quarantines_whole_race():
    r = synthetic_records(1)[0]
    loser = next(o for o in r["result"]["runners"] if o["finish_position"] == 6)
    loser.update(status="non_runner", finish_position=None)
    validate(r)
    assert training_issue(r) == "field_changed_after_snapshot"
    assert audit([r])["groups"]["AU/flat"]["eligible"] == 0


def test_dead_heat_and_dnf_labels_keep_unit_winner_mass():
    r = synthetic_records(1)[0]
    for o in r["result"]["runners"]:
        if o["finish_position"] == 2:
            o["finish_position"] = 1
        elif o["finish_position"] == 6:
            o.update(status="dnf", finish_position=None)
    validate(r)
    frame = build_features([r])
    assert frame.y.sum() == pytest.approx(1)
    assert sorted(frame.y) == [0, 0, 0, 0, .5, .5]


def test_attached_future_history_is_rejected():
    r = synthetic_records(1)[0]
    r["runners"][0]["history"] = [{"race_id": "synthetic:future", "off_at": "2025-01-02T12:00:00Z",
        "available_at": "2025-01-02T12:15:00Z", "country": "AU", "discipline": "flat",
        "distance_m": 1200, "field_size": 6, "status": "finished", "finish_position": 1}]
    with pytest.raises(DataError, match="historical result"):
        validate(r)


def test_target_and_future_results_do_not_change_pre_race_features():
    records = synthetic_records(5)
    before = build_features(records)
    changed = deepcopy(records)
    for r in changed[4:]:
        for o in r["result"]["runners"]:
            o["finish_position"] = 7-o["finish_position"]
    after = build_features(changed)
    # The fifth race itself also cannot use its own outcome.
    pd.testing.assert_frame_equal(before.loc[:29, NUMERIC+CATEGORICAL], after.loc[:29, NUMERIC+CATEGORICAL])
    assert not before.loc[30:, "history_wins"].equals(after.loc[30:, "history_wins"])


def test_delayed_results_only_enter_after_publication():
    records = synthetic_records(3)
    first = records[0]
    first["result"]["available_at"] = "2025-01-03T15:00:00Z"
    frame = build_features(records)
    third_day_first = frame.loc[frame.race_id == records[4]["race_id"]]
    # Its last-day first race is known; day-one result remains unknown.
    assert (third_day_first.history_starts == 1).all()


def test_imputation_uses_training_only_and_single_race_path_matches_batch():
    records = synthetic_records(6)
    frame = build_features(records)
    t = Transform().fit(frame.iloc[:36])
    original = deepcopy(t.to_dict())
    test = frame.iloc[36:].copy()
    test["official_rating"] = 10000
    t.apply(test)
    assert t.to_dict() == original
    first = t.apply(frame.iloc[36:42])
    pd.testing.assert_frame_equal(first, t.apply(frame).iloc[36:42])


def test_softmax_and_pl_conserve_probability_and_are_permutation_equivariant():
    p = race_softmax(np.array([10000., 9999., 9998., 9997.]), [slice(0, 4)])
    assert p.sum() == pytest.approx(1)
    q = pl_topk(p, 3)
    assert q.sum() == pytest.approx(3)
    assert np.all(q >= p - 1e-10)
    order = np.array([2, 0, 3, 1])
    np.testing.assert_allclose(pl_topk(p[order], 3), q[order])
    np.testing.assert_allclose(pl_topk(np.ones(6)/6, 3), np.ones(6)/2)
    np.testing.assert_allclose(pl_topk(np.array([.7, .3]), 3), np.ones(2))


def test_model_can_learn_race_signal():
    rng = np.random.default_rng(83)
    x = rng.normal(size=(1200, 1))
    ids = np.repeat(np.arange(200), 6)
    sl = groups(ids)
    p = race_softmax(x[:, 0] * 1.5, sl)
    y = np.zeros(len(x))
    for s in sl:
        y[s.start + rng.choice(6, p=p[s])] = 1
    fit = ConditionalLogit(l2=.001).fit(x, y, sl)
    assert 1.1 < fit.beta[0] < 1.9


def test_meeting_cannot_cross_training_boundary():
    frame = build_features(synthetic_records(5))
    with pytest.raises(DataError, match="crosses"):
        split_frame(frame, "2025-01-02T13:30:00Z", "2025-01-03T00:00:00Z", "2025-01-04T00:00:00Z", 1)


def test_ledger_idempotent_and_keeps_changed_snapshot(tmp_path):
    r = synthetic_records(1)[0]
    db = tmp_path / "ledger.sqlite"
    a = append(db, "snapshot", r)
    assert append(db, "snapshot", r) == a
    r["going"] = "SOFT"
    assert append(db, "snapshot", r) != a
    with sqlite3.connect(db) as con:
        assert con.execute("SELECT COUNT(*) FROM form_lab_records").fetchone()[0] == 2


def test_bridge_discards_markets_and_requires_real_ids():
    from common import Entry, RaceCard
    r = synthetic_records(1)[0]
    card = RaceCard(country="AUS", dist_m=1200, surface="TURF", entries=[
        Entry(number=i+1, name=f"H{i}", odds=2., tip_rank=i+1, official_rating=70+i, weight=55)
        for i in range(3)])
    names = {"race_id", "meeting_id", "course_id", "scheduled_off", "prediction_at", "available_at",
             "captured_at", "availability_basis", "source", "source_ref", "field_complete", "discipline"}
    metadata = {k: r[k] for k in names}
    identities = {str(i+1): {"runner_id": f"source:runner:{i}", "horse_id": f"source:horse:{i}", "horse_name": f"H{i}"} for i in range(3)}
    converted = from_legacy_card(card, metadata, identities)
    assert "odds" not in json.dumps(converted)
    assert all(e["carried_weight_kg"] is None for e in converted["runners"])
    identities["1"]["horse_name"] = "Different Horse"
    with pytest.raises(DataError, match="match"):
        from_legacy_card(card, metadata, identities)


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    root = tmp_path_factory.mktemp("trained") / "bundle"
    records = synthetic_records(100)
    m, report = train(records, root, country="AU", discipline="flat",
                      blend_start="2025-02-10T00:00:00Z", calibration_start="2025-03-02T00:00:00Z",
                      test_start="2025-03-22T00:00:00Z", iterations=35)
    return records, root, m, report


def test_training_serialisation_and_prediction_match_holdout(trained):
    records, root, m, report = trained
    assert sum(m["weights"]) == pytest.approx(1)
    assert report["test_metrics"]["ensemble"]["races"] == 40
    card = deepcopy(records[-1])
    card.pop("result")
    forecast = Bundle(root).predict(card, records[:-1])
    assert forecast["mode"] == "historical_replay"
    assert forecast["data_kind"] == "synthetic"
    saved = pd.read_csv(root / "holdout_predictions.csv").set_index("runner_id")
    for row in forecast["runners"]:
        assert row["win_probability"] == pytest.approx(saved.loc[row["runner_id"], "ensemble"], abs=1e-10)
    assert sum(r["win_probability"] for r in forecast["runners"]) == pytest.approx(1)
    assert sum(r["top3_probability_pl"] for r in forecast["runners"]) == pytest.approx(3)
    earlier = deepcopy(records[0])
    earlier.pop("result")
    with pytest.raises(DataError, match="knows outcomes"):
        Bundle(root).predict(earlier)


def test_model_checksum_prevents_loading_changed_weights(trained, tmp_path):
    import shutil
    _, root, _, _ = trained
    copy = tmp_path / "copy"
    shutil.copytree(root, copy)
    with (copy / "catboost.cbm").open("ab") as f:
        f.write(b"changed")
    with pytest.raises(DataError, match="checksum"):
        Bundle(copy)


def test_manifest_checksum_covers_calibration_and_coefficients(trained, tmp_path):
    import shutil
    _, root, _, _ = trained
    copy = tmp_path / "copy"
    shutil.copytree(root, copy)
    manifest = json.loads((copy / "manifest.json").read_text())
    manifest["temperature"] = 1.234
    (copy / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(DataError, match="manifest checksum"):
        Bundle(copy)


def test_dashboard_renders_trained_synthetic_card(trained, tmp_path, monkeypatch):
    from datetime import datetime, timedelta, timezone
    from pathlib import Path
    from streamlit.testing.v1 import AppTest
    records, root, _, _ = trained
    now = datetime.now(timezone.utc)
    card = deepcopy(records[-1])
    card.pop("result")
    card["race_id"] = "synthetic:ui-card"
    card["prediction_at"] = (now-timedelta(minutes=1)).isoformat()
    card["available_at"] = card["captured_at"] = (now-timedelta(minutes=2)).isoformat()
    card["scheduled_off"] = (now+timedelta(hours=1)).isoformat()
    for i, e in enumerate(card["runners"]):
        e["runner_id"] = f"synthetic:ui-card:{i}"
    cards = tmp_path / "cards.jsonl"
    history = tmp_path / "history.jsonl"
    cards.write_text(json.dumps(card) + "\n")
    history.write_text("\n".join(json.dumps(r) for r in records))
    monkeypatch.setenv("FORM_LAB_MODELS", str(root.parent))
    monkeypatch.setenv("FORM_LAB_CARDS", str(cards))
    monkeypatch.setenv("FORM_LAB_HISTORY", str(history))
    app = AppTest.from_file(str(Path(__file__).parents[1] / "app.py")).run(timeout=20)
    assert not app.exception
    assert not app.error
    assert len(app.dataframe) == 1
    assert any("Synthetic test model" in warning.value for warning in app.warning)
    assert app.dataframe[0].value["Combined win %"].is_monotonic_decreasing
