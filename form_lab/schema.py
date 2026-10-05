"""The same point-in-time race contract is used for training and prediction.

Availability timestamps are evidence supplied by a collector, never invented by
this module. Historical downloads need source-as-of evidence or an archived
pre-race capture. A horse's current career page is not a historical snapshot.
"""
from __future__ import annotations

import json
import math
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from . import SCHEMA_VERSION

COUNTRIES = {"AU", "GB", "ZA", "FR"}
DISCIPLINES = {"flat", "hurdle", "chase", "nh_flat", "trot", "mounted_trot"}
RACE_KEYS = {
    "schema_version", "race_id", "meeting_id", "country", "discipline",
    "course_id", "race_no", "scheduled_off", "prediction_at", "available_at",
    "captured_at", "availability_basis", "source", "source_ref", "field_complete",
    "distance_m", "surface", "going", "race_class", "runners", "result",
}
RUNNER_KEYS = {
    "runner_id", "horse_id", "horse_name", "number", "status", "age", "sex",
    "draw", "carried_weight_kg", "official_rating", "jockey_id", "trainer_id",
    "gear", "history",
}
HISTORY_KEYS = {
    "race_id", "off_at", "available_at", "country", "discipline", "course_id",
    "distance_m", "surface", "going", "field_size", "finish_position", "status",
    "beaten_lengths",
}
RESULT_KEYS = {"available_at", "status", "runners"}
OUTCOME_KEYS = {"runner_id", "status", "finish_position"}
FINISH_STATUSES = {"finished", "dnf", "dq", "non_runner"}


class DataError(ValueError):
    pass


def instant(value: str) -> datetime:
    try:
        d = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if d.tzinfo is None:
            raise ValueError("timezone missing")
        return d.astimezone(timezone.utc)
    except (AttributeError, TypeError, ValueError) as exc:
        raise DataError(f"Expected timezone-aware ISO timestamp: {value!r}") from exc


def iso(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def require(test, message):
    if not test:
        raise DataError(message)


def keys(obj, allowed, where):
    require(isinstance(obj, dict), f"{where}: expected an object")
    require(not (set(obj) - allowed), f"{where}: unapproved fields {sorted(set(obj) - allowed)}")


def identifier(value, where):
    require(isinstance(value, str) and bool(value.strip()), f"{where}: stable ID required")


def number(value, where, low=None, high=None, integer=False, optional=True):
    if value is None and optional:
        return
    require(isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value), f"{where}: expected finite number")
    require(low is None or value >= low, f"{where}: below {low}")
    require(high is None or value <= high, f"{where}: above {high}")
    require(not integer or value == int(value), f"{where}: expected integer")


def finish(obj, where, field_size):
    require(obj.get("status") in FINISH_STATUSES, f"{where}: unknown finish status")
    pos = obj.get("finish_position")
    if obj["status"] == "finished":
        number(pos, f"{where}.finish_position", 1, field_size, True, False)
    else:
        require(pos is None, f"{where}: non-finisher cannot have a finishing rank")


def validate(race: dict) -> dict:
    keys(race, RACE_KEYS, "race")
    require(race.get("schema_version") == SCHEMA_VERSION, "Unsupported schema_version")
    for key in ("race_id", "meeting_id", "course_id", "source", "source_ref"):
        identifier(race.get(key), key)
    rid = race["race_id"]
    require(race.get("country") in COUNTRIES, f"{rid}: country must be AU/GB/ZA/FR")
    require(race.get("discipline") in DISCIPLINES, f"{rid}: explicit discipline required")
    off, at, avail, capture = [instant(race.get(k)) for k in
                              ("scheduled_off", "prediction_at", "available_at", "captured_at")]
    require(avail <= at < off, f"{rid}: features must be available by prediction time, before off")
    require(capture >= avail, f"{rid}: capture precedes availability")
    basis = race.get("availability_basis")
    require(basis in {"archived_capture", "source_asof"}, f"{rid}: availability evidence required")
    if basis == "archived_capture":
        require(capture <= at, f"{rid}: late capture is not an archived pre-race snapshot")
    require(race.get("field_complete") is True, f"{rid}: full field must be confirmed")
    number(race.get("distance_m"), "distance_m", 200, 12000, True, False)
    for name in ("surface", "going", "race_class"):
        require(race.get(name) is None or isinstance(race[name], str), f"{name}: expected text or null")
    require(isinstance(race.get("runners"), list) and len(race["runners"]) >= 2,
            f"{rid}: at least two declarations required")
    runner_ids, horses = set(), set()
    for runner in race["runners"]:
        keys(runner, RUNNER_KEYS, "runner")
        for key in ("runner_id", "horse_id", "horse_name"):
            identifier(runner.get(key), key)
        for name in ("sex", "gear", "jockey_id", "trainer_id"):
            require(runner.get(name) is None or isinstance(runner[name], str), f"{name}: expected text or null")
        require(runner["runner_id"] not in runner_ids, f"{rid}: duplicate runner ID")
        require(runner["horse_id"] not in horses, f"{rid}: duplicate horse ID")
        runner_ids.add(runner["runner_id"])
        horses.add(runner["horse_id"])
        require(runner.get("status") in {"active", "scratched"}, f"{rid}: runner status required")
        for name, low, high in (("age", 1, 35), ("draw", 1, 100),
                                ("carried_weight_kg", 20, 100), ("official_rating", 0, 250)):
            number(runner.get(name), name, low, high)
        require(isinstance(runner.get("history", []), list), "history must be a list")
        seen = set()
        for h in runner.get("history", []):
            keys(h, HISTORY_KEYS, "history")
            identifier(h.get("race_id"), "history.race_id")
            require(h["race_id"] not in seen and h["race_id"] != rid, f"{rid}: duplicate/current historical run")
            seen.add(h["race_id"])
            when, known = instant(h.get("off_at")), instant(h.get("available_at"))
            require(when < known <= avail and when < off, f"{rid}: future/unknown historical result")
            require(h.get("country") in COUNTRIES and h.get("discipline") in DISCIPLINES,
                    f"{rid}: history needs country and discipline")
            number(h.get("field_size"), "history.field_size", 1, 100, True, False)
            number(h.get("distance_m"), "history.distance_m", 200, 12000, True, False)
            number(h.get("beaten_lengths"), "history.beaten_lengths", 0, 500)
            finish(h, "history", h["field_size"])
            if h.get("finish_position") == 1:
                require(h.get("beaten_lengths") in (None, 0), "Winner's beaten margin must be zero")
    require(sum(e["status"] == "active" for e in race["runners"]) >= 2, f"{rid}: walkover excluded")
    result = race.get("result")
    if result is not None:
        keys(result, RESULT_KEYS, "result")
        require(instant(result.get("available_at")) > off, f"{rid}: result precedes race")
        require(result.get("status") in {"official", "abandoned", "void"}, f"{rid}: result status required")
        require(isinstance(result.get("runners"), list), "result.runners must be a list")
        ids = []
        for o in result["runners"]:
            keys(o, OUTCOME_KEYS, "outcome")
            ids.append(o.get("runner_id"))
            finish(o, "outcome", len(runner_ids))
        require(len(ids) == len(set(ids)) and set(ids) == runner_ids,
                f"{rid}: results must map exactly to the full declared field")
        if result["status"] == "official":
            positions = sorted(o["finish_position"] for o in result["runners"] if o["status"] == "finished")
            require(positions and positions[0] == 1, f"{rid}: official winner missing")
            for i, pos in enumerate(positions):
                require(pos == (positions[i-1] if i and pos == positions[i-1] else i+1),
                        f"{rid}: invalid official finishing ranks")
    return race


def training_issue(race):
    result = race.get("result")
    if result is None:
        return "missing_result"
    if result["status"] != "official":
        return result["status"]
    statuses = {r["runner_id"]: r["status"] for r in result["runners"]}
    for e in race["runners"]:
        if (e["status"] == "active") != (statuses[e["runner_id"]] != "non_runner"):
            return "field_changed_after_snapshot"
    return None


def read_jsonl(path):
    records, seen = [], set()
    for n, line in enumerate(Path(path).read_text().splitlines(), 1):
        if not line.strip():
            continue
        try:
            r = validate(json.loads(line))
            require(r["race_id"] not in seen, "Select one prediction snapshot per race before modelling")
        except (ValueError, TypeError, KeyError) as exc:
            raise DataError(f"{path}:{n}: {exc}") from exc
        records.append(r)
        seen.add(r["race_id"])
    require(bool(records), "Empty dataset")
    return records


def audit(records):
    groups = {}
    for r in records:
        key = f"{r['country']}/{r['discipline']}"
        g = groups.setdefault(key, {"races": 0, "runners": 0, "eligible": 0,
                                    "excluded": Counter(), "basis": Counter(), "history_runs": 0,
                                    "no_attached_history": 0, "missing_fields": Counter(),
                                    "first_off": r["scheduled_off"], "last_off": r["scheduled_off"]})
        g["races"] += 1
        g["runners"] += sum(e["status"] == "active" for e in r["runners"])
        g["history_runs"] += sum(len(e.get("history", [])) for e in r["runners"])
        g["first_off"] = min((g["first_off"], r["scheduled_off"]), key=instant)
        g["last_off"] = max((g["last_off"], r["scheduled_off"]), key=instant)
        for e in r["runners"]:
            if e["status"] == "active":
                g["no_attached_history"] += not bool(e.get("history"))
                for name in ("draw", "carried_weight_kg", "official_rating", "jockey_id", "trainer_id", "age"):
                    if e.get(name) in (None, ""):
                        g["missing_fields"][name] += 1
        g["basis"][r["availability_basis"]] += 1
        reason = training_issue(r)
        if reason:
            g["excluded"][reason] += 1
        else:
            g["eligible"] += 1
    return {"schema_version": SCHEMA_VERSION, "groups": groups,
            "note": "Structural checks do not independently verify collector timestamp evidence."}
