from datetime import datetime, timedelta, timezone
import numpy as np


def synthetic_records(days=100):
    """Labelled synthetic races only. Not a racing performance benchmark."""
    rng = np.random.default_rng(197)
    start = datetime(2025, 1, 1, 12, tzinfo=timezone.utc)
    records = []
    for day in range(days):
        for race_no in (1, 2):
            off = start + timedelta(days=day, hours=race_no)
            stamp = lambda d: d.isoformat().replace("+00:00", "Z")
            race_id = f"synthetic:{day}:{race_no}"
            abilities = np.arange(6) * .3
            order = np.argsort(-(abilities + rng.gumbel(size=6)))
            finish = np.argsort(order) + 1
            race = {"schema_version": 1, "race_id": race_id, "meeting_id": f"synthetic:meeting:{day}",
                    "country": "AU", "discipline": "flat", "course_id": "synthetic:track",
                    "race_no": race_no, "scheduled_off": stamp(off), "prediction_at": stamp(off-timedelta(minutes=15)),
                    "available_at": stamp(off-timedelta(minutes=20)), "captured_at": stamp(off-timedelta(minutes=20)),
                    "availability_basis": "archived_capture", "source": "synthetic-test",
                    "source_ref": "synthetic fixture; not real racing data", "field_complete": True,
                    "distance_m": 1200 if day % 2 else 1600, "surface": "TURF", "going": "GOOD", "race_class": "TEST",
                    "runners": [], "result": {"available_at": stamp(off+timedelta(minutes=15)), "status": "official", "runners": []}}
            for i in range(6):
                runner_id = f"{race_id}:runner:{i}"
                race["runners"].append({"runner_id": runner_id, "horse_id": f"synthetic:horse:{race_no}:{i}",
                     "horse_name": f"Synthetic Horse {race_no}-{i}", "number": i+1, "status": "active", "age": 4,
                     "sex": "G", "draw": int(rng.integers(1, 7)), "carried_weight_kg": 54 + i,
                     "official_rating": 50 + 10 * abilities[i] + rng.normal(), "jockey_id": f"synthetic:jockey:{i}",
                     "trainer_id": f"synthetic:trainer:{i % 3}", "gear": "none", "history": []})
                race["result"]["runners"].append({"runner_id": runner_id, "status": "finished", "finish_position": int(finish[i])})
            records.append(race)
    return records
