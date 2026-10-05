"""Append-only snapshots and predictions in a separate SQLite database."""
from __future__ import annotations

import json
from pathlib import Path
import sqlite3

from .schema import validate
from .training import digest


def append(database, kind, record):
    if kind not in {"snapshot", "prediction"}:
        raise ValueError("Unknown ledger record kind")
    if kind == "snapshot":
        validate(record)
    stable = {k: v for k, v in record.items() if k != "created_at"}
    record_id = digest({"kind": kind, "record": stable})
    path = Path(database)
    path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(path) as con:
        con.execute("PRAGMA journal_mode=WAL")
        con.execute("CREATE TABLE IF NOT EXISTS form_lab_records "
                    "(id TEXT PRIMARY KEY, kind TEXT NOT NULL, race_id TEXT NOT NULL, "
                    "prediction_at TEXT NOT NULL, payload TEXT NOT NULL)")
        con.execute("INSERT OR IGNORE INTO form_lab_records VALUES (?, ?, ?, ?, ?)",
                    (record_id, kind, record["race_id"], record["prediction_at"],
                     json.dumps(record, sort_keys=True, allow_nan=False)))
    return record_id
