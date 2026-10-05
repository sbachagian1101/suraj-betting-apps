"""Read-only file/schema inventory; no credentials or row contents exported."""
from __future__ import annotations

import csv
import json
from pathlib import Path
import sqlite3
from urllib.parse import quote


def inventory(root):
    root = Path(root).resolve()
    if not root.is_dir():
        raise ValueError("Inventory root must be an existing directory")
    output = {"files": [], "note": "No row values or credentials included; source timestamp provenance still needs review."}
    suffixes = {".sqlite", ".sqlite3", ".db", ".json", ".jsonl", ".csv", ".parquet"}
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root)
        if any(p.startswith(".") or p in {"venv", "__pycache__", "node_modules"} for p in relative.parts):
            continue
        if path.is_symlink() or not path.is_file() or path.suffix.lower() not in suffixes:
            continue
        if any(word in path.name.lower() for word in ("secret", "credential", "token", "password")):
            continue
        item = {"path": str(relative), "bytes": path.stat().st_size}
        try:
            if path.suffix in {".db", ".sqlite", ".sqlite3"}:
                uri = "file:" + quote(str(path), safe="/") + "?mode=ro"
                with sqlite3.connect(uri, uri=True, timeout=2) as db:
                    db.execute("PRAGMA query_only=ON")
                    names = [r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'")]
                    item["tables"] = {name: [{"name": r[1], "type": r[2]} for r in
                        db.execute('PRAGMA table_info("' + name.replace('"', '""') + '")')]
                        for name in names}
            elif path.suffix == ".csv":
                with path.open(newline="") as f:
                    item["columns"] = next(csv.reader(f), [])
            elif path.suffix == ".jsonl":
                with path.open() as f:
                    value = json.loads(f.readline(2_000_000))
                item["keys"] = sorted(value) if isinstance(value, dict) else ["non-object"]
            elif path.suffix == ".json" and path.stat().st_size < 2_000_000:
                value = json.loads(path.read_text())
                item["keys"] = sorted(value) if isinstance(value, dict) else ["non-object"]
        except (OSError, ValueError, sqlite3.Error, UnicodeError):
            item["read_status"] = "unreadable_or_unsupported"
        output["files"].append(item)
        if len(output["files"]) >= 5000:
            output["truncated"] = True
            break
    return output


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root")
    print(json.dumps(inventory(parser.parse_args().root), indent=2))
