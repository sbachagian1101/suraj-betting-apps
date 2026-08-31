"""Persistence helpers for the Bet365 predictor."""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List
import json
import shutil
import zipfile

from bet365_model import default_model_state


def load_json(path: Path, default: Any) -> Any:
    try:
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        pass
    return default


def atomic_save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    temp.replace(path)


def load_model_state(data_dir: Path) -> Dict[str, Any]:
    return load_json(data_dir / "model_state.json", default_model_state())


def load_training_store(data_dir: Path) -> List[Dict[str, Any]]:
    value = load_json(data_dir / "training_store.json", [])
    return value if isinstance(value, list) else []


def load_training_history(data_dir: Path) -> List[Dict[str, Any]]:
    value = load_json(data_dir / "training_history.json", [])
    return value if isinstance(value, list) else []


def backup_model_state(data_dir: Path) -> Path | None:
    source = data_dir / "model_state.json"
    if not source.exists():
        return None
    backup_dir = data_dir / "model_state_backups"
    backup_dir.mkdir(parents=True, exist_ok=True)
    target = backup_dir / f"model_state_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    shutil.copy2(source, target)
    return target


def save_training_system(data_dir: Path, model_state: Dict[str, Any], training_store: List[Dict[str, Any]], history: List[Dict[str, Any]]) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)
    backup_model_state(data_dir)
    atomic_save_json(data_dir / "model_state.json", model_state)
    atomic_save_json(data_dir / "training_store.json", training_store)
    atomic_save_json(data_dir / "training_history.json", history)


def create_backup_zip(data_dir: Path, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    target = output_dir / f"Bet365_Predictor_Training_Backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip"
    with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as zf:
        if data_dir.exists():
            for path in data_dir.rglob("*"):
                if path.is_file():
                    zf.write(path, arcname=str(path.relative_to(data_dir.parent)))
    return target
