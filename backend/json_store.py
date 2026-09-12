# backend/json_store.py
"""
Atomic, locked, cached JSON storage for the AI Essay Evaluator backend.

Features:
- Atomic writes (temp file + os.replace)
- Per-file locking via filelock
- In-memory cache with mtime invalidation
- Rolling backups (.bak1, .bak2, .bak3)
- Optional in-memory indexes for fast lookups
"""

import json
import os
import shutil
import tempfile
from datetime import datetime
from pathlib import Path
from threading import Lock
from typing import Any, Dict, Iterable, List, Optional

from filelock import FileLock

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DATA_DIR = Path(os.environ.get("DATA_DIR", "/app/data"))
DATA_DIR.mkdir(parents=True, exist_ok=True)

BACKUP_KEEP = 3
_lock_registry: Dict[str, FileLock] = {}
_registry_lock = Lock()


def _path(filename: str) -> Path:
    return DATA_DIR / filename


def _get_lock(filename: str) -> FileLock:
    """Return a per-file lock, creating it on first use."""
    with _registry_lock:
        if filename not in _lock_registry:
            _lock_registry[filename] = FileLock(str(DATA_DIR / f"{filename}.lock"))
        return _lock_registry[filename]


# ---------------------------------------------------------------------------
# Atomic file operations
# ---------------------------------------------------------------------------

def _atomic_write_bytes(path: Path, data: bytes) -> None:
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except Exception:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise


def _rolling_backup(filename: str) -> None:
    """Keep the last BACKUP_KEEP versions of the file."""
    path = _path(filename)
    if not path.exists():
        return
    for i in range(BACKUP_KEEP - 1, 0, -1):
        older = _path(f"{filename}.bak{i}")
        newer = _path(f"{filename}.bak{i - 1}")
        if older.exists():
            older.unlink()
        if newer.exists():
            newer.rename(older)
    try:
        shutil.copy2(path, _path(f"{filename}.bak1"))
    except OSError:
        pass


# ---------------------------------------------------------------------------
# Core read / write
# ---------------------------------------------------------------------------

def load_json(filename: str, default: Any = None) -> Any:
    path = _path(filename)
    if not path.exists():
        return default if default is not None else []
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        if path.stat().st_size > 0:
            path.rename(_path(f"{filename}.bak-corrupt"))
        return default if default is not None else []


def save_json(filename: str, data: Any) -> None:
    _rolling_backup(filename)
    serialized = json.dumps(data, indent=2, ensure_ascii=False).encode("utf-8")
    _atomic_write_bytes(_path(filename), serialized)


# ---------------------------------------------------------------------------
# List operations
# ---------------------------------------------------------------------------

def _next_id(rows: Iterable[Dict[str, Any]]) -> int:
    return max((r.get("id", 0) for r in rows), default=0) + 1


def _now() -> str:
    return datetime.utcnow().isoformat()


def append_to_list(filename: str, item: Dict[str, Any]) -> Dict[str, Any]:
    with _get_lock(filename):
        rows = load_json(filename, [])
        item = dict(item)
        item.setdefault("id", _next_id(rows))
        item.setdefault("created_at", _now())
        rows.append(item)
        save_json(filename, rows)
        return item


def update_in_list(
    filename: str,
    match_key: str,
    match_value: Any,
    updates: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    with _get_lock(filename):
        rows = load_json(filename, [])
        updated = None
        for row in rows:
            if row.get(match_key) == match_value:
                row.update(updates)
                updated = row
                break
        if updated is not None:
            save_json(filename, rows)
        return updated


def upsert_in_list(
    filename: str,
    identity_keys: List[str],
    identity_values: List[Any],
    updates: Dict[str, Any],
) -> Dict[str, Any]:
    with _get_lock(filename):
        rows = load_json(filename, [])
        existing = next(
            (
                r for r in rows
                if all(r.get(k) == v for k, v in zip(identity_keys, identity_values))
            ),
            None,
        )
        if existing is not None:
            existing.update(updates)
            existing.setdefault("updated_at", _now())
            save_json(filename, rows)
            return existing
        new_row = dict(updates)
        for k, v in zip(identity_keys, identity_values):
            new_row[k] = v
        new_row.setdefault("id", _next_id(rows))
        new_row.setdefault("created_at", _now())
        rows.append(new_row)
        save_json(filename, rows)
        return new_row


def delete_from_list(filename: str, match_key: str, match_value: Any) -> bool:
    with _get_lock(filename):
        rows = load_json(filename, [])
        new_rows = [r for r in rows if r.get(match_key) != match_value]
        if len(new_rows) != len(rows):
            save_json(filename, new_rows)
            return True
        return False


def query_list(filename: str, **filters: Any) -> List[Dict[str, Any]]:
    rows = load_json(filename, [])
    if not filters:
        return rows
    return [
        r for r in rows
        if all(r.get(k) == v for k, v in filters.items())
    ]


def get_one(filename: str, **filters: Any) -> Optional[Dict[str, Any]]:
    rows = query_list(filename, **filters)
    return rows[0] if rows else None


# ---------------------------------------------------------------------------
# Simple in-memory index
# ---------------------------------------------------------------------------

class IndexedList:
    def __init__(self, filename: str, index_field: str):
        self.filename = filename
        self.index_field = index_field
        self._index: Dict[Any, List[Dict[str, Any]]] = {}
        self._mtime: float = 0
        self._lock = Lock()

    def _refresh_if_needed(self) -> None:
        path = _path(self.filename)
        mtime = path.stat().st_mtime if path.exists() else 0
        if mtime == self._mtime:
            return
        with self._lock:
            if mtime == self._mtime:
                return
            rows = load_json(self.filename, [])
            index: Dict[Any, List[Dict[str, Any]]] = {}
            for row in rows:
                key = row.get(self.index_field)
                index.setdefault(key, []).append(row)
            self._index = index
            self._mtime = mtime

    def by(self, value: Any) -> List[Dict[str, Any]]:
        self._refresh_if_needed()
        return list(self._index.get(value, []))