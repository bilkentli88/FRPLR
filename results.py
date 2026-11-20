"""Append-only JSONL result storage with torn-line repair and resume IDs."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Iterable


def repair_torn_final_line(path: str | Path) -> bool:
    """Remove only an invalid final partial line; return whether repair occurred."""

    target = Path(path)
    if not target.exists() or target.stat().st_size == 0:
        return False
    data = target.read_bytes()
    lines = data.splitlines(keepends=True)
    if not lines:
        return False
    final = lines[-1]
    if final.endswith((b"\n", b"\r")):
        return False
    try:
        json.loads(final.decode("utf-8"))
        with target.open("ab") as handle:
            handle.write(b"\n")
            handle.flush()
            os.fsync(handle.fileno())
        return True
    except (UnicodeDecodeError, json.JSONDecodeError):
        target.write_bytes(b"".join(lines[:-1]))
        return True


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    target = Path(path)
    if not target.exists():
        return []
    rows: list[dict[str, Any]] = []
    for number, line in enumerate(target.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSON at {target}:{number}") from exc
        if not isinstance(row, dict):
            raise ValueError(f"non-object JSON row at {target}:{number}")
        rows.append(row)
    return rows


def completed_ids(path: str | Path, id_field: str = "result_id") -> set[str]:
    ids: set[str] = set()
    for row in read_jsonl(path):
        value = row.get(id_field)
        if not isinstance(value, str) or not value:
            raise ValueError(f"row lacks non-empty {id_field}")
        if value in ids:
            raise ValueError(f"duplicate {id_field}: {value}")
        ids.add(value)
    return ids


class JsonlLedger:
    def __init__(self, path: str | Path, fsync_every: int = 1):
        if fsync_every < 1:
            raise ValueError("fsync_every must be positive")
        self.path = Path(path)
        self.fsync_every = fsync_every
        self._pending = 0
        self.path.parent.mkdir(parents=True, exist_ok=True)
        repair_torn_final_line(self.path)
        self._ids = completed_ids(self.path)

    @property
    def ids(self) -> frozenset[str]:
        return frozenset(self._ids)

    def append(self, row: dict[str, Any]) -> bool:
        result_id = row.get("result_id")
        if not isinstance(result_id, str) or not result_id:
            raise ValueError("row requires a non-empty result_id")
        if result_id in self._ids:
            return False
        payload = json.dumps(row, sort_keys=True, ensure_ascii=False, allow_nan=False)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(payload + "\n")
            handle.flush()
            self._pending += 1
            if self._pending >= self.fsync_every:
                os.fsync(handle.fileno())
                self._pending = 0
        self._ids.add(result_id)
        return True

    def append_many(self, rows: Iterable[dict[str, Any]]) -> int:
        return sum(int(self.append(row)) for row in rows)

