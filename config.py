"""Frozen configuration loading and hashing."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def load_config(path: str | Path) -> dict[str, Any]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise TypeError("configuration root must be an object")
    return data


def canonical_json(data: dict[str, Any]) -> str:
    return json.dumps(data, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def config_hash(data: dict[str, Any], length: int = 16) -> str:
    return hashlib.sha256(canonical_json(data).encode("utf-8")).hexdigest()[:length]


def assert_frozen(data: dict[str, Any]) -> None:
    if data.get("status") != "frozen":
        raise RuntimeError("configuration is not marked frozen")
    if not data.get("protocol_version"):
        raise RuntimeError("frozen configuration lacks protocol_version")

