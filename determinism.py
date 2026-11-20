"""Stable identifiers and pseudo-random streams independent of Python hashing."""

from __future__ import annotations

import hashlib
import random
from typing import Any


def stable_int(*parts: Any) -> int:
    """Return the v061-compatible unsigned 64-bit SHA-256 prefix."""

    blob = "|".join(map(str, parts)).encode("utf-8")
    return int.from_bytes(hashlib.sha256(blob).digest()[:8], "big")


def stable_rng(*parts: Any) -> random.Random:
    """Create a deterministic, locally scoped RNG."""

    return random.Random(stable_int(*parts))


def stable_hex(*parts: Any, length: int = 24) -> str:
    """Return a stable hexadecimal identifier for an experimental row."""

    if length < 8 or length > 64:
        raise ValueError("length must be between 8 and 64")
    blob = "|".join(map(str, parts)).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:length]

