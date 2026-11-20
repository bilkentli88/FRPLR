"""Factorial construction for token identity × positional geometry.

The implementation is a line-for-line structural port of the frozen Qwen v061
notebook.  It intentionally uses float32 positions because source alignment
places multiple fragmented pieces inside one canonical-token interval.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class UserEncoding:
    canonical_ids: tuple[int, ...]
    fragmented_ids: tuple[int, ...]
    canonical_labels: tuple[str, ...]
    fragmented_labels: tuple[str, ...]
    fragmented_source_positions: tuple[float, ...]

    def validate(self) -> None:
        if len(self.canonical_ids) != len(self.canonical_labels):
            raise ValueError("canonical IDs/labels length mismatch")
        if len(self.fragmented_ids) != len(self.fragmented_labels):
            raise ValueError("fragmented IDs/labels length mismatch")
        if len(self.fragmented_ids) != len(self.fragmented_source_positions):
            raise ValueError("fragmented IDs/source-position length mismatch")
        required = {"evidence", "filler", "query"}
        if not required.issubset(self.canonical_labels):
            raise ValueError("canonical labels lack an evidence/filler/query region")
        if not required.issubset(self.fragmented_labels):
            raise ValueError("fragmented labels lack an evidence/filler/query region")


@dataclass(frozen=True)
class Condition:
    code: str
    name: str
    ids: tuple[int, ...]
    positions: np.ndarray
    labels: tuple[str, ...]
    ids_hash: str
    position_hash: str
    gaps: Mapping[str, float]

    @property
    def realized_tokens(self) -> int:
        return len(self.ids)

    @property
    def position_max(self) -> float:
        return float(self.positions[-1])


@dataclass(frozen=True)
class FactorialPrompt:
    conditions: Mapping[str, Condition]
    prefix_len: int
    suffix_len: int

    def __getitem__(self, code: str) -> Condition:
        return self.conditions[code]


def _piece_intervals(pieces: Sequence[str]) -> tuple[list[tuple[int, int]], int]:
    intervals: list[tuple[int, int]] = []
    pos = 0
    for piece in pieces:
        intervals.append((pos, pos + len(piece)))
        pos += len(piece)
    return intervals, pos


def fractional_local_positions(
    canonical_pieces: Sequence[str], fragmented_pieces: Sequence[str]
) -> np.ndarray:
    """Map fine pieces to strictly increasing positions inside coarse tokens.

    Fragmented pieces may split a canonical token but must never cross a
    canonical-token boundary.  The output exactly matches v061.
    """

    if not canonical_pieces or not fragmented_pieces:
        raise ValueError("piece streams must be non-empty")
    if "".join(canonical_pieces) != "".join(fragmented_pieces):
        raise ValueError("piece streams reconstruct different text")

    c_intervals, c_total = _piece_intervals(canonical_pieces)
    f_intervals, f_total = _piece_intervals(fragmented_pieces)
    if c_total != f_total:
        raise ValueError("character extents disagree")

    assignment: list[int] = []
    c_idx = 0
    for f_start, f_end in f_intervals:
        while c_idx + 1 < len(c_intervals) and c_intervals[c_idx][1] <= f_start:
            c_idx += 1
        c_start, c_end = c_intervals[c_idx]
        if not (c_start <= f_start and f_end <= c_end):
            raise ValueError("fragment crossed a canonical-token boundary")
        assignment.append(c_idx)

    groups: dict[int, list[int]] = {}
    for fine_idx, coarse_idx in enumerate(assignment):
        groups.setdefault(coarse_idx, []).append(fine_idx)

    positions = np.empty(len(fragmented_pieces), dtype=np.float32)
    for coarse_idx, fine_indices in groups.items():
        width = len(fine_indices)
        for rank, fine_idx in enumerate(fine_indices):
            positions[fine_idx] = coarse_idx + rank / width

    if len(positions) > 1 and not np.all(np.diff(positions) > 0):
        raise ValueError("source-aligned local positions are not strictly increasing")
    return positions


def hash_ids(ids: Sequence[int]) -> str:
    return hashlib.sha256(np.asarray(ids, dtype=np.int64).tobytes()).hexdigest()[:24]


def hash_positions(positions: Sequence[float]) -> str:
    return hashlib.sha256(np.asarray(positions, dtype=np.float32).tobytes()).hexdigest()[:24]


def gap_diagnostics(
    positions: Sequence[float], prefix_len: int, labels: Sequence[str]
) -> dict[str, float]:
    labels_array = np.asarray(labels)
    evidence = np.where(labels_array == "evidence")[0]
    filler = np.where(labels_array == "filler")[0]
    query = np.where(labels_array == "query")[0]
    if not (len(evidence) and len(filler) and len(query)):
        raise ValueError("missing evidence/filler/query region")
    pos = np.asarray(positions, dtype=np.float32)
    evidence_last = float(pos[prefix_len + evidence[-1]])
    filler_first = float(pos[prefix_len + filler[0]])
    filler_last = float(pos[prefix_len + filler[-1]])
    query_first = float(pos[prefix_len + query[0]])
    query_last = float(pos[prefix_len + query[-1]])
    readout = float(pos[-1])
    return {
        "evidence_query": query_first - evidence_last,
        "evidence_readout": readout - evidence_last,
        "filler_span": filler_last - filler_first,
        "query_readout": readout - query_last,
    }


def _condition(
    code: str,
    name: str,
    ids: Sequence[int],
    positions: np.ndarray,
    labels: Sequence[str],
    prefix_len: int,
) -> Condition:
    pos = np.asarray(positions, dtype=np.float32)
    if len(ids) != len(pos):
        raise ValueError(f"{code}: IDs/positions length mismatch")
    if len(pos) > 1 and not np.all(np.diff(pos) > 0):
        bad = np.where(np.diff(pos) <= 0)[0][:10].tolist()
        raise ValueError(f"{code}: positions not strictly increasing at {bad}")
    return Condition(
        code=code,
        name=name,
        ids=tuple(int(x) for x in ids),
        positions=pos,
        labels=tuple(labels),
        ids_hash=hash_ids(ids),
        position_hash=hash_positions(pos),
        gaps=gap_diagnostics(pos, prefix_len, labels),
    )


def assemble_factorial(
    prefix_ids: Sequence[int], suffix_ids: Sequence[int], encoding: UserEncoding
) -> FactorialPrompt:
    """Build and validate the frozen `CC`, `CE`, `FC`, and `FE` conditions."""

    encoding.validate()
    c_user = list(encoding.canonical_ids)
    f_user = list(encoding.fragmented_ids)
    c_labels = np.asarray(encoding.canonical_labels)
    f_labels = np.asarray(encoding.fragmented_labels)
    prefix_len, suffix_len = len(prefix_ids), len(suffix_ids)
    c_full = list(prefix_ids) + c_user + list(suffix_ids)
    f_full = list(prefix_ids) + f_user + list(suffix_ids)

    cc_pos = np.arange(len(c_full), dtype=np.float32)
    fe_pos = np.arange(len(f_full), dtype=np.float32)
    prefix_pos = np.arange(prefix_len, dtype=np.float32)

    fc_user_pos = prefix_len + np.asarray(
        encoding.fragmented_source_positions, dtype=np.float32
    )
    canonical_suffix_start = prefix_len + len(c_user)
    fc_suffix_pos = np.arange(
        canonical_suffix_start, canonical_suffix_start + suffix_len, dtype=np.float32
    )
    fc_pos = np.concatenate([prefix_pos, fc_user_pos, fc_suffix_pos]).astype(np.float32)

    ce_user_pos = np.arange(len(c_user), dtype=np.float32)
    c_filler = np.where(c_labels == "filler")[0]
    c_query = np.where(c_labels == "query")[0]
    f_query = np.where(f_labels == "query")[0]
    if not (len(c_filler) and len(c_query) and len(f_query)):
        raise ValueError("missing canonical/fragmented regions")

    c_filler_first = int(c_filler[0])
    f_query_first = int(f_query[0])
    ce_user_pos[c_filler] = c_filler_first + np.arange(
        len(c_filler), dtype=np.float32
    ) * (f_query_first - c_filler_first) / len(c_filler)
    query_offset = f_query_first - int(c_query[0])
    ce_user_pos[c_query] = c_query.astype(np.float32) + query_offset

    expanded_suffix_start = prefix_len + len(f_user)
    ce_suffix_pos = np.arange(
        expanded_suffix_start, expanded_suffix_start + suffix_len, dtype=np.float32
    )
    ce_pos = np.concatenate(
        [prefix_pos, prefix_len + ce_user_pos, ce_suffix_pos]
    ).astype(np.float32)

    conditions = {
        "CC": _condition(
            "CC", "canonical_ids_canonical_pos", c_full, cc_pos, c_labels, prefix_len
        ),
        "CE": _condition(
            "CE", "canonical_ids_expanded_pos", c_full, ce_pos, c_labels, prefix_len
        ),
        "FC": _condition(
            "FC", "fragmented_ids_source_pos", f_full, fc_pos, f_labels, prefix_len
        ),
        "FE": _condition(
            "FE", "fragmented_ids_expanded_pos", f_full, fe_pos, f_labels, prefix_len
        ),
    }

    if conditions["CC"].ids_hash != conditions["CE"].ids_hash:
        raise AssertionError("CC and CE must have identical canonical token IDs")
    if conditions["FC"].ids_hash != conditions["FE"].ids_hash:
        raise AssertionError("FC and FE must have identical fragmented token IDs")
    if abs(
        conditions["CC"].gaps["evidence_readout"]
        - conditions["FC"].gaps["evidence_readout"]
    ) > 1e-3:
        raise AssertionError("CC and FC must share source evidence→readout geometry")
    if abs(
        conditions["CE"].gaps["evidence_readout"]
        - conditions["FE"].gaps["evidence_readout"]
    ) > 1e-3:
        raise AssertionError("CE and FE must share expanded evidence→readout geometry")
    if not (
        conditions["FE"].gaps["evidence_readout"]
        > conditions["FC"].gaps["evidence_readout"]
    ):
        raise AssertionError("expanded geometry did not increase evidence→readout distance")

    return FactorialPrompt(conditions, prefix_len, suffix_len)

