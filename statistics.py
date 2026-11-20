"""Exact estimands and trial-cluster bootstrap used in the manuscript."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .determinism import stable_int

ID_COLUMNS = ["filler_language", "target_chat_tokens", "semantic_seed", "trial_index"]
FACTORIAL_EFFECTS = (
    "position_effect_canonical_ids",
    "position_effect_fragmented_ids",
    "representation_effect_source_geometry",
    "representation_effect_expanded_geometry",
    "position_interaction",
    "total_fragmentation_shift",
)


def load_jsonl(path: str | Path) -> pd.DataFrame:
    rows = [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return pd.DataFrame(rows)


def validate_factorial_rows(frame: pd.DataFrame) -> None:
    required = set(ID_COLUMNS + ["factorial_code", "gold_logit_margin", "result_id"])
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"factorial rows lack columns: {sorted(missing)}")
    if frame["result_id"].duplicated().any():
        raise ValueError("duplicate result IDs")
    counts = frame.groupby(ID_COLUMNS)["factorial_code"].agg(lambda x: tuple(sorted(x)))
    bad = counts[counts != ("CC", "CE", "FC", "FE")]
    if len(bad):
        raise ValueError(f"{len(bad)} semantic instances lack exactly four factorial cells")


def factorial_effects(frame: pd.DataFrame) -> pd.DataFrame:
    validate_factorial_rows(frame)
    wide = frame.pivot_table(
        index=ID_COLUMNS,
        columns="factorial_code",
        values="gold_logit_margin",
        aggfunc="first",
    ).reset_index()
    out = wide[ID_COLUMNS].copy()
    out["position_effect_canonical_ids"] = wide.CE - wide.CC
    out["position_effect_fragmented_ids"] = wide.FE - wide.FC
    out["representation_effect_source_geometry"] = wide.FC - wide.CC
    out["representation_effect_expanded_geometry"] = wide.FE - wide.CE
    out["position_interaction"] = (wide.FE - wide.FC) - (wide.CE - wide.CC)
    out["total_fragmentation_shift"] = wide.FE - wide.CC
    return out


def cluster_ci(
    frame: pd.DataFrame,
    column: str,
    seed: int,
    reps: int = 10_000,
    cluster: str = "trial_index",
) -> tuple[float, float, float]:
    clean = frame[[cluster, column]].dropna()
    clusters = np.asarray(sorted(clean[cluster].unique()))
    if not len(clusters):
        raise ValueError("no clusters")
    grouped = {
        item: clean.loc[clean[cluster] == item, column].to_numpy(float)
        for item in clusters
    }
    rng = np.random.default_rng(seed)
    boots = np.empty(reps)
    for index in range(reps):
        sample = rng.choice(clusters, size=len(clusters), replace=True)
        boots[index] = np.concatenate([grouped[item] for item in sample]).mean()
    return (
        float(clean[column].mean()),
        float(np.quantile(boots, 0.025)),
        float(np.quantile(boots, 0.975)),
    )


def summarize_factorial(
    frame: pd.DataFrame, bootstrap_seed: int = 20260828, reps: int = 10_000
) -> dict[str, dict[str, float]]:
    effects = factorial_effects(frame)
    summary: dict[str, dict[str, float]] = {}
    for column in FACTORIAL_EFFECTS:
        seed = bootstrap_seed + stable_int("pooled", column) % 100000
        mean, lo, hi = cluster_ci(effects, column, seed, reps=reps)
        summary[column] = {"mean": mean, "ci_lo": lo, "ci_hi": hi}
    return summary


def canonical_metrics(frame: pd.DataFrame) -> dict[str, float]:
    canonical = frame[frame["factorial_code"] == "CC"]
    counts = canonical["predicted_digit"].value_counts()
    return {
        "canonical_accuracy": float(canonical["correct"].mean()),
        "canonical_max_pred_share": float(counts.max() / counts.sum()),
    }


def compare_reported(
    observed: dict[str, dict[str, float]], expected: dict[str, Any], atol: float = 1e-12
) -> list[str]:
    mismatches: list[str] = []
    expected_effects = expected["effects"]
    for effect, values in observed.items():
        for field, value in values.items():
            target = float(expected_effects[effect][field])
            if not np.isclose(value, target, atol=atol, rtol=0):
                mismatches.append(f"{effect}.{field}: observed={value!r}, expected={target!r}")
    return mismatches

