"""Outcome-independent model and tokenizer qualification gates."""

from __future__ import annotations

import inspect
from typing import Any, Iterable, Mapping

import numpy as np


def static_model_checks(
    model: Any,
    tokenizer: Any,
    required_context: int = 8192,
    candidate_token_ids: Mapping[str, int] | None = None,
) -> dict[str, Any]:
    backbone = getattr(model, "model", model)
    signatures = [inspect.signature(backbone.forward)]
    if backbone is not model and hasattr(model, "forward"):
        signatures.append(inspect.signature(model.forward))
    config = model.config
    position_config = getattr(config, "text_config", config)
    max_context = int(
        getattr(position_config, "max_position_embeddings", 0)
        or getattr(position_config, "n_positions", 0)
        or 0
    )
    digit_ids: dict[str, int | None] = {}
    if candidate_token_ids is not None:
        if set(candidate_token_ids) != set("1234"):
            raise ValueError("candidate_token_ids must have exactly keys 1, 2, 3, 4")
        digit_ids = {digit: int(candidate_token_ids[digit]) for digit in "1234"}
    else:
        for digit in "1234":
            ids = tokenizer(digit, add_special_tokens=False)["input_ids"]
            digit_ids[digit] = int(ids[0]) if len(ids) == 1 else None
    result = {
        "model_type": getattr(config, "model_type", type(config).__name__),
        "max_context": max_context,
        "required_context": required_context,
        "context_gate": max_context >= required_context,
        "position_ids_exposed": any(
            "position_ids" in signature.parameters for signature in signatures
        ),
        "digit_ids": digit_ids,
        "digit_gate": all(value is not None for value in digit_ids.values())
        and len(set(digit_ids.values())) == 4,
    }
    result["passed"] = bool(
        result["context_gate"] and result["position_ids_exposed"] and result["digit_gate"]
    )
    return result


def select_fragmentation_keep(
    observations: Iterable[dict[str, Any]],
    target_ratio: float = 1.70,
    acceptable: tuple[float, float] = (1.25, 2.10),
) -> dict[str, Any]:
    """Choose a keep value using only token-count inflation, never outcomes."""

    rows = list(observations)
    if not rows:
        raise ValueError("no calibration observations")
    by_keep: dict[int, list[float]] = {}
    for row in rows:
        keep = int(row["keep_pct"])
        ratio = float(row["fragmented_tokens"]) / float(row["canonical_tokens"])
        by_keep.setdefault(keep, []).append(ratio)
    candidates = []
    for keep, ratios in by_keep.items():
        candidates.append(
            {
                "keep_pct": keep,
                "mean_ratio": float(np.mean(ratios)),
                "min_ratio": float(np.min(ratios)),
                "max_ratio": float(np.max(ratios)),
                "within_range": min(ratios) >= acceptable[0] and max(ratios) <= acceptable[1],
            }
        )
    viable = [item for item in candidates if item["within_range"]]
    pool = viable or candidates
    selected = min(pool, key=lambda item: (abs(item["mean_ratio"] - target_ratio), -item["keep_pct"]))
    return {
        "target_ratio": target_ratio,
        "acceptable_range": list(acceptable),
        "selected": selected,
        "passed": bool(viable),
        "candidates": sorted(candidates, key=lambda item: item["keep_pct"], reverse=True),
    }
