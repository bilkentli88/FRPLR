"""Resume-safe execution of a frozen symbolic factorial grid."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from .determinism import stable_rng
from .generator import build_instance, position_rank_for_trial
from .geometry import Condition, assemble_factorial
from .results import JsonlLedger
from .scoring import flatten_score


def make_result_id(metadata: dict[str, Any]) -> str:
    blob = json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:24]


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def run_symbolic_factorial(
    config: dict[str, Any],
    adapter: Any,
    model: Any,
    output_path: str | Path,
    scorer: Callable[..., dict[str, Any]],
    limit_new_rows: int | None = None,
    row_provenance: dict[str, Any] | None = None,
) -> int:
    """Execute a frozen four-cell grid and append only missing rows."""

    ledger = JsonlLedger(output_path)
    row_provenance = dict(row_provenance or {})
    written = 0
    conditions = list(config.get("conditions", ["CC", "CE", "FC", "FE"]))
    seeds = config["semantic_seeds"]
    for filler_language in config["filler_languages"]:
        for target_tokens in config["target_lengths"]:
            for trial_index, semantic_seed in enumerate(seeds):
                instance = build_instance(
                    filler_language=filler_language,
                    semantic_seed=int(semantic_seed),
                    trial_index=trial_index,
                    target_chat_tokens=int(target_tokens),
                    count_chat_tokens=adapter.count_chat_tokens,
                    verify_chat_shell=adapter.verify_chat_shell,
                    n_records=int(config["n_records"]),
                )
                encoding = adapter.encode_instance(instance)
                factorial = assemble_factorial(adapter.prefix_ids, adapter.suffix_ids, encoding)
                order = conditions[:]
                stable_rng(
                    config["condition_order_namespace"],
                    filler_language,
                    target_tokens,
                    semantic_seed,
                    trial_index,
                ).shuffle(order)
                evidence_text = "\n".join(instance.evidence_lines)
                query_text = "\n".join(instance.query_lines)
                for order_index, code in enumerate(order):
                    condition: Condition = factorial[code]
                    metadata = {
                        "phase": config["phase"],
                        "model": config["model"],
                        "task_version": config["task_version"],
                        "scorer_version": config["scorer_version"],
                        "position_intervention": config["position_intervention"],
                        "filler_language": filler_language,
                        "target_chat_tokens": int(target_tokens),
                        "semantic_seed": int(semantic_seed),
                        "trial_index": trial_index,
                        "n_records": int(config["n_records"]),
                        "factorial_code": code,
                        "condition": condition.name,
                        "frag_keep": (
                            None
                            if code in {"CC", "CE"}
                            else config["frag_keep_by_language"][filler_language]
                        ),
                    }
                    result_id = make_result_id(metadata)
                    if result_id in ledger.ids:
                        continue
                    scored = scorer(
                        model=model,
                        condition=condition,
                        gold_digit=instance.gold_digit,
                        digit_ids=adapter.digit_ids,
                        tokenizer=adapter.tokenizer,
                    )
                    row = {
                        "result_id": result_id,
                        **metadata,
                        **row_provenance,
                        "condition_order_index": order_index,
                        "gold_digit": instance.gold_digit,
                        "target_key": instance.target_key,
                        "position_rank": position_rank_for_trial(trial_index),
                        "target_record_position": instance.target_record_position,
                        "realized_full_tokens": condition.realized_tokens,
                        "position_max": condition.position_max,
                        "evidence_query_position_gap": condition.gaps["evidence_query"],
                        "evidence_readout_position_gap": condition.gaps["evidence_readout"],
                        "filler_position_span": condition.gaps["filler_span"],
                        "ids_hash": condition.ids_hash,
                        "position_hash": condition.position_hash,
                        "prompt_sha256": _sha256(instance.user_content),
                        "evidence_sha256": _sha256(evidence_text),
                        "query_sha256": _sha256(query_text),
                        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                        **flatten_score(scored),
                    }
                    ledger.append(row)
                    written += 1
                    print(
                        f"fill={filler_language} L={target_tokens:5d} trial={trial_index:02d} "
                        f"{code} gold={instance.gold_digit} pred={scored['predicted_digit']} "
                        f"margin={scored['gold_logit_margin']:+8.3f} "
                        f"n={condition.realized_tokens:5d} "
                        f"gap={condition.gaps['evidence_readout']:8.1f}"
                    )
                    if limit_new_rows is not None and written >= limit_new_rows:
                        return written
    return written
