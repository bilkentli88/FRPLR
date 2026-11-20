#!/usr/bin/env python3
"""Reproduce every numerical effect reported in the revised manuscript.

No GPU, model download, network connection, or Hugging Face account is required.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


BOOTSTRAP_REPS = 10_000
BOOTSTRAP_SEED = 20260828
ID_COLS = ["filler_language", "target_chat_tokens", "semantic_seed", "trial_index"]


def stable_int(*parts: object) -> int:
    blob = "|".join(map(str, parts)).encode("utf-8")
    return int.from_bytes(hashlib.sha256(blob).digest()[:8], "big")


def config_hash(config: dict[str, object]) -> str:
    payload = json.dumps(
        config, sort_keys=True, ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


def load_jsonl(path: Path) -> pd.DataFrame:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return pd.DataFrame(rows)


def cluster_ci(frame: pd.DataFrame, column: str, seed: int) -> tuple[float, float, float]:
    clean = frame[["trial_index", column]].dropna()
    clusters = np.asarray(sorted(clean.trial_index.unique()))
    grouped = {
        cluster: clean.loc[clean.trial_index == cluster, column].to_numpy(float)
        for cluster in clusters
    }
    rng = np.random.default_rng(seed)
    bootstrap = np.empty(BOOTSTRAP_REPS)
    for index in range(BOOTSTRAP_REPS):
        sample = rng.choice(clusters, size=len(clusters), replace=True)
        bootstrap[index] = np.concatenate([grouped[item] for item in sample]).mean()
    return (
        float(clean[column].mean()),
        float(np.quantile(bootstrap, 0.025)),
        float(np.quantile(bootstrap, 0.975)),
    )


def validate_complete_factorial(frame: pd.DataFrame, experiment: str) -> None:
    required = set(ID_COLS + ["factorial_code", "gold_logit_margin", "result_id"])
    missing = required - set(frame.columns)
    if missing:
        raise AssertionError(f"{experiment}: missing fields {sorted(missing)}")
    if frame.result_id.duplicated().any():
        raise AssertionError(f"{experiment}: duplicate result IDs")
    observed = frame.groupby(ID_COLS).factorial_code.agg(lambda x: tuple(sorted(x)))
    if not (observed == ("CC", "CE", "FC", "FE")).all():
        raise AssertionError(f"{experiment}: incomplete factorial instance")


def v060_effects(frame: pd.DataFrame) -> pd.DataFrame:
    wide = frame.pivot_table(
        index=ID_COLS, columns="condition", values="forced_choice_nll", aggfunc="first"
    ).reset_index()
    out = wide[ID_COLS].copy()
    out["fragmentation_penalty"] = wide.filler_frag_normal - wide.canonical_normal
    out["block_affine_rescue"] = wide.filler_frag_normal - wide.filler_frag_block_affine
    out["fractional_rescue"] = wide.filler_frag_normal - wide.filler_frag_fractional
    out["fractional_advantage_over_block"] = (
        wide.filler_frag_block_affine - wide.filler_frag_fractional
    )
    out["fractional_residual_vs_canonical"] = (
        wide.filler_frag_fractional - wide.canonical_normal
    )
    return out


def factorial_effects(frame: pd.DataFrame, experiment: str) -> pd.DataFrame:
    validate_complete_factorial(frame, experiment)
    wide = frame.pivot_table(
        index=ID_COLS, columns="factorial_code", values="gold_logit_margin", aggfunc="first"
    ).reset_index()
    out = wide[ID_COLS].copy()
    out["position_effect_canonical_ids"] = wide.CE - wide.CC
    out["position_effect_fragmented_ids"] = wide.FE - wide.FC
    out["representation_effect_source_geometry"] = wide.FC - wide.CC
    out["representation_effect_expanded_geometry"] = wide.FE - wide.CE
    out["position_interaction"] = (wide.FE - wide.FC) - (wide.CE - wide.CC)
    out["total_fragmentation_shift"] = wide.FE - wide.CC
    return out


def localization_effects(frame: pd.DataFrame) -> pd.DataFrame:
    wide = frame.pivot_table(
        index=ID_COLS, columns="geometry_code", values="gold_logit_margin", aggfunc="first"
    ).reset_index()
    out = wide[ID_COLS].copy()
    out["query_shift_canonical"] = wide.CQ - wide.CC
    out["query_shift_fragmented"] = wide.FQ - wide.FC
    out["filler_spread_canonical"] = wide.CE - wide.CQ
    out["filler_spread_fragmented"] = wide.FE - wide.FQ
    out["total_expansion_canonical"] = wide.CE - wide.CC
    out["total_expansion_fragmented"] = wide.FE - wide.FC
    out["representation_source"] = wide.FC - wide.CC
    out["representation_query_shift"] = wide.FQ - wide.CQ
    out["representation_spread"] = wide.FE - wide.CE
    out["query_shift_interaction"] = (wide.FQ - wide.FC) - (wide.CQ - wide.CC)
    out["filler_spread_interaction"] = (wide.FE - wide.FQ) - (wide.CE - wide.CQ)
    if not np.array_equal(
        out.total_expansion_canonical.to_numpy(),
        (out.query_shift_canonical + out.filler_spread_canonical).to_numpy(),
    ):
        raise AssertionError("canonical localization components do not add exactly")
    if not np.array_equal(
        out.total_expansion_fragmented.to_numpy(),
        (out.query_shift_fragmented + out.filler_spread_fragmented).to_numpy(),
    ):
        raise AssertionError("fragmented localization components do not add exactly")
    return out


def canonical_metrics(frame: pd.DataFrame, code_column: str, code: str) -> tuple[float, float]:
    canonical = frame[frame[code_column] == code]
    counts = canonical.predicted_digit.value_counts()
    return float(canonical.correct.mean()), float(counts.max() / counts.sum())


def assert_triplet(
    experiment: str,
    effect: str,
    observed: tuple[float, float, float],
    expected: dict[str, float],
) -> None:
    target = [expected["mean"], expected["ci_lo"], expected["ci_hi"]]
    if not np.allclose(observed, target, atol=1e-12, rtol=0):
        raise AssertionError(
            f"{experiment}/{effect} mismatch: observed={observed!r}, expected={target!r}"
        )


def summarize_effects(
    experiment: str,
    role: str,
    effects: pd.DataFrame,
    expected: dict[str, dict[str, float]],
    seed_prefix: str,
    bootstrap_seed: int = BOOTSTRAP_SEED,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    effect_names = [column for column in effects.columns if column not in ID_COLS]
    for effect in effect_names:
        seed = bootstrap_seed + stable_int(seed_prefix, effect) % 100000
        observed = cluster_ci(effects, effect, seed)
        assert_triplet(experiment, effect, observed, expected[effect])
        rows.append(
            {
                "experiment": experiment,
                "article_role": role,
                "effect": effect,
                "mean": observed[0],
                "ci_lo": observed[1],
                "ci_hi": observed[2],
            }
        )
    return rows


def analyze(root: Path) -> pd.DataFrame:
    legacy_targets = json.loads(
        (root / "summaries/original_experiments_reported_results.json").read_text()
    )
    legacy_specs = {
        "qwen_v060": ("data/qwen/v060_mechanism_4k8k.jsonl", "v060", "appendix"),
        "qwen_v061": ("data/qwen/v061_factorial_4k8k.jsonl", "factorial", "appendix"),
        "qwen_v062": (
            "data/qwen/v062_geometry_localization_4k8k.jsonl",
            "localization",
            "main_localization",
        ),
        "gemma_v071": (
            "data/gemma/v071_gemma_factorial_4k8k.jsonl",
            "factorial",
            "appendix",
        ),
        "gemma_v072": (
            "data/gemma/v072_gemma_geometry_localization_4k8k.jsonl",
            "localization",
            "main_localization",
        ),
    }
    new_specs = {
        "qwen_harmonized": (
            "data/qwen/qwen_harmonized_robustness_v1.jsonl",
            "summaries/qwen_harmonized_robustness_v1_summary.json",
            "configs/qwen_harmonized_robustness_v1.json",
            "audits/qwen_harmonized_engineering_audit_v1.json",
            "engineering_audit_sha256",
        ),
        "gemma_harmonized": (
            "data/gemma/gemma_harmonized_robustness_v1.jsonl",
            "summaries/gemma_harmonized_robustness_v1_summary.json",
            "configs/gemma_harmonized_robustness_v1.json",
            "audits/gemma_harmonized_engineering_audit_v1.json",
            "engineering_audit_sha256",
        ),
        "phi_confirmatory": (
            "data/phi/phi_factorial_confirmatory_v1.jsonl",
            "summaries/phi_factorial_confirmatory_v1_summary.json",
            "configs/phi_factorial_confirmatory_v1.json",
            "audits/phi_engineering_audit.json",
            "qualification_audit_sha256",
        ),
        "mistral_confirmatory": (
            "data/mistral/mistral_factorial_confirmatory_v1.jsonl",
            "summaries/mistral_factorial_confirmatory_v1_summary.json",
            "configs/mistral_factorial_confirmatory_v1.json",
            "audits/mistral_engineering_audit.json",
            "qualification_audit_sha256",
        ),
    }

    output: list[dict[str, object]] = []
    for experiment, (relative_path, kind, role) in legacy_specs.items():
        frame = load_jsonl(root / relative_path)
        if frame.result_id.duplicated().any():
            raise AssertionError(f"{experiment}: duplicate result IDs")
        if kind == "v060":
            effects = v060_effects(frame)
            accuracy, share = canonical_metrics(frame, "condition", "canonical_normal")
            seed_prefix = "forced_choice_nll"
        elif kind == "factorial":
            effects = factorial_effects(frame, experiment)
            accuracy, share = canonical_metrics(frame, "factorial_code", "CC")
            seed_prefix = "pooled"
        else:
            effects = localization_effects(frame)
            accuracy, share = canonical_metrics(frame, "geometry_code", "CC")
            seed_prefix = "pooled"
        target = legacy_targets[experiment]
        if not np.allclose(
            [accuracy, share],
            [target["canonical_accuracy"], target["canonical_max_pred_share"]],
            atol=1e-12,
            rtol=0,
        ):
            raise AssertionError(f"{experiment}: canonical substrate mismatch")
        output.extend(
            summarize_effects(
                experiment, role, effects, target["effects"], seed_prefix
            )
        )

    for experiment, spec in new_specs.items():
        data_path, summary_path, config_path, audit_path, audit_field = spec
        frame = load_jsonl(root / data_path)
        summary = json.loads((root / summary_path).read_text())
        config = json.loads((root / config_path).read_text())
        observed_config_hash = config_hash(config)
        if observed_config_hash != summary["factorial_config_hash"]:
            raise AssertionError(f"{experiment}: frozen configuration hash mismatch")
        if set(frame.factorial_config_hash) != {observed_config_hash}:
            raise AssertionError(f"{experiment}: row configuration provenance mismatch")
        observed_audit_hash = hashlib.sha256((root / audit_path).read_bytes()).hexdigest()
        if summary[audit_field] != observed_audit_hash:
            raise AssertionError(f"{experiment}: summary audit provenance mismatch")
        if set(frame[audit_field]) != {observed_audit_hash}:
            raise AssertionError(f"{experiment}: row audit provenance mismatch")
        if set(frame.runtime_fingerprint) != {summary["runtime_fingerprint"]}:
            raise AssertionError(f"{experiment}: runtime provenance mismatch")
        if len(frame) != int(summary["rows"]):
            raise AssertionError(f"{experiment}: unexpected row count")
        effects = factorial_effects(frame, experiment)
        accuracy, share = canonical_metrics(frame, "factorial_code", "CC")
        metrics = summary["canonical_metrics"]
        if not np.allclose(
            [accuracy, share],
            [metrics["canonical_accuracy"], metrics["canonical_max_pred_share"]],
            atol=1e-12,
            rtol=0,
        ):
            raise AssertionError(f"{experiment}: canonical substrate mismatch")
        output.extend(
            summarize_effects(
                experiment,
                "main_factorial",
                effects,
                summary["pooled_effects"],
                "pooled",
                int(config["bootstrap"]["seed"]),
            )
        )

    return pd.DataFrame(output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repo-root", type=Path, default=Path(__file__).resolve().parents[1]
    )
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    results = analyze(args.repo_root)
    print(results.to_string(index=False))
    if args.write:
        output_dir = args.repo_root / "results"
        output_dir.mkdir(exist_ok=True)
        results.to_csv(output_dir / "reported_effects_reproduced.csv", index=False)
    print(
        "\nPASS: all reported numerical effects reproduce exactly from "
        "the committed raw rows."
    )


if __name__ == "__main__":
    main()
