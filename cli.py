"""Small command line surface for CPU reproducibility checks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import config_hash, load_config
from .statistics import canonical_metrics, compare_reported, load_jsonl, summarize_factorial


def reproduce(args: argparse.Namespace) -> int:
    repo = args.repo_root.resolve()
    if args.experiment != "qwen_v061":
        raise ValueError("this release currently exposes qwen_v061")
    frame = load_jsonl(repo / "data/qwen/v061_factorial_4k8k.jsonl")
    summary = summarize_factorial(frame, reps=args.bootstrap_reps)
    metrics = canonical_metrics(frame)
    expected = json.loads((repo / "config/reported_results.json").read_text(encoding="utf-8"))[
        "qwen_v061"
    ]
    mismatches = compare_reported(summary, expected)
    if args.bootstrap_reps == 10_000:
        for field, value in metrics.items():
            if value != float(expected[field]):
                mismatches.append(f"{field}: observed={value!r}, expected={expected[field]!r}")
    print(json.dumps({"metrics": metrics, "effects": summary}, indent=2, sort_keys=True))
    if args.bootstrap_reps != 10_000:
        print("NOTE: exact reported-CI comparison requires --bootstrap-reps 10000")
        return 0
    if mismatches:
        print("FAIL")
        print("\n".join(mismatches))
        return 1
    print("PASS: qwen_v061 matches every reported mean and confidence interval exactly.")
    return 0


def show_config_hash(args: argparse.Namespace) -> int:
    data = load_config(args.config)
    print(config_hash(data))
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(prog="tacl-factorization")
    subparsers = parser.add_subparsers(dest="command", required=True)
    repro = subparsers.add_parser("reproduce")
    repro.add_argument("--repo-root", type=Path, required=True)
    repro.add_argument("--experiment", default="qwen_v061")
    repro.add_argument("--bootstrap-reps", type=int, default=10_000)
    repro.set_defaults(function=reproduce)
    config_parser = subparsers.add_parser("config-hash")
    config_parser.add_argument("config", type=Path)
    config_parser.set_defaults(function=show_config_hash)
    args = parser.parse_args()
    raise SystemExit(args.function(args))


if __name__ == "__main__":
    main()

