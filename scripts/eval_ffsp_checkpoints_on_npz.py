from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from eval_downloaded_routing_checkpoints import (
    REPO_ROOT,
    build_model,
    load_manifest,
    parse_problem_key,
    run_ffsp_augmented_evaluation,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate downloaded FFSP checkpoints on explicit NPZ run_time test sets."
    )
    parser.add_argument("--manifest", type=Path, default=REPO_ROOT / "downloads" / "manifest.json")
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--problems", type=str, default="ffsp50,ffsp100")
    parser.add_argument("--methods", type=str, default="")
    parser.add_argument("--ffsp50-file", type=Path, default=None)
    parser.add_argument("--ffsp100-file", type=Path, default=None)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--test-batch-size", type=int, default=64)
    parser.add_argument("--ffsp-aug-factor", type=int, default=128)
    parser.add_argument("--ffsp-aug-batch-size", type=int, default=128)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    problem_filter = {token.strip().lower() for token in args.problems.split(",") if token.strip()}
    method_filter = {token.strip().lower() for token in args.methods.split(",") if token.strip()}
    data_files = {"ffsp50": args.ffsp50_file, "ffsp100": args.ffsp100_file}

    rows: list[dict[str, object]] = []
    for entry in load_manifest(args.manifest, REPO_ROOT):
        if entry.problem_key not in problem_filter:
            continue
        if method_filter and entry.method not in method_filter:
            continue
        env_name, _ = parse_problem_key(entry.problem_key)
        if env_name != "ffsp":
            continue
        data_file = data_files.get(entry.problem_key)
        if data_file is None:
            raise ValueError(f"Missing data file for {entry.problem_key}")
        data_file = data_file.resolve()
        with np.load(data_file) as payload:
            test_data_size = int(payload["run_time"].shape[0])

        model, hparams, _ = build_model(entry, REPO_ROOT)
        model.env.test_file = str(data_file)
        model.data_cfg["test_data_size"] = test_data_size
        model.data_cfg["test_batch_size"] = int(args.test_batch_size)
        model.test_metrics = ["reward", "max_reward", "max_aug_reward"]

        result = run_ffsp_augmented_evaluation(
            model=model,
            hparams=hparams,
            device=args.device,
            ffsp_aug_factor=int(args.ffsp_aug_factor),
            ffsp_aug_batch_size=int(args.ffsp_aug_batch_size),
        )
        row = {
            "problem_key": entry.problem_key,
            "method": entry.method,
            "checkpoint_path": str(entry.checkpoint_path),
            "test_file": str(data_file),
            **result,
        }
        rows.append(row)
        print(
            f"{entry.problem_key}/{entry.method}: "
            f"max_reward={result['test_max_reward']} "
            f"max_aug_reward={result['test_max_aug_reward']}"
        )

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with args.output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
