from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rl4co.utils.ops import batchify, unbatchify
from scripts.eval_downloaded_routing_checkpoints import build_model, load_manifest


METHOD_NAMES = {
    "po": "PO4COPs",
    "bopo": "BOPO",
    "loss_only": "USW",
    "weighting": "ASW",
}


def _tokens(raw: str) -> set[str]:
    return {token.strip().lower() for token in str(raw).split(",") if token.strip()}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def evaluate_entry(
    entry,
    *,
    data_file: Path,
    device: torch.device,
    test_batch_size: int,
    augment_factor: int,
    augment_batch_size: int,
    eval_seed: int | None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    with np.load(data_file) as payload:
        count = int(payload["run_time"].shape[0])

    model, hparams, _ = build_model(entry, REPO_ROOT)
    model.env.test_file = str(data_file)
    model.data_cfg["test_data_size"] = count
    model.data_cfg["test_batch_size"] = int(test_batch_size)
    model = model.to(device)
    model.eval()
    seed = int(hparams.get("seed", 1234) if eval_seed is None else eval_seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    model.setup(stage="test")

    rows: list[dict[str, Any]] = []
    offset = 0
    started = time.perf_counter()
    with torch.inference_mode():
        for batch in model.test_dataloader():
            batch = batch.to(device)
            best_aug_reward = None
            remaining = int(augment_factor)
            while remaining > 0:
                chunk = min(int(augment_batch_size), remaining)
                batch_aug = batchify(batch, chunk)
                td_aug = model.env.reset(batch_aug)
                num_starts = int(model.num_starts or model.env.get_num_starts(td_aug))
                out = model.policy(
                    td_aug.clone(),
                    model.env,
                    phase="test",
                    num_starts=num_starts,
                    return_actions=False,
                )
                reward_ms = unbatchify(out["reward"], (0, num_starts))
                reward_aug = unbatchify(reward_ms, chunk)
                chunk_best = reward_aug.amax(dim=(-1, -2))
                best_aug_reward = (
                    chunk_best
                    if best_aug_reward is None
                    else torch.maximum(best_aug_reward, chunk_best)
                )
                remaining -= chunk

            if best_aug_reward is None:
                raise RuntimeError("No FFSP augmentation rewards were produced")
            costs = -best_aug_reward.detach().cpu()
            for local_index, cost in enumerate(costs.tolist()):
                index = offset + local_index
                rows.append(
                    {
                        "problem": entry.problem_key,
                        "method": METHOD_NAMES[entry.method],
                        "legacy_method": entry.method,
                        "instance_id": f"{entry.problem_key}_{index:05d}",
                        "instance_index": index,
                        "cost": float(cost),
                        "num_starts": int(model.num_starts or 0),
                        "num_augment": int(augment_factor),
                        "checkpoint": entry.checkpoint_path.relative_to(REPO_ROOT).as_posix(),
                    }
                )
            offset += int(costs.numel())

    elapsed = time.perf_counter() - started
    checkpoint_path = entry.checkpoint_path.resolve()
    summary = {
        "problem": entry.problem_key,
        "method": METHOD_NAMES[entry.method],
        "legacy_method": entry.method,
        "num_instances": len(rows),
        "mean_cost": sum(row["cost"] for row in rows) / len(rows),
        "elapsed_sec": elapsed,
        "num_starts": int(model.num_starts or 0),
        "num_augment": int(augment_factor),
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": _sha256(checkpoint_path),
        "test_file": str(data_file),
        "test_file_sha256": _sha256(data_file),
        "seed": seed,
    }
    del model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return rows, summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate paper FFSP checkpoints and save the best augmented cost per instance."
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=REPO_ROOT
        / "paper_materials"
        / "statistical_analysis"
        / "paper_eval_manifest.json",
    )
    parser.add_argument("--problems", default="ffsp50,ffsp100")
    parser.add_argument("--methods", default="po,bopo,loss_only,weighting")
    parser.add_argument("--ffsp50-file", type=Path, required=True)
    parser.add_argument("--ffsp100-file", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--test-batch-size", type=int, default=64)
    parser.add_argument("--augment-factor", type=int, default=128)
    parser.add_argument("--augment-batch-size", type=int, default=128)
    parser.add_argument(
        "--eval-seed",
        type=int,
        default=None,
        help="Override the checkpoint seed for a shared evaluation/augmentation random stream.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "paper_materials" / "statistical_analysis" / "ffsp_eval",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    problem_filter = _tokens(args.problems)
    method_filter = _tokens(args.methods)
    data_files = {
        "ffsp50": args.ffsp50_file.resolve(),
        "ffsp100": args.ffsp100_file.resolve(),
    }
    for problem in problem_filter:
        if problem not in data_files or not data_files[problem].is_file():
            raise FileNotFoundError(f"Missing data file for {problem}: {data_files.get(problem)}")

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    entries = [
        entry
        for entry in load_manifest(args.manifest.resolve(), REPO_ROOT)
        if entry.problem_key in problem_filter and entry.method in method_filter
    ]
    expected = {
        (problem, method)
        for problem in problem_filter
        for method in method_filter
    }
    present = {(entry.problem_key, entry.method) for entry in entries}
    missing = sorted(expected - present)
    if missing:
        raise ValueError(f"Manifest is missing requested entries: {missing}")

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for entry in entries:
        print(f"[eval] {entry.problem_key}/{entry.method}", flush=True)
        rows, summary = evaluate_entry(
            entry,
            data_file=data_files[entry.problem_key],
            device=device,
            test_batch_size=int(args.test_batch_size),
            augment_factor=int(args.augment_factor),
            augment_batch_size=int(args.augment_batch_size),
            eval_seed=args.eval_seed,
        )
        all_rows.extend(rows)
        summaries.append(summary)
        _write_rows(output_dir / "per_instance.csv", all_rows)
        (output_dir / "summary.json").write_text(
            json.dumps({"evaluations": summaries}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(
            f"[done] {entry.problem_key}/{entry.method} "
            f"n={summary['num_instances']} mean={summary['mean_cost']:.9f}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
