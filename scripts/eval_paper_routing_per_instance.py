from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tensordict import TensorDict


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rl4co.utils.ops import unbatchify
from scripts.eval_downloaded_routing_checkpoints import (
    build_model,
    load_manifest,
    parse_problem_key,
)


METHOD_NAMES = {
    "po": "PO4COPs",
    "sll": "SLL",
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
    device: torch.device,
    test_batch_size: int,
    num_instances: int | None,
    tsp_start_node: str,
    tsp_logit_clipping: float | None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    env_name, size = parse_problem_key(entry.problem_key)
    if env_name not in {"tsp", "cvrp"}:
        raise ValueError(f"Routing evaluator received {entry.problem_key}")

    model, hparams, test_file = build_model(entry, REPO_ROOT)
    canonical_prefix = "vrp" if env_name == "cvrp" else "tsp"
    canonical_dir = "vrp" if env_name == "cvrp" else "tsp"
    canonical_test_file = (
        REPO_ROOT
        / "data"
        / canonical_dir
        / f"{canonical_prefix}{size}_test_seed{int(hparams.get('seed', 1234))}.npz"
    )
    if canonical_test_file.exists():
        test_file = str(canonical_test_file.resolve())
    model.data_cfg["test_batch_size"] = int(test_batch_size)
    if test_file is not None:
        model.env.test_file = test_file
    if num_instances is not None:
        model.data_cfg["test_data_size"] = int(num_instances)
    if env_name == "tsp":
        model.policy.start_node = str(tsp_start_node)
        model.policy.eval_type = "argmax"
        model.policy.test_decode_type = "greedy"
        if tsp_logit_clipping is not None:
            for decoder_layer in model.policy.decoder.layers:
                decoder_layer.logit_clipping = float(tsp_logit_clipping)
    model = model.to(device)
    model.eval()

    explicit_batches = None
    if test_file is not None and Path(test_file).suffix.lower() == ".npz":
        # Delegate NPZ parsing to the environment so CVRP's depot, demand and
        # capacity fields receive the same normalization as normal test data.
        dataset = model.env.load_data(Path(test_file))
        if len(dataset.batch_size) != 1:
            raise ValueError(
                f"Expected a rank-1 test dataset, got {dataset.batch_size}"
            )
        if num_instances is not None:
            dataset = dataset[: int(num_instances)]
        count = int(dataset.batch_size[0])
        explicit_batches = [
            dataset[offset : offset + test_batch_size]
            for offset in range(0, count, test_batch_size)
        ]
    else:
        model.setup(stage="test")

    num_starts = int(model.num_starts or size)
    num_augment = int(model.num_augment or 1)
    rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    offset = 0
    with torch.inference_mode():
        batches = explicit_batches if explicit_batches is not None else model.test_dataloader()
        for batch in batches:
            batch = batch.to(device)
            td = model.env.reset(batch).to(device)
            if num_augment > 1:
                if model.augment is None:
                    raise RuntimeError(
                        f"{entry.problem_key}/{entry.method} requests augmentation without a transform"
                    )
                td = model.augment(td)
            out = model.policy(
                td,
                model.env,
                phase="test",
                num_starts=num_starts,
                return_actions=False,
            )
            rewards = unbatchify(out["reward"], (num_augment, num_starts))
            if rewards.ndim != 3:
                raise RuntimeError(
                    f"Expected [batch, augment, starts] reward tensor, got {tuple(rewards.shape)}"
                )
            costs = -rewards.amax(dim=(-1, -2)).detach().cpu()
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
                        "num_starts": num_starts,
                        "num_augment": num_augment,
                        "checkpoint": entry.checkpoint_path.relative_to(REPO_ROOT).as_posix(),
                    }
                )
            offset += int(costs.numel())

    elapsed = time.perf_counter() - started
    checkpoint_path = entry.checkpoint_path.resolve()
    test_path = Path(test_file).resolve() if test_file else None
    summary = {
        "problem": entry.problem_key,
        "method": METHOD_NAMES[entry.method],
        "legacy_method": entry.method,
        "num_instances": len(rows),
        "mean_cost": sum(row["cost"] for row in rows) / len(rows),
        "elapsed_sec": elapsed,
        "num_starts": num_starts,
        "num_augment": num_augment,
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": _sha256(checkpoint_path),
        "test_file": str(test_path) if test_path else None,
        "test_file_sha256": _sha256(test_path) if test_path and test_path.is_file() else None,
        "seed": int(hparams.get("seed", 1234)),
    }
    del model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return rows, summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate paper routing checkpoints and save one augmented best cost per instance."
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=REPO_ROOT
        / "paper_materials"
        / "statistical_analysis"
        / "paper_eval_manifest.json",
    )
    parser.add_argument("--problems", default="tsp50,tsp100,cvrp50,cvrp100")
    parser.add_argument("--methods", default="po,sll,bopo,loss_only,weighting")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--test-batch-size", type=int, default=256)
    parser.add_argument("--num-instances", type=int, default=None)
    parser.add_argument(
        "--tsp-start-node",
        choices=("same", "random", "pomo"),
        default="pomo",
    )
    parser.add_argument("--tsp-logit-clipping", type=float, default=None)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "paper_materials" / "statistical_analysis" / "routing_eval",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    problem_filter = _tokens(args.problems)
    method_filter = _tokens(args.methods)
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
            device=device,
            test_batch_size=int(args.test_batch_size),
            num_instances=args.num_instances,
            tsp_start_node=args.tsp_start_node,
            tsp_logit_clipping=args.tsp_logit_clipping,
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
