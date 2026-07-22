from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
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

from rl4co.utils.ops import batchify, unbatchify
from scripts.eval_downloaded_routing_checkpoints import build_model, load_manifest


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export augmented FFSP50/100 neural schedules as CP-SAT incumbents."
    )
    parser.add_argument("--problem", choices=("ffsp50", "ffsp100"), required=True)
    parser.add_argument("--method", choices=("po", "bopo", "loss_only", "weighting"), required=True)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=REPO_ROOT
        / "paper_materials"
        / "statistical_analysis"
        / "paper_eval_manifest.json",
    )
    parser.add_argument("--test-file", type=Path, required=True)
    parser.add_argument("--test-file-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--augment-factor", type=int, default=128)
    parser.add_argument("--augment-batch-size", type=int, default=4)
    parser.add_argument("--eval-seed", type=int, default=1234)
    parser.add_argument("--max-instances", type=int, default=0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    test_file = args.test_file.resolve()
    observed_test_sha = _sha256(test_file)
    if observed_test_sha != args.test_file_sha256.lower():
        raise ValueError("FFSP test-file SHA256 mismatch")
    with np.load(test_file) as payload:
        run_times = np.asarray(payload["run_time"], dtype=np.int64)
    expected_jobs = int(args.problem.removeprefix("ffsp"))
    if run_times.ndim != 3 or run_times.shape[1:] != (expected_jobs, 12):
        raise ValueError(f"Unexpected {args.problem} test shape: {run_times.shape}")
    count = run_times.shape[0] if args.max_instances <= 0 else min(args.max_instances, run_times.shape[0])
    run_times = run_times[:count]
    if count < 1 or args.batch_size < 1 or args.augment_factor < 1 or args.augment_batch_size < 1:
        raise ValueError("Counts and batch sizes must be positive")

    entries = [
        entry
        for entry in load_manifest(args.manifest.resolve(), REPO_ROOT)
        if entry.problem_key == args.problem and entry.method == args.method
    ]
    if len(entries) != 1:
        raise ValueError(f"Expected one manifest entry for {args.problem}/{args.method}, got {len(entries)}")
    entry = entries[0]
    checkpoint_sha = _sha256(entry.checkpoint_path)
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite FFSP hint output: {output_dir}")
    output_dir.mkdir(parents=True)
    hint_dir = output_dir / "hints"
    hint_dir.mkdir()

    model, hparams, _ = build_model(entry, REPO_ROOT)
    model.env.test_file = str(test_file)
    model.data_cfg["test_data_size"] = count
    model.data_cfg["test_batch_size"] = int(args.batch_size)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    model = model.to(device).eval()
    random.seed(args.eval_seed)
    np.random.seed(args.eval_seed)
    torch.manual_seed(args.eval_seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.eval_seed)
    rows: list[dict[str, Any]] = []
    offset = 0
    total_started = time.perf_counter()
    with torch.inference_mode():
        while offset < count:
            stop = min(offset + int(args.batch_size), count)
            batch_np = run_times[offset:stop]
            batch_count = stop - offset
            batch = TensorDict(
                {"run_time": torch.from_numpy(batch_np.copy())}, batch_size=[batch_count]
            ).to(device)
            best_rewards = torch.full((batch_count,), -torch.inf, device=device)
            best_schedules: list[torch.Tensor | None] = [None] * batch_count
            remaining = int(args.augment_factor)
            batch_started = time.perf_counter()
            while remaining > 0:
                chunk = min(int(args.augment_batch_size), remaining)
                batch_aug = batchify(batch, chunk)
                initial = model.env.reset(batch_aug)
                num_starts = int(model.num_starts or model.env.get_num_starts(initial))
                out = model.policy(
                    initial.clone(),
                    model.env,
                    phase="test",
                    num_starts=num_starts,
                    return_actions=False,
                    return_schedule=True,
                )
                rewards = unbatchify(unbatchify(out["reward"], (0, num_starts)), chunk)
                schedules = unbatchify(unbatchify(out["schedule"], (0, num_starts)), chunk)
                flat_rewards = rewards.reshape(batch_count, chunk * num_starts)
                chunk_rewards, chunk_indices = flat_rewards.max(dim=1)
                flat_schedules = schedules.reshape(
                    batch_count, chunk * num_starts, *schedules.shape[3:]
                )
                for local_index in range(batch_count):
                    if chunk_rewards[local_index] > best_rewards[local_index]:
                        best_rewards[local_index] = chunk_rewards[local_index]
                        best_schedules[local_index] = flat_schedules[
                            local_index, int(chunk_indices[local_index])
                        ].detach().cpu().long()
                remaining -= chunk

            batch_elapsed = time.perf_counter() - batch_started
            batch_run_time = batch_np
            for local_index in range(batch_count):
                index = offset + local_index
                schedule_tensor = best_schedules[local_index]
                if schedule_tensor is None:
                    raise RuntimeError(f"No schedule selected for instance {index}")
                objective = int(round(float(-best_rewards[local_index].cpu())))
                assignments = []
                for stage in range(3):
                    for job in range(expected_jobs):
                        chosen = []
                        for machine in range(4):
                            global_machine = stage * 4 + machine
                            start_value = int(schedule_tensor[global_machine, job])
                            if start_value >= 0:
                                chosen.append((machine, start_value))
                        if len(chosen) != 1:
                            raise RuntimeError(
                                f"Expected one machine for instance={index}, job={job}, stage={stage}; got {chosen}"
                            )
                        machine, start_value = chosen[0]
                        duration = int(batch_run_time[local_index, job, stage * 4 + machine])
                        assignments.append(
                            {
                                "job": job,
                                "stage": stage,
                                "machine": machine,
                                "start": start_value,
                                "end": start_value + duration,
                            }
                        )
                reconstructed = max(item["end"] for item in assignments)
                if reconstructed != objective:
                    raise RuntimeError(
                        f"Hint reconstruction mismatch at {index}: reward={objective}, schedule={reconstructed}"
                    )
                hint_payload = {
                    "instance_index": index,
                    "test_file_sha256": observed_test_sha,
                    "checkpoint_sha256": checkpoint_sha,
                    "problem": args.problem,
                    "method": args.method,
                    "objective": objective,
                    "schedule": assignments,
                }
                _write_json(hint_dir / f"instance_{index:05d}.json", hint_payload)
                row = {
                    "instance_index": index,
                    "objective": objective,
                    "batch_time_sec_per_instance": batch_elapsed / batch_count,
                }
                rows.append(row)
                print(json.dumps({"event": "hint", **row}), flush=True)
            offset += batch_count

    with (output_dir / "per_instance.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "protocol": f"{args.problem}_test{count}_starts{int(model.num_starts or 0)}_augment{args.augment_factor}_cp_sat_hints_v1",
        "problem": args.problem,
        "method": args.method,
        "count": count,
        "mean_objective": float(np.mean([row["objective"] for row in rows])),
        "total_elapsed_sec": time.perf_counter() - total_started,
        "checkpoint": entry.checkpoint_path.as_posix(),
        "checkpoint_sha256": checkpoint_sha,
        "test_file": test_file.as_posix(),
        "test_file_sha256": observed_test_sha,
        "num_starts": int(model.num_starts or 0),
        "num_augment": int(args.augment_factor),
        "seed": int(args.eval_seed),
        "checkpoint_seed": int(hparams.get("seed", 1234)),
    }
    _write_json(output_dir / "summary.json", summary)
    print(json.dumps({"event": "summary", **summary}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
