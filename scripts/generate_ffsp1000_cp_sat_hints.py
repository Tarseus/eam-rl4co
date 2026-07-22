from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from tensordict import TensorDict


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rl4co.utils.ops import batchify, unbatchify

try:
    from scripts.train_ffsp1000_objectives_complete_rows import (
        _autocast,
        _build_model,
        _resolve,
        _seed,
    )
except ImportError:
    from scripts.train_ffsp1000_objectives import _autocast, _build_model, _resolve, _seed


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export FFSP1000 neural schedules as CP-SAT hints.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--method", choices=("rl", "po", "sll", "bopo", "usw", "asw"), required=True)
    parser.add_argument("--test-file", type=Path, required=True)
    parser.add_argument("--test-file-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--precision", default="bf16")
    parser.add_argument("--usw-pair", type=Path, required=True)
    parser.add_argument("--asw-pair", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    checkpoint = _resolve(args.checkpoint)
    test_file = _resolve(args.test_file)
    output_dir = _resolve(args.output_dir)
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite FFSP hint output: {output_dir}")
    if _sha256(checkpoint) != args.checkpoint_sha256.lower():
        raise ValueError("Checkpoint SHA256 mismatch")
    if _sha256(test_file) != args.test_file_sha256.lower():
        raise ValueError("Test-file SHA256 mismatch")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    step = int(payload.get("optimizer_step", payload.get("global_step", -1)))
    if step != 100:
        raise ValueError(f"Expected final step-100 checkpoint, observed {step}")
    with np.load(test_file) as data:
        run_time = np.asarray(data["run_time"], dtype=np.int64)
    if run_time.shape != (100, 1000, 12):
        raise ValueError(f"Unexpected FFSP1000 test shape: {run_time.shape}")
    build_args = argparse.Namespace(
        target_jobs=1000,
        num_stages=3,
        num_machines=4,
        num_starts=24,
        bopo_select_k=6,
        alpha=float(payload.get("large_scale_config", {}).get("alpha", 1.0)),
        method=args.method,
        seed=12345678,
        usw_pair=args.usw_pair,
        asw_pair=args.asw_pair,
    )
    model, env, _ = _build_model(build_args, checkpoint)
    device = torch.device(args.device)
    model = model.to(device).eval()
    output_dir.mkdir(parents=True)
    hint_dir = output_dir / "hints"
    hint_dir.mkdir()
    rows = []
    total_started = time.perf_counter()
    for offset in range(0, 100, args.batch_size):
        stop = min(offset + args.batch_size, 100)
        batch_np = run_time[offset:stop]
        batch = TensorDict(
            {"run_time": torch.from_numpy(batch_np.copy())}, batch_size=[stop - offset]
        ).to(device)
        _seed(12345678 + offset * 1_000_003, device)
        started = time.perf_counter()
        initial = env.reset(batch.clone()).to(device)
        with torch.inference_mode(), _autocast(device, args.precision):
            out = model.policy(
                initial, env, phase="test", num_starts=24, return_actions=True
            )

        replay = env.reset(batch.clone()).to(device)
        replay = batchify(replay, 24)
        replay = env.pre_step(replay)
        for step_index in range(out["actions"].shape[1]):
            replay.set("action", out["actions"][:, step_index])
            replay = env.step(replay)["next"]
        if not replay["done"].all():
            raise RuntimeError("Forced FFSP schedule replay did not finish")
        rewards = unbatchify(out["reward"], (0, 24))
        schedules = unbatchify(replay["schedule"], (0, 24))
        best_starts = rewards.argmax(dim=1)
        elapsed = time.perf_counter() - started
        for local_index in range(stop - offset):
            index = offset + local_index
            start_index = int(best_starts[local_index])
            objective = int(round(float(-rewards[local_index, start_index].cpu())))
            schedule_tensor = schedules[local_index, start_index].detach().cpu().long()
            assignments = []
            for stage in range(3):
                for job in range(1000):
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
                    duration = int(batch_np[local_index, job, stage * 4 + machine])
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
                    f"FFSP hint reconstruction mismatch at {index}: reward={objective}, schedule={reconstructed}"
                )
            hint_payload = {
                "instance_index": index,
                "test_file_sha256": args.test_file_sha256.lower(),
                "checkpoint_sha256": args.checkpoint_sha256.lower(),
                "method": args.method,
                "optimizer_step": step,
                "objective": objective,
                "schedule": assignments,
            }
            (hint_dir / f"instance_{index:05d}.json").write_text(
                json.dumps(hint_payload), encoding="utf-8"
            )
            row = {
                "instance_index": index,
                "objective": objective,
                "batch_time_sec_per_instance": elapsed / (stop - offset),
            }
            rows.append(row)
            print(json.dumps({"event": "hint", **row}), flush=True)
    with (output_dir / "per_instance.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "protocol": "ffsp1000_generated100_starts24_cp_sat_hints_v1",
        "count": 100,
        "mean_objective": float(np.mean([row["objective"] for row in rows])),
        "total_elapsed_sec": time.perf_counter() - total_started,
        "checkpoint": checkpoint.as_posix(),
        "checkpoint_sha256": args.checkpoint_sha256.lower(),
        "optimizer_step": step,
        "test_file_sha256": args.test_file_sha256.lower(),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"event": "summary", **summary}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
