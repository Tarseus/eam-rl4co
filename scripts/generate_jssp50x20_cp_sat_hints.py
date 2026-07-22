from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rl4co.models.zoo.mgl_jssp.data import load_instance
from rl4co.models.zoo.mgl_jssp.sampling import solve_jsp

try:
    from scripts.train_jssp_large_objectives_complete_rows import _build_model, _resolve
except ImportError:
    from scripts.train_jssp_large_objectives import _build_model, _resolve


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _seed(value: int, device: torch.device) -> None:
    random.seed(value)
    np.random.seed(value % (2**32))
    torch.manual_seed(value)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(value)


def _schedule(instance: dict, trajectory: list[int]) -> tuple[int, list[dict[str, int]]]:
    num_jobs = int(instance["j"])
    num_machines = int(instance["m"])
    machines = instance["machines"].view(num_jobs, num_machines).cpu().long()
    durations = instance["costs"].view(num_jobs, num_machines).cpu().long()
    counts = [0] * num_jobs
    for job in trajectory:
        counts[job] += 1
    remaining = [job for job, count in enumerate(counts) if count == num_machines - 1]
    if len(remaining) != 1 or any(count not in {num_machines - 1, num_machines} for count in counts):
        raise ValueError(f"Invalid incomplete MGL trajectory counts: {counts}")
    dispatch = trajectory + remaining
    job_ready = [0] * num_jobs
    machine_ready = [0] * num_machines
    operation_index = [0] * num_jobs
    schedule: list[dict[str, int]] = []
    for job in dispatch:
        operation = operation_index[job]
        machine = int(machines[job, operation])
        duration = int(durations[job, operation])
        start = max(job_ready[job], machine_ready[machine])
        end = start + duration
        schedule.append(
            {
                "job": job,
                "operation": operation,
                "machine": machine,
                "duration": duration,
                "start": start,
                "end": end,
            }
        )
        job_ready[job] = end
        machine_ready[machine] = end
        operation_index[job] += 1
    if operation_index != [num_machines] * num_jobs:
        raise RuntimeError("Dispatch did not schedule every operation exactly once")
    return max(job_ready), schedule


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export H8 JSSP50x20 schedules as CP-SAT hints.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--instance-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--usw-pair", type=Path, required=True)
    parser.add_argument("--asw-pair", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    checkpoint = _resolve(args.checkpoint)
    instance_dir = _resolve(args.instance_dir)
    output_dir = _resolve(args.output_dir)
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite hint output: {output_dir}")
    if _sha256(checkpoint) != args.checkpoint_sha256:
        raise ValueError("Checkpoint SHA256 mismatch")
    files = sorted(instance_dir.glob("50x20_*.jsp"))
    if len(files) != 100:
        raise ValueError(f"Expected 100 aligned instances, found {len(files)}")
    device = torch.device(args.device)
    build_args = argparse.Namespace(
        num_jobs=50,
        num_machines=20,
        method="usw",
        rollouts=128,
        select_k=16,
        po_alpha=0.25,
        alpha=1.0,
        eval_rollouts=128,
        greedy=0,
        usw_pair=args.usw_pair,
        asw_pair=args.asw_pair,
    )
    model, payload = _build_model(build_args, checkpoint)
    step = int(payload.get("optimizer_step", payload.get("global_step", -1)))
    if step != 500:
        raise ValueError(f"Expected H8 step 500, observed {step}")
    model = model.to(device).eval()
    output_dir.mkdir(parents=True)
    hint_dir = output_dir / "hints"
    hint_dir.mkdir()
    rows = []
    total_started = time.perf_counter()
    for index, path in enumerate(files):
        instance = load_instance(path.as_posix(), device="cpu")
        seed = 12345678 + index
        _seed(seed, device)
        started = time.perf_counter()
        with torch.inference_mode():
            trajectories, _, makespans, _ = solve_jsp(
                [instance],
                batch_size_per_instance=128,
                device=str(device),
                encoder=model.encoder,
                decoder=model.decoder,
                use_greedy=False,
            )
        best_index = int(makespans.argmin())
        model_cost = float(makespans[best_index].cpu())
        trajectory = [int(value) for value in trajectories[best_index].cpu().tolist()]
        schedule_cost, schedule = _schedule(instance, trajectory)
        if not math.isclose(model_cost, schedule_cost, abs_tol=0.02):
            raise RuntimeError(
                f"Schedule reconstruction mismatch for {path.name}: model={model_cost}, reconstructed={schedule_cost}"
            )
        elapsed = time.perf_counter() - started
        payload_out = {
            "instance": path.name,
            "instance_sha256": _sha256(path),
            "checkpoint_sha256": args.checkpoint_sha256,
            "optimizer_step": step,
            "sampling_seed": seed,
            "B": 128,
            "greedy": False,
            "objective": schedule_cost,
            "schedule": schedule,
        }
        (hint_dir / f"{path.stem}.json").write_text(
            json.dumps(payload_out), encoding="utf-8"
        )
        row = {
            "instance": path.name,
            "instance_sha256": payload_out["instance_sha256"],
            "objective": schedule_cost,
            "time_sec": elapsed,
        }
        rows.append(row)
        print(json.dumps({"event": "hint", **row}), flush=True)
    with (output_dir / "per_instance.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "protocol": "jssp50x20_generated100_h8_b128_cp_sat_hints_v1",
        "count": len(rows),
        "mean_objective": sum(row["objective"] for row in rows) / len(rows),
        "total_elapsed_sec": time.perf_counter() - total_started,
        "checkpoint": checkpoint.as_posix(),
        "checkpoint_sha256": args.checkpoint_sha256,
        "optimizer_step": step,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"event": "summary", **summary}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
