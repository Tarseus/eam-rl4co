from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from tensordict import TensorDict


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rl4co.data.transforms import StateAugmentation
from rl4co.envs import get_env
from scripts.standard_routing_policy import METHODS, PROBLEMS, build_policy


@dataclass(frozen=True)
class InstanceResult:
    method: str
    problem: str
    instance_index: int
    target_size: int
    capacity: int | None
    num_starts: int
    num_augment: int
    cost: float
    elapsed_sec: float
    peak_memory_allocated_gib: float | None
    peak_memory_reserved_gib: float | None


def _resolve(path: str | Path) -> Path:
    value = Path(path).expanduser()
    return (value if value.is_absolute() else REPO_ROOT / value).resolve()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _autocast(device: torch.device, precision: str):
    value = precision.lower()
    if value in {"32", "32-true", "fp32", "float32"}:
        return nullcontext()
    if device.type != "cuda":
        raise ValueError(f"precision={precision} requires CUDA")
    if value in {"bf16", "bf16-mixed", "bfloat16"}:
        return torch.autocast("cuda", dtype=torch.bfloat16, cache_enabled=False)
    if value in {"16", "16-mixed", "fp16", "float16"}:
        return torch.autocast("cuda", dtype=torch.float16, cache_enabled=False)
    raise ValueError(f"Unsupported precision={precision}")


def _write_csv(path: Path, rows: list[InstanceResult]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(rows[0])))
        writer.writeheader()
        writer.writerows(asdict(row) for row in rows)


def _load_dataset(path: Path, *, problem: str, env) -> TensorDict:
    if problem == "cvrp":
        return env.load_data(path)
    with np.load(path) as payload:
        if "locs" not in payload.files:
            raise ValueError(f"TSP dataset {path} does not contain locs")
        locs = torch.from_numpy(np.asarray(payload["locs"], dtype=np.float32))
    return TensorDict({"locs": locs}, batch_size=[locs.shape[0]])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate scale-1000 AM/SymNCO checkpoints on a locked dataset."
    )
    parser.add_argument("--method", required=True, choices=METHODS)
    parser.add_argument("--problem", required=True, choices=PROBLEMS)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--target-size", type=int, default=1000)
    parser.add_argument("--capacity", type=int, default=50)
    parser.add_argument("--num-instances", type=int, default=None)
    parser.add_argument("--num-starts", type=int, default=None)
    parser.add_argument("--num-augment", type=int, default=8)
    parser.add_argument("--precision", default="32-true")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    checkpoint = _resolve(args.checkpoint)
    dataset_path = _resolve(args.dataset)
    output_dir = _resolve(args.output_dir)
    if not checkpoint.exists() or not dataset_path.exists():
        raise FileNotFoundError(checkpoint if not checkpoint.exists() else dataset_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    num_instances = args.num_instances or (100 if args.problem == "tsp" else 128)
    num_starts = args.num_starts or (args.target_size if args.problem == "tsp" else 100)
    if num_starts > args.target_size:
        raise ValueError("num-starts cannot exceed target-size")
    if args.num_augment not in {1, 8}:
        raise ValueError("num-augment must be 1 or 8")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if device.type == "cuda":
        torch.backends.cuda.enable_cudnn_sdp(False)
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        torch.backends.cuda.enable_math_sdp(True)

    env = get_env(
        args.problem,
        generator_params={
            "num_loc": args.target_size,
            **({"capacity": float(args.capacity)} if args.problem == "cvrp" else {}),
        },
        seed=1234,
    )
    policy, init_metadata = build_policy(
        method=args.method,
        problem=args.problem,
        init_checkpoint=checkpoint,
    )
    policy = policy.to(device).eval()
    dataset = _load_dataset(dataset_path, problem=args.problem, env=env)
    if int(dataset.batch_size[0]) < num_instances:
        raise ValueError(
            f"Dataset has {dataset.batch_size[0]} rows, fewer than requested {num_instances}"
        )
    augment = (
        StateAugmentation(num_augment=8, augment_fn="dihedral8")
        if args.num_augment == 8
        else None
    )
    rows: list[InstanceResult] = []
    for index in range(num_instances):
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
        started = time.perf_counter()
        raw = dataset[index : index + 1].to(device)
        td = env.reset(raw).to(device)
        if augment is not None:
            td = augment(td)
        with torch.inference_mode(), _autocast(device, args.precision):
            out = policy(
                td,
                env,
                phase="test",
                num_starts=num_starts,
                return_actions=False,
            )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            gib = float(1024**3)
            peak_allocated = float(torch.cuda.max_memory_allocated(device)) / gib
            peak_reserved = float(torch.cuda.max_memory_reserved(device)) / gib
        else:
            peak_allocated = peak_reserved = None
        cost = float(-out["reward"].max().detach().cpu())
        row = InstanceResult(
            method=args.method,
            problem=args.problem,
            instance_index=index,
            target_size=args.target_size,
            capacity=args.capacity if args.problem == "cvrp" else None,
            num_starts=num_starts,
            num_augment=args.num_augment,
            cost=cost,
            elapsed_sec=time.perf_counter() - started,
            peak_memory_allocated_gib=peak_allocated,
            peak_memory_reserved_gib=peak_reserved,
        )
        rows.append(row)
        _write_csv(output_dir / "per_instance.csv", rows)
        print(
            f"[result] {args.method} {args.problem} row={index} cost={cost:.6f} "
            f"elapsed={row.elapsed_sec:.3f}s",
            flush=True,
        )

    costs = np.asarray([row.cost for row in rows], dtype=np.float64)
    elapsed = np.asarray([row.elapsed_sec for row in rows], dtype=np.float64)
    summary = {
        "protocol": "scale1000_standard_baseline_locked_eval_v1",
        "method": args.method,
        "problem": args.problem,
        "target_size": args.target_size,
        "capacity": args.capacity if args.problem == "cvrp" else None,
        "num_instances": num_instances,
        "num_starts": num_starts,
        "num_augment": args.num_augment,
        "precision": args.precision,
        "dataset": str(dataset_path),
        "dataset_sha256": _sha256(dataset_path),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": _sha256(checkpoint),
        "checkpoint_metadata": init_metadata,
        "mean_cost": float(costs.mean()),
        "std_cost": float(costs.std(ddof=1)) if len(costs) > 1 else 0.0,
        "min_cost": float(costs.min()),
        "max_cost": float(costs.max()),
        "mean_elapsed_sec": float(elapsed.mean()),
        "rows": [asdict(row) for row in rows],
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("[done] " + json.dumps({k: summary[k] for k in ("mean_cost", "std_cost")}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
