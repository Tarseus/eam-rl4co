from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import sys
import time
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rl4co.utils.ops import unbatchify
from scripts.eval_downloaded_routing_checkpoints import DownloadEntry, load_manifest
from scripts.train_cvrp1000_objectives import _build_model, _load_fixed


@dataclass(frozen=True)
class InstanceResult:
    method: str
    instance_index: int
    target_size: int
    capacity: int
    num_starts: int
    num_augment: int
    instance_batch_size: int
    cost: float
    elapsed_sec: float
    peak_memory_allocated_gib: float | None
    peak_memory_reserved_gib: float | None


def _parse_csv(raw: str) -> list[str]:
    return [token.strip() for token in str(raw).split(",") if token.strip()]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _autocast(device: torch.device, precision: str):
    value = str(precision).strip().lower()
    if value in {"32", "32-true", "fp32", "float32"}:
        return nullcontext()
    if device.type != "cuda":
        raise ValueError(f"precision={precision} requires CUDA")
    if value in {"16", "16-mixed", "fp16", "float16"}:
        return torch.autocast("cuda", dtype=torch.float16, cache_enabled=False)
    if value in {"bf16", "bf16-mixed", "bfloat16"}:
        return torch.autocast("cuda", dtype=torch.bfloat16, cache_enabled=False)
    raise ValueError(f"Unsupported precision: {precision}")


def _select_entries(manifest: Path, methods: list[str]) -> list[DownloadEntry]:
    requested = set(methods)
    entries = [
        entry
        for entry in load_manifest(manifest, REPO_ROOT)
        if entry.problem_key == "cvrp100" and entry.method in requested
    ]
    by_method = {entry.method: entry for entry in entries}
    missing = [method for method in methods if method not in by_method]
    if missing:
        raise ValueError(f"Missing cvrp100 manifest entries for methods: {missing}")
    selected = [by_method[method] for method in methods]
    for entry in selected:
        if not entry.checkpoint_path.exists():
            raise FileNotFoundError(entry.checkpoint_path)
    return selected


def _reset_peak_memory(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)


def _peak_memory(device: torch.device) -> tuple[float | None, float | None]:
    if device.type != "cuda":
        return None, None
    torch.cuda.synchronize(device)
    gib = float(1024**3)
    return (
        float(torch.cuda.max_memory_allocated(device)) / gib,
        float(torch.cuda.max_memory_reserved(device)) / gib,
    )


def _write_csv(path: Path, rows: list[InstanceResult]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(rows[0])))
        writer.writeheader()
        writer.writerows(asdict(row) for row in rows)


def _aggregate(rows: list[InstanceResult]) -> list[dict[str, Any]]:
    grouped: dict[str, list[InstanceResult]] = {}
    for row in rows:
        grouped.setdefault(row.method, []).append(row)
    summaries = []
    for method, group in grouped.items():
        costs = np.asarray([row.cost for row in group], dtype=np.float64)
        elapsed = np.asarray([row.elapsed_sec for row in group], dtype=np.float64)
        summaries.append(
            {
                "method": method,
                "num_instances": len(group),
                "mean_cost": float(costs.mean()),
                "std_cost": float(costs.std(ddof=1)) if len(group) > 1 else 0.0,
                "median_cost": float(np.median(costs)),
                "min_cost": float(costs.min()),
                "max_cost": float(costs.max()),
                "mean_elapsed_sec": float(elapsed.mean()),
                "max_peak_memory_allocated_gib": max(
                    row.peak_memory_allocated_gib or 0.0 for row in group
                ),
                "max_peak_memory_reserved_gib": max(
                    row.peak_memory_reserved_gib or 0.0 for row in group
                ),
            }
        )
    return summaries


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate saved CVRP100 checkpoints on a fixed larger CVRP dataset."
    )
    parser.add_argument("--manifest", type=Path, default=REPO_ROOT / "downloads" / "manifest.json")
    parser.add_argument("--methods", default="po,sll,bopo,loss_only,weighting")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--target-size", type=int, default=1000)
    parser.add_argument("--capacity", type=int, default=50)
    parser.add_argument("--num-instances", type=int, default=128)
    parser.add_argument("--num-starts", type=int, default=100)
    parser.add_argument("--num-augment", type=int, default=8)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--precision", default="32-true")
    parser.add_argument(
        "--allow-standard-pomo-policy",
        action="store_true",
        help="Allow a recovered standard POMO policy instead of requiring PO4COPsCVRPPolicy.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "logs" / "cvrp_scale_streaming" / datetime.now().strftime("%Y%m%d-%H%M%S"),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.num_instances < 1:
        raise ValueError("num-instances must be >= 1")
    if args.num_starts < 1 or args.num_starts > args.target_size:
        raise ValueError("num-starts must be in [1, target-size]")
    if args.num_augment not in {1, 8}:
        raise ValueError("num-augment must be 1 or 8")
    dataset_path = args.dataset.resolve()
    if not dataset_path.exists():
        raise FileNotFoundError(dataset_path)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    methods = _parse_csv(args.methods)
    entries = _select_entries(args.manifest.resolve(), methods)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[InstanceResult] = []
    checkpoints: dict[str, dict[str, str]] = {}

    for entry in entries:
        print(f"[load] method={entry.method} checkpoint={entry.checkpoint_path}", flush=True)
        model, env, _ = _build_model(
            checkpoint_path=entry.checkpoint_path,
            method="po",
            target_size=args.target_size,
            capacity=args.capacity,
            num_starts=args.num_starts,
            bopo_select_k=10,
            alpha=0.05,
            seed=args.seed,
            usw_pair_path=REPO_ROOT / "unused_usw_pair.json",
            asw_pair_path=REPO_ROOT / "unused_asw_pair.json",
            require_po4cops_policy=not args.allow_standard_pomo_policy,
        )
        model = model.to(device)
        model.eval()
        dataset = _load_fixed(dataset_path, env)
        count = min(args.num_instances, int(dataset.batch_size[0]))
        if count != args.num_instances:
            raise ValueError(
                f"Dataset has {dataset.batch_size[0]} rows, fewer than requested {args.num_instances}"
            )
        checkpoints[entry.method] = {
            "path": str(entry.checkpoint_path.resolve()),
            "sha256": _sha256(entry.checkpoint_path),
        }

        for index in range(count):
            _reset_peak_memory(device)
            started = time.perf_counter()
            raw = dataset[index : index + 1].to(device)
            td = env.reset(raw).to(device)
            if args.num_augment > 1:
                td = model.augment(td)
            with torch.inference_mode(), _autocast(device, args.precision):
                out = model.policy(
                    td,
                    env,
                    phase="test",
                    num_starts=args.num_starts,
                    return_actions=False,
                    return_entropy=False,
                    return_sum_log_likelihood=True,
                )
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            elapsed = time.perf_counter() - started
            peak_allocated, peak_reserved = _peak_memory(device)
            reward = unbatchify(out["reward"], (args.num_augment, args.num_starts))
            cost = float(-reward.max().detach().cpu())
            rows.append(
                InstanceResult(
                    method=entry.method,
                    instance_index=index,
                    target_size=args.target_size,
                    capacity=args.capacity,
                    num_starts=args.num_starts,
                    num_augment=args.num_augment,
                    instance_batch_size=1,
                    cost=cost,
                    elapsed_sec=elapsed,
                    peak_memory_allocated_gib=peak_allocated,
                    peak_memory_reserved_gib=peak_reserved,
                )
            )
            _write_csv(args.output_dir / "per_instance.csv", rows)
            print(
                f"[result] method={entry.method} instance={index} cost={cost:.6f} "
                f"elapsed={elapsed:.2f}s peak_allocated={peak_allocated}GiB",
                flush=True,
            )

        del dataset, model
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    summary = {
        "config": {
            "methods": methods,
            "dataset": str(dataset_path),
            "dataset_sha256": _sha256(dataset_path),
            "target_size": args.target_size,
            "capacity": args.capacity,
            "num_instances": args.num_instances,
            "num_starts": args.num_starts,
            "num_augment": args.num_augment,
            "instance_batch_size": 1,
            "precision": args.precision,
            "device": str(device),
            "seed": args.seed,
            "allow_standard_pomo_policy": args.allow_standard_pomo_policy,
        },
        "checkpoints": checkpoints,
        "aggregates": _aggregate(rows),
        "rows": [asdict(row) for row in rows],
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"[done] output_dir={args.output_dir.resolve()}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
