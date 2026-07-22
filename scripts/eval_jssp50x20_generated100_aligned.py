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
from rl4co.models.zoo.mgl_jssp.sampling import sampling
try:
    # The remote shared checkout keeps tracked user edits untouched; Codex syncs
    # the complete-row trainer under this unique module name.
    from scripts.train_jssp_large_objectives_complete_rows import _build_model, _resolve
except ImportError:
    from scripts.train_jssp_large_objectives import _build_model, _resolve


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _dataset_order_sha256(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(bytes.fromhex(_sha256(path)))
    return digest.hexdigest()


def _seed(value: int, device: torch.device) -> None:
    random.seed(value)
    np.random.seed(value % (2**32))
    torch.manual_seed(value)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(value)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate locked JSSP50x20 checkpoints under the paper's generated-100 protocol."
    )
    parser.add_argument(
        "--label",
        choices=("rl", "po", "sll", "bopo", "h8_usw", "h13_asw", "weighting_only"),
        required=True,
    )
    parser.add_argument(
        "--method", choices=("rl", "po", "sll", "bopo", "usw", "asw"), required=True
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--expected-step", type=int, required=True)
    parser.add_argument("--instance-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--sampling-seed", type=int, default=12345678)
    parser.add_argument("--B", type=int, default=128)
    parser.add_argument("--greedy", type=int, choices=(0, 1), default=0)
    parser.add_argument("--usw-pair", type=Path, required=True)
    parser.add_argument("--asw-pair", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    locked = (args.count, args.sampling_seed, args.B, args.greedy)
    if locked != (100, 12345678, 128, 0):
        raise ValueError(f"Aligned generated-100 protocol mismatch: {locked}")

    checkpoint = _resolve(args.checkpoint)
    instance_dir = _resolve(args.instance_dir)
    output_dir = _resolve(args.output_dir)
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite aligned evaluation output: {output_dir}")

    observed_sha = _sha256(checkpoint)
    if observed_sha != args.checkpoint_sha256:
        raise ValueError(
            f"Checkpoint SHA256 mismatch: expected {args.checkpoint_sha256}, observed {observed_sha}"
        )
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    saved_method = str(payload.get("method", payload.get("large_scale_config", {}).get("method")))
    saved_step = int(payload.get("optimizer_step", payload.get("global_step", -1)))
    if saved_method != args.method or (
        args.expected_step >= 0 and saved_step != args.expected_step
    ):
        raise ValueError(f"Checkpoint metadata mismatch: method={saved_method}, step={saved_step}")

    scale_config = payload.get("large_scale_config", {})
    shape_value = str(scale_config.get("shape", "50x20"))
    shape_jobs, shape_machines = shape_value.lower().split("x", maxsplit=1)
    num_jobs = int(scale_config.get("num_jobs", shape_jobs))
    num_machines = int(scale_config.get("num_machines", shape_machines))
    shape_name = f"{num_jobs}x{num_machines}"
    files = sorted(instance_dir.glob(f"{shape_name}_*.jsp"))
    if len(files) != 100:
        raise ValueError(
            f"Expected exactly 100 aligned {shape_name} instances, found {len(files)}"
        )
    dataset_order_sha = _dataset_order_sha256(files)

    alpha = float(payload.get("large_scale_config", {}).get("alpha", 0.0))
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    build_args = argparse.Namespace(
        num_jobs=num_jobs,
        num_machines=num_machines,
        method=args.method,
        rollouts=128,
        select_k=16,
        po_alpha=0.25,
        alpha=alpha,
        eval_rollouts=128,
        greedy=0,
        usw_pair=args.usw_pair,
        asw_pair=args.asw_pair,
    )
    model, _ = _build_model(build_args, checkpoint)
    model = model.to(device)
    model.eval()
    output_dir.mkdir(parents=True, exist_ok=False)

    rows: list[dict[str, object]] = []
    evaluation_started = time.perf_counter()
    for index, path in enumerate(files):
        instance = load_instance(path.as_posix(), device="cpu")
        shape = (int(instance["j"]), int(instance["m"]))
        if shape != (num_jobs, num_machines):
            raise ValueError(f"Generated shape mismatch at {path}: {shape}")
        sampling_seed = 12345678 + index
        _seed(sampling_seed, device)
        started = time.perf_counter()
        with torch.inference_mode():
            makespans, _, _ = sampling(
                [instance],
                model.encoder,
                model.decoder,
                bs=128,
                use_greedy=False,
                device=str(device),
            )
        elapsed = time.perf_counter() - started
        cost = float(makespans.view(-1).min().detach().cpu())
        if not math.isfinite(cost):
            raise FloatingPointError(f"Non-finite cost at index {index}")
        row = {
            "label": args.label,
            "method": args.method,
            "instance": path.name,
            "instance_index": index,
            "instance_sha256": _sha256(path),
            "sampling_seed": sampling_seed,
            "cost": cost,
            "time_sec": elapsed,
        }
        rows.append(row)
        print(json.dumps({"event": "instance", **row}), flush=True)

    total_elapsed = time.perf_counter() - evaluation_started
    with (output_dir / "per_instance.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "protocol": f"jssp{shape_name}_paper_aligned_generated100_b128_g0_v1",
        "label": args.label,
        "method": args.method,
        "checkpoint": checkpoint.as_posix(),
        "checkpoint_sha256": observed_sha,
        "optimizer_step": saved_step,
        "instance_dir": instance_dir.as_posix(),
        "instance_count": 100,
        "dataset_order_sha256": dataset_order_sha,
        "sampling_seed_base": 12345678,
        "B": 128,
        "greedy_injected": False,
        "augmentation_factor": 1,
        "physical_instance_batch": 1,
        "mean_cost": float(np.mean([float(row["cost"]) for row in rows])),
        "total_elapsed_sec": total_elapsed,
        "mean_elapsed_sec_per_instance": total_elapsed / len(rows),
        "test_only_no_selection": True,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"event": "summary", **summary}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
