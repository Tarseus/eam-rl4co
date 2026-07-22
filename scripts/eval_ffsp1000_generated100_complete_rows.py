from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
from tensordict import TensorDict


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rl4co.utils.ops import unbatchify

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
    parser = argparse.ArgumentParser(
        description="Evaluate one locked FFSP1000 checkpoint on the fixed generated-100 set."
    )
    parser.add_argument("--label", choices=("rl", "po", "sll", "bopo", "usw", "asw"), required=True)
    parser.add_argument("--method", choices=("rl", "po", "sll", "bopo", "usw", "asw"), required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--expected-step", type=int, default=100)
    parser.add_argument("--test-file", type=Path, required=True)
    parser.add_argument("--test-file-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--precision", default="bf16")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--end-index", type=int, default=100)
    parser.add_argument("--usw-pair", type=Path, required=True)
    parser.add_argument("--asw-pair", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    checkpoint = _resolve(args.checkpoint)
    test_file = _resolve(args.test_file)
    output_dir = _resolve(args.output_dir)
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite aligned FFSP output: {output_dir}")
    if _sha256(checkpoint) != args.checkpoint_sha256:
        raise ValueError("Checkpoint SHA256 mismatch")
    if _sha256(test_file) != args.test_file_sha256:
        raise ValueError("Test-file SHA256 mismatch")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    saved_method = str(payload.get("method", payload.get("large_scale_config", {}).get("method")))
    saved_step = int(payload.get("optimizer_step", payload.get("global_step", -1)))
    if saved_method != args.method or (
        args.expected_step >= 0 and saved_step != args.expected_step
    ):
        raise ValueError(f"Checkpoint metadata mismatch: method={saved_method}, step={saved_step}")
    with np.load(test_file) as data:
        run_time = np.asarray(data["run_time"], dtype=np.int64)
    if run_time.shape != (100, 1000, 12):
        raise ValueError(f"Expected fixed FFSP1000 generated-100 shape, observed {run_time.shape}")
    if not 0 <= args.start_index < args.end_index <= 100:
        raise ValueError(
            f"Expected 0 <= start-index < end-index <= 100, observed "
            f"[{args.start_index}, {args.end_index})"
        )

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
    rows: list[dict[str, object]] = []
    total_started = time.perf_counter()
    for offset in range(args.start_index, args.end_index, args.batch_size):
        stop = min(offset + args.batch_size, args.end_index)
        batch = TensorDict(
            {"run_time": torch.from_numpy(run_time[offset:stop].copy())},
            batch_size=[stop - offset],
        ).to(device)
        _seed(12345678 + offset * 1_000_003, device)
        started = time.perf_counter()
        td = env.reset(batch.clone()).to(device)
        with torch.inference_mode(), _autocast(device, args.precision):
            out = model.policy(td, env, phase="test", num_starts=24, return_actions=False)
        reward = unbatchify(out["reward"], (0, 24))
        costs = (-reward.max(dim=-1).values).detach().float().cpu().tolist()
        elapsed = time.perf_counter() - started
        for local_index, cost in enumerate(costs):
            if not math.isfinite(float(cost)):
                raise FloatingPointError("Non-finite FFSP evaluation cost")
            index = offset + local_index
            row = {
                "label": args.label,
                "method": args.method,
                "instance_index": index,
                "cost": float(cost),
                "batch_time_sec_per_instance": elapsed / len(costs),
            }
            rows.append(row)
            print(json.dumps({"event": "instance", **row}), flush=True)
    total_elapsed = time.perf_counter() - total_started
    with (output_dir / "per_instance.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "protocol": "ffsp1000_fixed_generated100_seed12345678_starts24_aug1_v1",
        "label": args.label,
        "method": args.method,
        "checkpoint": checkpoint.as_posix(),
        "checkpoint_sha256": args.checkpoint_sha256,
        "optimizer_step": saved_step,
        "test_file": test_file.as_posix(),
        "test_file_sha256": args.test_file_sha256,
        "instance_count": len(rows),
        "start_index": args.start_index,
        "end_index": args.end_index,
        "num_starts": 24,
        "augmentation_factor": 1,
        "mean_cost": float(np.mean([row["cost"] for row in rows])),
        "total_elapsed_sec": total_elapsed,
        "mean_elapsed_sec_per_instance": total_elapsed / len(rows),
        "test_only_no_selection": True,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"event": "summary", **summary}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
