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

from rl4co.models.zoo.mgl_jssp.sampling import sampling
from scripts.train_jssp_large_objectives import _build_model, _dynamic_instance, _resolve


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _instance_sha(instance: dict[str, object]) -> str:
    digest = hashlib.sha256()
    for key in ("costs", "machines"):
        tensor = instance[key]
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"Expected tensor for {key}")
        array = tensor.detach().cpu().contiguous().numpy()
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(str(array.shape).encode("ascii"))
        digest.update(array.tobytes())
    return digest.hexdigest()


def _seed(value: int, device: torch.device) -> None:
    random.seed(value)
    np.random.seed(value % (2**32))
    torch.manual_seed(value)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(value)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", choices=("bopo", "h8_usw", "h13_asw"), required=True)
    parser.add_argument("--method", choices=("bopo", "usw", "asw"), required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--expected-step", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--count", type=int, default=256)
    parser.add_argument("--generator-seed", type=int, default=314159265)
    parser.add_argument("--sampling-seed", type=int, default=271828182)
    parser.add_argument("--B", type=int, default=128)
    parser.add_argument("--greedy", type=int, choices=(0, 1), default=1)
    parser.add_argument("--usw-pair", type=Path, required=True)
    parser.add_argument("--asw-pair", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    locked = (args.count, args.generator_seed, args.sampling_seed, args.B, args.greedy)
    if locked != (256, 314159265, 271828182, 128, 1):
        raise ValueError(f"Locked generated-test protocol mismatch: {locked}")
    checkpoint = _resolve(args.checkpoint)
    output_dir = _resolve(args.output_dir)
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite locked test output: {output_dir}")
    observed_sha = _sha256(checkpoint)
    if observed_sha != args.checkpoint_sha256:
        raise ValueError("Checkpoint SHA256 mismatch")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    saved_method = str(payload.get("method", payload.get("large_scale_config", {}).get("method")))
    saved_step = int(payload.get("optimizer_step", payload.get("global_step", -1)))
    if saved_method != args.method or saved_step != args.expected_step:
        raise ValueError(
            f"Checkpoint metadata mismatch: method={saved_method}, step={saved_step}"
        )
    alpha = float(payload.get("large_scale_config", {}).get("alpha", 0.0))

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    build_args = argparse.Namespace(
        num_jobs=50,
        num_machines=20,
        method=args.method,
        rollouts=128,
        select_k=16,
        po_alpha=0.25,
        alpha=alpha,
        eval_rollouts=128,
        greedy=1,
        usw_pair=args.usw_pair,
        asw_pair=args.asw_pair,
    )
    model, _ = _build_model(build_args, checkpoint)
    model = model.to(device)
    model.eval()
    output_dir.mkdir(parents=True, exist_ok=False)

    rows: list[dict[str, object]] = []
    for index in range(256):
        instance = _dynamic_instance(
            jobs=50,
            machines=20,
            seed=314159265,
            instance_index=index,
        )
        if (int(instance["j"]), int(instance["m"])) != (50, 20):
            raise ValueError(f"Generated shape mismatch at index {index}")
        sampling_seed = 271828182 + index
        _seed(sampling_seed, device)
        started = time.perf_counter()
        with torch.inference_mode():
            makespans, _, _ = sampling(
                [instance],
                model.encoder,
                model.decoder,
                bs=128,
                use_greedy=True,
                device=str(device),
            )
        elapsed = time.perf_counter() - started
        cost = float(makespans.view(-1).min().detach().cpu())
        if not math.isfinite(cost):
            raise FloatingPointError(f"Non-finite cost at index {index}")
        row = {
            "label": args.label,
            "method": args.method,
            "instance": f"generated_{index:05d}",
            "instance_index": index,
            "instance_sha256": _instance_sha(instance),
            "generator_seed": 314159265,
            "sampling_seed": sampling_seed,
            "cost": cost,
            "time_sec": elapsed,
        }
        rows.append(row)
        print(json.dumps({"event": "instance", **row}), flush=True)

    with (output_dir / "per_instance.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "protocol": "jssp50x20_locked_generated256_b128_v1",
        "label": args.label,
        "method": args.method,
        "checkpoint": checkpoint.as_posix(),
        "checkpoint_sha256": observed_sha,
        "optimizer_step": saved_step,
        "instance_count": 256,
        "generator_seed": 314159265,
        "sampling_seed": 271828182,
        "B_prime": 128,
        "greedy_injected": True,
        "physical_instance_batch": 1,
        "mean_cost": float(np.mean([float(row["cost"]) for row in rows])),
        "test_only_no_selection": True,
        "official_h14_replaced": False,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps({"event": "summary", **summary}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

