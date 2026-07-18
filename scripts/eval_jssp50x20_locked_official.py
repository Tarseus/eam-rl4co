from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import subprocess
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
from scripts.train_jssp_large_objectives import _build_model, _resolve


DATASET_COMMIT = "2739fbfe39a478755173c892b80c4b06f9f59b05"
EXPECTED_FILES = {
    "TA": [f"ta{index:02d}.jsp" for index in range(61, 71)],
    "DMU": [
        *[f"dmu{index:02d}.jsp" for index in range(36, 41)],
        *[f"dmu{index:02d}.jsp" for index in range(76, 81)],
    ],
}


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


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate one locked JSSP50x20 checkpoint on official TA+DMU test instances."
    )
    parser.add_argument("--label", choices=("bopo", "h8_usw", "h13_asw"), required=True)
    parser.add_argument("--method", choices=("bopo", "usw", "asw"), required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--expected-step", type=int, required=True)
    parser.add_argument("--benchmark-root", type=Path, required=True)
    parser.add_argument("--dataset-repo", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--B", type=int, default=128)
    parser.add_argument("--greedy", type=int, choices=(0, 1), default=1)
    parser.add_argument("--sampling-seed", type=int, default=12345678)
    parser.add_argument("--usw-pair", type=Path, required=True)
    parser.add_argument("--asw-pair", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if args.B != 128 or args.greedy != 1 or args.sampling_seed != 12345678:
        raise ValueError("Locked protocol requires B=128, greedy=1, seed=12345678")

    checkpoint = _resolve(args.checkpoint)
    benchmark_root = args.benchmark_root.resolve()
    dataset_repo = args.dataset_repo.resolve()
    output_dir = _resolve(args.output_dir)
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite locked test output: {output_dir}")
    if _sha256(checkpoint) != args.checkpoint_sha256:
        raise ValueError("Checkpoint SHA256 mismatch")
    commit = subprocess.check_output(
        ["git", "-C", str(dataset_repo), "rev-parse", "HEAD"], text=True
    ).strip()
    if commit != DATASET_COMMIT:
        raise ValueError(f"Dataset commit mismatch: {commit}")

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
    file_hashes: dict[str, str] = {}
    combined_index = 0
    for set_name, names in EXPECTED_FILES.items():
        set_dir = benchmark_root / set_name
        for filename in names:
            path = set_dir / filename
            if not path.is_file():
                raise FileNotFoundError(path)
            instance = load_instance(path.as_posix(), device="cpu")
            shape = (int(instance["j"]), int(instance["m"]))
            if shape != (50, 20):
                raise ValueError(f"Expected 50x20, got {shape} for {path}")
            seed = int(args.sampling_seed) + combined_index
            _seed(seed, device)
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
            predicted = float(makespans.view(-1).min().detach().cpu())
            reference = float(instance["makespan"])
            gap = (predicted / reference - 1.0) * 100.0
            if not all(math.isfinite(value) for value in (predicted, reference, gap)):
                raise FloatingPointError(f"Non-finite result for {set_name}/{filename}")
            row = {
                "label": args.label,
                "method": args.method,
                "set": set_name,
                "instance": filename,
                "combined_index": combined_index,
                "reference_makespan": reference,
                "predicted_makespan": predicted,
                "gap_percent": gap,
                "sampling_seed": seed,
                "time_sec": elapsed,
            }
            rows.append(row)
            file_hashes[f"{set_name}/{filename}"] = _sha256(path)
            print(json.dumps({"event": "instance", **row}), flush=True)
            combined_index += 1

    if len(rows) != 20:
        raise RuntimeError(f"Expected 20 official instances, got {len(rows)}")
    with (output_dir / "per_instance.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "protocol": "jssp50x20_locked_official_ta_dmu_b128_v1",
        "label": args.label,
        "method": args.method,
        "checkpoint": checkpoint.as_posix(),
        "checkpoint_sha256": _sha256(checkpoint),
        "optimizer_step": saved_step,
        "dataset_repo": dataset_repo.as_posix(),
        "dataset_commit": commit,
        "instance_count": len(rows),
        "B_prime": 128,
        "greedy_injected": True,
        "sampling_seed": int(args.sampling_seed),
        "mean_gap_percent": float(np.mean([float(row["gap_percent"]) for row in rows])),
        "set_mean_gap_percent": {
            set_name: float(
                np.mean(
                    [
                        float(row["gap_percent"])
                        for row in rows
                        if row["set"] == set_name
                    ]
                )
            )
            for set_name in EXPECTED_FILES
        },
        "file_sha256": file_hashes,
        "test_only_no_selection": True,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=True), encoding="utf-8"
    )
    print(json.dumps({"event": "summary", **summary}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

