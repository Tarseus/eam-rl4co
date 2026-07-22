from __future__ import annotations

import argparse
import csv
import hashlib
import json
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
from scripts.train_jssp_large_objectives import DEFAULT_PAIR_PATHS, _build_model


PAPER_GAPS = {128: 8.0, 512: 7.4}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a lightweight large-JSSP checkpoint on official TA50x20."
    )
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--method", choices=("po", "bopo", "usw", "asw"), required=True)
    parser.add_argument("--benchmark-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--B", type=int, choices=tuple(PAPER_GAPS), default=128)
    parser.add_argument("--greedy", type=int, choices=(0, 1), default=1)
    parser.add_argument("--sampling-seed", type=int, default=12345678)
    parser.add_argument("--alert-margin-pp", type=float, default=2.0)
    parser.add_argument("--usw-pair", type=Path, default=DEFAULT_PAIR_PATHS["usw"])
    parser.add_argument("--asw-pair", type=Path, default=DEFAULT_PAIR_PATHS["asw"])
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_lightweight_model(args: argparse.Namespace, checkpoint: Path, device: torch.device):
    build_args = argparse.Namespace(
        num_jobs=50,
        num_machines=20,
        method=args.method,
        rollouts=int(args.B),
        select_k=16,
        po_alpha=0.25,
        eval_rollouts=int(args.B),
        greedy=int(args.greedy),
        usw_pair=args.usw_pair,
        asw_pair=args.asw_pair,
    )
    model, payload = _build_model(build_args, checkpoint)
    saved_method = payload.get("method")
    if saved_method is not None and str(saved_method) != args.method:
        raise ValueError(
            f"Checkpoint method mismatch: requested {args.method}, saved {saved_method}"
        )
    saved_step = int(payload.get("optimizer_step", payload.get("global_step", -1)))
    model = model.to(device)
    model.eval()
    return model, saved_step


def main() -> int:
    args = _parse_args()
    model_path = args.model_path.resolve()
    benchmark_dir = args.benchmark_dir.resolve()
    output_dir = args.output_dir.resolve()
    if not model_path.is_file():
        raise FileNotFoundError(model_path)
    files = sorted(benchmark_dir.glob("*.jsp"))
    if len(files) != 10:
        raise ValueError(f"Expected exactly 10 TA50x20 instances, found {len(files)}")

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    model, optimizer_step = _load_lightweight_model(args, model_path, device)
    output_dir.mkdir(parents=True, exist_ok=False)

    rows: list[dict[str, object]] = []
    for index, file_path in enumerate(files):
        instance = load_instance(file_path.as_posix(), device="cpu")
        shape = (int(instance["j"]), int(instance["m"]))
        if shape != (50, 20):
            raise ValueError(f"Mixed or unsupported benchmark shape {shape} in {file_path}")

        seed = int(args.sampling_seed) + index
        random.seed(seed)
        np.random.seed(seed % (2**32))
        torch.manual_seed(seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(seed)

        started = time.perf_counter()
        with torch.inference_mode():
            makespans, _, _ = sampling(
                [instance],
                model.encoder,
                model.decoder,
                bs=int(args.B),
                use_greedy=bool(args.greedy),
                device=str(device),
            )
        elapsed = time.perf_counter() - started
        predicted = float(makespans.view(-1).min().detach().cpu())
        reference = float(instance["makespan"])
        gap = (predicted / reference - 1.0) * 100.0
        row = {
            "instance": str(instance["name"]),
            "shape": "50x20",
            "reference_makespan": reference,
            "predicted_makespan": predicted,
            "gap_percent": gap,
            "time_sec": elapsed,
            "sampling_seed": seed,
        }
        rows.append(row)
        print(json.dumps({"event": "instance", **row}), flush=True)

    mean_gap = sum(float(row["gap_percent"]) for row in rows) / len(rows)
    paper_gap = PAPER_GAPS[int(args.B)] if args.method == "bopo" else None
    delta_pp = mean_gap - paper_gap if paper_gap is not None else None
    alert = bool(delta_pp is not None and delta_pp > float(args.alert_margin_pp))
    summary = {
        "protocol": "official_bopo_ta50x20_v1",
        "method": args.method,
        "checkpoint": model_path.as_posix(),
        "checkpoint_sha256": _sha256(model_path),
        "optimizer_step": optimizer_step,
        "benchmark_dir": benchmark_dir.as_posix(),
        "instance_count": len(rows),
        "B_prime": int(args.B),
        "greedy_injected": bool(args.greedy),
        "mean_gap_percent": mean_gap,
        "paper_bopo_gap_percent": paper_gap,
        "delta_from_paper_pp": delta_pp,
        "alert_margin_pp": float(args.alert_margin_pp),
        "paper_alignment_alert": alert,
    }

    with (output_dir / "per_instance.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=True)
    print(json.dumps({"event": "summary", **summary}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
