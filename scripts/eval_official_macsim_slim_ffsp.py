from __future__ import annotations

import argparse
import csv
import hashlib
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from tensordict import TensorDict


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a published MACSIM SLIM FFSP checkpoint on the paper's "
            "fixed NPZ instances with an explicit best-of-samples budget."
        )
    )
    parser.add_argument("--macsim-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-file", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--problem", choices=("ffsp50", "ffsp100"), required=True)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--end-index", type=int, default=None)
    parser.add_argument("--physical-batch-size", type=int, default=1)
    parser.add_argument("--num-starts", type=int, default=24)
    parser.add_argument("--num-augment", type=int, default=128)
    parser.add_argument("--sample-offset", type=int, default=0)
    parser.add_argument(
        "--sample-count",
        type=int,
        default=None,
        help="Evaluate only this many candidates per instance (for candidate-axis sharding).",
    )
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--precision", choices=("fp32", "fp16", "bf16"), default="fp32")
    parser.add_argument(
        "--matmul-precision",
        choices=("highest", "high", "medium"),
        default="highest",
        help="PyTorch float32 matmul precision; 'high' enables TF32-style acceleration on Ampere.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    macsim_root = args.macsim_root.resolve()
    checkpoint = args.checkpoint.resolve()
    data_file = args.data_file.resolve()
    output_csv = args.output_csv.resolve()
    if not checkpoint.is_file() or not data_file.is_file():
        raise FileNotFoundError(f"checkpoint={checkpoint}, data_file={data_file}")
    if args.physical_batch_size < 1:
        raise ValueError("physical_batch_size must be positive")

    total_candidates = int(args.num_starts) * int(args.num_augment)
    sample_count = total_candidates if args.sample_count is None else int(args.sample_count)
    if sample_count < 1 or args.sample_offset < 0:
        raise ValueError("sample_count must be positive and sample_offset nonnegative")
    if args.sample_offset + sample_count > total_candidates:
        raise ValueError("candidate shard exceeds num_starts * num_augment")

    sys.path.insert(0, str(macsim_root))
    os.environ["PROJECT_ROOT"] = str(macsim_root)
    from macsim.algorithms.base import EvalModule
    from macsim.envs.env_args import FFSPParams
    from macsim.utils.config import TestParams

    num_jobs = 50 if args.problem == "ffsp50" else 100
    env_params = FFSPParams(
        num_jobs=num_jobs,
        num_stage=3,
        min_ma_per_stage=4,
        max_ma_per_stage=4,
        min_processing_time=2,
        max_processing_time=10,
    )
    test_params = TestParams(
        batch_size=int(args.physical_batch_size),
        dataset_size=0,
        checkpoint=str(checkpoint),
        seed=int(args.seed),
        decode_type="sampling",
        num_starts=sample_count,
        num_augment=1,
        select_best=True,
        hybrid_decoding=False,
    )
    model, _ = EvalModule.init_from_checkpoint(
        str(checkpoint), env_params, test_params=test_params
    )
    device = torch.device(args.device)
    model = model.to(device).eval()
    model.policy.set_decode_type(test_params.decoding)

    with np.load(data_file) as payload:
        run_time = np.asarray(payload["run_time"])
    if run_time.ndim != 3 or run_time.shape[1:] != (num_jobs, 12):
        raise ValueError(
            f"Unexpected run_time shape {run_time.shape}; expected [N,{num_jobs},12]"
        )
    begin = int(args.start_index)
    end = len(run_time) if args.end_index is None else int(args.end_index)
    if not (0 <= begin < end <= len(run_time)):
        raise ValueError(f"Invalid instance range [{begin}, {end}) for N={len(run_time)}")

    random.seed(args.seed + args.sample_offset)
    np.random.seed(args.seed + args.sample_offset)
    torch.manual_seed(args.seed + args.sample_offset)
    torch.cuda.manual_seed_all(args.seed + args.sample_offset)
    torch.set_float32_matmul_precision(args.matmul_precision)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "problem",
        "instance_id",
        "instance_index",
        "cost",
        "sample_offset",
        "sample_count",
        "total_candidates",
        "seed",
        "checkpoint",
        "checkpoint_sha256",
        "test_file",
        "test_file_sha256",
        "precision",
        "matmul_precision",
        "elapsed_sec",
    ]
    checkpoint_hash = sha256(checkpoint)
    data_hash = sha256(data_file)
    started = time.perf_counter()
    rows_written = 0
    autocast_dtype = {"fp16": torch.float16, "bf16": torch.bfloat16}.get(args.precision)

    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        handle.flush()
        with torch.inference_mode():
            for batch_begin in range(begin, end, int(args.physical_batch_size)):
                batch_end = min(end, batch_begin + int(args.physical_batch_size))
                proc_times = torch.as_tensor(
                    run_time[batch_begin:batch_end], dtype=torch.long, device=device
                )
                batch_size = proc_times.shape[0]
                machine_cnt = torch.full(
                    (batch_size, 3), 4, dtype=torch.long, device=device
                )
                stage_table = torch.arange(3, device=device).repeat_interleave(4)
                stage_table = stage_table.unsqueeze(0).expand(batch_size, -1).contiguous()
                batch = TensorDict(
                    {
                        "proc_times": proc_times,
                        "machine_cnt": machine_cnt,
                        "stage_table": stage_table,
                    },
                    batch_size=(batch_size,),
                    device=device,
                )
                state, _ = model.env.reset(batch)
                with torch.autocast(
                    device_type="cuda",
                    dtype=autocast_dtype or torch.float16,
                    enabled=autocast_dtype is not None,
                ):
                    reward = model.policy(state, model.env)["reward"]
                costs = (-reward).detach().float().cpu().tolist()
                batch_elapsed = time.perf_counter() - started
                for local_index, cost in enumerate(costs):
                    index = batch_begin + local_index
                    writer.writerow(
                        {
                            "problem": args.problem,
                            "instance_id": f"{args.problem}_{index:05d}",
                            "instance_index": index,
                            "cost": float(cost),
                            "sample_offset": int(args.sample_offset),
                            "sample_count": sample_count,
                            "total_candidates": total_candidates,
                            "seed": int(args.seed),
                            "checkpoint": str(checkpoint),
                            "checkpoint_sha256": checkpoint_hash,
                            "test_file": str(data_file),
                            "test_file_sha256": data_hash,
                            "precision": args.precision,
                            "matmul_precision": args.matmul_precision,
                            "elapsed_sec": batch_elapsed,
                        }
                    )
                    rows_written += 1
                handle.flush()
                print(
                    f"progress={rows_written}/{end-begin} "
                    f"last_index={batch_end-1} elapsed={batch_elapsed:.3f}s",
                    flush=True,
                )

    print(
        f"done problem={args.problem} instances={rows_written} "
        f"samples={sample_count}/{total_candidates} elapsed={time.perf_counter()-started:.3f}s",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
