from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate canonical FFSP evaluation sets as NPZ files with a run_time array."
    )
    parser.add_argument("--output-dir", type=Path, default=Path("data/ffsp"))
    parser.add_argument("--shapes", nargs="+", default=["50", "100"], help="Job counts, e.g. 50 100.")
    parser.add_argument("--count", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--num-stage", type=int, default=3)
    parser.add_argument("--num-machine", type=int, default=4)
    parser.add_argument("--min-time", type=int, default=2)
    parser.add_argument("--max-time", type=int, default=10)
    parser.add_argument(
        "--prefix",
        type=str,
        default="ffsp",
        help="Output name prefix. Files are <prefix><jobs>_test_seed<seed>_torch.npz.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for shape in args.shapes:
        num_job = int(str(shape).lower().replace("ffsp", ""))
        torch.manual_seed(int(args.seed))
        run_time = torch.randint(
            low=int(args.min_time),
            high=int(args.max_time),
            size=(int(args.count), num_job, int(args.num_stage) * int(args.num_machine)),
            dtype=torch.int64,
        )
        output_path = args.output_dir / f"{args.prefix}{num_job}_test_seed{args.seed}_torch.npz"
        np.savez(output_path, run_time=run_time.cpu().numpy().astype(np.int64))
        print(
            f"wrote {output_path} shape={tuple(run_time.shape)} "
            f"min={int(run_time.min())} max={int(run_time.max())}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
