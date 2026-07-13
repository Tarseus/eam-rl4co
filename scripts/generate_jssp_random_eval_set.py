from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def _parse_shape(raw: str) -> tuple[int, int]:
    left, right = raw.lower().split("x", 1)
    return int(left), int(right)


def _write_instance(path: Path, num_jobs: int, num_machines: int, rng: np.random.Generator) -> None:
    lines = [f"{num_jobs} {num_machines}"]
    for _ in range(num_jobs):
        machines = rng.permutation(num_machines)
        durations = rng.integers(1, 100, size=num_machines)
        row: list[str] = []
        for machine, duration in zip(machines, durations, strict=True):
            row.extend([str(int(machine)), str(int(duration))])
        lines.append(" ".join(row))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate random JSSP eval sets matching the MGL training data format.")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--shapes", nargs="+", default=["10x10", "15x15"])
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--seed", type=int, default=12345678)
    parser.add_argument("--clean", action="store_true", help="Delete existing *.jsp files in each target shape directory.")
    args = parser.parse_args()

    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    for shape_idx, shape in enumerate(args.shapes):
        num_jobs, num_machines = _parse_shape(shape)
        shape_name = f"{num_jobs}x{num_machines}"
        shape_dir = output_root / shape_name
        shape_dir.mkdir(parents=True, exist_ok=True)
        if args.clean:
            for path in shape_dir.glob("*.jsp"):
                path.unlink()
        rng = np.random.default_rng(int(args.seed) + shape_idx * 1_000_003 + num_jobs * 1000 + num_machines)
        for idx in range(int(args.count)):
            _write_instance(shape_dir / f"{shape_name}_{idx:04d}.jsp", num_jobs, num_machines, rng)
        print(f"{shape_name}: wrote {args.count} files to {shape_dir.as_posix()}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
