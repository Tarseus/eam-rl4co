from __future__ import annotations

import argparse
import csv
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.classical_solver_benchmark import _parse_lkh_tour, run_tsp_lkh


@dataclass(frozen=True)
class ReferenceResult:
    instance_index: int
    status: str
    tour_length: float | None
    continuous_tour_length: float | None
    elapsed_sec: float
    notes: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run LKH on TSP locations stored in an NPZ file."
    )
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-instances", type=int, default=None)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--time-limit-sec", type=float, default=60.0)
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--max-trials", type=int, default=None)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--coordinate-scale", type=float, default=100_000.0)
    return parser.parse_args()


def _write_csv(path: Path, rows: list[ReferenceResult]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(rows[0])))
        writer.writeheader()
        writer.writerows(asdict(row) for row in rows)


def main() -> int:
    args = parse_args()
    if args.max_instances is not None and args.max_instances < 1:
        raise ValueError("max_instances must be >= 1")
    if args.time_limit_sec <= 0:
        raise ValueError("time_limit_sec must be > 0")
    if args.runs < 1:
        raise ValueError("runs must be >= 1")
    if args.max_trials is not None and args.max_trials < 1:
        raise ValueError("max_trials must be >= 1")
    if args.workers < 1:
        raise ValueError("workers must be >= 1")
    if args.coordinate_scale <= 0:
        raise ValueError("coordinate_scale must be > 0")

    with np.load(args.dataset.resolve()) as payload:
        if "locs" not in payload:
            raise KeyError(f"Dataset has no 'locs' array: {args.dataset}")
        locations = np.asarray(payload["locs"], dtype=np.float32)
    if locations.ndim != 3 or locations.shape[-1] != 2:
        raise ValueError(f"Expected locs with shape [B, N, 2], got {locations.shape}")
    if args.max_instances is not None:
        locations = locations[: args.max_instances]

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    scenario = f"tsp{locations.shape[1]}"
    rows_by_index: dict[int, ReferenceResult] = {}

    def solve(instance_index: int) -> ReferenceResult:
        result = run_tsp_lkh(
            coords=locations[instance_index],
            seed=int(args.seed) + instance_index,
            instance_id=f"instance_{instance_index:05d}",
            scenario_name=scenario,
            time_limit_sec=float(args.time_limit_sec),
            runs=int(args.runs),
            output_dir=output_dir,
            max_trials=args.max_trials,
        )
        tour_length = (
            None
            if result.objective is None
            else float(result.objective) / float(args.coordinate_scale)
        )
        continuous_tour_length = None
        if result.status == "ok":
            tour_path = (
                output_dir
                / "artifacts"
                / scenario
                / "lkh"
                / f"instance_{instance_index:05d}.tour"
            )
            tour = np.asarray(_parse_lkh_tour(tour_path), dtype=np.int64)
            ordered = np.asarray(locations[instance_index], dtype=np.float64)[tour]
            continuous_tour_length = float(
                np.linalg.norm(ordered - np.roll(ordered, -1, axis=0), axis=1).sum()
            )
        return ReferenceResult(
            instance_index=instance_index,
            status=result.status,
            tour_length=tour_length,
            continuous_tour_length=continuous_tour_length,
            elapsed_sec=float(result.elapsed_s),
            notes=result.notes,
        )

    with ThreadPoolExecutor(max_workers=min(args.workers, len(locations))) as executor:
        futures = {
            executor.submit(solve, instance_index): instance_index
            for instance_index in range(len(locations))
        }
        for future in as_completed(futures):
            row = future.result()
            rows_by_index[row.instance_index] = row
            rows = [rows_by_index[index] for index in sorted(rows_by_index)]
            _write_csv(output_dir / "per_instance.csv", rows)
            print(
                f"[result] instance={row.instance_index} status={row.status} "
                f"length={row.tour_length} elapsed={row.elapsed_sec:.2f}s",
                flush=True,
            )

    rows = [rows_by_index[index] for index in sorted(rows_by_index)]
    successful = [row for row in rows if row.status == "ok" and row.tour_length is not None]
    summary = {
        "config": {
            "dataset": str(args.dataset.resolve()),
            "num_instances": len(locations),
            "num_nodes": int(locations.shape[1]),
            "seed": args.seed,
            "time_limit_sec": args.time_limit_sec,
            "runs": args.runs,
            "max_trials": args.max_trials,
            "workers": args.workers,
            "coordinate_scale": args.coordinate_scale,
        },
        "num_successful": len(successful),
        "mean_tour_length": (
            float(np.mean([row.tour_length for row in successful]))
            if successful
            else None
        ),
        "mean_continuous_tour_length": (
            float(
                np.mean(
                    [
                        row.continuous_tour_length
                        for row in successful
                        if row.continuous_tour_length is not None
                    ]
                )
            )
            if successful
            else None
        ),
        "mean_elapsed_sec": (
            float(np.mean([row.elapsed_sec for row in successful]))
            if successful
            else None
        ),
        "rows": [asdict(row) for row in rows],
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"[done] output_dir={output_dir}", flush=True)
    return 0 if len(successful) == len(rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
