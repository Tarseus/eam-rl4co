from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class ReferenceResult:
    instance_index: int
    status: str
    tour_length: float | None
    concorde_scaled_length: float | None
    concorde_integer_length: int | None
    quantization_delta: float | None
    elapsed_sec: float
    success: bool
    found_tour: bool
    hit_timebound: bool
    tour: list[int] | None
    error: str | None = None


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_locations(path: Path) -> np.ndarray:
    with np.load(path) as payload:
        if "locs" not in payload:
            raise KeyError(f"Dataset has no 'locs' array: {path}")
        locations = np.asarray(payload["locs"], dtype=np.float32)
    if locations.ndim != 3 or locations.shape[-1] != 2:
        raise ValueError(f"Expected locs with shape [B, N, 2], got {locations.shape}")
    return locations


def cycle_length_float(coords: np.ndarray, tour: np.ndarray) -> float:
    ordered = np.asarray(coords, dtype=np.float64)[tour]
    deltas = ordered - np.roll(ordered, -1, axis=0)
    return float(np.sqrt(np.square(deltas).sum(axis=1)).sum())


def cycle_length_euc_2d(coords: np.ndarray, tour: np.ndarray) -> int:
    ordered = np.asarray(coords, dtype=np.int64)[tour]
    deltas = ordered - np.roll(ordered, -1, axis=0)
    distances = np.sqrt(np.square(deltas).sum(axis=1, dtype=np.int64))
    return int(np.floor(distances + 0.5).sum(dtype=np.float64))


def _validate_tour(tour: np.ndarray, node_count: int) -> None:
    if tour.shape != (node_count,):
        raise ValueError(f"Expected tour shape {(node_count,)}, got {tour.shape}")
    if not np.array_equal(np.sort(tour), np.arange(node_count)):
        raise ValueError("Concorde returned a non-permutation tour")


def solve_instance(
    instance_index: int,
    coords: np.ndarray,
    coordinate_scale: int,
    seed: int,
    time_bound_sec: float,
) -> ReferenceResult:
    started_at = time.perf_counter()
    try:
        from concorde.tsp import TSPSolver

        scaled = np.rint(
            np.asarray(coords, dtype=np.float64) * float(coordinate_scale)
        ).astype(np.int32)
        solver = TSPSolver.from_data(
            scaled[:, 0],
            scaled[:, 1],
            norm="EUC_2D",
            name=f"fixed_tsp_{instance_index:05d}",
        )
        solution = solver.solve(
            time_bound=float(time_bound_sec),
            verbose=False,
            random_seed=int(seed) + int(instance_index),
        )
        elapsed = time.perf_counter() - started_at
        if not solution.found_tour:
            return ReferenceResult(
                instance_index=instance_index,
                status="timebound" if solution.hit_timebound else "no_tour",
                tour_length=None,
                concorde_scaled_length=None,
                concorde_integer_length=None,
                quantization_delta=None,
                elapsed_sec=elapsed,
                success=bool(solution.success),
                found_tour=False,
                hit_timebound=bool(solution.hit_timebound),
                tour=None,
            )

        tour = np.asarray(solution.tour, dtype=np.int64)
        _validate_tour(tour, len(coords))
        integer_length = cycle_length_euc_2d(scaled, tour)
        solver_integer_length = int(round(float(solution.optimal_value)))
        if integer_length != solver_integer_length:
            raise RuntimeError(
                "Tour length mismatch: "
                f"recomputed={integer_length}, concorde={solver_integer_length}"
            )
        float_length = cycle_length_float(coords, tour)
        scaled_length = float(integer_length) / float(coordinate_scale)
        optimal = bool(solution.success) and not bool(solution.hit_timebound)
        return ReferenceResult(
            instance_index=instance_index,
            status="ok" if optimal else "feasible_not_proven",
            tour_length=float_length,
            concorde_scaled_length=scaled_length,
            concorde_integer_length=integer_length,
            quantization_delta=float_length - scaled_length,
            elapsed_sec=elapsed,
            success=bool(solution.success),
            found_tour=True,
            hit_timebound=bool(solution.hit_timebound),
            tour=[int(node) for node in tour],
        )
    except Exception as exc:  # noqa: BLE001
        return ReferenceResult(
            instance_index=instance_index,
            status="error",
            tour_length=None,
            concorde_scaled_length=None,
            concorde_integer_length=None,
            quantization_delta=None,
            elapsed_sec=time.perf_counter() - started_at,
            success=False,
            found_tour=False,
            hit_timebound=False,
            tour=None,
            error=f"{type(exc).__name__}: {exc}",
        )


def _write_csv(path: Path, rows: list[ReferenceResult]) -> None:
    if not rows:
        return
    columns = [field for field in asdict(rows[0]) if field != "tour"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            payload = asdict(row)
            payload.pop("tour")
            writer.writerow(payload)


def _summary(
    *,
    dataset: Path,
    dataset_sha256: str,
    locations: np.ndarray,
    rows: list[ReferenceResult],
    coordinate_scale: int,
    seed: int,
    workers: int,
    time_bound_sec: float,
) -> dict[str, Any]:
    exact = [row for row in rows if row.status == "ok" and row.tour_length is not None]
    quantization_bound = locations.shape[1] * 0.5 / float(coordinate_scale)
    return {
        "protocol": "tsp_concorde_exact_reference_v1",
        "config": {
            "dataset": str(dataset),
            "dataset_sha256": dataset_sha256,
            "num_instances": int(len(locations)),
            "num_nodes": int(locations.shape[1]),
            "coordinate_scale": int(coordinate_scale),
            "distance_type": "TSPLIB EUC_2D",
            "seed": int(seed),
            "workers": int(workers),
            "time_bound_sec": float(time_bound_sec),
            "float_tour_length_rounding_error_bound": quantization_bound,
        },
        "num_completed": len(rows),
        "num_exact": len(exact),
        "mean_tour_length": (
            float(np.mean([row.tour_length for row in exact])) if exact else None
        ),
        "max_abs_quantization_delta": (
            max(abs(float(row.quantization_delta)) for row in exact) if exact else None
        ),
        "rows": [asdict(row) for row in rows],
    }


def _write_outputs(output_dir: Path, summary: dict[str, Any], rows: list[ReferenceResult]) -> None:
    _write_csv(output_dir / "per_instance.csv", rows)
    temp_path = output_dir / "summary.json.tmp"
    temp_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temp_path.replace(output_dir / "summary.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute exact Concorde references for a fixed TSP NPZ dataset."
    )
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-instances", type=int, default=None)
    parser.add_argument("--coordinate-scale", type=int, default=1_000_000)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--time-bound-sec", type=float, default=-1.0)
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Reuse exact rows already present in output-dir/summary.json.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    dataset = args.dataset.resolve()
    locations = load_locations(dataset)
    if args.max_instances is not None:
        if args.max_instances < 1:
            raise ValueError("max-instances must be >= 1")
        locations = locations[: args.max_instances]
    if args.coordinate_scale < 1 or args.workers < 1:
        raise ValueError("coordinate-scale and workers must be >= 1")
    if not math.isfinite(args.time_bound_sec):
        raise ValueError("time-bound-sec must be finite")

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_sha256 = _sha256_file(dataset)
    rows_by_index: dict[int, ReferenceResult] = {}
    summary_path = output_dir / "summary.json"
    if args.resume and summary_path.is_file():
        previous = json.loads(summary_path.read_text(encoding="utf-8"))
        previous_config = previous.get("config", {})
        if previous_config.get("dataset_sha256") != dataset_sha256:
            raise ValueError("Cannot resume: dataset SHA-256 changed")
        if int(previous_config.get("coordinate_scale", -1)) != int(args.coordinate_scale):
            raise ValueError("Cannot resume: coordinate scale changed")
        for payload in previous.get("rows", []):
            row = ReferenceResult(**payload)
            if row.status == "ok" and row.instance_index < len(locations):
                rows_by_index[row.instance_index] = row

    pending_indices = [
        index for index in range(len(locations)) if index not in rows_by_index
    ]
    if rows_by_index:
        print(
            f"[resume] exact={len(rows_by_index)} pending={len(pending_indices)}",
            flush=True,
        )
    if not pending_indices:
        rows = [rows_by_index[index] for index in sorted(rows_by_index)]
        summary = _summary(
            dataset=dataset,
            dataset_sha256=dataset_sha256,
            locations=locations,
            rows=rows,
            coordinate_scale=int(args.coordinate_scale),
            seed=int(args.seed),
            workers=int(args.workers),
            time_bound_sec=float(args.time_bound_sec),
        )
        _write_outputs(output_dir, summary, rows)
        print(f"[done] exact={len(rows)}/{len(rows)} output_dir={output_dir}", flush=True)
        return 0

    with ProcessPoolExecutor(max_workers=min(args.workers, len(locations))) as executor:
        futures = {
            executor.submit(
                solve_instance,
                index,
                locations[index],
                int(args.coordinate_scale),
                int(args.seed),
                float(args.time_bound_sec),
            ): index
            for index in pending_indices
        }
        for future in as_completed(futures):
            row = future.result()
            rows_by_index[row.instance_index] = row
            rows = [rows_by_index[index] for index in sorted(rows_by_index)]
            summary = _summary(
                dataset=dataset,
                dataset_sha256=dataset_sha256,
                locations=locations,
                rows=rows,
                coordinate_scale=int(args.coordinate_scale),
                seed=int(args.seed),
                workers=int(args.workers),
                time_bound_sec=float(args.time_bound_sec),
            )
            _write_outputs(output_dir, summary, rows)
            print(
                f"[reference] instance={row.instance_index} status={row.status} "
                f"length={row.tour_length} elapsed={row.elapsed_sec:.2f}s",
                flush=True,
            )

    rows = [rows_by_index[index] for index in sorted(rows_by_index)]
    exact_count = sum(row.status == "ok" for row in rows)
    print(f"[done] exact={exact_count}/{len(rows)} output_dir={output_dir}", flush=True)
    return 0 if exact_count == len(rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
