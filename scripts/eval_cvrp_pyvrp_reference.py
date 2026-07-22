from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import classical_solver_benchmark as bench


@dataclass(frozen=True)
class ReferenceResult:
    instance_index: int
    source_index: int
    status: str
    cost: float | None
    scaled_integer_cost: float | None
    elapsed_sec: float
    num_routes: int | None
    max_route_load: int | None
    all_customers_once: bool | None
    capacity_feasible: bool | None
    recomputed_cost_matches: bool | None
    notes: str
    error: str | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run PyVRP on a fixed CVRP NPZ dataset."
    )
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-instances", type=int, default=None)
    parser.add_argument("--time-limit-sec", type=float, default=60.0)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--coordinate-scale", type=float, default=100_000.0)
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Reuse completed rows from output-dir/summary.json.",
    )
    return parser.parse_args()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_dataset(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as payload:
        required = {"depot", "locs", "demand", "capacity"}
        missing = required.difference(payload.files)
        if missing:
            raise KeyError(f"Dataset is missing arrays: {sorted(missing)}")
        arrays = {key: np.asarray(payload[key]) for key in required}
        arrays["source_index"] = (
            np.asarray(payload["source_index"], dtype=np.int64)
            if "source_index" in payload.files
            else np.arange(len(arrays["locs"]), dtype=np.int64)
        )

    count = len(arrays["locs"])
    expected = {
        "depot": (count, 2),
        "demand": (count, arrays["locs"].shape[1]),
        "capacity": (count,),
        "source_index": (count,),
    }
    if arrays["locs"].ndim != 3 or arrays["locs"].shape[-1] != 2:
        raise ValueError(f"Expected locs [B, N, 2], got {arrays['locs'].shape}")
    for key, shape in expected.items():
        if arrays[key].shape != shape:
            raise ValueError(f"Expected {key} shape {shape}, got {arrays[key].shape}")
    if not np.all(np.isfinite(arrays["depot"])) or not np.all(
        np.isfinite(arrays["locs"])
    ):
        raise ValueError("Coordinates contain non-finite values")
    if np.any(arrays["demand"] <= 0) or np.any(arrays["capacity"] <= 0):
        raise ValueError("Demand and capacity must be positive")
    return arrays


def _solve(
    instance_index: int,
    source_index: int,
    depot: np.ndarray,
    locs: np.ndarray,
    demand: np.ndarray,
    capacity: float,
    time_limit_sec: float,
    coordinate_scale: float,
) -> ReferenceResult:
    try:
        if (
            bench._PYVRP_IMPORT_ERROR is not None
            or bench.pyvrp_solve is None
            or bench.MaxRuntime is None
        ):
            raise RuntimeError("PyVRP is unavailable") from bench._PYVRP_IMPORT_ERROR

        scaled = np.rint(
            np.concatenate([depot[None, :], locs], axis=0) * coordinate_scale
        ).astype(np.int64)
        delta = scaled[:, None, :] - scaled[None, :, :]
        matrix = np.floor(
            np.sqrt(np.square(delta).sum(axis=-1, dtype=np.int64)) + 0.5
        ).astype(np.int64)
        clients = [
            bench.Client(
                x=int(scaled[index, 0]),
                y=int(scaled[index, 1]),
                delivery=[int(demand[index - 1])],
                pickup=[0],
                service_duration=0,
                tw_early=0,
                tw_late=bench.PYVRP_MAX_VALUE,
            )
            for index in range(1, len(scaled))
        ]
        problem = bench.ProblemData(
            clients,
            [bench.Depot(x=int(scaled[0, 0]), y=int(scaled[0, 1]))],
            [
                bench.VehicleType(
                    num_available=len(locs),
                    capacity=[int(round(capacity))],
                    max_distance=bench.PYVRP_MAX_VALUE,
                    tw_early=0,
                    tw_late=bench.PYVRP_MAX_VALUE,
                )
            ],
            [matrix],
            [matrix],
        )
        started_at = time.perf_counter()
        result = bench.pyvrp_solve(
            problem,
            bench.MaxRuntime(time_limit_sec),
            seed=source_index,
        )
        elapsed_sec = time.perf_counter() - started_at
        routes = result.best.routes()
        visits = [int(client) for route in routes for client in route.visits()]
        all_customers_once = sorted(visits) == list(range(1, len(locs) + 1))
        route_loads = [
            sum(int(demand[client - 1]) for client in route.visits())
            for route in routes
        ]
        max_route_load = max(route_loads, default=0)
        capacity_feasible = max_route_load <= int(round(capacity))
        recomputed_cost = 0
        for route in routes:
            sequence = [0, *[int(client) for client in route.visits()], 0]
            recomputed_cost += sum(
                int(matrix[src, dst]) for src, dst in zip(sequence, sequence[1:])
            )
        scaled_cost = float(result.cost())
        recomputed_cost_matches = int(round(scaled_cost)) == recomputed_cost
        cost = scaled_cost / coordinate_scale
        if not math.isfinite(cost) or cost <= 0:
            raise RuntimeError(f"Invalid PyVRP objective: {scaled_cost}")
        if not all_customers_once or not capacity_feasible or not recomputed_cost_matches:
            raise RuntimeError(
                "Route audit failed: "
                f"all_customers_once={all_customers_once}, "
                f"capacity_feasible={capacity_feasible}, "
                f"recomputed_cost_matches={recomputed_cost_matches}"
            )
        return ReferenceResult(
            instance_index=instance_index,
            source_index=source_index,
            status="ok",
            cost=cost,
            scaled_integer_cost=scaled_cost,
            elapsed_sec=elapsed_sec,
            num_routes=len(routes),
            max_route_load=max_route_load,
            all_customers_once=all_customers_once,
            capacity_feasible=capacity_feasible,
            recomputed_cost_matches=recomputed_cost_matches,
            notes=f"seed={source_index}",
        )
    except Exception as exc:  # noqa: BLE001
        return ReferenceResult(
            instance_index=instance_index,
            source_index=source_index,
            status="error",
            cost=None,
            scaled_integer_cost=None,
            elapsed_sec=0.0,
            num_routes=None,
            max_route_load=None,
            all_customers_once=None,
            capacity_feasible=None,
            recomputed_cost_matches=None,
            notes="",
            error=f"{type(exc).__name__}: {exc}",
        )


def _write_csv(path: Path, rows: list[ReferenceResult]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(rows[0])))
        writer.writeheader()
        writer.writerows(asdict(row) for row in rows)


def _make_summary(
    *,
    dataset: Path,
    dataset_sha256: str,
    arrays: dict[str, np.ndarray],
    rows: list[ReferenceResult],
    time_limit_sec: float,
    workers: int,
    coordinate_scale: float,
) -> dict[str, Any]:
    successful = [row for row in rows if row.status == "ok" and row.cost is not None]
    return {
        "protocol": "cvrp_pyvrp_fixed_reference_v1",
        "config": {
            "dataset": str(dataset),
            "dataset_sha256": dataset_sha256,
            "num_instances": int(len(arrays["locs"])),
            "num_customers": int(arrays["locs"].shape[1]),
            "capacity_values": sorted(
                {float(value) for value in arrays["capacity"].tolist()}
            ),
            "time_limit_sec": float(time_limit_sec),
            "workers": int(workers),
            "coordinate_scale": float(coordinate_scale),
            "distance_type": "TSPLIB EUC_2D after coordinate scaling",
        },
        "num_completed": len(rows),
        "num_successful": len(successful),
        "mean_cost": (
            float(np.mean([row.cost for row in successful])) if successful else None
        ),
        "mean_elapsed_sec": (
            float(np.mean([row.elapsed_sec for row in successful]))
            if successful
            else None
        ),
        "rows": [asdict(row) for row in rows],
    }


def _write_outputs(
    output_dir: Path, rows: list[ReferenceResult], summary: dict[str, Any]
) -> None:
    _write_csv(output_dir / "per_instance.csv", rows)
    temp = output_dir / "summary.json.tmp"
    temp.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    temp.replace(output_dir / "summary.json")


def main() -> int:
    args = parse_args()
    if args.max_instances is not None and args.max_instances < 1:
        raise ValueError("max-instances must be >= 1")
    if args.time_limit_sec <= 0 or args.workers < 1 or args.coordinate_scale <= 0:
        raise ValueError("time-limit-sec, workers, and coordinate-scale must be positive")

    dataset = args.dataset.resolve()
    arrays = _load_dataset(dataset)
    if args.max_instances is not None:
        arrays = {key: value[: args.max_instances] for key, value in arrays.items()}
    dataset_sha256 = _sha256_file(dataset)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    rows_by_index: dict[int, ReferenceResult] = {}
    summary_path = output_dir / "summary.json"
    if args.resume and summary_path.is_file():
        previous = json.loads(summary_path.read_text(encoding="utf-8"))
        if previous.get("config", {}).get("dataset_sha256") != dataset_sha256:
            raise ValueError("Existing summary uses a different dataset SHA256")
        for raw in previous.get("rows", []):
            row = ReferenceResult(**raw)
            if row.status == "ok":
                rows_by_index[row.instance_index] = row

    pending = [index for index in range(len(arrays["locs"])) if index not in rows_by_index]
    with ProcessPoolExecutor(max_workers=min(args.workers, max(1, len(pending)))) as executor:
        futures = {
            executor.submit(
                _solve,
                index,
                int(arrays["source_index"][index]),
                arrays["depot"][index],
                arrays["locs"][index],
                arrays["demand"][index],
                float(arrays["capacity"][index]),
                float(args.time_limit_sec),
                float(args.coordinate_scale),
            ): index
            for index in pending
        }
        for future in as_completed(futures):
            row = future.result()
            rows_by_index[row.instance_index] = row
            rows = [rows_by_index[index] for index in sorted(rows_by_index)]
            summary = _make_summary(
                dataset=dataset,
                dataset_sha256=dataset_sha256,
                arrays=arrays,
                rows=rows,
                time_limit_sec=args.time_limit_sec,
                workers=args.workers,
                coordinate_scale=args.coordinate_scale,
            )
            _write_outputs(output_dir, rows, summary)
            print(
                f"[result] instance={row.source_index} status={row.status} "
                f"cost={row.cost} elapsed={row.elapsed_sec:.2f}s error={row.error}",
                flush=True,
            )

    rows = [rows_by_index[index] for index in sorted(rows_by_index)]
    summary = _make_summary(
        dataset=dataset,
        dataset_sha256=dataset_sha256,
        arrays=arrays,
        rows=rows,
        time_limit_sec=args.time_limit_sec,
        workers=args.workers,
        coordinate_scale=args.coordinate_scale,
    )
    _write_outputs(output_dir, rows, summary)
    print(
        f"[done] successful={summary['num_successful']}/{len(arrays['locs'])} "
        f"mean_cost={summary['mean_cost']} output_dir={output_dir}",
        flush=True,
    )
    return 0 if summary["num_successful"] == len(arrays["locs"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
