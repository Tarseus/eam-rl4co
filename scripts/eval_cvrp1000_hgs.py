from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from ctypes import POINTER, Structure, byref, c_char, c_double, c_int, sizeof
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = REPO_ROOT / "data/vrp/agfn_vrp1000_capacity50_test128.npz"
DEFAULT_LIBRARY = (
    REPO_ROOT
    / "rl4co/envs/routing/cvrp/HGS-CVRP/build/libhgscvrp.so"
)


class AlgorithmParameters(Structure):
    _fields_ = [
        ("nbGranular", c_int),
        ("mu", c_int),
        ("lambda_", c_int),
        ("nbElite", c_int),
        ("nbClose", c_int),
        ("nbIterPenaltyManagement", c_int),
        ("targetFeasible", c_double),
        ("penaltyDecrease", c_double),
        ("penaltyIncrease", c_double),
        ("seed", c_int),
        ("nbIter", c_int),
        ("nbIterTraces", c_int),
        ("timeLimit", c_double),
        ("useSwapStar", c_int),
    ]


class SolutionRoute(Structure):
    _fields_ = [("length", c_int), ("path", POINTER(c_int))]


class Solution(Structure):
    _fields_ = [
        ("cost", c_double),
        ("time", c_double),
        ("n_routes", c_int),
        ("routes", POINTER(SolutionRoute)),
    ]


@dataclass(frozen=True)
class SeedResult:
    seed: int
    cost: float
    elapsed_sec: float
    n_routes: int
    max_route_load: int
    reported_cost: float
    recompute_delta: float


@dataclass(frozen=True)
class InstanceResult:
    instance_index: int
    source_index: int
    best_cost: float
    best_seed: int
    best_elapsed_sec: float
    best_n_routes: int
    max_route_load: int
    total_elapsed_sec: float
    runs: tuple[SeedResult, ...]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_library(path: str):
    from ctypes import CDLL

    library = CDLL(path)
    library.default_algorithm_parameters.argtypes = []
    library.default_algorithm_parameters.restype = AlgorithmParameters
    library.solve_cvrp.argtypes = [
        c_int,
        POINTER(c_double),
        POINTER(c_double),
        POINTER(c_double),
        POINTER(c_double),
        c_double,
        c_double,
        c_char,
        c_char,
        c_int,
        POINTER(AlgorithmParameters),
        c_char,
    ]
    library.solve_cvrp.restype = POINTER(Solution)
    library.delete_solution.argtypes = [POINTER(Solution)]
    library.delete_solution.restype = None
    return library


def _route_cost(coords: np.ndarray, routes: list[list[int]]) -> float:
    total = 0.0
    for route in routes:
        sequence = np.asarray([0, *route, 0], dtype=np.int64)
        delta = coords[sequence[1:]] - coords[sequence[:-1]]
        total += float(np.sqrt(np.square(delta).sum(axis=1)).sum(dtype=np.float64))
    return total


def _validate_routes(
    routes: list[list[int]], demands: np.ndarray, capacity: int
) -> tuple[int, int]:
    customers = [node for route in routes for node in route]
    expected = list(range(1, len(demands)))
    if sorted(customers) != expected:
        raise RuntimeError("HGS route set is not an exact customer permutation")
    loads = [int(demands[np.asarray(route, dtype=np.int64)].sum()) for route in routes]
    if not loads or max(loads) > int(capacity):
        raise RuntimeError(f"HGS produced an infeasible route load: {max(loads, default=-1)}")
    return len(routes), max(loads)


def _solve_seed(
    *,
    library_path: str,
    coords: np.ndarray,
    demands: np.ndarray,
    capacity: int,
    seed: int,
    time_limit_sec: float,
    nb_iter: int,
) -> SeedResult:
    library = _load_library(library_path)
    coords = np.ascontiguousarray(coords, dtype=np.float64)
    demands = np.ascontiguousarray(demands, dtype=np.float64)
    service = np.zeros(len(demands), dtype=np.float64)
    x = np.ascontiguousarray(coords[:, 0], dtype=np.float64)
    y = np.ascontiguousarray(coords[:, 1], dtype=np.float64)
    params = library.default_algorithm_parameters()
    params.seed = int(seed)
    params.nbIter = int(nb_iter)
    params.timeLimit = float(time_limit_sec)
    params.useSwapStar = 1
    max_vehicles = 2 ** (sizeof(c_int) * 8 - 1) - 1
    started = time.perf_counter()
    pointer = library.solve_cvrp(
        len(demands),
        x.ctypes.data_as(POINTER(c_double)),
        y.ctypes.data_as(POINTER(c_double)),
        service.ctypes.data_as(POINTER(c_double)),
        demands.ctypes.data_as(POINTER(c_double)),
        float(capacity),
        float(np.finfo(np.float64).max),
        b"\x00",  # Continuous Euclidean distances; no integer rounding.
        b"\x00",  # No route-duration constraint.
        max_vehicles,
        byref(params),
        b"\x00",
    )
    wall_elapsed = time.perf_counter() - started
    if not pointer:
        raise RuntimeError("HGS returned a null solution pointer")
    try:
        solution = pointer.contents
        routes = [
            [int(solution.routes[idx].path[j]) for j in range(solution.routes[idx].length)]
            for idx in range(int(solution.n_routes))
        ]
        n_routes, max_route_load = _validate_routes(routes, demands, capacity)
        recomputed = _route_cost(coords, routes)
        delta = recomputed - float(solution.cost)
        if abs(delta) > 1e-7:
            raise RuntimeError(
                f"HGS objective mismatch: reported={solution.cost}, recomputed={recomputed}"
            )
        return SeedResult(
            seed=int(seed),
            cost=float(recomputed),
            elapsed_sec=float(wall_elapsed),
            n_routes=int(n_routes),
            max_route_load=int(max_route_load),
            reported_cost=float(solution.cost),
            recompute_delta=float(delta),
        )
    finally:
        library.delete_solution(pointer)


def _solve_instance(task: dict[str, Any]) -> dict[str, Any]:
    output_path = Path(task["output_path"])
    if output_path.exists():
        return json.loads(output_path.read_text(encoding="utf-8"))
    instance_index = int(task["instance_index"])
    coords = np.asarray(task["coords"], dtype=np.float64)
    demands = np.asarray(task["demands"], dtype=np.float64)
    runs = []
    for replicate in range(int(task["num_replicates"])):
        seed = int(task["base_seed"]) + instance_index * 104_729 + replicate * 1_000_003
        runs.append(
            _solve_seed(
                library_path=str(task["library_path"]),
                coords=coords,
                demands=demands,
                capacity=int(task["capacity"]),
                seed=seed,
                time_limit_sec=float(task["time_limit_sec"]),
                nb_iter=int(task["nb_iter"]),
            )
        )
    best = min(runs, key=lambda row: (row.cost, row.seed))
    result = InstanceResult(
        instance_index=instance_index,
        source_index=int(task["source_index"]),
        best_cost=best.cost,
        best_seed=best.seed,
        best_elapsed_sec=best.elapsed_sec,
        best_n_routes=best.n_routes,
        max_route_load=best.max_route_load,
        total_elapsed_sec=float(sum(row.elapsed_sec for row in runs)),
        runs=tuple(runs),
    )
    payload = asdict(result)
    temporary = output_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(temporary, output_path)
    return payload


def _write_aggregate(output_dir: Path, rows: list[dict[str, Any]], config: dict[str, Any]) -> None:
    rows = sorted(rows, key=lambda row: int(row["instance_index"]))
    with (output_dir / "per_instance.csv").open("w", encoding="utf-8", newline="") as handle:
        fieldnames = [
            "instance_index",
            "source_index",
            "best_cost",
            "best_seed",
            "best_elapsed_sec",
            "best_n_routes",
            "max_route_load",
            "total_elapsed_sec",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row[key] for key in fieldnames})
    costs = np.asarray([float(row["best_cost"]) for row in rows], dtype=np.float64)
    summary = {
        "protocol": "agfn_cvrp1000_capacity50_hgs_continuous_euclidean_v1",
        "config": config,
        "num_completed": len(rows),
        "mean_cost": float(costs.mean()) if len(costs) else None,
        "std_cost": float(costs.std(ddof=1)) if len(costs) > 1 else 0.0,
        "min_cost": float(costs.min()) if len(costs) else None,
        "max_cost": float(costs.max()) if len(costs) else None,
        "sum_solver_wall_sec": float(sum(float(row["total_elapsed_sec"]) for row in rows)),
        "rows": rows,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate HGS-CVRP on the official AGFN CVRP1000 set.")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--library", type=Path, default=DEFAULT_LIBRARY)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--time-limit-sec", type=float, default=600.0)
    parser.add_argument("--replicates", type=int, default=3)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--max-instances", type=int, default=None)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--nb-iter", type=int, default=20000)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.time_limit_sec <= 0 or args.replicates < 1 or args.workers < 1:
        raise ValueError("time-limit-sec, replicates, and workers must be positive")
    dataset = args.dataset.resolve()
    library = args.library.resolve()
    if not dataset.exists() or not library.exists():
        raise FileNotFoundError(f"Missing dataset or HGS library: {dataset}, {library}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    instance_dir = args.output_dir / "instances"
    instance_dir.mkdir(parents=True, exist_ok=True)
    with np.load(dataset) as payload:
        depot = np.asarray(payload["depot"], dtype=np.float64)
        locs = np.asarray(payload["locs"], dtype=np.float64)
        demand = np.asarray(payload["demand"], dtype=np.float64)
        capacity = np.asarray(payload["capacity"], dtype=np.float64)
        source_index = np.asarray(payload["source_index"], dtype=np.int64)
    count = len(depot) if args.max_instances is None else min(len(depot), args.max_instances)
    if locs.shape[1:] != (1000, 2) or count < 1:
        raise ValueError(f"Unexpected CVRP1000 dataset shape: {locs.shape}")
    config = {
        "dataset": str(dataset),
        "dataset_sha256": _sha256(dataset),
        "library": str(library),
        "library_sha256": _sha256(library),
        "hgs_git_commit": "c06dd08ab3909f3da4fc31276c46e18eba690ffd",
        "distance": "continuous Euclidean on original float32 coordinates",
        "capacity": 50,
        "num_instances": count,
        "time_limit_sec_per_seed": float(args.time_limit_sec),
        "replicates_per_instance": int(args.replicates),
        "workers": int(args.workers),
        "base_seed": int(args.seed),
        "nb_iter_without_improvement": int(args.nb_iter),
        "fleet": "unspecified; HGS safety upper bound, variable nonempty route count",
    }
    tasks = []
    for idx in range(count):
        coords = np.concatenate([depot[idx : idx + 1], locs[idx]], axis=0)
        demands = np.concatenate([np.zeros(1, dtype=np.float64), demand[idx]], axis=0)
        tasks.append(
            {
                "instance_index": idx,
                "source_index": int(source_index[idx]),
                "coords": coords,
                "demands": demands,
                "capacity": int(round(float(capacity[idx]))),
                "base_seed": int(args.seed),
                "num_replicates": int(args.replicates),
                "time_limit_sec": float(args.time_limit_sec),
                "nb_iter": int(args.nb_iter),
                "library_path": str(library),
                "output_path": str(instance_dir / f"instance_{idx:03d}.json"),
            }
        )
    results = []
    with ProcessPoolExecutor(max_workers=min(args.workers, count)) as executor:
        futures = {executor.submit(_solve_instance, task): task["instance_index"] for task in tasks}
        for future in as_completed(futures):
            row = future.result()
            results.append(row)
            _write_aggregate(args.output_dir, results, config)
            print(
                f"[result] instance={row['instance_index']} cost={row['best_cost']:.9f} "
                f"routes={row['best_n_routes']} completed={len(results)}/{count}",
                flush=True,
            )
    _write_aggregate(args.output_dir, results, config)
    print(f"[done] output_dir={args.output_dir.resolve()}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
