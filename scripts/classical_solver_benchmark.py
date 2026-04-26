from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import json
import math
import os
import re
import subprocess
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SOLVER_SITE = REPO_ROOT / ".solver_site"
if SOLVER_SITE.is_dir():
    sys.path.insert(0, str(SOLVER_SITE))

os.environ.setdefault("PANDAS_NO_IMPORT_NUMEXPR", "1")
os.environ.setdefault("PANDAS_NO_IMPORT_BOTTLENECK", "1")
sys.modules.setdefault("numexpr", None)
sys.modules.setdefault("bottleneck", None)

import numpy as np

from ortools.sat.python import cp_model
from pyvrp import Client, Depot, ProblemData, VehicleType, solve as pyvrp_solve
from pyvrp.constants import MAX_VALUE as PYVRP_MAX_VALUE
from pyvrp.stop import MaxRuntime


@dataclass(frozen=True)
class ScenarioConfig:
    name: str
    problem: str
    size: int
    test_file: Path | None
    generated_count: int
    jssp_shape: str | None = None
    ffsp_stages: int | None = None
    ffsp_machines: int | None = None
    cp_sat_time_limit_s: float | None = None
    pyvrp_time_limit_s: float | None = None
    lkh_time_limit_s: float | None = None
    lkh_runs: int = 10
    solvers: tuple[str, ...] = ()


SCENARIOS: dict[str, ScenarioConfig] = {
    "tsp50": ScenarioConfig(
        name="tsp50",
        problem="tsp",
        size=50,
        test_file=REPO_ROOT / "data" / "tsp" / "tsp50_test_seed1234.npz",
        generated_count=10000,
        lkh_time_limit_s=60.0,
        lkh_runs=20,
        solvers=("lkh", "concorde"),
    ),
    "tsp100": ScenarioConfig(
        name="tsp100",
        problem="tsp",
        size=100,
        test_file=REPO_ROOT / "data" / "tsp" / "tsp100_test_seed1234.npz",
        generated_count=10000,
        lkh_time_limit_s=180.0,
        lkh_runs=20,
        solvers=("lkh", "concorde"),
    ),
    "cvrp50": ScenarioConfig(
        name="cvrp50",
        problem="cvrp",
        size=50,
        test_file=REPO_ROOT / "data" / "vrp" / "vrp50_test_seed1234.npz",
        generated_count=10000,
        pyvrp_time_limit_s=60.0,
        solvers=("pyvrp",),
    ),
    "cvrp100": ScenarioConfig(
        name="cvrp100",
        problem="cvrp",
        size=100,
        test_file=REPO_ROOT / "data" / "vrp" / "vrp100_test_seed1234.npz",
        generated_count=10000,
        pyvrp_time_limit_s=180.0,
        solvers=("pyvrp",),
    ),
    "ffsp50": ScenarioConfig(
        name="ffsp50",
        problem="ffsp",
        size=50,
        test_file=None,
        generated_count=1000,
        ffsp_stages=3,
        ffsp_machines=4,
        cp_sat_time_limit_s=180.0,
        solvers=("ortools_cp_sat", "sjf"),
    ),
    "ffsp100": ScenarioConfig(
        name="ffsp100",
        problem="ffsp",
        size=100,
        test_file=None,
        generated_count=1000,
        ffsp_stages=3,
        ffsp_machines=4,
        cp_sat_time_limit_s=600.0,
        solvers=("ortools_cp_sat", "sjf"),
    ),
    "jssp10x10": ScenarioConfig(
        name="jssp10x10",
        problem="jssp",
        size=10,
        test_file=None,
        generated_count=0,
        jssp_shape="10x10",
        cp_sat_time_limit_s=180.0,
        solvers=("ortools_cp_sat", "spt", "mor", "mwr"),
    ),
    "jssp15x15": ScenarioConfig(
        name="jssp15x15",
        problem="jssp",
        size=15,
        test_file=None,
        generated_count=0,
        jssp_shape="15x15",
        cp_sat_time_limit_s=600.0,
        solvers=("ortools_cp_sat", "spt", "mor", "mwr"),
    ),
}


@dataclass
class PerInstanceResult:
    scenario: str
    solver: str
    instance_id: str
    status: str
    objective: float | None
    elapsed_s: float
    notes: str = ""


@dataclass(frozen=True)
class SolverBatchSummary:
    scenario: str
    solver: str
    wall_clock_s: float
    workers: int
    solver_threads: int


_WARNED_DATA_FALLBACKS: set[str] = set()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch benchmark classical solvers for routing and scheduling scenarios."
    )
    parser.add_argument(
        "--scenarios",
        nargs="+",
        default=list(SCENARIOS.keys()),
        choices=list(SCENARIOS.keys()),
        help="Scenario names to evaluate.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "logs" / "classical_benchmark",
        help="Output directory for per-instance and summary CSV/JSON files.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1234,
        help="Seed used when benchmark datasets are generated on the fly.",
    )
    parser.add_argument(
        "--max-instances",
        type=int,
        default=None,
        help="Optional cap on the number of instances per scenario.",
    )
    parser.add_argument(
        "--tsp-solvers",
        nargs="+",
        default=None,
        choices=["lkh", "concorde"],
        help="Override TSP solver list.",
    )
    parser.add_argument(
        "--require-test-data",
        action="store_true",
        help="Fail instead of generating fallback routing instances when the expected test file is missing.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=os.cpu_count() or 1,
        help="Maximum logical CPUs to use for each independent solver batch.",
    )
    parser.add_argument(
        "--cp-sat-search-workers",
        type=int,
        default=1,
        help="OR-Tools CP-SAT threads per instance. Keep at 1 when maximizing outer parallelism across instances.",
    )
    return parser.parse_args()


def _tsplib_euc_2d_distance(a: np.ndarray, b: np.ndarray) -> int:
    return int(np.floor(np.linalg.norm(a - b) + 0.5))


def _cycle_length(coords: np.ndarray, tour: list[int]) -> int:
    cost = 0
    for idx in range(len(tour)):
        cost += _tsplib_euc_2d_distance(coords[tour[idx]], coords[tour[(idx + 1) % len(tour)]])
    return cost


def _parse_lkh_tour(tour_path: Path) -> list[int]:
    nodes: list[int] = []
    in_section = False
    for raw_line in tour_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if line == "TOUR_SECTION":
            in_section = True
            continue
        if not in_section:
            continue
        if line in {"-1", "EOF"}:
            break
        nodes.append(int(line) - 1)
    if not nodes:
        raise ValueError(f"Failed to parse TOUR_SECTION from {tour_path}")
    return nodes


def _routing_workdir(output_dir: Path, scenario_name: str, solver: str) -> Path:
    work_dir = output_dir / "artifacts" / scenario_name / solver
    work_dir.mkdir(parents=True, exist_ok=True)
    return work_dir


def _resolve_parallelism(args: argparse.Namespace, solver: str, instance_count: int) -> tuple[int, int]:
    if instance_count <= 0:
        return 1, 1

    requested_workers = max(1, int(args.workers))
    solver_threads = 1
    if solver == "ortools_cp_sat":
        solver_threads = max(1, int(args.cp_sat_search_workers))
        requested_workers = max(1, requested_workers // solver_threads)

    workers = max(1, min(instance_count, requested_workers))
    return workers, solver_threads


def _available_file(path: Path | None) -> Path | None:
    return path if path is not None and path.is_file() else None


def _warn_data_fallback(cfg: ScenarioConfig) -> None:
    message = (
        f"[warning] Missing benchmark file for {cfg.name}: {cfg.test_file}. "
        f"Falling back to generated instances, which is not directly comparable to neural-model test results."
    )
    if message not in _WARNED_DATA_FALLBACKS:
        print(message, file=sys.stderr, flush=True)
        _WARNED_DATA_FALLBACKS.add(message)


def _load_tsp_instances(
    cfg: ScenarioConfig,
    seed: int,
    max_instances: int | None,
    require_test_data: bool,
) -> list[np.ndarray]:
    dataset_path = _available_file(cfg.test_file)
    if dataset_path is not None:
        data = np.load(dataset_path)
        locs = np.asarray(data["locs"], dtype=np.float32)
    else:
        if require_test_data:
            raise FileNotFoundError(f"Missing required test file for {cfg.name}: {cfg.test_file}")
        _warn_data_fallback(cfg)
        rng = np.random.default_rng(seed)
        count = cfg.generated_count if max_instances is None else min(cfg.generated_count, max_instances)
        locs = rng.uniform(size=(count, cfg.size, 2)).astype(np.float32)
    if max_instances is not None:
        locs = locs[:max_instances]
    return [instance for instance in locs]


def _generate_vrp_data(dataset_size: int, vrp_size: int, seed: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    capacities = {
        10: 20.0,
        15: 25.0,
        20: 30.0,
        30: 33.0,
        40: 37.0,
        50: 40.0,
        60: 43.0,
        75: 45.0,
        100: 50.0,
        125: 55.0,
        150: 60.0,
        200: 70.0,
        500: 100.0,
        1000: 150.0,
    }
    return {
        "depot": rng.uniform(size=(dataset_size, 2)).astype(np.float32),
        "locs": rng.uniform(size=(dataset_size, vrp_size, 2)).astype(np.float32),
        "demand": rng.integers(1, 10, size=(dataset_size, vrp_size), endpoint=False).astype(np.float32),
        "capacity": np.full(dataset_size, capacities[vrp_size], dtype=np.float32),
    }


def _load_cvrp_instances(
    cfg: ScenarioConfig,
    seed: int,
    max_instances: int | None,
    require_test_data: bool,
) -> list[dict[str, np.ndarray]]:
    dataset_path = _available_file(cfg.test_file)
    if dataset_path is not None:
        data = dict(np.load(dataset_path))
    else:
        if require_test_data:
            raise FileNotFoundError(f"Missing required test file for {cfg.name}: {cfg.test_file}")
        _warn_data_fallback(cfg)
        count = cfg.generated_count if max_instances is None else min(cfg.generated_count, max_instances)
        data = _generate_vrp_data(count, cfg.size, seed)
    total = data["locs"].shape[0] if max_instances is None else min(max_instances, data["locs"].shape[0])
    return [
        {
            "depot": np.asarray(data["depot"][idx], dtype=np.float32),
            "locs": np.asarray(data["locs"][idx], dtype=np.float32),
            "demand": np.asarray(data["demand"][idx], dtype=np.float32),
            "capacity": float(data["capacity"][idx]),
        }
        for idx in range(total)
    ]


def _load_ffsp_instances(cfg: ScenarioConfig, seed: int, max_instances: int | None) -> list[np.ndarray]:
    assert cfg.ffsp_stages is not None and cfg.ffsp_machines is not None
    count = cfg.generated_count if max_instances is None else min(cfg.generated_count, max_instances)
    rng = np.random.default_rng(seed)
    run_time = rng.integers(
        low=2,
        high=10,
        size=(count, cfg.size, cfg.ffsp_stages * cfg.ffsp_machines),
        endpoint=False,
    )
    return [instance.astype(np.int32) for instance in run_time]


def _load_jssp_instances(cfg: ScenarioConfig, max_instances: int | None) -> list[Path]:
    assert cfg.jssp_shape is not None
    root = REPO_ROOT / "data" / "jssp_bopo" / "validation"
    files = sorted(root.glob(f"{cfg.jssp_shape}_*.jsp"))
    if max_instances is not None:
        files = files[:max_instances]
    return files


def run_tsp_lkh(
    coords: np.ndarray,
    seed: int,
    instance_id: str,
    scenario_name: str,
    time_limit_sec: float,
    runs: int,
    output_dir: Path,
) -> PerInstanceResult:
    lkh_path = REPO_ROOT / "tools" / "LKH-3.0.13" / "LKH"
    if not lkh_path.is_file():
        return PerInstanceResult(scenario_name, "lkh", instance_id, "missing", None, 0.0, notes="LKH missing")

    work_dir = _routing_workdir(output_dir, scenario_name, "lkh")
    tsp_path = work_dir / f"{instance_id}.tsp"
    par_path = work_dir / f"{instance_id}.par"
    tour_path = work_dir / f"{instance_id}.tour"

    scaled = np.rint(coords * 100_000).astype(int)
    lines = [
        f"NAME : {instance_id}",
        "TYPE : TSP",
        f"DIMENSION : {scaled.shape[0]}",
        "EDGE_WEIGHT_TYPE : EUC_2D",
        "NODE_COORD_SECTION",
    ]
    for idx, (x_coord, y_coord) in enumerate(scaled, start=1):
        lines.append(f"{idx} {int(x_coord)} {int(y_coord)}")
    lines.append("EOF")
    tsp_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    par_lines = [
        f"PROBLEM_FILE = {tsp_path}",
        f"TOUR_FILE = {tour_path}",
        f"TIME_LIMIT = {max(1, int(math.ceil(time_limit_sec)))}",
        f"SEED = {seed}",
        f"RUNS = {runs}",
        "TRACE_LEVEL = 0",
    ]
    par_path.write_text("\n".join(par_lines) + "\n", encoding="utf-8")

    t0 = time.perf_counter()
    completed = subprocess.run(
        [str(lkh_path), str(par_path)],
        cwd=work_dir,
        capture_output=True,
        text=True,
        check=True,
    )
    elapsed = time.perf_counter() - t0
    objective = float(_cycle_length(scaled, _parse_lkh_tour(tour_path)))
    notes = completed.stdout.strip().splitlines()[-1] if completed.stdout.strip() else ""
    return PerInstanceResult(scenario_name, "lkh", instance_id, "ok", objective, elapsed, notes=notes[:300])


def run_tsp_concorde(
    coords: np.ndarray,
    instance_id: str,
    scenario_name: str,
    output_dir: Path,
) -> PerInstanceResult:
    concorde_path = REPO_ROOT / "tools" / "concorde" / "TSP" / "concorde"
    if not concorde_path.is_file():
        return PerInstanceResult(scenario_name, "concorde", instance_id, "missing", None, 0.0, notes="Concorde missing")

    work_dir = _routing_workdir(output_dir, scenario_name, "concorde")
    tsp_path = work_dir / f"{instance_id}.tsp"
    sol_path = work_dir / f"{instance_id}.sol"
    scaled = np.rint(coords * 100_000).astype(int)
    lines = [
        f"NAME : {instance_id}",
        "TYPE : TSP",
        f"DIMENSION : {scaled.shape[0]}",
        "EDGE_WEIGHT_TYPE : EUC_2D",
        "NODE_COORD_SECTION",
    ]
    for idx, (x_coord, y_coord) in enumerate(scaled, start=1):
        lines.append(f"{idx} {int(x_coord)} {int(y_coord)}")
    lines.append("EOF")
    tsp_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    t0 = time.perf_counter()
    completed = subprocess.run(
        [str(concorde_path), "-x", "-o", str(sol_path), str(tsp_path)],
        cwd=work_dir,
        capture_output=True,
        text=True,
        check=False,
    )
    elapsed = time.perf_counter() - t0
    match = re.search(r"Optimal Solution:\s*([0-9]+(?:\.[0-9]+)?)", completed.stdout)
    if match is None:
        return PerInstanceResult(
            scenario_name,
            "concorde",
            instance_id,
            f"error_{completed.returncode}",
            None,
            elapsed,
            notes=((completed.stderr or completed.stdout).strip()[:500]),
        )
    objective = float(match.group(1))
    notes = f"returncode={completed.returncode}"
    return PerInstanceResult(scenario_name, "concorde", instance_id, "ok", objective, elapsed, notes=notes)


def _build_pyvrp_problem(instance: dict[str, np.ndarray]) -> ProblemData:
    depot_coord = np.asarray(instance["depot"], dtype=np.float32)
    locs = np.asarray(instance["locs"], dtype=np.float32)
    demand = np.asarray(instance["demand"], dtype=np.float32)
    capacity = int(round(float(instance["capacity"])))

    coords = np.concatenate([depot_coord[None, :], locs], axis=0)
    scaled = np.rint(coords * 100_000).astype(int)
    matrix = np.zeros((scaled.shape[0], scaled.shape[0]), dtype=int)
    for src in range(scaled.shape[0]):
        for dst in range(scaled.shape[0]):
            matrix[src, dst] = _tsplib_euc_2d_distance(scaled[src], scaled[dst])

    depot = Depot(x=int(scaled[0, 0]), y=int(scaled[0, 1]))
    clients = [
        Client(
            x=int(scaled[idx, 0]),
            y=int(scaled[idx, 1]),
            delivery=[int(round(float(demand[idx - 1])))],
            pickup=[0],
            service_duration=0,
            tw_early=0,
            tw_late=PYVRP_MAX_VALUE,
        )
        for idx in range(1, scaled.shape[0])
    ]
    vehicle_type = VehicleType(
        num_available=scaled.shape[0] - 1,
        capacity=[capacity],
        max_distance=PYVRP_MAX_VALUE,
        tw_early=0,
        tw_late=PYVRP_MAX_VALUE,
    )
    return ProblemData(clients, [depot], [vehicle_type], [matrix], [matrix])


def run_cvrp_pyvrp(
    instance: dict[str, np.ndarray],
    instance_id: str,
    scenario_name: str,
    time_limit_sec: float,
) -> PerInstanceResult:
    t0 = time.perf_counter()
    result = pyvrp_solve(_build_pyvrp_problem(instance), MaxRuntime(time_limit_sec))
    elapsed = time.perf_counter() - t0
    return PerInstanceResult(
        scenario_name,
        "pyvrp",
        instance_id,
        "ok",
        float(result.cost()),
        elapsed,
        notes=f"routes={len(result.best.routes())}",
    )


def ffsp_sjf_makespan(run_time: np.ndarray, num_stage: int, num_machine: int) -> int:
    num_job = run_time.shape[0]
    prev_finish = np.zeros(num_job, dtype=int)
    for stage in range(num_stage):
        stage_times = run_time[:, stage * num_machine : (stage + 1) * num_machine]
        machine_ready = np.zeros(num_machine, dtype=int)
        stage_finish = np.zeros(num_job, dtype=int)
        order = sorted(range(num_job), key=lambda job: (int(stage_times[job].min()), int(stage_times[job].sum()), job))
        for job in order:
            best_machine = min(
                range(num_machine),
                key=lambda machine: max(machine_ready[machine], prev_finish[job]) + int(stage_times[job, machine]),
            )
            start = max(machine_ready[best_machine], prev_finish[job])
            end = start + int(stage_times[job, best_machine])
            machine_ready[best_machine] = end
            stage_finish[job] = end
        prev_finish = stage_finish
    return int(prev_finish.max())


def ffsp_cp_sat_makespan(
    run_time: np.ndarray,
    num_stage: int,
    num_machine: int,
    time_limit_sec: float,
    num_search_workers: int,
) -> tuple[str, int | None]:
    num_job = run_time.shape[0]
    horizon = int(run_time.sum())
    model = cp_model.CpModel()
    start: dict[tuple[int, int], cp_model.IntVar] = {}
    end: dict[tuple[int, int], cp_model.IntVar] = {}
    machine_intervals: dict[tuple[int, int], list[cp_model.IntervalVar]] = defaultdict(list)

    for job in range(num_job):
        for stage in range(num_stage):
            start[(job, stage)] = model.NewIntVar(0, horizon, f"start_{job}_{stage}")
            end[(job, stage)] = model.NewIntVar(0, horizon, f"end_{job}_{stage}")
            selectors = []
            for machine in range(num_machine):
                duration = int(run_time[job, stage * num_machine + machine])
                selected = model.NewBoolVar(f"sel_{job}_{stage}_{machine}")
                local_start = model.NewIntVar(0, horizon, f"ls_{job}_{stage}_{machine}")
                local_end = model.NewIntVar(0, horizon, f"le_{job}_{stage}_{machine}")
                interval = model.NewOptionalIntervalVar(
                    local_start,
                    duration,
                    local_end,
                    selected,
                    f"int_{job}_{stage}_{machine}",
                )
                model.Add(local_start == start[(job, stage)]).OnlyEnforceIf(selected)
                model.Add(local_end == end[(job, stage)]).OnlyEnforceIf(selected)
                machine_intervals[(stage, machine)].append(interval)
                selectors.append(selected)
            model.AddExactlyOne(selectors)
            if stage > 0:
                model.Add(start[(job, stage)] >= end[(job, stage - 1)])

    for intervals in machine_intervals.values():
        model.AddNoOverlap(intervals)

    makespan = model.NewIntVar(0, horizon, "makespan")
    for job in range(num_job):
        model.Add(makespan >= end[(job, num_stage - 1)])
    model.Minimize(makespan)

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = float(time_limit_sec)
    solver.parameters.num_search_workers = max(1, int(num_search_workers))
    status = solver.Solve(model)
    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return solver.StatusName(status).lower(), int(solver.Value(makespan))
    return solver.StatusName(status).lower(), None


def parse_jssp_file(path: Path) -> tuple[int, list[list[tuple[int, int]]]]:
    rows = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        stripped = raw_line.strip()
        if stripped:
            rows.append([int(token) for token in stripped.split()])
    num_jobs, num_machines = rows[0][0], rows[0][1]
    jobs = []
    for line in rows[1 : 1 + num_jobs]:
        operations = []
        for idx in range(0, len(line), 2):
            machine = int(line[idx])
            duration = int(line[idx + 1])
            operations.append((machine - 1 if machine >= 1 else machine, duration))
        jobs.append(operations)
    return num_machines, jobs


def jssp_dispatch_makespan(jobs: list[list[tuple[int, int]]], num_machines: int, rule: str) -> int:
    machine_ready = [0] * num_machines
    job_ready = [0] * len(jobs)
    next_op = [0] * len(jobs)
    total_ops = sum(len(job) for job in jobs)
    scheduled_ops = 0

    while scheduled_ops < total_ops:
        candidates = []
        for job_idx, operations in enumerate(jobs):
            op_idx = next_op[job_idx]
            if op_idx >= len(operations):
                continue
            machine, duration = operations[op_idx]
            remaining_ops = len(operations) - op_idx
            remaining_work = sum(d for _, d in operations[op_idx:])
            candidates.append((job_idx, machine, duration, remaining_ops, remaining_work))

        if rule == "spt":
            chosen = min(candidates, key=lambda item: (item[2], item[0]))
        elif rule == "mor":
            chosen = max(candidates, key=lambda item: (item[3], -item[0]))
        elif rule == "mwr":
            chosen = max(candidates, key=lambda item: (item[4], -item[0]))
        else:
            raise ValueError(f"Unsupported dispatching rule: {rule}")

        job_idx, machine, duration, _, _ = chosen
        start = max(job_ready[job_idx], machine_ready[machine])
        end = start + duration
        job_ready[job_idx] = end
        machine_ready[machine] = end
        next_op[job_idx] += 1
        scheduled_ops += 1

    return max(job_ready)


def jssp_cp_sat_makespan(
    jobs: list[list[tuple[int, int]]],
    num_machines: int,
    time_limit_sec: float,
    num_search_workers: int,
) -> tuple[str, int | None]:
    horizon = sum(duration for job in jobs for _, duration in job)
    model = cp_model.CpModel()
    intervals: dict[int, list[cp_model.IntervalVar]] = {machine: [] for machine in range(num_machines)}
    task_ends = []

    for job_idx, job in enumerate(jobs):
        prev_end = None
        for op_idx, (machine, duration) in enumerate(job):
            start = model.NewIntVar(0, horizon, f"start_{job_idx}_{op_idx}")
            end = model.NewIntVar(0, horizon, f"end_{job_idx}_{op_idx}")
            interval = model.NewIntervalVar(start, duration, end, f"int_{job_idx}_{op_idx}")
            intervals[machine].append(interval)
            if prev_end is not None:
                model.Add(start >= prev_end)
            prev_end = end
        if prev_end is not None:
            task_ends.append(prev_end)

    for machine_intervals in intervals.values():
        model.AddNoOverlap(machine_intervals)

    makespan = model.NewIntVar(0, horizon, "makespan")
    model.AddMaxEquality(makespan, task_ends)
    model.Minimize(makespan)

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = float(time_limit_sec)
    solver.parameters.num_search_workers = max(1, int(num_search_workers))
    status = solver.Solve(model)
    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return solver.StatusName(status).lower(), int(solver.Value(makespan))
    return solver.StatusName(status).lower(), None


def _run_single_tsp_task(
    solver: str,
    cfg: ScenarioConfig,
    coords: np.ndarray,
    idx: int,
    seed: int,
    output_dir: Path,
) -> PerInstanceResult:
    instance_id = f"{cfg.name}_{idx:05d}"
    if solver == "lkh":
        return run_tsp_lkh(
            coords,
            seed=seed + idx,
            instance_id=instance_id,
            scenario_name=cfg.name,
            time_limit_sec=float(cfg.lkh_time_limit_s or 30.0),
            runs=int(cfg.lkh_runs),
            output_dir=output_dir,
        )
    if solver == "concorde":
        return run_tsp_concorde(coords, instance_id, cfg.name, output_dir)
    raise ValueError(f"Unsupported TSP solver: {solver}")


def _run_single_cvrp_task(
    cfg: ScenarioConfig,
    instance: dict[str, np.ndarray],
    idx: int,
) -> PerInstanceResult:
    return run_cvrp_pyvrp(
        instance,
        f"{cfg.name}_{idx:05d}",
        cfg.name,
        time_limit_sec=float(cfg.pyvrp_time_limit_s or 30.0),
    )


def _run_single_ffsp_task(
    solver: str,
    cfg: ScenarioConfig,
    run_time: np.ndarray,
    idx: int,
    cp_sat_search_workers: int,
) -> PerInstanceResult:
    assert cfg.ffsp_stages is not None and cfg.ffsp_machines is not None
    instance_id = f"{cfg.name}_{idx:05d}"
    if solver == "sjf":
        t0 = time.perf_counter()
        objective = ffsp_sjf_makespan(run_time, cfg.ffsp_stages, cfg.ffsp_machines)
        return PerInstanceResult(cfg.name, "sjf", instance_id, "ok", float(objective), time.perf_counter() - t0)
    if solver == "ortools_cp_sat":
        assert cfg.cp_sat_time_limit_s is not None
        t0 = time.perf_counter()
        status, objective = ffsp_cp_sat_makespan(
            run_time,
            cfg.ffsp_stages,
            cfg.ffsp_machines,
            cfg.cp_sat_time_limit_s,
            cp_sat_search_workers,
        )
        return PerInstanceResult(
            cfg.name,
            "ortools_cp_sat",
            instance_id,
            status,
            float(objective) if objective is not None else None,
            time.perf_counter() - t0,
        )
    raise ValueError(f"Unsupported FFSP solver: {solver}")


def _run_single_jssp_task(
    solver: str,
    cfg: ScenarioConfig,
    path: Path,
    cp_sat_search_workers: int,
) -> PerInstanceResult:
    assert cfg.cp_sat_time_limit_s is not None
    num_machines, jobs = parse_jssp_file(path)
    if solver in {"spt", "mor", "mwr"}:
        t0 = time.perf_counter()
        objective = jssp_dispatch_makespan(jobs, num_machines, solver)
        return PerInstanceResult(cfg.name, solver, path.name, "ok", float(objective), time.perf_counter() - t0)
    if solver == "ortools_cp_sat":
        t0 = time.perf_counter()
        status, objective = jssp_cp_sat_makespan(jobs, num_machines, cfg.cp_sat_time_limit_s, cp_sat_search_workers)
        return PerInstanceResult(
            cfg.name,
            solver,
            path.name,
            status,
            float(objective) if objective is not None else None,
            time.perf_counter() - t0,
        )
    raise ValueError(f"Unsupported JSSP solver: {solver}")


def _execute_parallel(
    solver: str,
    tasks: list[tuple],
    worker_fn,
    args: argparse.Namespace,
) -> tuple[list[PerInstanceResult], SolverBatchSummary]:
    workers, solver_threads = _resolve_parallelism(args, solver, len(tasks))
    t0 = time.perf_counter()
    rows: list[PerInstanceResult] = []
    if workers == 1:
        for task in tasks:
            rows.append(worker_fn(*task))
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(worker_fn, *task) for task in tasks]
            for future in as_completed(futures):
                rows.append(future.result())
    wall_clock = time.perf_counter() - t0
    rows.sort(key=lambda row: row.instance_id)
    return rows, SolverBatchSummary(
        scenario=rows[0].scenario if rows else "",
        solver=solver,
        wall_clock_s=wall_clock,
        workers=workers,
        solver_threads=solver_threads,
    )


def benchmark_scenario(
    cfg: ScenarioConfig,
    args: argparse.Namespace,
) -> tuple[list[PerInstanceResult], list[SolverBatchSummary]]:
    scenario_rows: list[PerInstanceResult] = []
    summaries: list[SolverBatchSummary] = []

    if cfg.problem == "tsp":
        solvers = tuple(args.tsp_solvers) if args.tsp_solvers is not None else cfg.solvers
        instances = _load_tsp_instances(cfg, args.seed, args.max_instances, args.require_test_data)
        for solver in solvers:
            tasks = [(solver, cfg, coords, idx, args.seed, args.output_dir) for idx, coords in enumerate(instances)]
            print(f"[{cfg.name}/{solver}] launching {len(tasks)} instances", flush=True)
            rows, summary = _execute_parallel(solver, tasks, _run_single_tsp_task, args)
            scenario_rows.extend(rows)
            summaries.append(summary)
        return scenario_rows, summaries

    if cfg.problem == "cvrp":
        instances = _load_cvrp_instances(cfg, args.seed, args.max_instances, args.require_test_data)
        tasks = [(cfg, instance, idx) for idx, instance in enumerate(instances)]
        print(f"[{cfg.name}/pyvrp] launching {len(tasks)} instances", flush=True)
        rows, summary = _execute_parallel("pyvrp", tasks, _run_single_cvrp_task, args)
        scenario_rows.extend(rows)
        summaries.append(summary)
        return scenario_rows, summaries

    if cfg.problem == "ffsp":
        instances = _load_ffsp_instances(cfg, args.seed, args.max_instances)
        for solver in cfg.solvers:
            tasks = [(solver, cfg, run_time, idx, args.cp_sat_search_workers) for idx, run_time in enumerate(instances)]
            print(f"[{cfg.name}/{solver}] launching {len(tasks)} instances", flush=True)
            rows, summary = _execute_parallel(solver, tasks, _run_single_ffsp_task, args)
            scenario_rows.extend(rows)
            summaries.append(summary)
        return scenario_rows, summaries

    if cfg.problem == "jssp":
        files = _load_jssp_instances(cfg, args.max_instances)
        for solver in cfg.solvers:
            tasks = [(solver, cfg, path, args.cp_sat_search_workers) for path in files]
            print(f"[{cfg.name}/{solver}] launching {len(tasks)} instances", flush=True)
            rows, summary = _execute_parallel(solver, tasks, _run_single_jssp_task, args)
            scenario_rows.extend(rows)
            summaries.append(summary)
        return scenario_rows, summaries

    raise ValueError(f"Unsupported scenario problem: {cfg.problem}")


def write_outputs(rows: list[PerInstanceResult], batch_summaries: list[SolverBatchSummary], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    per_instance_csv = output_dir / "per_instance.csv"
    summary_csv = output_dir / "summary.csv"
    summary_json = output_dir / "summary.json"

    with per_instance_csv.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=["scenario", "solver", "instance_id", "status", "objective", "elapsed_s", "notes"],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row.__dict__)

    grouped: dict[tuple[str, str], list[PerInstanceResult]] = defaultdict(list)
    for row in rows:
        grouped[(row.scenario, row.solver)].append(row)
    batch_lookup = {(row.scenario, row.solver): row for row in batch_summaries}

    summary_records = []
    for (scenario, solver), solver_rows in sorted(grouped.items()):
        ok_rows = [row for row in solver_rows if row.status == "ok"]
        objective_rows = [row for row in ok_rows if row.objective is not None]
        sum_instance_elapsed = float(sum(row.elapsed_s for row in solver_rows))
        batch_summary = batch_lookup.get((scenario, solver))
        total_elapsed = batch_summary.wall_clock_s if batch_summary is not None else sum_instance_elapsed
        summary_records.append(
            {
                "scenario": scenario,
                "solver": solver,
                "count": len(solver_rows),
                "ok_count": len(ok_rows),
                "total_elapsed_s": total_elapsed,
                "sum_instance_elapsed_s": sum_instance_elapsed,
                "avg_elapsed_s": (sum_instance_elapsed / len(solver_rows)) if solver_rows else None,
                "avg_objective_ok": (
                    float(sum(row.objective for row in objective_rows) / len(objective_rows))
                    if objective_rows
                    else None
                ),
                "best_objective_ok": (float(min(row.objective for row in objective_rows)) if objective_rows else None),
                "workers": batch_summary.workers if batch_summary is not None else 1,
                "solver_threads": batch_summary.solver_threads if batch_summary is not None else 1,
            }
        )

    with summary_csv.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "scenario",
                "solver",
                "count",
                "ok_count",
                "total_elapsed_s",
                "sum_instance_elapsed_s",
                "avg_elapsed_s",
                "avg_objective_ok",
                "best_objective_ok",
                "workers",
                "solver_threads",
            ],
        )
        writer.writeheader()
        for record in summary_records:
            writer.writerow(record)

    summary_json.write_text(
        json.dumps(
            {
                "records": summary_records,
                "created_at_epoch_s": time.time(),
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def main() -> int:
    args = parse_args()
    rows: list[PerInstanceResult] = []
    batch_summaries: list[SolverBatchSummary] = []
    for scenario_name in args.scenarios:
        cfg = SCENARIOS[scenario_name]
        scenario_rows, scenario_batch_summaries = benchmark_scenario(cfg, args)
        rows.extend(scenario_rows)
        batch_summaries.extend(scenario_batch_summaries)
        print(f"[{scenario_name}] finished {len(scenario_rows)} solver-runs", flush=True)

    write_outputs(rows, batch_summaries, args.output_dir.resolve())
    print(f"Wrote benchmark outputs to {args.output_dir.resolve()}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
