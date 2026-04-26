from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SOLVER_SITE = REPO_ROOT / ".solver_site"
if SOLVER_SITE.is_dir():
    sys.path.insert(0, str(SOLVER_SITE))

# Avoid pandas pulling binary extensions from the host environment that may be
# compiled against an incompatible NumPy ABI.
os.environ.setdefault("PANDAS_NO_IMPORT_NUMEXPR", "1")
os.environ.setdefault("PANDAS_NO_IMPORT_BOTTLENECK", "1")
sys.modules.setdefault("numexpr", None)
sys.modules.setdefault("bottleneck", None)

import numpy as np

from ortools.sat.python import cp_model
from pyvrp import Client, Depot, ProblemData, VehicleType, solve as pyvrp_solve
from pyvrp.constants import MAX_VALUE as PYVRP_MAX_VALUE
from pyvrp.stop import MaxRuntime


@dataclass
class ResultRow:
    problem: str
    solver: str
    status: str
    objective: float | None
    elapsed_s: float
    instance: str
    notes: str = ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run lightweight classical-solver smoke tests and record solve times."
    )
    parser.add_argument(
        "--problems",
        nargs="+",
        default=["tsp", "cvrp", "ffsp", "jssp"],
        choices=["tsp", "cvrp", "ffsp", "jssp"],
        help="Problems to execute.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "logs" / "solver_smoke",
        help="Directory for CSV/JSON output and temporary artifacts.",
    )
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--tsp-size", type=int, default=20)
    parser.add_argument(
        "--tsp-solvers",
        nargs="+",
        default=["lkh", "concorde"],
        choices=["lkh", "concorde"],
        help="TSP solvers to try. Unavailable solvers are skipped.",
    )
    parser.add_argument("--cvrp-size", type=int, default=20)
    parser.add_argument("--ffsp-jobs", type=int, default=6)
    parser.add_argument("--ffsp-stages", type=int, default=3)
    parser.add_argument("--ffsp-machines", type=int, default=2)
    parser.add_argument(
        "--jssp-instance",
        type=Path,
        default=REPO_ROOT / "data" / "jssp_bopo" / "LA" / "la01.jsp",
        help="JSSP instance file to use for smoke testing.",
    )
    parser.add_argument("--time-limit-sec", type=float, default=5.0)
    return parser.parse_args()


def _tsplib_euc_2d_distance(a: np.ndarray, b: np.ndarray) -> int:
    return int(np.floor(np.linalg.norm(a - b) + 0.5))


def _cycle_length(coords: np.ndarray, tour: list[int]) -> int:
    cost = 0
    for idx in range(len(tour)):
        src = coords[tour[idx]]
        dst = coords[tour[(idx + 1) % len(tour)]]
        cost += _tsplib_euc_2d_distance(src, dst)
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


def run_tsp_lkh(seed: int, size: int, time_limit_sec: float, output_dir: Path) -> ResultRow:
    lkh_path = REPO_ROOT / "tools" / "LKH-3.0.13" / "LKH"
    if not lkh_path.is_file():
        raise FileNotFoundError(f"Missing LKH executable: {lkh_path}")

    rng = np.random.default_rng(seed)
    coords = rng.integers(0, 10_000, size=(size, 2), endpoint=False)

    work_dir = output_dir / "artifacts" / "tsp_lkh"
    work_dir.mkdir(parents=True, exist_ok=True)
    tsp_path = work_dir / "smoke.tsp"
    par_path = work_dir / "smoke.par"
    tour_path = work_dir / "smoke.tour"

    tsp_lines = [
        "NAME : smoke_tsp",
        "TYPE : TSP",
        f"DIMENSION : {size}",
        "EDGE_WEIGHT_TYPE : EUC_2D",
        "NODE_COORD_SECTION",
    ]
    for idx, (x_coord, y_coord) in enumerate(coords, start=1):
        tsp_lines.append(f"{idx} {int(x_coord)} {int(y_coord)}")
    tsp_lines.append("EOF")
    tsp_path.write_text("\n".join(tsp_lines) + "\n", encoding="utf-8")

    par_lines = [
        f"PROBLEM_FILE = {tsp_path}",
        f"TOUR_FILE = {tour_path}",
        f"TIME_LIMIT = {max(1, int(math.ceil(time_limit_sec)))}",
        f"SEED = {seed}",
        "RUNS = 1",
        "TRACE_LEVEL = 1",
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
    tour = _parse_lkh_tour(tour_path)
    objective = float(_cycle_length(coords, tour))
    return ResultRow(
        problem="tsp",
        solver="lkh",
        status="ok",
        objective=objective,
        elapsed_s=elapsed,
        instance=f"random_{size}",
        notes=completed.stdout.strip().splitlines()[-1] if completed.stdout.strip() else "",
    )


def run_tsp_concorde(seed: int, size: int, output_dir: Path) -> ResultRow:
    concorde_path = REPO_ROOT / "tools" / "concorde" / "TSP" / "concorde"
    if not concorde_path.is_file():
        raise FileNotFoundError(f"Missing Concorde executable: {concorde_path}")

    rng = np.random.default_rng(seed)
    coords = rng.integers(0, 10_000, size=(size, 2), endpoint=False)

    work_dir = output_dir / "artifacts" / "tsp_concorde"
    work_dir.mkdir(parents=True, exist_ok=True)
    tsp_path = work_dir / "smoke.tsp"
    sol_path = work_dir / "smoke.sol"

    tsp_lines = [
        "NAME : smoke_tsp",
        "TYPE : TSP",
        f"DIMENSION : {size}",
        "EDGE_WEIGHT_TYPE : EUC_2D",
        "NODE_COORD_SECTION",
    ]
    for idx, (x_coord, y_coord) in enumerate(coords, start=1):
        tsp_lines.append(f"{idx} {int(x_coord)} {int(y_coord)}")
    tsp_lines.append("EOF")
    tsp_path.write_text("\n".join(tsp_lines) + "\n", encoding="utf-8")

    t0 = time.perf_counter()
    completed = subprocess.run(
        [str(concorde_path), "-x", "-o", str(sol_path), str(tsp_path)],
        cwd=work_dir,
        capture_output=True,
        text=True,
        check=False,
    )
    elapsed = time.perf_counter() - t0
    objective = None
    match = re.search(r"Optimal Solution:\s*([0-9]+(?:\.[0-9]+)?)", completed.stdout)
    if match:
        objective = float(match.group(1))

    if completed.returncode != 0 and objective is None:
        raise subprocess.CalledProcessError(
            completed.returncode,
            completed.args,
            output=completed.stdout,
            stderr=completed.stderr,
        )

    return ResultRow(
        problem="tsp",
        solver="concorde",
        status="ok",
        objective=objective,
        elapsed_s=elapsed,
        instance=f"random_{size}",
        notes=(
            f"returncode={completed.returncode}; "
            + (completed.stdout.strip().splitlines()[-1] if completed.stdout.strip() else "")
        ),
    )


def _build_pyvrp_problem(seed: int, size: int) -> ProblemData:
    rng = np.random.default_rng(seed)
    coords = rng.integers(0, 10_000, size=(size + 1, 2), endpoint=False)
    demands = rng.integers(1, 10, size=size, endpoint=False)
    capacity = int(max(20, math.ceil(demands.sum() / 3)))

    matrix = np.zeros((size + 1, size + 1), dtype=int)
    for src in range(size + 1):
        for dst in range(size + 1):
            matrix[src, dst] = _tsplib_euc_2d_distance(coords[src], coords[dst])

    depot = Depot(x=int(coords[0, 0]), y=int(coords[0, 1]))
    clients = [
        Client(
            x=int(coords[idx, 0]),
            y=int(coords[idx, 1]),
            delivery=[int(demands[idx - 1])],
            pickup=[0],
            service_duration=0,
            tw_early=0,
            tw_late=PYVRP_MAX_VALUE,
        )
        for idx in range(1, size + 1)
    ]
    vehicle_type = VehicleType(
        num_available=size,
        capacity=[capacity],
        max_distance=PYVRP_MAX_VALUE,
        tw_early=0,
        tw_late=PYVRP_MAX_VALUE,
    )
    return ProblemData(clients, [depot], [vehicle_type], [matrix], [matrix])


def run_cvrp_pyvrp(seed: int, size: int, time_limit_sec: float) -> ResultRow:
    data = _build_pyvrp_problem(seed, size)
    t0 = time.perf_counter()
    result = pyvrp_solve(data, MaxRuntime(time_limit_sec))
    elapsed = time.perf_counter() - t0
    return ResultRow(
        problem="cvrp",
        solver="pyvrp",
        status="ok",
        objective=float(result.cost()),
        elapsed_s=elapsed,
        instance=f"random_{size}",
        notes=f"routes={len(result.best.routes())}",
    )


def _generate_ffsp_times(seed: int, jobs: int, stages: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(1, 20, size=(jobs, stages), endpoint=False)


def ffsp_sjf_makespan(times: np.ndarray, machines: int) -> int:
    jobs, stages = times.shape
    prev_finish = [0] * jobs
    for stage in range(stages):
        machine_ready = [0] * machines
        stage_finish = [0] * jobs
        order = sorted(range(jobs), key=lambda job: (int(times[job, stage]), job))
        for job in order:
            best_machine = min(
                range(machines),
                key=lambda machine: max(machine_ready[machine], prev_finish[job]) + int(times[job, stage]),
            )
            start = max(machine_ready[best_machine], prev_finish[job])
            end = start + int(times[job, stage])
            machine_ready[best_machine] = end
            stage_finish[job] = end
        prev_finish = stage_finish
    return max(prev_finish)


def ffsp_cp_sat_makespan(times: np.ndarray, machines: int, time_limit_sec: float) -> tuple[str, int | None]:
    jobs, stages = times.shape
    horizon = int(times.sum())
    model = cp_model.CpModel()
    start: dict[tuple[int, int], cp_model.IntVar] = {}
    end: dict[tuple[int, int], cp_model.IntVar] = {}
    machine_intervals: dict[tuple[int, int], list[cp_model.IntervalVar]] = {}

    for stage in range(stages):
        for machine in range(machines):
            machine_intervals[(stage, machine)] = []

    for job in range(jobs):
        for stage in range(stages):
            duration = int(times[job, stage])
            start[(job, stage)] = model.NewIntVar(0, horizon, f"start_{job}_{stage}")
            end[(job, stage)] = model.NewIntVar(0, horizon, f"end_{job}_{stage}")
            selectors = []
            for machine in range(machines):
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
    for job in range(jobs):
        model.Add(makespan >= end[(job, stages - 1)])
    model.Minimize(makespan)

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = float(time_limit_sec)
    status = solver.Solve(model)
    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return solver.StatusName(status), int(solver.Value(makespan))
    return solver.StatusName(status), None


def run_ffsp(seed: int, jobs: int, stages: int, machines: int, time_limit_sec: float) -> list[ResultRow]:
    times = _generate_ffsp_times(seed, jobs, stages)
    rows: list[ResultRow] = []

    t0 = time.perf_counter()
    heuristic_obj = ffsp_sjf_makespan(times, machines)
    rows.append(
        ResultRow(
            problem="ffsp",
            solver="sjf",
            status="ok",
            objective=float(heuristic_obj),
            elapsed_s=time.perf_counter() - t0,
            instance=f"random_{jobs}j_{stages}s_{machines}m",
        )
    )

    t0 = time.perf_counter()
    status, objective = ffsp_cp_sat_makespan(times, machines, time_limit_sec)
    rows.append(
        ResultRow(
            problem="ffsp",
            solver="ortools_cp_sat",
            status=status.lower(),
            objective=float(objective) if objective is not None else None,
            elapsed_s=time.perf_counter() - t0,
            instance=f"random_{jobs}j_{stages}s_{machines}m",
        )
    )
    return rows


def parse_jssp_file(path: Path) -> tuple[int, list[list[tuple[int, int]]]]:
    rows = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        stripped = raw_line.strip()
        if stripped:
            rows.append([int(token) for token in stripped.split()])
    num_jobs, num_machines = rows[0][0], rows[0][1]
    jobs: list[list[tuple[int, int]]] = []
    for line in rows[1 : 1 + num_jobs]:
        if len(line) % 2 != 0:
            raise ValueError(f"Malformed JSSP line in {path}: {line}")
        operations = []
        for idx in range(0, len(line), 2):
            machine = int(line[idx])
            duration = int(line[idx + 1])
            operations.append((machine - 1 if machine >= 1 else machine, duration))
        jobs.append(operations)
    return num_machines, jobs


def jssp_dispatch_makespan(
    jobs: list[list[tuple[int, int]]],
    num_machines: int,
    rule: str,
) -> int:
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
) -> tuple[str, int | None]:
    horizon = sum(duration for job in jobs for _, duration in job)
    model = cp_model.CpModel()
    intervals: dict[int, list[cp_model.IntervalVar]] = {machine: [] for machine in range(num_machines)}
    task_ends: list[cp_model.IntVar] = []

    for job_idx, job in enumerate(jobs):
        prev_end: cp_model.IntVar | None = None
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
    status = solver.Solve(model)
    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return solver.StatusName(status), int(solver.Value(makespan))
    return solver.StatusName(status), None


def run_jssp(instance_path: Path, time_limit_sec: float) -> list[ResultRow]:
    num_machines, jobs = parse_jssp_file(instance_path)
    rows: list[ResultRow] = []

    for rule in ("spt", "mor", "mwr"):
        t0 = time.perf_counter()
        objective = jssp_dispatch_makespan(jobs, num_machines, rule)
        rows.append(
            ResultRow(
                problem="jssp",
                solver=rule,
                status="ok",
                objective=float(objective),
                elapsed_s=time.perf_counter() - t0,
                instance=instance_path.name,
            )
        )

    t0 = time.perf_counter()
    status, objective = jssp_cp_sat_makespan(jobs, num_machines, time_limit_sec)
    rows.append(
        ResultRow(
            problem="jssp",
            solver="ortools_cp_sat",
            status=status.lower(),
            objective=float(objective) if objective is not None else None,
            elapsed_s=time.perf_counter() - t0,
            instance=instance_path.name,
        )
    )
    return rows


def write_outputs(rows: list[ResultRow], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "results.csv"
    json_path = output_dir / "results.json"

    with csv_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=["problem", "solver", "status", "objective", "elapsed_s", "instance", "notes"],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row.__dict__)

    summary = {
        "results": [row.__dict__ for row in rows],
        "created_at_epoch_s": time.time(),
    }
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def main() -> int:
    args = parse_args()
    rows: list[ResultRow] = []
    output_dir = args.output_dir.resolve()

    try:
        if "tsp" in args.problems:
            for solver in args.tsp_solvers:
                try:
                    if solver == "lkh":
                        rows.append(run_tsp_lkh(args.seed, args.tsp_size, args.time_limit_sec, output_dir))
                    elif solver == "concorde":
                        rows.append(run_tsp_concorde(args.seed, args.tsp_size, output_dir))
                except FileNotFoundError:
                    continue
                except subprocess.CalledProcessError as exc:
                    notes = (exc.stdout or "").strip()
                    if exc.stderr:
                        notes = f"{notes} | stderr={exc.stderr.strip()}".strip(" |")
                    rows.append(
                        ResultRow(
                            problem="tsp",
                            solver=solver,
                            status=f"error_{exc.returncode}",
                            objective=None,
                            elapsed_s=0.0,
                            instance=f"random_{args.tsp_size}",
                            notes=notes[:500],
                        )
                    )
        if "cvrp" in args.problems:
            rows.append(run_cvrp_pyvrp(args.seed + 1, args.cvrp_size, args.time_limit_sec))
        if "ffsp" in args.problems:
            rows.extend(
                run_ffsp(
                    args.seed + 2,
                    args.ffsp_jobs,
                    args.ffsp_stages,
                    args.ffsp_machines,
                    args.time_limit_sec,
                )
            )
        if "jssp" in args.problems:
            rows.extend(run_jssp(args.jssp_instance.resolve(), args.time_limit_sec))
    except Exception as exc:  # noqa: BLE001
        print(f"[classical_solver_smoke] ERROR: {exc}", file=sys.stderr)
        return 1

    if "tsp" in args.problems and not any(
        row.problem == "tsp" and row.status == "ok" for row in rows
    ):
        print(
            "[classical_solver_smoke] ERROR: no TSP solver available. Install LKH-3 or Concorde first.",
            file=sys.stderr,
        )
        return 1

    write_outputs(rows, output_dir)
    for row in rows:
        print(
            f"{row.problem:>4} | {row.solver:<16} | status={row.status:<10} "
            f"| objective={row.objective!s:<8} | time={row.elapsed_s:.3f}s | {row.instance}"
        )
    print(f"Wrote outputs to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
