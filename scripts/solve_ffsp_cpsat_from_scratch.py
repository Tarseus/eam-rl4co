from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
SOLVER_SITE = REPO_ROOT / ".solver_site"
if SOLVER_SITE.is_dir():
    sys.path.insert(0, str(SOLVER_SITE))

from ortools.sat.python import cp_model


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def _solve_one(task: dict[str, Any]) -> dict[str, Any]:
    run_time = np.asarray(task["run_time"], dtype=np.int32)
    num_jobs, num_stages, num_machines = run_time.shape[0], 3, 4

    # Scheduling every operation sequentially on its fastest eligible machine is
    # always feasible.  Its duration sum is therefore a safe model horizon; no
    # schedule or incumbent is supplied to CP-SAT.
    fastest = run_time.reshape(num_jobs, num_stages, num_machines).min(axis=2)
    stage_bounds = [
        math.ceil(int(fastest[:, stage].sum()) / num_machines)
        for stage in range(num_stages)
    ]
    job_bound = int(fastest.sum(axis=1).max())
    simple_lower_bound = max(*stage_bounds, job_bound)
    horizon = int(fastest.sum())
    target_ratio = float(task.get("target_ratio", 0.0))
    if target_ratio > 0:
        horizon = min(horizon, math.ceil(simple_lower_bound * target_ratio))
    model = cp_model.CpModel()
    starts: dict[tuple[int, int], cp_model.IntVar] = {}
    ends: dict[tuple[int, int], cp_model.IntVar] = {}
    durations: dict[tuple[int, int], cp_model.IntVar] = {}
    selected: dict[tuple[int, int, int], cp_model.BoolVar] = {}
    machine_intervals: dict[tuple[int, int], list[cp_model.IntervalVar]] = defaultdict(list)
    stage_intervals: dict[int, list[cp_model.IntervalVar]] = defaultdict(list)

    min_heads = np.zeros((num_jobs, num_stages), dtype=np.int32)
    min_tails = np.zeros((num_jobs, num_stages), dtype=np.int32)
    for stage in range(1, num_stages):
        min_heads[:, stage] = min_heads[:, stage - 1] + fastest[:, stage - 1]
    for stage in range(num_stages - 2, -1, -1):
        min_tails[:, stage] = min_tails[:, stage + 1] + fastest[:, stage + 1]

    for job in range(num_jobs):
        for stage in range(num_stages):
            earliest = int(min_heads[job, stage])
            latest_end = horizon - int(min_tails[job, stage])
            starts[(job, stage)] = model.NewIntVar(
                earliest, latest_end, f"start_{job}_{stage}"
            )
            ends[(job, stage)] = model.NewIntVar(
                earliest + int(fastest[job, stage]), latest_end, f"end_{job}_{stage}"
            )
            stage_times = run_time[
                job, stage * num_machines : (stage + 1) * num_machines
            ]
            durations[(job, stage)] = model.NewIntVar(
                int(stage_times.min()), int(stage_times.max()), f"duration_{job}_{stage}"
            )
            selectors = []
            for machine in range(num_machines):
                key = (job, stage, machine)
                duration = int(run_time[job, stage * num_machines + machine])
                selected[key] = model.NewBoolVar(f"selected_{job}_{stage}_{machine}")
                interval = model.NewOptionalIntervalVar(
                    starts[(job, stage)],
                    duration,
                    ends[(job, stage)],
                    selected[key],
                    f"interval_{job}_{stage}_{machine}",
                )
                machine_intervals[(stage, machine)].append(interval)
                selectors.append(selected[key])
            model.AddExactlyOne(selectors)
            model.Add(
                durations[(job, stage)]
                == sum(
                    int(stage_times[machine]) * selected[(job, stage, machine)]
                    for machine in range(num_machines)
                )
            )
            stage_intervals[stage].append(
                model.NewIntervalVar(
                    starts[(job, stage)],
                    durations[(job, stage)],
                    ends[(job, stage)],
                    f"stage_interval_{job}_{stage}",
                )
            )
            if stage:
                model.Add(starts[(job, stage)] >= ends[(job, stage - 1)])

    for intervals in machine_intervals.values():
        model.AddNoOverlap(intervals)
    for stage in range(num_stages):
        model.AddCumulative(
            stage_intervals[stage], [1] * len(stage_intervals[stage]), num_machines
        )

    makespan = model.NewIntVar(0, horizon, "makespan")
    model.AddMaxEquality(
        makespan, [ends[(job, num_stages - 1)] for job in range(num_jobs)]
    )

    stage_floor = fastest.min(axis=0)
    for stage in range(num_stages):
        head_tail_floor = int(stage_floor[:stage].sum() + stage_floor[stage + 1 :].sum())
        total_selected_duration = 0
        for machine in range(num_machines):
            machine_load = sum(
                int(run_time[job, stage * num_machines + machine])
                * selected[(job, stage, machine)]
                for job in range(num_jobs)
            )
            model.Add(makespan >= machine_load + head_tail_floor)
            total_selected_duration += machine_load
        model.Add(
            num_machines * makespan
            >= total_selected_duration + num_machines * head_tail_floor
        )

    model.Add(makespan >= simple_lower_bound)
    model.Minimize(makespan)

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = float(task["budget_sec"])
    solver.parameters.num_search_workers = int(task["search_workers"])
    solver.parameters.random_seed = int(task["random_seed"])
    solver.parameters.symmetry_level = 2
    solver.parameters.cp_model_presolve = True
    solver.parameters.stop_after_first_solution = bool(task.get("first_feasible", False))
    started = time.perf_counter()
    status_code = solver.Solve(model)
    elapsed = time.perf_counter() - started
    status = solver.StatusName(status_code).lower()
    has_solution = status_code in (cp_model.OPTIMAL, cp_model.FEASIBLE)
    return {
        "instance_index": int(task["instance_index"]),
        "status": status,
        "objective": int(round(solver.ObjectiveValue())) if has_solution else None,
        "best_bound": float(solver.BestObjectiveBound()),
        "elapsed_sec": elapsed,
        "conflicts": int(solver.NumConflicts()),
        "branches": int(solver.NumBranches()),
        "wall_time_sec": float(solver.WallTime()),
        "simple_lower_bound": int(simple_lower_bound),
        "target_cap": int(horizon) if target_ratio > 0 else None,
    }


def _write_outputs(
    output_dir: Path,
    results: dict[int, dict[str, Any]],
    *,
    problem: str,
    test_sha: str,
    budget_sec: float,
    search_workers: int,
    target_ratio: float,
    first_feasible: bool,
    seed_offset: int,
) -> None:
    rows = [results[index] for index in sorted(results)]
    if rows:
        temporary = output_dir / "per_instance.csv.tmp"
        with temporary.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temporary, output_dir / "per_instance.csv")
    solved = [row for row in rows if row["objective"] is not None]
    _atomic_json(
        output_dir / "summary.json",
        {
            "protocol": f"{problem}_cp_sat_from_scratch_v3",
            "problem": problem,
            "test_file_sha256": test_sha,
            "requested_instance_count": len(results),
            "solved_count": len(solved),
            "optimal_count": sum(row["status"] == "optimal" for row in solved),
            "feasible_count": sum(row["status"] == "feasible" for row in solved),
            "mean_objective": (
                float(np.mean([int(row["objective"]) for row in solved])) if solved else None
            ),
            "sum_solver_elapsed_sec": sum(float(row["elapsed_sec"]) for row in rows),
            "budget_sec_per_instance": float(budget_sec),
            "search_workers_per_instance": int(search_workers),
            "target_ratio": float(target_ratio),
            "first_feasible": bool(first_feasible),
            "seed_offset": int(seed_offset),
            "external_incumbent": False,
            "solution_hint": False,
            "neural_checkpoint_loaded": False,
            "time_definition": "sum of per-instance CP-SAT elapsed times",
        },
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Solve FFSP50/100 with CP-SAT from scratch.")
    parser.add_argument("--problem", choices=("ffsp50", "ffsp100"), required=True)
    parser.add_argument("--test-file", type=Path, required=True)
    parser.add_argument("--test-file-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--outer-workers", type=int, default=16)
    parser.add_argument("--search-workers", type=int, default=4)
    parser.add_argument("--budget-sec", type=float, default=60.0)
    parser.add_argument("--max-instances", type=int, default=0)
    parser.add_argument(
        "--target-ratio",
        type=float,
        default=0.0,
        help="Optional makespan cap as ceil(target_ratio * instance lower bound).",
    )
    parser.add_argument(
        "--first-feasible",
        action="store_true",
        help="Stop CP-SAT after its first solution satisfying the model and optional cap.",
    )
    parser.add_argument(
        "--seed-offset",
        type=int,
        default=0,
        help="Offset added to deterministic per-instance CP-SAT random seeds.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    test_file = args.test_file.resolve()
    observed_sha = _sha256(test_file)
    if observed_sha != args.test_file_sha256.lower():
        raise ValueError("FFSP test-file SHA256 mismatch")
    with np.load(test_file) as payload:
        run_times = np.asarray(payload["run_time"], dtype=np.int32)
    expected_jobs = int(args.problem.removeprefix("ffsp"))
    if run_times.ndim != 3 or run_times.shape[1:] != (expected_jobs, 12):
        raise ValueError(f"Unexpected {args.problem} test shape: {run_times.shape}")
    count = run_times.shape[0] if args.max_instances <= 0 else min(args.max_instances, len(run_times))
    run_times = run_times[:count]
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    results: dict[int, dict[str, Any]] = {}
    per_instance_path = output_dir / "per_instance.csv"
    if per_instance_path.exists():
        with per_instance_path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                index = int(row["instance_index"])
                results[index] = {
                    "instance_index": index,
                    "status": row["status"],
                    "objective": int(row["objective"]) if row["objective"] else None,
                    "best_bound": float(row["best_bound"]),
                    "elapsed_sec": float(row["elapsed_sec"]),
                    "conflicts": int(row["conflicts"]),
                    "branches": int(row["branches"]),
                    "wall_time_sec": float(row["wall_time_sec"]),
                    "simple_lower_bound": int(row["simple_lower_bound"]),
                    "target_cap": int(row["target_cap"]) if row["target_cap"] else None,
                }
    tasks = [
        {
            "instance_index": index,
            "run_time": run_times[index],
            "budget_sec": float(args.budget_sec),
            "search_workers": int(args.search_workers),
            "random_seed": 12345678 + int(args.seed_offset) + index * 1009,
            "target_ratio": float(args.target_ratio),
            "first_feasible": bool(args.first_feasible),
        }
        for index in range(count)
        if index not in results
    ]
    print(
        json.dumps(
            {
                "event": "start",
                "problem": args.problem,
                "count": len(tasks),
                "external_incumbent": False,
                "solution_hint": False,
            }
        ),
        flush=True,
    )
    with ProcessPoolExecutor(max_workers=int(args.outer_workers)) as executor:
        futures = {executor.submit(_solve_one, task): task for task in tasks}
        for future in as_completed(futures):
            result = future.result()
            results[result["instance_index"]] = result
            _write_outputs(
                output_dir,
                results,
                problem=args.problem,
                test_sha=observed_sha,
                budget_sec=float(args.budget_sec),
                search_workers=int(args.search_workers),
                target_ratio=float(args.target_ratio),
                first_feasible=bool(args.first_feasible),
                seed_offset=int(args.seed_offset),
            )
            print(json.dumps({"event": "instance_done", **result}), flush=True)
    _write_outputs(
        output_dir,
        results,
        problem=args.problem,
        test_sha=observed_sha,
        budget_sec=float(args.budget_sec),
        search_workers=int(args.search_workers),
        target_ratio=float(args.target_ratio),
        first_feasible=bool(args.first_feasible),
        seed_offset=int(args.seed_offset),
    )
    return 0 if len(results) == count else 2


if __name__ == "__main__":
    raise SystemExit(main())
