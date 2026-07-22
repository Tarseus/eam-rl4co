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

from scripts.classical_solver_benchmark import ffsp_best_sjf_schedule


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


def _heuristic_hint(run_time: np.ndarray) -> tuple[int, list[dict[str, int]]]:
    objective, assignments = ffsp_best_sjf_schedule(run_time, 3, 4)
    return int(objective), [
        {
            "job": int(job),
            "stage": int(stage),
            "machine": int(machine),
            "start": int(start),
            "end": int(start + duration),
        }
        for job, stage, machine, start, duration in assignments
    ]


def _solve_one(task: dict[str, Any]) -> dict[str, Any]:
    run_time = np.asarray(task["run_time"], dtype=np.int32)
    num_jobs, num_stages, num_machines = run_time.shape[0], 3, 4
    incumbent = int(task["incumbent"])
    # A verified incumbent is already supplied for every instance, so no
    # selected operation can finish after it.  Keeping all time-variable
    # domains inside this exact horizon avoids carrying the much looser sum of
    # all processing times through presolve and subsequent proof rounds.
    horizon = incumbent
    model = cp_model.CpModel()
    starts: dict[tuple[int, int], cp_model.IntVar] = {}
    ends: dict[tuple[int, int], cp_model.IntVar] = {}
    selected: dict[tuple[int, int, int], cp_model.BoolVar] = {}
    local_starts: dict[tuple[int, int, int], cp_model.IntVar] = {}
    local_ends: dict[tuple[int, int, int], cp_model.IntVar] = {}
    intervals: dict[tuple[int, int], list[cp_model.IntervalVar]] = defaultdict(list)
    for job in range(num_jobs):
        for stage in range(num_stages):
            starts[(job, stage)] = model.NewIntVar(0, horizon, f"start_{job}_{stage}")
            ends[(job, stage)] = model.NewIntVar(0, horizon, f"end_{job}_{stage}")
            selectors = []
            for machine in range(num_machines):
                key = (job, stage, machine)
                duration = int(run_time[job, stage * num_machines + machine])
                selected[key] = model.NewBoolVar(f"selected_{job}_{stage}_{machine}")
                local_starts[key] = model.NewIntVar(0, horizon, f"ls_{job}_{stage}_{machine}")
                local_ends[key] = model.NewIntVar(0, horizon, f"le_{job}_{stage}_{machine}")
                interval = model.NewOptionalIntervalVar(
                    local_starts[key], duration, local_ends[key], selected[key],
                    f"interval_{job}_{stage}_{machine}",
                )
                model.Add(local_starts[key] == starts[(job, stage)]).OnlyEnforceIf(selected[key])
                model.Add(local_ends[key] == ends[(job, stage)]).OnlyEnforceIf(selected[key])
                intervals[(stage, machine)].append(interval)
                selectors.append(selected[key])
            model.AddExactlyOne(selectors)
            if stage:
                model.Add(starts[(job, stage)] >= ends[(job, stage - 1)])
    for machine_intervals in intervals.values():
        model.AddNoOverlap(machine_intervals)
    makespan = model.NewIntVar(0, horizon, "makespan")
    model.AddMaxEquality(makespan, [ends[(job, num_stages - 1)] for job in range(num_jobs)])
    # Optional intervals alone give CP-SAT a very weak objective bound for
    # FFSP1000.  Explicit per-machine energy inequalities expose the unrelated
    # parallel-machine load relaxation at every stage.
    stage_duration_floors = [
        int(run_time[:, stage * num_machines : (stage + 1) * num_machines].min())
        for stage in range(num_stages)
    ]
    for stage in range(num_stages):
        head_tail_floor = sum(stage_duration_floors[:stage]) + sum(
            stage_duration_floors[stage + 1 :]
        )
        for machine in range(num_machines):
            load = sum(
                int(run_time[job, stage * num_machines + machine])
                * selected[(job, stage, machine)]
                for job in range(num_jobs)
            )
            model.Add(makespan >= load + head_tail_floor)
    simple_stage_bounds = [
        math.ceil(
            sum(
                int(run_time[job, stage * num_machines : (stage + 1) * num_machines].min())
                for job in range(num_jobs)
            )
            / num_machines
        )
        for stage in range(num_stages)
    ]
    simple_job_bound = max(
        sum(
            int(run_time[job, stage * num_machines : (stage + 1) * num_machines].min())
            for stage in range(num_stages)
        )
        for job in range(num_jobs)
    )
    model.Add(makespan >= max(*simple_stage_bounds, simple_job_bound))
    prior_lower_bound = task.get("prior_lower_bound")
    if prior_lower_bound is not None and math.isfinite(float(prior_lower_bound)):
        model.Add(makespan >= math.ceil(float(prior_lower_bound) - 1e-9))
    model.Add(makespan <= incumbent)
    model.Minimize(makespan)

    hint = task["hint"]
    assigned = {(item["job"], item["stage"]): item["machine"] for item in hint}
    for item in hint:
        job, stage, machine = item["job"], item["stage"], item["machine"]
        key = (job, stage, machine)
        model.AddHint(starts[(job, stage)], item["start"])
        model.AddHint(ends[(job, stage)], item["end"])
        model.AddHint(selected[key], 1)
        model.AddHint(local_starts[key], item["start"])
        model.AddHint(local_ends[key], item["end"])
    for job in range(num_jobs):
        for stage in range(num_stages):
            for machine in range(num_machines):
                if assigned[(job, stage)] != machine:
                    model.AddHint(selected[(job, stage, machine)], 0)
    model.AddHint(makespan, incumbent)

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = float(task["budget_sec"])
    solver.parameters.num_search_workers = int(task["search_workers"])
    solver.parameters.random_seed = int(task["random_seed"])
    started = time.perf_counter()
    status_code = solver.Solve(model)
    elapsed = time.perf_counter() - started
    status = solver.StatusName(status_code).lower()
    has_solution = status_code in (cp_model.OPTIMAL, cp_model.FEASIBLE)
    objective = int(round(solver.ObjectiveValue())) if has_solution else None
    schedule = None
    if has_solution:
        schedule = []
        for job in range(num_jobs):
            for stage in range(num_stages):
                chosen = [
                    machine
                    for machine in range(num_machines)
                    if solver.BooleanValue(selected[(job, stage, machine)])
                ]
                if len(chosen) != 1:
                    raise RuntimeError("CP-SAT did not choose exactly one FFSP machine")
                machine = chosen[0]
                schedule.append(
                    {
                        "job": job,
                        "stage": stage,
                        "machine": machine,
                        "start": int(solver.Value(starts[(job, stage)])),
                        "end": int(solver.Value(ends[(job, stage)])),
                    }
                )
    return {
        "instance_index": int(task["instance_index"]),
        "attempt": int(task["attempt"]),
        "budget_sec": float(task["budget_sec"]),
        "search_workers": int(task["search_workers"]),
        "status": status,
        "objective": objective,
        "best_bound": float(solver.BestObjectiveBound()),
        "elapsed_sec": elapsed,
        "schedule": schedule,
    }


def _write_outputs(output_dir: Path, states: list[dict[str, Any]], data_sha: str) -> None:
    rows = [
        {
            "instance_index": state["instance_index"],
            "status": "optimal" if state["optimal"] else "unresolved",
            "objective": state["objective"],
            "best_bound": state["best_bound"],
            "attempts": state["attempts"],
            "total_elapsed_sec": state["total_elapsed_sec"],
        }
        for state in sorted(states, key=lambda value: value["instance_index"])
    ]
    temporary = output_dir / "per_instance.csv.tmp"
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, output_dir / "per_instance.csv")
    optimal = [row for row in rows if row["status"] == "optimal"]
    _atomic_json(
        output_dir / "summary.json",
        {
            "protocol": "ffsp1000_generated100_cp_sat_until_proven_optimal_v1",
            "test_file_sha256": data_sha,
            "instance_count": len(rows),
            "optimal_count": len(optimal),
            "unresolved_count": len(rows) - len(optimal),
            "all_optimal": len(optimal) == len(rows),
            "mean_optimal_objective": (
                sum(int(row["objective"]) for row in optimal) / len(optimal) if optimal else None
            ),
            "sum_solver_elapsed_sec": sum(float(row["total_elapsed_sec"]) for row in rows),
            "paper_gap_allowed": len(optimal) == len(rows),
        },
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Resume FFSP1000 CP-SAT until 100/100 are optimal.")
    parser.add_argument("--test-file", type=Path, required=True)
    parser.add_argument("--test-file-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--hint-dir", type=Path)
    parser.add_argument("--outer-workers", type=int, default=4)
    parser.add_argument("--search-workers", type=int, default=16)
    parser.add_argument("--initial-budget-sec", type=float, default=1200.0)
    parser.add_argument("--budget-multiplier", type=float, default=2.0)
    parser.add_argument("--max-budget-sec", type=float, default=0.0)
    parser.add_argument("--stop-after-rounds", type=int, default=0)
    parser.add_argument("--max-instances", type=int, default=100)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    test_file = args.test_file.resolve()
    observed_sha = _sha256(test_file)
    if observed_sha != args.test_file_sha256.lower():
        raise ValueError("FFSP test-file SHA256 mismatch")
    with np.load(test_file) as payload:
        run_times = np.asarray(payload["run_time"], dtype=np.int32)
    if run_times.shape != (100, 1000, 12):
        raise ValueError(f"Unexpected FFSP1000 test shape: {run_times.shape}")
    if not 1 <= args.max_instances <= 100:
        raise ValueError("max-instances must be in [1, 100]")
    run_times = run_times[: args.max_instances]
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    state_dir = output_dir / "state"
    state_dir.mkdir(exist_ok=True)
    states: dict[int, dict[str, Any]] = {}
    for index, run_time in enumerate(run_times):
        path = state_dir / f"instance_{index:05d}.json"
        if path.exists():
            state = json.loads(path.read_text(encoding="utf-8"))
        else:
            objective, schedule = _heuristic_hint(run_time)
            state = {
                "instance_index": index,
                "attempts": 0,
                "optimal": False,
                "objective": objective,
                "best_bound": None,
                "total_elapsed_sec": 0.0,
                "schedule": schedule,
            }
            _atomic_json(path, state)
        states[index] = state
    if args.hint_dir is not None:
        hint_dir = args.hint_dir.resolve()
        for index, state in states.items():
            hint_path = hint_dir / f"instance_{index:05d}.json"
            hint = json.loads(hint_path.read_text(encoding="utf-8"))
            if hint["test_file_sha256"] != observed_sha:
                raise ValueError(f"FFSP hint test-file hash mismatch: {hint_path}")
            if int(hint["objective"]) <= int(state["objective"]):
                state["objective"] = int(hint["objective"])
                state["schedule"] = hint["schedule"]
                _atomic_json(state_dir / f"instance_{index:05d}.json", state)
    _write_outputs(output_dir, list(states.values()), observed_sha)

    round_index = 0
    while True:
        unresolved = sorted(
            (index for index, state in states.items() if not state["optimal"]),
            key=lambda index: (int(states[index]["attempts"]), index),
        )
        if not unresolved:
            print(json.dumps({"event": "complete", "optimal_count": len(states)}), flush=True)
            return 0
        if args.stop_after_rounds and round_index >= args.stop_after_rounds:
            print(json.dumps({"event": "round_limit", "unresolved_count": len(unresolved)}), flush=True)
            return 2
        tasks = []
        for index in unresolved:
            state = states[index]
            budget = args.initial_budget_sec * args.budget_multiplier ** int(state["attempts"])
            if args.max_budget_sec > 0:
                budget = min(budget, args.max_budget_sec)
            tasks.append(
                {
                    "instance_index": index,
                    "run_time": run_times[index],
                    "attempt": int(state["attempts"]) + 1,
                    "budget_sec": budget,
                    "search_workers": args.search_workers,
                    "random_seed": 12345678 + index * 1009 + int(state["attempts"]),
                    "incumbent": state["objective"],
                    "prior_lower_bound": state["best_bound"],
                    "hint": state["schedule"],
                }
            )
        print(
            json.dumps(
                {
                    "event": "round_start",
                    "round": round_index + 1,
                    "unresolved_count": len(tasks),
                    "min_budget_sec": min(task["budget_sec"] for task in tasks),
                    "max_budget_sec": max(task["budget_sec"] for task in tasks),
                }
            ),
            flush=True,
        )
        with ProcessPoolExecutor(max_workers=args.outer_workers) as executor:
            futures = {executor.submit(_solve_one, task): task for task in tasks}
            for future in as_completed(futures):
                result = future.result()
                state = states[result["instance_index"]]
                state["attempts"] = result["attempt"]
                state["total_elapsed_sec"] += result["elapsed_sec"]
                if math.isfinite(result["best_bound"]):
                    state["best_bound"] = (
                        result["best_bound"]
                        if state["best_bound"] is None
                        else max(float(state["best_bound"]), result["best_bound"])
                    )
                if result["objective"] is not None and result["objective"] <= state["objective"]:
                    state["objective"] = int(result["objective"])
                    state["schedule"] = result["schedule"]
                state["optimal"] = bool(
                    result["status"] == "optimal"
                    or (
                        state["best_bound"] is not None
                        and math.ceil(float(state["best_bound"]) - 1e-9) >= int(state["objective"])
                    )
                )
                _atomic_json(state_dir / f"instance_{result['instance_index']:05d}.json", state)
                with (output_dir / "attempts.jsonl").open("a", encoding="utf-8") as handle:
                    logged = dict(result)
                    logged.pop("schedule", None)
                    handle.write(json.dumps(logged) + "\n")
                _write_outputs(output_dir, list(states.values()), observed_sha)
                print(
                    json.dumps(
                        {
                            "event": "instance_done",
                            "instance_index": result["instance_index"],
                            "solver_status": result["status"],
                            "objective": state["objective"],
                            "best_bound": state["best_bound"],
                            "optimal": state["optimal"],
                            "optimal_count": sum(item["optimal"] for item in states.values()),
                        }
                    ),
                    flush=True,
                )
        round_index += 1


if __name__ == "__main__":
    raise SystemExit(main())
