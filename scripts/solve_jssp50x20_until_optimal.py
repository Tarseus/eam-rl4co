from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
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


def _parse_jssp(path: Path) -> tuple[int, list[list[tuple[int, int]]]]:
    rows = [
        [int(token) for token in line.split()]
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    num_jobs, num_machines = rows[0][:2]
    if len(rows) != num_jobs + 1:
        raise ValueError(f"Malformed JSSP instance {path}: expected {num_jobs} job rows")
    ids = [row[i] for row in rows[1:] for i in range(0, len(row), 2)]
    offset = 0 if min(ids) == 0 else 1
    if min(ids) - offset != 0 or max(ids) - offset != num_machines - 1:
        raise ValueError(f"Unsupported machine indexing in {path}")
    jobs: list[list[tuple[int, int]]] = []
    for row in rows[1:]:
        if len(row) != 2 * num_machines:
            raise ValueError(f"Malformed operation row in {path}")
        jobs.append(
            [(int(row[i]) - offset, int(row[i + 1])) for i in range(0, len(row), 2)]
        )
    return num_machines, jobs


def _solve_one(task: dict[str, Any]) -> dict[str, Any]:
    path = Path(task["path"])
    num_machines, jobs = _parse_jssp(path)
    incumbent = int(task["incumbent"])
    # The incumbent schedule is verified before solving.  Every selected JSSP
    # operation must therefore finish within this exact horizon; using it
    # directly keeps resumed proof rounds compact.
    horizon = incumbent
    model = cp_model.CpModel()
    starts: dict[tuple[int, int], cp_model.IntVar] = {}
    ends: dict[tuple[int, int], cp_model.IntVar] = {}
    machine_intervals: dict[int, list[cp_model.IntervalVar]] = {
        machine: [] for machine in range(num_machines)
    }
    job_ends: list[cp_model.IntVar] = []
    for job_index, job in enumerate(jobs):
        previous_end = None
        for operation_index, (machine, duration) in enumerate(job):
            key = (job_index, operation_index)
            start = model.NewIntVar(0, horizon, f"start_{job_index}_{operation_index}")
            end = model.NewIntVar(0, horizon, f"end_{job_index}_{operation_index}")
            interval = model.NewIntervalVar(
                start, duration, end, f"interval_{job_index}_{operation_index}"
            )
            starts[key] = start
            ends[key] = end
            machine_intervals[machine].append(interval)
            if previous_end is not None:
                model.Add(start >= previous_end)
            previous_end = end
        assert previous_end is not None
        job_ends.append(previous_end)
    for intervals in machine_intervals.values():
        model.AddNoOverlap(intervals)

    makespan = model.NewIntVar(0, horizon, "makespan")
    model.AddMaxEquality(makespan, job_ends)
    # These redundant integer lower bounds are inexpensive and preserve useful
    # proof progress across resumed CP-SAT attempts.  For a machine, every one
    # of its operations must run serially; at least the smallest job prefix is
    # required before that serial block and at least the smallest suffix after
    # it.  CP-SAT can derive related bounds, but stating them explicitly makes
    # the large 50x20 proof considerably easier to resume.
    model.Add(makespan >= max(sum(duration for _, duration in job) for job in jobs))
    for machine in range(num_machines):
        machine_load = 0
        heads: list[int] = []
        tails: list[int] = []
        for job in jobs:
            operation_index = next(
                index for index, (candidate, _) in enumerate(job) if candidate == machine
            )
            machine_load += job[operation_index][1]
            heads.append(sum(duration for _, duration in job[:operation_index]))
            tails.append(sum(duration for _, duration in job[operation_index + 1 :]))
        model.Add(makespan >= machine_load + min(heads) + min(tails))

    prior_lower_bound = task.get("prior_lower_bound")
    if prior_lower_bound is not None and math.isfinite(float(prior_lower_bound)):
        model.Add(makespan >= math.ceil(float(prior_lower_bound) - 1e-9))
    model.Add(makespan <= incumbent)
    model.Minimize(makespan)

    hint = task.get("hint")
    if hint:
        for item in hint:
            key = (int(item["job"]), int(item["operation"]))
            model.AddHint(starts[key], int(item["start"]))
            model.AddHint(ends[key], int(item["end"]))
        model.AddHint(makespan, incumbent)

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = float(task["budget_sec"])
    solver.parameters.num_search_workers = int(task["search_workers"])
    solver.parameters.random_seed = int(task["random_seed"])
    solver.parameters.log_search_progress = False
    started = time.perf_counter()
    status_code = solver.Solve(model)
    elapsed = time.perf_counter() - started
    status = solver.StatusName(status_code).lower()
    has_solution = status_code in (cp_model.OPTIMAL, cp_model.FEASIBLE)
    objective = int(round(solver.ObjectiveValue())) if has_solution else None
    best_bound = float(solver.BestObjectiveBound())
    schedule = None
    if has_solution:
        schedule = []
        for job_index, job in enumerate(jobs):
            for operation_index, (_, duration) in enumerate(job):
                start_value = int(solver.Value(starts[(job_index, operation_index)]))
                schedule.append(
                    {
                        "job": job_index,
                        "operation": operation_index,
                        "start": start_value,
                        "end": start_value + duration,
                    }
                )
    return {
        "instance": path.name,
        "instance_sha256": task["instance_sha256"],
        "attempt": int(task["attempt"]),
        "budget_sec": float(task["budget_sec"]),
        "search_workers": int(task["search_workers"]),
        "random_seed": int(task["random_seed"]),
        "status": status,
        "objective": objective,
        "best_bound": best_bound,
        "elapsed_sec": elapsed,
        "schedule": schedule,
    }


def _read_incumbents(path: Path) -> dict[str, int]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    incumbents = {
        row["instance_id"]: int(round(float(row["objective"])))
        for row in rows
        if row.get("objective") not in (None, "")
    }
    if len(incumbents) != 100:
        raise ValueError(f"Expected 100 incumbent objectives, found {len(incumbents)}")
    return incumbents


def _load_state(path: Path, instance: Path, incumbent: int) -> dict[str, Any]:
    if path.exists():
        state = json.loads(path.read_text(encoding="utf-8"))
        if state["instance_sha256"] != _sha256(instance):
            raise ValueError(f"Instance changed since state was written: {instance}")
        return state
    return {
        "instance": instance.name,
        "instance_sha256": _sha256(instance),
        "attempts": 0,
        "optimal": False,
        "objective": int(incumbent),
        "best_bound": None,
        "total_elapsed_sec": 0.0,
        "schedule": None,
    }


def _write_outputs(output_dir: Path, states: list[dict[str, Any]]) -> None:
    rows = []
    for state in sorted(states, key=lambda item: item["instance"]):
        rows.append(
            {
                "instance": state["instance"],
                "instance_sha256": state["instance_sha256"],
                "status": "optimal" if state["optimal"] else "unresolved",
                "objective": state["objective"],
                "best_bound": state["best_bound"],
                "attempts": state["attempts"],
                "total_elapsed_sec": state["total_elapsed_sec"],
            }
        )
    temporary = output_dir / "per_instance.csv.tmp"
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, output_dir / "per_instance.csv")
    optimal = [row for row in rows if row["status"] == "optimal"]
    summary = {
        "protocol": "jssp50x20_generated100_cp_sat_until_proven_optimal_v1",
        "instance_count": len(rows),
        "optimal_count": len(optimal),
        "unresolved_count": len(rows) - len(optimal),
        "all_optimal": len(optimal) == len(rows),
        "mean_optimal_objective": (
            sum(int(row["objective"]) for row in optimal) / len(optimal) if optimal else None
        ),
        "sum_solver_elapsed_sec": sum(float(row["total_elapsed_sec"]) for row in rows),
        "paper_gap_allowed": len(optimal) == len(rows),
    }
    _atomic_json(output_dir / "summary.json", summary)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Resume CP-SAT on generated JSSP50x20 until every instance is proven optimal."
    )
    parser.add_argument("--instance-dir", type=Path, required=True)
    parser.add_argument("--incumbent-csv", type=Path, required=True)
    parser.add_argument(
        "--hint-dir",
        type=Path,
        help="Optional per-instance JSON schedules; lower objectives replace CSV incumbents.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--outer-workers", type=int, default=16)
    parser.add_argument("--search-workers", type=int, default=8)
    parser.add_argument("--initial-budget-sec", type=float, default=1200.0)
    parser.add_argument("--budget-multiplier", type=float, default=2.0)
    parser.add_argument("--max-budget-sec", type=float, default=0.0)
    parser.add_argument(
        "--stop-after-rounds",
        type=int,
        default=0,
        help="0 means keep increasing budgets until all instances are optimal.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.outer_workers < 1 or args.search_workers < 1:
        raise ValueError("Worker counts must be positive")
    if args.initial_budget_sec <= 0 or args.budget_multiplier < 1:
        raise ValueError("Invalid CP-SAT budget schedule")
    instance_dir = args.instance_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    state_dir = output_dir / "state"
    state_dir.mkdir(exist_ok=True)
    instances = sorted(instance_dir.glob("50x20_*.jsp"))
    if len(instances) != 100:
        raise ValueError(f"Expected exactly 100 JSSP50x20 instances, found {len(instances)}")
    incumbents = _read_incumbents(args.incumbent_csv.resolve())
    states = {
        instance.name: _load_state(
            state_dir / f"{instance.stem}.json", instance, incumbents[instance.name]
        )
        for instance in instances
    }
    if args.hint_dir is not None:
        hint_dir = args.hint_dir.resolve()
        for instance in instances:
            state = states[instance.name]
            if state["attempts"] != 0:
                continue
            hint_path = hint_dir / f"{instance.stem}.json"
            hint = json.loads(hint_path.read_text(encoding="utf-8"))
            if hint["instance_sha256"] != state["instance_sha256"]:
                raise ValueError(f"Hint instance hash mismatch: {hint_path}")
            if int(hint["objective"]) <= int(state["objective"]):
                state["objective"] = int(hint["objective"])
                state["schedule"] = hint["schedule"]
                _atomic_json(state_dir / f"{instance.stem}.json", state)
    _write_outputs(output_dir, list(states.values()))

    round_index = 0
    while True:
        unresolved = [instance for instance in instances if not states[instance.name]["optimal"]]
        if not unresolved:
            print(json.dumps({"event": "complete", "optimal_count": 100}), flush=True)
            return 0
        if args.stop_after_rounds and round_index >= args.stop_after_rounds:
            print(
                json.dumps({"event": "round_limit", "unresolved_count": len(unresolved)}),
                flush=True,
            )
            return 2
        tasks = []
        for instance in unresolved:
            state = states[instance.name]
            budget = args.initial_budget_sec * args.budget_multiplier ** int(state["attempts"])
            if args.max_budget_sec > 0:
                budget = min(budget, args.max_budget_sec)
            tasks.append(
                {
                    "path": str(instance),
                    "instance_sha256": state["instance_sha256"],
                    "attempt": int(state["attempts"]) + 1,
                    "budget_sec": budget,
                    "search_workers": args.search_workers,
                    "random_seed": 12345678 + instances.index(instance) * 1009 + int(state["attempts"]),
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
                state = states[result["instance"]]
                state["attempts"] = result["attempt"]
                state["total_elapsed_sec"] += result["elapsed_sec"]
                bound = result["best_bound"]
                if math.isfinite(bound):
                    previous = state["best_bound"]
                    state["best_bound"] = bound if previous is None else max(float(previous), bound)
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
                _atomic_json(state_dir / f"{Path(result['instance']).stem}.json", state)
                with (output_dir / "attempts.jsonl").open("a", encoding="utf-8") as handle:
                    log_result = dict(result)
                    log_result.pop("schedule", None)
                    handle.write(json.dumps(log_result) + "\n")
                _write_outputs(output_dir, list(states.values()))
                print(
                    json.dumps(
                        {
                            "event": "instance_done",
                            "instance": result["instance"],
                            "attempt": result["attempt"],
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
