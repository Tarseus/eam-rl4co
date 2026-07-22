from __future__ import annotations

import argparse
import csv
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np

from solve_ffsp_cpsat_from_scratch import _sha256, _solve_one


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def _solve_multistart_one(task: dict[str, Any]) -> dict[str, Any]:
    trace = []
    winner = None
    for attempt_index, attempt in enumerate(task["attempt_plan"]):
        result = _solve_one(
            {
                "instance_index": task["instance_index"],
                "run_time": task["run_time"],
                "budget_sec": task["budget_sec"],
                "search_workers": task["search_workers"],
                "random_seed": (
                    12345678
                    + int(task["instance_index"]) * 1009
                    + int(attempt["seed_offset"])
                ),
                "target_ratio": float(attempt["target_ratio"]),
                "first_feasible": bool(attempt["target_ratio"] > 0),
            }
        )
        trace.append(
            {
                "attempt_index": attempt_index,
                "target_ratio": float(attempt["target_ratio"]),
                "seed_offset": int(attempt["seed_offset"]),
                "status": result["status"],
                "objective": result["objective"],
                "best_bound": result["best_bound"],
                "elapsed_sec": result["elapsed_sec"],
            }
        )
        if result["objective"] is not None:
            winner = result
            break
    if winner is None:
        winner = result
    return {
        "instance_index": int(task["instance_index"]),
        "status": winner["status"],
        "objective": winner["objective"],
        "best_bound": winner["best_bound"],
        "total_solver_elapsed_sec": sum(float(item["elapsed_sec"]) for item in trace),
        "attempts_run": len(trace),
        "winning_target_ratio": (
            float(trace[-1]["target_ratio"]) if winner["objective"] is not None else None
        ),
        "winning_seed_offset": (
            int(trace[-1]["seed_offset"]) if winner["objective"] is not None else None
        ),
        "simple_lower_bound": winner["simple_lower_bound"],
        "target_cap": winner["target_cap"],
        "attempt_trace_json": json.dumps(trace, separators=(",", ":")),
    }


def _write_outputs(
    output_dir: Path,
    results: dict[int, dict[str, Any]],
    *,
    problem: str,
    test_sha: str,
    budget_sec: float,
    search_workers: int,
    attempt_plan: list[dict[str, float | int]],
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
            "protocol": f"{problem}_cp_sat_multistart_from_scratch_v1",
            "problem": problem,
            "test_file_sha256": test_sha,
            "requested_instance_count": len(rows),
            "solved_count": len(solved),
            "mean_objective": (
                float(np.mean([int(row["objective"]) for row in solved])) if solved else None
            ),
            "sum_solver_elapsed_sec": sum(
                float(row["total_solver_elapsed_sec"]) for row in rows
            ),
            "budget_sec_per_attempt": float(budget_sec),
            "search_workers_per_attempt": int(search_workers),
            "attempt_plan": attempt_plan,
            "external_incumbent": False,
            "solution_hint": False,
            "neural_checkpoint_loaded": False,
            "time_definition": "sum of every attempted per-instance CP-SAT solve",
        },
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a progressive pure CP-SAT multistart protocol for FFSP100."
    )
    parser.add_argument("--problem", choices=("ffsp100",), required=True)
    parser.add_argument("--test-file", type=Path, required=True)
    parser.add_argument("--test-file-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--outer-workers", type=int, default=32)
    parser.add_argument("--search-workers", type=int, default=4)
    parser.add_argument("--budget-sec", type=float, default=60.0)
    parser.add_argument("--max-instances", type=int, default=0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    test_file = args.test_file.resolve()
    observed_sha = _sha256(test_file)
    if observed_sha != args.test_file_sha256.lower():
        raise ValueError("FFSP test-file SHA256 mismatch")
    with np.load(test_file) as payload:
        run_times = np.asarray(payload["run_time"], dtype=np.int32)
    if run_times.ndim != 3 or run_times.shape[1:] != (100, 12):
        raise ValueError(f"Unexpected ffsp100 test shape: {run_times.shape}")
    count = len(run_times) if args.max_instances <= 0 else min(args.max_instances, len(run_times))
    run_times = run_times[:count]
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    seed_offsets = (0, 1000000, 2000000, 3000000)
    attempt_plan = [
        {"target_ratio": ratio, "seed_offset": seed_offset}
        for ratio in (1.10, 1.12)
        for seed_offset in seed_offsets
    ]
    attempt_plan.append({"target_ratio": 0.0, "seed_offset": 4000000})

    results: dict[int, dict[str, Any]] = {}
    per_instance_path = output_dir / "per_instance.csv"
    if per_instance_path.exists():
        with per_instance_path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                index = int(row["instance_index"])
                if row["objective"]:
                    results[index] = {
                        "instance_index": index,
                        "status": row["status"],
                        "objective": int(row["objective"]),
                        "best_bound": float(row["best_bound"]),
                        "total_solver_elapsed_sec": float(row["total_solver_elapsed_sec"]),
                        "attempts_run": int(row["attempts_run"]),
                        "winning_target_ratio": float(row["winning_target_ratio"]),
                        "winning_seed_offset": int(row["winning_seed_offset"]),
                        "simple_lower_bound": int(row["simple_lower_bound"]),
                        "target_cap": int(row["target_cap"]) if row["target_cap"] else None,
                        "attempt_trace_json": row["attempt_trace_json"],
                    }

    tasks = [
        {
            "instance_index": index,
            "run_time": run_times[index],
            "budget_sec": float(args.budget_sec),
            "search_workers": int(args.search_workers),
            "attempt_plan": attempt_plan,
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
                "attempt_plan": attempt_plan,
                "external_incumbent": False,
                "solution_hint": False,
                "neural_checkpoint_loaded": False,
            }
        ),
        flush=True,
    )
    with ProcessPoolExecutor(max_workers=int(args.outer_workers)) as executor:
        futures = {executor.submit(_solve_multistart_one, task): task for task in tasks}
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
                attempt_plan=attempt_plan,
            )
            print(json.dumps({"event": "instance_done", **result}), flush=True)
    _write_outputs(
        output_dir,
        results,
        problem=args.problem,
        test_sha=observed_sha,
        budget_sec=float(args.budget_sec),
        search_workers=int(args.search_workers),
        attempt_plan=attempt_plan,
    )
    return 0 if len(results) == count and all(row["objective"] is not None for row in results.values()) else 2


if __name__ == "__main__":
    raise SystemExit(main())
