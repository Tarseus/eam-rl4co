from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.solve_ffsp1000_until_optimal import _solve_one


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


def _write_outputs(
    output_dir: Path,
    states: dict[int, dict[str, Any]],
    *,
    problem: str,
    data_sha: str,
    hint_summary: dict[str, Any],
) -> None:
    rows = [
        {
            "instance_index": state["instance_index"],
            "status": "optimal" if state["optimal"] else "feasible",
            "objective": state["objective"],
            "best_bound": state["best_bound"],
            "attempts": state["attempts"],
            "total_solver_elapsed_sec": state["total_solver_elapsed_sec"],
        }
        for state in sorted(states.values(), key=lambda value: value["instance_index"])
    ]
    temporary = output_dir / "per_instance.csv.tmp"
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, output_dir / "per_instance.csv")
    solver_elapsed = sum(float(row["total_solver_elapsed_sec"]) for row in rows)
    neural_elapsed = float(hint_summary["total_elapsed_sec"])
    objectives = [int(row["objective"]) for row in rows]
    initial_mean = float(hint_summary["mean_objective"])
    final_mean = float(np.mean(objectives))
    _atomic_json(
        output_dir / "summary.json",
        {
            "protocol": f"{problem}_neural_warm_start_cp_sat_test{len(rows)}_v1",
            "problem": problem,
            "test_file_sha256": data_sha,
            "instance_count": len(rows),
            "optimal_count": sum(row["status"] == "optimal" for row in rows),
            "initial_neural_method": hint_summary["method"],
            "initial_mean_objective": initial_mean,
            "mean_objective": final_mean,
            "improvement": initial_mean - final_mean,
            "neural_hint_elapsed_sec": neural_elapsed,
            "sum_solver_elapsed_sec": solver_elapsed,
            "total_serial_equivalent_elapsed_sec": neural_elapsed + solver_elapsed,
            "time_definition": "neural batched inference wall time plus sum of per-instance CP-SAT elapsed times",
            "paper_gap_reference_only": True,
        },
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Improve FFSP50/100 neural incumbents with CP-SAT.")
    parser.add_argument("--problem", choices=("ffsp50", "ffsp100"), required=True)
    parser.add_argument("--test-file", type=Path, required=True)
    parser.add_argument("--test-file-sha256", required=True)
    parser.add_argument("--hint-output-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--outer-workers", type=int, default=8)
    parser.add_argument("--search-workers", type=int, default=4)
    parser.add_argument("--budget-sec", type=float, default=30.0)
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
    expected_jobs = int(args.problem.removeprefix("ffsp"))
    if run_times.ndim != 3 or run_times.shape[1:] != (expected_jobs, 12):
        raise ValueError(f"Unexpected {args.problem} test shape: {run_times.shape}")
    count = run_times.shape[0] if args.max_instances <= 0 else min(args.max_instances, run_times.shape[0])
    run_times = run_times[:count]
    hint_output_dir = args.hint_output_dir.resolve()
    hint_summary = json.loads((hint_output_dir / "summary.json").read_text(encoding="utf-8"))
    if hint_summary["test_file_sha256"] != observed_sha or int(hint_summary["count"]) < count:
        raise ValueError("Hint summary does not match the requested test set")
    with (hint_output_dir / "per_instance.csv").open(newline="", encoding="utf-8") as handle:
        hint_rows = list(csv.DictReader(handle))
    if len(hint_rows) < count:
        raise ValueError("Hint per-instance file is shorter than the requested test set")
    hint_summary = dict(hint_summary)
    hint_summary["mean_objective"] = float(
        np.mean([float(row["objective"]) for row in hint_rows[:count]])
    )

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    state_dir = output_dir / "state"
    state_dir.mkdir(exist_ok=True)
    states: dict[int, dict[str, Any]] = {}
    for index in range(count):
        state_path = state_dir / f"instance_{index:05d}.json"
        if state_path.exists():
            state = json.loads(state_path.read_text(encoding="utf-8"))
        else:
            hint_path = hint_output_dir / "hints" / f"instance_{index:05d}.json"
            hint = json.loads(hint_path.read_text(encoding="utf-8"))
            if hint["test_file_sha256"] != observed_sha:
                raise ValueError(f"Hint test-file hash mismatch: {hint_path}")
            state = {
                "instance_index": index,
                "attempts": 0,
                "optimal": False,
                "objective": int(hint["objective"]),
                "best_bound": None,
                "total_solver_elapsed_sec": 0.0,
                "schedule": hint["schedule"],
            }
            _atomic_json(state_path, state)
        states[index] = state
    _write_outputs(
        output_dir, states, problem=args.problem, data_sha=observed_sha, hint_summary=hint_summary
    )

    tasks = []
    for index, state in states.items():
        if state["optimal"] or state["attempts"] > 0:
            continue
        tasks.append(
            {
                "instance_index": index,
                "run_time": run_times[index],
                "attempt": 1,
                "budget_sec": float(args.budget_sec),
                "search_workers": int(args.search_workers),
                "random_seed": 12345678 + index * 1009,
                "incumbent": int(state["objective"]),
                "prior_lower_bound": state["best_bound"],
                "hint": state["schedule"],
            }
        )
    print(json.dumps({"event": "start", "problem": args.problem, "count": len(tasks)}), flush=True)
    with ProcessPoolExecutor(max_workers=args.outer_workers) as executor:
        futures = {executor.submit(_solve_one, task): task for task in tasks}
        for future in as_completed(futures):
            result = future.result()
            state = states[result["instance_index"]]
            state["attempts"] = 1
            state["total_solver_elapsed_sec"] += float(result["elapsed_sec"])
            if math.isfinite(result["best_bound"]):
                state["best_bound"] = float(result["best_bound"])
            if result["objective"] is not None and int(result["objective"]) <= int(state["objective"]):
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
            _write_outputs(
                output_dir,
                states,
                problem=args.problem,
                data_sha=observed_sha,
                hint_summary=hint_summary,
            )
            print(
                json.dumps(
                    {
                        "event": "instance_done",
                        "instance_index": result["instance_index"],
                        "objective": state["objective"],
                        "solver_status": result["status"],
                    }
                ),
                flush=True,
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
