from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = REPO_ROOT / "scripts" / "minizinc_models"
SUCCESS_STATUSES = {"ok", "optimal", "feasible"}


@dataclass(frozen=True)
class ScenarioConfig:
    name: str
    problem: str
    size: int
    generated_count: int
    time_limit_s: float
    jssp_shape: str | None = None
    ffsp_stages: int | None = None
    ffsp_machines: int | None = None


SCENARIOS: dict[str, ScenarioConfig] = {
    "ffsp50": ScenarioConfig(
        name="ffsp50",
        problem="ffsp",
        size=50,
        generated_count=1000,
        time_limit_s=180.0,
        ffsp_stages=3,
        ffsp_machines=4,
    ),
    "ffsp100": ScenarioConfig(
        name="ffsp100",
        problem="ffsp",
        size=100,
        generated_count=1000,
        time_limit_s=600.0,
        ffsp_stages=3,
        ffsp_machines=4,
    ),
    "jssp10x10": ScenarioConfig(
        name="jssp10x10",
        problem="jssp",
        size=10,
        generated_count=0,
        time_limit_s=180.0,
        jssp_shape="10x10",
    ),
    "jssp15x15": ScenarioConfig(
        name="jssp15x15",
        problem="jssp",
        size=15,
        generated_count=0,
        time_limit_s=600.0,
        jssp_shape="15x15",
    ),
    "jssp20x20": ScenarioConfig(
        name="jssp20x20",
        problem="jssp",
        size=20,
        generated_count=0,
        time_limit_s=1200.0,
        jssp_shape="20x20",
    ),
}

SOLVER_CANDIDATES: dict[str, tuple[str, ...]] = {
    "chuffed": ("org.chuffed.chuffed", "chuffed"),
    "scip": ("org.minizinc.scip", "scip"),
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark MiniZinc-backed classical solvers for JSSP and FFSP, "
            "currently targeting Chuffed and SCIP."
        )
    )
    parser.add_argument(
        "--scenarios",
        nargs="+",
        default=list(SCENARIOS.keys()),
        choices=list(SCENARIOS.keys()),
        help="Scenario names to evaluate.",
    )
    parser.add_argument(
        "--solvers",
        nargs="+",
        default=["chuffed", "scip"],
        choices=sorted(SOLVER_CANDIDATES.keys()),
        help="Requested MiniZinc solver backends.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "logs" / "classical_benchmark_mzn",
        help="Output directory for per-instance and summary CSV/JSON files.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1234,
        help="Seed used when FFSP instances are generated on the fly.",
    )
    parser.add_argument(
        "--max-instances",
        type=int,
        default=None,
        help="Optional cap on the number of instances per scenario.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Max number of outer worker processes.",
    )
    parser.add_argument(
        "--minizinc-bin",
        type=str,
        default="minizinc",
        help="MiniZinc executable to invoke.",
    )
    parser.add_argument(
        "--timeout-grace-sec",
        type=float,
        default=5.0,
        help="Extra subprocess grace period beyond the scenario solve limit.",
    )
    return parser.parse_args()


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
            operations.append((machine, duration))
        jobs.append(operations)
    return num_machines, jobs


def _load_jssp_instances(cfg: ScenarioConfig, max_instances: int | None) -> list[Path]:
    assert cfg.jssp_shape is not None
    root = REPO_ROOT / "data" / "jssp_bopo" / "validation"
    files = sorted(root.glob(f"{cfg.jssp_shape}_*.jsp"))
    if max_instances is not None:
        files = files[:max_instances]
    return files


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


def _routing_workdir(output_dir: Path, scenario_name: str, solver: str) -> Path:
    work_dir = output_dir / "artifacts" / scenario_name / solver
    work_dir.mkdir(parents=True, exist_ok=True)
    return work_dir


def _format_flat_array(values: list[int]) -> str:
    return ", ".join(str(value) for value in values)


def _write_jssp_data_file(path: Path, num_machines: int, jobs: list[list[tuple[int, int]]]) -> None:
    num_jobs = len(jobs)
    num_ops = len(jobs[0]) if jobs else 0
    machine_values = [machine for job in jobs for machine, _ in job]
    duration_values = [duration for job in jobs for _, duration in job]
    text = "\n".join(
        [
            f"num_jobs = {num_jobs};",
            f"num_machines = {num_machines};",
            f"num_ops = {num_ops};",
            f"machine_of = array2d(1..{num_jobs}, 1..{num_ops}, [{_format_flat_array(machine_values)}]);",
            f"duration = array2d(1..{num_jobs}, 1..{num_ops}, [{_format_flat_array(duration_values)}]);",
            "",
        ]
    )
    path.write_text(text, encoding="utf-8")


def _write_ffsp_data_file(path: Path, run_time: np.ndarray, num_stages: int, num_machines: int) -> None:
    num_jobs = int(run_time.shape[0])
    values: list[int] = []
    for job in range(num_jobs):
        for stage in range(num_stages):
            for machine in range(num_machines):
                values.append(int(run_time[job, stage * num_machines + machine]))
    text = "\n".join(
        [
            f"num_jobs = {num_jobs};",
            f"num_stages = {num_stages};",
            f"num_machines = {num_machines};",
            (
                f"duration = array3d(1..{num_jobs}, 1..{num_stages}, 1..{num_machines}, "
                f"[{_format_flat_array(values)}]);"
            ),
            "",
        ]
    )
    path.write_text(text, encoding="utf-8")


def _load_available_solver_configs(minizinc_bin: str) -> list[dict[str, object]]:
    try:
        completed = subprocess.run(
            [minizinc_bin, "--solvers-json"],
            capture_output=True,
            text=True,
            check=True,
        )
        payload = json.loads(completed.stdout)
        if isinstance(payload, list):
            return [cfg for cfg in payload if isinstance(cfg, dict)]
    except Exception:
        pass

    try:
        completed = subprocess.run(
            [minizinc_bin, "--solvers"],
            capture_output=True,
            text=True,
            check=True,
        )
    except Exception:
        return []

    configs: list[dict[str, object]] = []
    for line in completed.stdout.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        configs.append({"id": stripped, "name": stripped, "tags": []})
    return configs


def resolve_solver_ids(minizinc_bin: str, requested: list[str]) -> dict[str, str | None]:
    configs = _load_available_solver_configs(minizinc_bin)
    resolved: dict[str, str | None] = {}
    normalized = []
    for cfg in configs:
        id_value = str(cfg.get("id", "")).strip()
        name_value = str(cfg.get("name", "")).strip()
        tags = [str(tag).strip() for tag in cfg.get("tags", []) if str(tag).strip()]
        version = str(cfg.get("version", "")).strip()
        haystacks = [id_value.lower(), name_value.lower(), version.lower(), " ".join(tag.lower() for tag in tags)]
        normalized.append((cfg, haystacks))

    for solver_name in requested:
        resolved_id: str | None = None
        for candidate in SOLVER_CANDIDATES[solver_name]:
            candidate_lower = candidate.lower()
            match = next(
                (
                    cfg
                    for cfg, haystacks in normalized
                    if any(candidate_lower in haystack for haystack in haystacks if haystack)
                ),
                None,
            )
            if match is not None:
                resolved_id = str(match.get("id") or match.get("name") or candidate)
                break
        resolved[solver_name] = resolved_id
    return resolved


def _parse_json_stream(stdout_text: str) -> tuple[str | None, float | None, str]:
    final_status: str | None = None
    objective: float | None = None
    last_output = ""

    for raw_line in stdout_text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        try:
            message = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(message, dict):
            continue
        msg_type = str(message.get("type", ""))
        if msg_type == "solution":
            output = message.get("output", {})
            if isinstance(output, dict):
                raw_output = output.get("raw")
                if raw_output is None:
                    raw_output = output.get("default")
                if isinstance(raw_output, str):
                    last_output = raw_output
                    match = re.search(r"makespan\s*=\s*(-?\d+(?:\.\d+)?)", raw_output)
                    if match is not None:
                        objective = float(match.group(1))
        elif msg_type == "status":
            final_status = str(message.get("status", "")).strip() or None

    return final_status, objective, last_output


def _status_from_minizinc(final_status: str | None, objective: float | None, timed_out: bool) -> str:
    if final_status == "OPTIMAL_SOLUTION":
        return "optimal"
    if final_status == "ALL_SOLUTIONS":
        return "optimal"
    if final_status == "UNSATISFIABLE":
        return "unsat"
    if final_status == "UNBOUNDED":
        return "unbounded"
    if final_status == "UNSAT_OR_UNBOUNDED":
        return "unsat_or_unbounded"
    if final_status == "ERROR":
        return "error"
    if objective is not None:
        return "feasible"
    if timed_out:
        return "timeout"
    if final_status == "UNKNOWN":
        return "unknown"
    return "unknown"


def _run_minizinc_instance(
    *,
    solver_name: str,
    solver_id: str | None,
    scenario_name: str,
    instance_id: str,
    model_path: Path,
    data_path: Path,
    minizinc_bin: str,
    time_limit_s: float,
    timeout_grace_s: float,
) -> PerInstanceResult:
    if solver_id is None:
        return PerInstanceResult(
            scenario_name,
            solver_name,
            instance_id,
            "missing",
            None,
            0.0,
            notes="solver_not_registered_in_minizinc",
        )

    cmd = [
        minizinc_bin,
        "--solver",
        solver_id,
        "--json-stream",
        "--output-time",
        model_path.as_posix(),
        data_path.as_posix(),
    ]

    t0 = time.perf_counter()
    timed_out = False
    stdout_text = ""
    stderr_text = ""
    return_code = 0
    try:
        completed = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=max(1.0, float(time_limit_s + timeout_grace_s)),
            check=False,
        )
        stdout_text = completed.stdout or ""
        stderr_text = completed.stderr or ""
        return_code = int(completed.returncode)
    except subprocess.TimeoutExpired as exc:
        timed_out = True
        stdout_text = (exc.stdout or "") if isinstance(exc.stdout, str) else (exc.stdout or b"").decode(errors="ignore")
        stderr_text = (exc.stderr or "") if isinstance(exc.stderr, str) else (exc.stderr or b"").decode(errors="ignore")
        return_code = -9
    elapsed = time.perf_counter() - t0

    final_status, objective, last_output = _parse_json_stream(stdout_text)
    status = _status_from_minizinc(final_status, objective, timed_out)
    notes_parts = [f"solver_id={solver_id}"]
    if final_status is not None:
        notes_parts.append(f"mzn_status={final_status}")
    if timed_out:
        notes_parts.append("subprocess_timeout")
    if return_code != 0 and status not in {"optimal", "feasible"}:
        notes_parts.append(f"returncode={return_code}")
    if stderr_text.strip():
        notes_parts.append(stderr_text.strip().replace("\n", " ")[:240])
    elif stdout_text.strip() and status not in {"optimal", "feasible"}:
        notes_parts.append(stdout_text.strip().replace("\n", " ")[:240])
    elif last_output.strip() and status in {"optimal", "feasible"}:
        notes_parts.append(last_output.strip()[:120])

    return PerInstanceResult(
        scenario_name,
        solver_name,
        instance_id,
        status,
        objective,
        elapsed,
        notes="; ".join(notes_parts)[:500],
    )


def _run_single_jssp_task(
    solver_name: str,
    solver_id: str | None,
    cfg: ScenarioConfig,
    path: Path,
    output_dir: Path,
    minizinc_bin: str,
    timeout_grace_s: float,
) -> PerInstanceResult:
    num_machines, jobs = parse_jssp_file(path)
    work_dir = _routing_workdir(output_dir, cfg.name, solver_name)
    data_path = work_dir / f"{path.stem}.dzn"
    _write_jssp_data_file(data_path, num_machines, jobs)
    return _run_minizinc_instance(
        solver_name=solver_name,
        solver_id=solver_id,
        scenario_name=cfg.name,
        instance_id=path.name,
        model_path=MODELS_DIR / "jssp_disjunctive.mzn",
        data_path=data_path,
        minizinc_bin=minizinc_bin,
        time_limit_s=cfg.time_limit_s,
        timeout_grace_s=timeout_grace_s,
    )


def _run_single_ffsp_task(
    solver_name: str,
    solver_id: str | None,
    cfg: ScenarioConfig,
    run_time: np.ndarray,
    idx: int,
    output_dir: Path,
    minizinc_bin: str,
    timeout_grace_s: float,
) -> PerInstanceResult:
    assert cfg.ffsp_stages is not None and cfg.ffsp_machines is not None
    instance_id = f"{cfg.name}_{idx:05d}"
    work_dir = _routing_workdir(output_dir, cfg.name, solver_name)
    data_path = work_dir / f"{instance_id}.dzn"
    _write_ffsp_data_file(data_path, run_time, cfg.ffsp_stages, cfg.ffsp_machines)
    return _run_minizinc_instance(
        solver_name=solver_name,
        solver_id=solver_id,
        scenario_name=cfg.name,
        instance_id=instance_id,
        model_path=MODELS_DIR / "ffsp_parallel_machine_assignment.mzn",
        data_path=data_path,
        minizinc_bin=minizinc_bin,
        time_limit_s=cfg.time_limit_s,
        timeout_grace_s=timeout_grace_s,
    )


def _execute_parallel(
    solver_name: str,
    tasks: list[tuple],
    worker_fn,
    workers: int,
) -> tuple[list[PerInstanceResult], SolverBatchSummary]:
    max_workers = max(1, min(int(workers), len(tasks) if tasks else 1))
    t0 = time.perf_counter()
    rows: list[PerInstanceResult] = []
    if max_workers == 1:
        for task in tasks:
            rows.append(worker_fn(*task))
    else:
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(worker_fn, *task) for task in tasks]
            for future in as_completed(futures):
                rows.append(future.result())
    wall_clock = time.perf_counter() - t0
    rows.sort(key=lambda row: row.instance_id)
    return rows, SolverBatchSummary(
        scenario=rows[0].scenario if rows else "",
        solver=solver_name,
        wall_clock_s=wall_clock,
        workers=max_workers,
        solver_threads=1,
    )


def benchmark_scenario(
    cfg: ScenarioConfig,
    args: argparse.Namespace,
    solver_ids: dict[str, str | None],
) -> tuple[list[PerInstanceResult], list[SolverBatchSummary]]:
    scenario_rows: list[PerInstanceResult] = []
    summaries: list[SolverBatchSummary] = []

    if cfg.problem == "jssp":
        files = _load_jssp_instances(cfg, args.max_instances)
        for solver_name in args.solvers:
            solver_id = solver_ids.get(solver_name)
            tasks = [
                (
                    solver_name,
                    solver_id,
                    cfg,
                    path,
                    args.output_dir.resolve(),
                    args.minizinc_bin,
                    args.timeout_grace_sec,
                )
                for path in files
            ]
            print(f"[{cfg.name}/{solver_name}] launching {len(tasks)} instances", flush=True)
            rows, summary = _execute_parallel(solver_name, tasks, _run_single_jssp_task, args.workers)
            scenario_rows.extend(rows)
            summaries.append(summary)
        return scenario_rows, summaries

    if cfg.problem == "ffsp":
        instances = _load_ffsp_instances(cfg, args.seed, args.max_instances)
        for solver_name in args.solvers:
            solver_id = solver_ids.get(solver_name)
            tasks = [
                (
                    solver_name,
                    solver_id,
                    cfg,
                    run_time,
                    idx,
                    args.output_dir.resolve(),
                    args.minizinc_bin,
                    args.timeout_grace_sec,
                )
                for idx, run_time in enumerate(instances)
            ]
            print(f"[{cfg.name}/{solver_name}] launching {len(tasks)} instances", flush=True)
            rows, summary = _execute_parallel(solver_name, tasks, _run_single_ffsp_task, args.workers)
            scenario_rows.extend(rows)
            summaries.append(summary)
        return scenario_rows, summaries

    raise ValueError(f"Unsupported scenario problem: {cfg.problem}")


def write_outputs(rows: list[PerInstanceResult], batch_summaries: list[SolverBatchSummary], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    per_instance_csv = output_dir / "per_instance.csv"
    summary_csv = output_dir / "summary.csv"
    summary_json = output_dir / "summary.json"

    with per_instance_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
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
        accepted_rows = [row for row in solver_rows if row.status in SUCCESS_STATUSES]
        objective_rows = [row for row in accepted_rows if row.objective is not None]
        sum_instance_elapsed = float(sum(row.elapsed_s for row in solver_rows))
        batch_summary = batch_lookup.get((scenario, solver))
        total_elapsed = batch_summary.wall_clock_s if batch_summary is not None else sum_instance_elapsed
        summary_records.append(
            {
                "scenario": scenario,
                "solver": solver,
                "count": len(solver_rows),
                "ok_count": len(accepted_rows),
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

    with summary_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
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
                "success_statuses": sorted(SUCCESS_STATUSES),
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def main() -> int:
    args = parse_args()
    solver_ids = resolve_solver_ids(args.minizinc_bin, list(args.solvers))

    rows: list[PerInstanceResult] = []
    batch_summaries: list[SolverBatchSummary] = []
    for scenario_name in args.scenarios:
        cfg = SCENARIOS[scenario_name]
        scenario_rows, scenario_summaries = benchmark_scenario(cfg, args, solver_ids)
        rows.extend(scenario_rows)
        batch_summaries.extend(scenario_summaries)
        print(f"[{scenario_name}] finished {len(scenario_rows)} solver-runs", flush=True)

    write_outputs(rows, batch_summaries, args.output_dir.resolve())
    print(f"Wrote benchmark outputs to {args.output_dir.resolve()}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
