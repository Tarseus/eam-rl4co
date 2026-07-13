from __future__ import annotations

import argparse
import csv
import json
import re
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_DIR = REPO_ROOT / "logs" / "classical_benchmark_mzn_chuffed_heavy"
SUCCESS_STATUSES = {"ok", "optimal", "feasible"}
SKIP_SUFFIXES = {".dzn", ".mzn", ".fzn", ".ozn"}
KNOWN_ARTIFACT_SUFFIXES = (
    ".dzn",
    ".mzn",
    ".fzn",
    ".ozn",
    ".solution",
    ".solutions",
    ".stdout",
    ".stderr",
    ".output",
    ".out",
    ".jsonl",
    ".json",
    ".log",
    ".txt",
)
FAILURE_MARKERS = (
    "=====ERROR=====",
    "ERROR:",
    "MINIZINC: ERROR",
    "TYPE ERROR",
    "EVALUATION ERROR",
    "ASSERTION FAILED",
    "FAILED",
    "EXCEPTION",
)


@dataclass(frozen=True)
class ArtifactResult:
    scenario: str
    solver: str
    instance_id: str
    status: str
    objective: float | None
    elapsed_s: float | None
    source_path: str
    notes: str = ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build per-instance and aggregate CSV summaries from Chuffed solution "
            "artifacts under logs/classical_benchmark_mzn_chuffed_heavy/artifacts."
        )
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help="Benchmark output directory containing artifacts/<scenario>/<solver>.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for per_instance.csv, summary.csv, and summary.json. Defaults to --input-dir.",
    )
    parser.add_argument(
        "--solver",
        default="chuffed",
        help="Solver subdirectory to scan below every scenario directory.",
    )
    parser.add_argument(
        "--scenarios",
        nargs="+",
        default=None,
        help="Optional scenario names to scan. Defaults to every artifacts/*/<solver> directory.",
    )
    return parser.parse_args()


def _parse_objective_from_text(text: str) -> float | None:
    patterns = (
        r"\bmakespan\s*=\s*(-?\d+(?:\.\d+)?)",
        r"\bobjective\s*=\s*(-?\d+(?:\.\d+)?)",
        r"\bobjective\s*:\s*(-?\d+(?:\.\d+)?)",
        r"\bobj(?:ective)?\s*=\s*(-?\d+(?:\.\d+)?)",
        r"\bcost\s*=\s*(-?\d+(?:\.\d+)?)",
    )
    for pattern in patterns:
        matches = re.findall(pattern, text, flags=re.IGNORECASE)
        if matches:
            return float(matches[-1])
    return None


def _parse_elapsed_from_text(text: str) -> float | None:
    patterns = (
        r"\belapsed_s\s*=\s*(\d+(?:\.\d+)?)",
        r"\belapsed(?:\s+time)?\s*[:=]\s*(\d+(?:\.\d+)?)\s*s\b",
        r"\btime(?:\s+elapsed)?\s*[:=]\s*(\d+(?:\.\d+)?)\s*s\b",
        r"\bsolveTime\s*[:=]\s*(\d+(?:\.\d+)?)",
    )
    for pattern in patterns:
        matches = re.findall(pattern, text, flags=re.IGNORECASE)
        if matches:
            return float(matches[-1])
    return None


def _status_from_minizinc(final_status: str | None, objective: float | None, text: str) -> str:
    normalized = (final_status or "").strip().upper()
    upper_text = text.upper()
    if normalized in {"OPTIMAL_SOLUTION", "ALL_SOLUTIONS", "OPTIMAL"}:
        return "optimal"
    if normalized in {"SATISFIED", "SATISFIABLE", "FEASIBLE"}:
        return "feasible"
    if normalized == "UNSATISFIABLE" or "UNSATISFIABLE" in upper_text:
        return "unsat"
    if normalized == "UNBOUNDED" or "UNBOUNDED" in upper_text:
        return "unbounded"
    if normalized == "ERROR" or any(marker in upper_text for marker in FAILURE_MARKERS):
        return "error"
    if "TIMEOUT" in upper_text or "TIME LIMIT" in upper_text or "TIMED OUT" in upper_text:
        return "timeout"
    if objective is not None:
        if "==========" in text:
            return "optimal"
        return "feasible"
    if normalized == "UNKNOWN" or "UNKNOWN" in upper_text:
        return "unknown"
    return "unknown"


def _parse_json_stream(text: str) -> tuple[str | None, float | None, float | None, str]:
    final_status: str | None = None
    objective: float | None = None
    elapsed_s: float | None = None
    last_output = ""

    for raw_line in text.splitlines():
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
                    parsed = _parse_objective_from_text(raw_output)
                    if parsed is not None:
                        objective = parsed
        elif msg_type == "status":
            final_status = str(message.get("status", "")).strip() or None
        elif msg_type == "error":
            final_status = "ERROR"
            message_text = message.get("message")
            if isinstance(message_text, str):
                last_output = message_text
        elif msg_type == "statistics":
            stats = message.get("statistics", {})
            if isinstance(stats, dict):
                for key in ("time", "solveTime", "flatTime"):
                    value = stats.get(key)
                    if isinstance(value, (int, float)):
                        elapsed_s = float(value)

    return final_status, objective, elapsed_s, last_output


def parse_artifact_file(path: Path, scenario: str, solver: str, root: Path) -> ArtifactResult | None:
    if path.suffix.lower() in SKIP_SUFFIXES:
        return None
    if not path.is_file():
        return None

    text = path.read_text(encoding="utf-8", errors="ignore")
    if not text.strip():
        return None

    final_status, objective, elapsed_s, last_output = _parse_json_stream(text)
    if objective is None:
        objective = _parse_objective_from_text(text)
    if elapsed_s is None:
        elapsed_s = _parse_elapsed_from_text(text)

    status = _status_from_minizinc(final_status, objective, text)
    if objective is None and status == "unknown":
        return None

    source_path = path.relative_to(root).as_posix()
    notes_parts = []
    if final_status:
        notes_parts.append(f"mzn_status={final_status}")
    if last_output.strip():
        notes_parts.append(last_output.strip().replace("\n", " ")[:160])

    return ArtifactResult(
        scenario=scenario,
        solver=solver,
        instance_id=_derive_instance_id(path),
        status=status,
        objective=objective,
        elapsed_s=elapsed_s,
        source_path=source_path,
        notes="; ".join(notes_parts)[:500],
    )


def missing_output_result(path: Path, scenario: str, solver: str, root: Path) -> ArtifactResult:
    return ArtifactResult(
        scenario=scenario,
        solver=solver,
        instance_id=_derive_instance_id(path),
        status="missing_output",
        objective=None,
        elapsed_s=None,
        source_path=path.relative_to(root).as_posix(),
        notes="data_file_without_parsed_solver_output",
    )


def _derive_instance_id(path: Path) -> str:
    name = path.name
    changed = True
    while changed:
        changed = False
        lower = name.lower()
        for suffix in KNOWN_ARTIFACT_SUFFIXES:
            if lower.endswith(suffix):
                name = name[: -len(suffix)]
                changed = True
                break
    return name


def _discover_scenarios(artifact_root: Path, solver: str) -> list[str]:
    if not artifact_root.is_dir():
        return []
    scenarios = []
    for path in artifact_root.iterdir():
        if path.is_dir() and (path / solver).is_dir():
            scenarios.append(path.name)
    return sorted(scenarios)


def _prefer_result(current: ArtifactResult | None, candidate: ArtifactResult) -> ArtifactResult:
    if current is None:
        return candidate
    if current.status == "missing_output" and candidate.status != "missing_output":
        return candidate
    if current.status != "missing_output" and candidate.status == "missing_output":
        return current
    current_ok = current.status in SUCCESS_STATUSES
    candidate_ok = candidate.status in SUCCESS_STATUSES
    if candidate_ok and not current_ok:
        return candidate
    if current_ok and not candidate_ok:
        return current
    if candidate.objective is not None and current.objective is None:
        return candidate
    if current.objective is not None and candidate.objective is None:
        return current
    if current.objective is not None and candidate.objective is not None:
        if candidate.objective < current.objective:
            return candidate
        if candidate.objective > current.objective:
            return current
    if candidate.status == "optimal" and current.status != "optimal":
        return candidate
    return current


def collect_artifact_results(input_dir: Path, solver: str, scenarios: list[str] | None) -> list[ArtifactResult]:
    artifact_root = input_dir / "artifacts"
    scenario_names = scenarios if scenarios is not None else _discover_scenarios(artifact_root, solver)
    deduped: dict[tuple[str, str, str], ArtifactResult] = {}

    for scenario in scenario_names:
        solver_dir = artifact_root / scenario / solver
        if not solver_dir.is_dir():
            continue
        for path in sorted(p for p in solver_dir.rglob("*") if p.is_file()):
            if path.suffix.lower() in SKIP_SUFFIXES:
                parsed = missing_output_result(path, scenario, solver, input_dir)
                key = (parsed.scenario, parsed.solver, parsed.instance_id)
                deduped[key] = _prefer_result(deduped.get(key), parsed)
                continue
            parsed = parse_artifact_file(path, scenario, solver, input_dir)
            if parsed is None:
                continue
            key = (parsed.scenario, parsed.solver, parsed.instance_id)
            deduped[key] = _prefer_result(deduped.get(key), parsed)

    return sorted(deduped.values(), key=lambda row: (row.scenario, row.solver, row.instance_id))


def build_summary_records(rows: list[ArtifactResult]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str], list[ArtifactResult]] = defaultdict(list)
    for row in rows:
        grouped[(row.scenario, row.solver)].append(row)

    records: list[dict[str, object]] = []
    for (scenario, solver), solver_rows in sorted(grouped.items()):
        accepted_rows = [row for row in solver_rows if row.status in SUCCESS_STATUSES]
        objective_rows = [row for row in accepted_rows if row.objective is not None]
        elapsed_values = [row.elapsed_s for row in solver_rows if row.elapsed_s is not None]
        sum_instance_elapsed = float(sum(elapsed_values)) if elapsed_values else None
        records.append(
            {
                "scenario": scenario,
                "solver": solver,
                "count": len(solver_rows),
                "ok_count": len(accepted_rows),
                "total_elapsed_s": sum_instance_elapsed,
                "sum_instance_elapsed_s": sum_instance_elapsed,
                "avg_elapsed_s": (
                    float(sum(elapsed_values) / len(elapsed_values)) if elapsed_values else None
                ),
                "avg_objective_ok": (
                    float(sum(row.objective for row in objective_rows) / len(objective_rows))
                    if objective_rows
                    else None
                ),
                "best_objective_ok": (
                    float(min(row.objective for row in objective_rows)) if objective_rows else None
                ),
                "workers": 1,
                "solver_threads": 1,
            }
        )
    return records


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_outputs(rows: list[ArtifactResult], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    per_instance_rows = [
        {
            "scenario": row.scenario,
            "solver": row.solver,
            "instance_id": row.instance_id,
            "status": row.status,
            "objective": row.objective,
            "elapsed_s": row.elapsed_s,
            "source_path": row.source_path,
            "notes": row.notes,
        }
        for row in rows
    ]
    _write_csv(
        output_dir / "per_instance.csv",
        ["scenario", "solver", "instance_id", "status", "objective", "elapsed_s", "source_path", "notes"],
        per_instance_rows,
    )

    summary_records = build_summary_records(rows)
    summary_fields = [
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
    ]
    _write_csv(output_dir / "summary.csv", summary_fields, summary_records)
    (output_dir / "summary.json").write_text(
        json.dumps(
            {
                "records": summary_records,
                "created_at_epoch_s": time.time(),
                "success_statuses": sorted(SUCCESS_STATUSES),
                "source": "summarize_chuffed_artifacts.py",
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def main() -> int:
    args = parse_args()
    input_dir = args.input_dir.resolve()
    output_dir = (args.output_dir or input_dir).resolve()
    rows = collect_artifact_results(input_dir, args.solver, args.scenarios)
    write_outputs(rows, output_dir)
    print(
        f"Wrote {len(rows)} per-instance rows and {len(build_summary_records(rows))} summary rows to {output_dir}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
