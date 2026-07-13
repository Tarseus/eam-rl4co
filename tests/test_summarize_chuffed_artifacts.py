from __future__ import annotations

import csv
import json
import shutil
import sys
import uuid
from pathlib import Path

from scripts import summarize_chuffed_artifacts as mod


def _workspace_tmp(name: str) -> Path:
    path = Path("tmp") / f"{name}_{uuid.uuid4().hex}"
    path.mkdir(parents=True)
    return path


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_collects_json_stream_and_plain_output_artifacts(monkeypatch) -> None:
    root = _workspace_tmp("summarize_chuffed")
    try:
        input_dir = root / "classical_benchmark_mzn_chuffed_heavy"
        ffsp_dir = input_dir / "artifacts" / "ffsp50" / "chuffed"
        jssp_dir = input_dir / "artifacts" / "jssp15x15" / "chuffed"
        ffsp_dir.mkdir(parents=True)
        jssp_dir.mkdir(parents=True)

        (ffsp_dir / "ffsp50_00000.jsonl").write_text(
            "\n".join(
                [
                    json.dumps({"type": "solution", "output": {"raw": "makespan = 120\n"}}),
                    json.dumps({"type": "status", "status": "UNKNOWN"}),
                    json.dumps({"type": "statistics", "statistics": {"solveTime": 2.5}}),
                ]
            ),
            encoding="utf-8",
        )
        (ffsp_dir / "ffsp50_00000.dzn").write_text("num_jobs = 50;", encoding="utf-8")
        (jssp_dir / "15x15_001.out").write_text("makespan = 915\n==========\n", encoding="utf-8")

        monkeypatch.setattr(
            sys,
            "argv",
            [
                "summarize_chuffed_artifacts.py",
                "--input-dir",
                str(input_dir),
                "--scenarios",
                "ffsp50",
                "jssp15x15",
            ],
        )

        assert mod.main() == 0

        per_instance = _read_csv(input_dir / "per_instance.csv")
        assert [(row["scenario"], row["instance_id"], row["status"], row["objective"]) for row in per_instance] == [
            ("ffsp50", "ffsp50_00000", "feasible", "120.0"),
            ("jssp15x15", "15x15_001", "optimal", "915.0"),
        ]

        summary = _read_csv(input_dir / "summary.csv")
        ffsp_row = next(row for row in summary if row["scenario"] == "ffsp50")
        assert ffsp_row["solver"] == "chuffed"
        assert ffsp_row["count"] == "1"
        assert ffsp_row["ok_count"] == "1"
        assert ffsp_row["avg_objective_ok"] == "120.0"
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_duplicate_artifacts_keep_best_objective() -> None:
    root = _workspace_tmp("summarize_chuffed")
    try:
        input_dir = root / "run"
        solver_dir = input_dir / "artifacts" / "ffsp100" / "chuffed"
        solver_dir.mkdir(parents=True)
        (solver_dir / "ffsp100_00001.out").write_text("makespan = 240\n", encoding="utf-8")
        (solver_dir / "ffsp100_00001.solution.txt").write_text("makespan = 230\n", encoding="utf-8")

        rows = mod.collect_artifact_results(input_dir, "chuffed", None)

        assert len(rows) == 1
        assert rows[0].instance_id == "ffsp100_00001"
        assert rows[0].objective == 230.0
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_data_files_without_outputs_are_counted_as_failures() -> None:
    root = _workspace_tmp("summarize_chuffed")
    try:
        input_dir = root / "run"
        solver_dir = input_dir / "artifacts" / "ffsp100" / "chuffed"
        solver_dir.mkdir(parents=True)
        (solver_dir / "ffsp100_00001.dzn").write_text("num_jobs = 100;", encoding="utf-8")
        (solver_dir / "ffsp100_00002.stderr").write_text("MiniZinc: error: solver failed\n", encoding="utf-8")

        rows = mod.collect_artifact_results(input_dir, "chuffed", None)

        assert [(row.instance_id, row.status) for row in rows] == [
            ("ffsp100_00001", "missing_output"),
            ("ffsp100_00002", "error"),
        ]
        summary = mod.build_summary_records(rows)
        assert summary[0]["count"] == 2
        assert summary[0]["ok_count"] == 0
        assert summary[0]["avg_objective_ok"] is None
    finally:
        shutil.rmtree(root, ignore_errors=True)
