from __future__ import annotations

import csv
import json
from pathlib import Path

from scripts import classical_solver_minizinc_benchmark as mod


def test_parse_json_stream_marks_unknown_with_solution_as_feasible() -> None:
    stream = "\n".join(
        [
            json.dumps(
                {
                    "type": "solution",
                    "output": {
                        "default": "makespan=123\n",
                        "raw": "makespan=123\n",
                    },
                }
            ),
            json.dumps({"type": "status", "status": "UNKNOWN"}),
        ]
    )

    final_status, objective, raw = mod._parse_json_stream(stream)

    assert final_status == "UNKNOWN"
    assert objective == 123.0
    assert raw == "makespan=123\n"
    assert mod._status_from_minizinc(final_status, objective, timed_out=False) == "feasible"


def test_resolve_solver_ids_prefers_registered_ids(monkeypatch) -> None:
    monkeypatch.setattr(
        mod,
        "_load_available_solver_configs",
        lambda _: [
            {"id": "org.chuffed.chuffed", "name": "Chuffed", "tags": ["cp"]},
            {"id": "org.minizinc.scip", "name": "SCIP", "tags": ["mip"]},
        ],
    )

    resolved = mod.resolve_solver_ids("minizinc", ["chuffed", "scip"])

    assert resolved == {
        "chuffed": "org.chuffed.chuffed",
        "scip": "org.minizinc.scip",
    }


def test_write_outputs_counts_optimal_and_feasible_as_success(tmp_path: Path) -> None:
    rows = [
        mod.PerInstanceResult("jssp10x10", "chuffed", "a.jsp", "optimal", 100.0, 1.0, ""),
        mod.PerInstanceResult("jssp10x10", "chuffed", "b.jsp", "feasible", 110.0, 2.0, ""),
        mod.PerInstanceResult("jssp10x10", "scip", "a.jsp", "missing", None, 0.0, ""),
    ]
    summaries = [
        mod.SolverBatchSummary("jssp10x10", "chuffed", 2.5, 1, 1),
        mod.SolverBatchSummary("jssp10x10", "scip", 0.1, 1, 1),
    ]

    mod.write_outputs(rows, summaries, tmp_path)

    with (tmp_path / "summary.csv").open("r", encoding="utf-8", newline="") as handle:
        summary_rows = list(csv.DictReader(handle))

    chuffed_row = next(row for row in summary_rows if row["solver"] == "chuffed")
    assert chuffed_row["ok_count"] == "2"
    assert chuffed_row["avg_objective_ok"] == "105.0"
    assert chuffed_row["best_objective_ok"] == "100.0"

    scip_row = next(row for row in summary_rows if row["solver"] == "scip")
    assert scip_row["ok_count"] == "0"
    assert scip_row["avg_objective_ok"] == ""
