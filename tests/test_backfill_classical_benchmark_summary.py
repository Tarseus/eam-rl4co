from __future__ import annotations

import csv
import sys
from pathlib import Path

from scripts import backfill_classical_benchmark_summary as mod


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def test_main_backfills_optimal_and_feasible_rows(tmp_path: Path, monkeypatch) -> None:
    input_dir = tmp_path / "classical_benchmark"
    per_instance_csv = input_dir / "per_instance.csv"
    summary_csv = input_dir / "summary.csv"

    _write_csv(
        per_instance_csv,
        [
            {
                "scenario": "ffsp50",
                "solver": "ortools_cp_sat",
                "instance_id": "ffsp50_00000",
                "status": "feasible",
                "objective": "51",
                "elapsed_s": "180.0",
                "notes": "",
            },
            {
                "scenario": "ffsp50",
                "solver": "ortools_cp_sat",
                "instance_id": "ffsp50_00001",
                "status": "feasible",
                "objective": "61",
                "elapsed_s": "180.0",
                "notes": "",
            },
            {
                "scenario": "tsp50",
                "solver": "lkh",
                "instance_id": "tsp50_00000",
                "status": "ok",
                "objective": "475346",
                "elapsed_s": "0.2",
                "notes": "",
            },
        ],
    )
    _write_csv(
        summary_csv,
        [
            {
                "scenario": "ffsp50",
                "solver": "ortools_cp_sat",
                "count": "2",
                "ok_count": "0",
                "total_elapsed_s": "123.0",
                "sum_instance_elapsed_s": "360.0",
                "avg_elapsed_s": "180.0",
                "avg_objective_ok": "",
                "best_objective_ok": "",
                "workers": "128",
                "solver_threads": "2",
            },
            {
                "scenario": "tsp50",
                "solver": "lkh",
                "count": "1",
                "ok_count": "1",
                "total_elapsed_s": "9.0",
                "sum_instance_elapsed_s": "0.2",
                "avg_elapsed_s": "0.2",
                "avg_objective_ok": "475346.0",
                "best_objective_ok": "475346.0",
                "workers": "256",
                "solver_threads": "1",
            },
        ],
    )

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "backfill_classical_benchmark_summary.py",
            "--input-dir",
            str(input_dir),
        ],
    )

    exit_code = mod.main()

    assert exit_code == 0
    output_csv = input_dir / "summary_backfilled.csv"
    with output_csv.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    ffsp_row = next(
        row for row in rows if row["scenario"] == "ffsp50" and row["solver"] == "ortools_cp_sat"
    )
    assert ffsp_row["ok_count"] == "2"
    assert ffsp_row["avg_objective_ok"] == "56.0"
    assert ffsp_row["best_objective_ok"] == "51.0"
    assert ffsp_row["total_elapsed_s"] == "123.0"
    assert ffsp_row["workers"] == "128"
    assert ffsp_row["solver_threads"] == "2"

    tsp_row = next(row for row in rows if row["scenario"] == "tsp50" and row["solver"] == "lkh")
    assert tsp_row["ok_count"] == "1"
    assert tsp_row["avg_objective_ok"] == "475346.0"
