from __future__ import annotations

import argparse
import csv
import json
import time
from collections import defaultdict
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_DIR = REPO_ROOT / "logs" / "classical_benchmark"
DEFAULT_SUCCESS_STATUSES = ("ok", "optimal", "feasible")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Backfill classical benchmark summary metrics from per-instance results, "
            "treating statuses like optimal/feasible as successful solves."
        )
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help="Directory containing per_instance.csv and the existing summary files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for repaired summary files. Defaults to <input-dir> unless --in-place is omitted.",
    )
    parser.add_argument(
        "--in-place",
        action="store_true",
        help="Overwrite summary.csv and summary.json inside --input-dir.",
    )
    parser.add_argument(
        "--success-status",
        nargs="+",
        default=list(DEFAULT_SUCCESS_STATUSES),
        help="Statuses that should count as successful solves when aggregating summary metrics.",
    )
    return parser.parse_args()


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _to_float(value: str | None) -> float | None:
    if value is None:
        return None
    stripped = str(value).strip()
    if not stripped:
        return None
    return float(stripped)


def _to_int(value: str | None) -> int | None:
    if value is None:
        return None
    stripped = str(value).strip()
    if not stripped:
        return None
    return int(stripped)


def load_legacy_summary(summary_csv: Path) -> dict[tuple[str, str], dict[str, object]]:
    if not summary_csv.is_file():
        return {}

    lookup: dict[tuple[str, str], dict[str, object]] = {}
    for row in _read_csv_rows(summary_csv):
        key = (row["scenario"], row["solver"])
        lookup[key] = {
            "ok_count": _to_int(row.get("ok_count")),
            "total_elapsed_s": _to_float(row.get("total_elapsed_s")),
            "avg_objective_ok": _to_float(row.get("avg_objective_ok")),
            "workers": _to_int(row.get("workers")),
            "solver_threads": _to_int(row.get("solver_threads")),
        }
    return lookup


def build_summary_records(
    per_instance_rows: list[dict[str, str]],
    legacy_summary: dict[tuple[str, str], dict[str, object]],
    success_statuses: set[str],
) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in per_instance_rows:
        grouped[(row["scenario"], row["solver"])].append(row)

    records: list[dict[str, object]] = []
    for (scenario, solver), solver_rows in sorted(grouped.items()):
        accepted_rows = [row for row in solver_rows if row["status"] in success_statuses]
        objective_values = [
            objective
            for objective in (_to_float(row.get("objective")) for row in accepted_rows)
            if objective is not None
        ]
        sum_instance_elapsed = float(
            sum(_to_float(row.get("elapsed_s")) or 0.0 for row in solver_rows)
        )
        legacy = legacy_summary.get((scenario, solver), {})
        total_elapsed = legacy.get("total_elapsed_s")
        if total_elapsed is None:
            total_elapsed = sum_instance_elapsed

        records.append(
            {
                "scenario": scenario,
                "solver": solver,
                "count": len(solver_rows),
                "ok_count": len(accepted_rows),
                "total_elapsed_s": float(total_elapsed),
                "sum_instance_elapsed_s": sum_instance_elapsed,
                "avg_elapsed_s": (
                    sum_instance_elapsed / len(solver_rows) if solver_rows else None
                ),
                "avg_objective_ok": (
                    float(sum(objective_values) / len(objective_values))
                    if objective_values
                    else None
                ),
                "best_objective_ok": (
                    float(min(objective_values)) if objective_values else None
                ),
                "workers": int(legacy.get("workers") or 1),
                "solver_threads": int(legacy.get("solver_threads") or 1),
            }
        )
    return records


def write_summary_outputs(
    records: list[dict[str, object]],
    output_dir: Path,
    stem: str,
    success_statuses: set[str],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_csv = output_dir / f"{stem}.csv"
    summary_json = output_dir / f"{stem}.json"

    fieldnames = [
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

    with summary_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow(record)

    summary_json.write_text(
        json.dumps(
            {
                "records": records,
                "created_at_epoch_s": time.time(),
                "success_statuses": sorted(success_statuses),
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def main() -> int:
    args = parse_args()
    input_dir = args.input_dir.resolve()
    output_dir = (
        input_dir if args.in_place else (args.output_dir.resolve() if args.output_dir else input_dir)
    )
    stem = "summary" if args.in_place else "summary_backfilled"

    per_instance_csv = input_dir / "per_instance.csv"
    legacy_summary_csv = input_dir / "summary.csv"
    if not per_instance_csv.is_file():
        raise FileNotFoundError(f"Missing per-instance results: {per_instance_csv}")

    success_statuses = {status.strip().lower() for status in args.success_status if status.strip()}
    per_instance_rows = _read_csv_rows(per_instance_csv)
    legacy_summary = load_legacy_summary(legacy_summary_csv)
    records = build_summary_records(per_instance_rows, legacy_summary, success_statuses)
    write_summary_outputs(records, output_dir, stem, success_statuses)

    repaired_rows = [row for row in records if row["avg_objective_ok"] is not None]
    newly_filled_rows = [
        row
        for row in records
        if row["avg_objective_ok"] is not None
        and (row["scenario"], row["solver"]) in legacy_summary
        and legacy_summary[(row["scenario"], row["solver"])].get("avg_objective_ok") is None
    ]
    print(
        f"Wrote {len(records)} summary rows to {output_dir / (stem + '.csv')} "
        f"using success statuses: {sorted(success_statuses)}",
        flush=True,
    )
    print(f"Rows with successful objective aggregates: {len(repaired_rows)}", flush=True)
    print(f"Rows newly backfilled from missing legacy objectives: {len(newly_filled_rows)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
