from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
DEFAULT_DATASET = REPO_ROOT / "data/vrp/agfn_vrp1000_capacity50_test128.npz"


@dataclass(frozen=True)
class ReferenceResult:
    instance_index: int
    source_index: int
    status: str
    cost: float | None
    scaled_integer_cost: float | None
    elapsed_sec: float
    notes: str
    error: str | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run LKH-3 on a fixed CVRP NPZ dataset with route feasibility audits."
    )
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-instances", type=int, default=None)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--time-limit-sec", type=float, default=600.0)
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument(
        "--max-trials",
        type=int,
        default=1000,
        help="LKH MAX_TRIALS per run; use 100/1000/10000 for the paper-style variants.",
    )
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Reuse successful rows from output-dir/summary.json when the protocol matches.",
    )
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_dataset(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as payload:
        required = {"depot", "locs", "demand", "capacity"}
        missing = required.difference(payload.files)
        if missing:
            raise KeyError(f"Dataset is missing arrays: {sorted(missing)}")
        arrays = {key: np.asarray(payload[key]) for key in required}
        arrays["source_index"] = (
            np.asarray(payload["source_index"], dtype=np.int64)
            if "source_index" in payload.files
            else np.arange(len(arrays["locs"]), dtype=np.int64)
        )
    count = len(arrays["locs"])
    if arrays["locs"].ndim != 3 or arrays["locs"].shape[-1] != 2:
        raise ValueError(f"Expected locs [B, N, 2], got {arrays['locs'].shape}")
    expected = {
        "depot": (count, 2),
        "demand": (count, arrays["locs"].shape[1]),
        "capacity": (count,),
        "source_index": (count,),
    }
    for key, shape in expected.items():
        if arrays[key].shape != shape:
            raise ValueError(f"Expected {key} shape {shape}, got {arrays[key].shape}")
    if not np.isfinite(arrays["depot"]).all() or not np.isfinite(arrays["locs"]).all():
        raise ValueError("Coordinates contain non-finite values")
    return arrays


def _solve(
    *,
    instance_index: int,
    source_index: int,
    depot: np.ndarray,
    locs: np.ndarray,
    demand: np.ndarray,
    capacity: float,
    seed: int,
    time_limit_sec: float,
    runs: int,
    max_trials: int,
    output_dir: Path,
) -> ReferenceResult:
    try:
        from scripts.classical_solver_benchmark import run_cvrp_lkh

        result = run_cvrp_lkh(
            {
                "depot": depot,
                "locs": locs,
                "demand": demand,
                "capacity": capacity,
            },
            seed=seed + source_index,
            instance_id=f"cvrp{len(locs)}_{instance_index:05d}",
            scenario_name=f"cvrp{len(locs)}",
            time_limit_sec=time_limit_sec,
            runs=runs,
            output_dir=output_dir,
            max_trials=max_trials,
        )
        scaled_cost = float(result.objective) if result.objective is not None else None
        return ReferenceResult(
            instance_index=instance_index,
            source_index=source_index,
            status=result.status,
            cost=scaled_cost / 100_000.0 if scaled_cost is not None else None,
            scaled_integer_cost=scaled_cost,
            elapsed_sec=float(result.elapsed_s),
            notes=result.notes,
        )
    except Exception as exc:  # noqa: BLE001
        return ReferenceResult(
            instance_index=instance_index,
            source_index=source_index,
            status="error",
            cost=None,
            scaled_integer_cost=None,
            elapsed_sec=0.0,
            notes="",
            error=f"{type(exc).__name__}: {exc}",
        )


def _make_summary(
    *, config: dict[str, Any], rows: list[ReferenceResult]
) -> dict[str, Any]:
    successful = [row for row in rows if row.status == "ok" and row.cost is not None]
    return {
        "protocol": "cvrp_lkh_euc2d_fixed_reference_v1",
        "config": config,
        "num_completed": len(rows),
        "num_successful": len(successful),
        "mean_cost": (
            float(np.mean([row.cost for row in successful])) if successful else None
        ),
        "mean_elapsed_sec": (
            float(np.mean([row.elapsed_sec for row in successful]))
            if successful
            else None
        ),
        "rows": [asdict(row) for row in sorted(rows, key=lambda item: item.instance_index)],
    }


def _write_outputs(output_dir: Path, summary: dict[str, Any]) -> None:
    rows = summary["rows"]
    if rows:
        with (output_dir / "per_instance.csv").open(
            "w", encoding="utf-8", newline=""
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    temporary = output_dir / "summary.json.tmp"
    temporary.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(output_dir / "summary.json")


def main() -> int:
    args = parse_args()
    if args.max_instances is not None and args.max_instances < 1:
        raise ValueError("max-instances must be >= 1")
    if (
        args.time_limit_sec <= 0
        or args.runs < 1
        or args.max_trials < 1
        or args.workers < 1
    ):
        raise ValueError("time-limit-sec, runs, max-trials, and workers must be positive")

    dataset = args.dataset.resolve()
    arrays = _load_dataset(dataset)
    count = len(arrays["locs"])
    if args.max_instances is not None:
        count = min(count, args.max_instances)
        arrays = {key: value[:count] for key, value in arrays.items()}
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "dataset": str(dataset),
        "dataset_sha256": _sha256(dataset),
        "num_instances": count,
        "num_customers": int(arrays["locs"].shape[1]),
        "capacity_values": sorted({float(value) for value in arrays["capacity"]}),
        "seed": int(args.seed),
        "time_limit_sec": float(args.time_limit_sec),
        "runs": int(args.runs),
        "max_trials": int(args.max_trials),
        "workers": int(args.workers),
        "coordinate_scale": 100_000,
        "distance_type": "TSPLIB EUC_2D after coordinate scaling",
    }

    rows_by_index: dict[int, ReferenceResult] = {}
    summary_path = output_dir / "summary.json"
    if args.resume and summary_path.is_file():
        previous = json.loads(summary_path.read_text(encoding="utf-8"))
        previous_config = previous.get("config", {})
        signature_keys = (
            "dataset_sha256",
            "num_instances",
            "seed",
            "time_limit_sec",
            "runs",
            "max_trials",
            "coordinate_scale",
        )
        if any(previous_config.get(key) != config[key] for key in signature_keys):
            raise ValueError("Existing summary uses a different LKH protocol")
        for raw in previous.get("rows", []):
            row = ReferenceResult(**raw)
            if row.status == "ok":
                rows_by_index[row.instance_index] = row

    pending = [index for index in range(count) if index not in rows_by_index]
    if pending:
        with ProcessPoolExecutor(max_workers=min(args.workers, len(pending))) as executor:
            futures = {
                executor.submit(
                    _solve,
                    instance_index=index,
                    source_index=int(arrays["source_index"][index]),
                    depot=arrays["depot"][index],
                    locs=arrays["locs"][index],
                    demand=arrays["demand"][index],
                    capacity=float(arrays["capacity"][index]),
                    seed=int(args.seed),
                    time_limit_sec=float(args.time_limit_sec),
                    runs=int(args.runs),
                    max_trials=int(args.max_trials),
                    output_dir=output_dir,
                ): index
                for index in pending
            }
            for future in as_completed(futures):
                row = future.result()
                rows_by_index[row.instance_index] = row
                summary = _make_summary(config=config, rows=list(rows_by_index.values()))
                _write_outputs(output_dir, summary)
                print(
                    f"[result] instance={row.source_index} status={row.status} "
                    f"cost={row.cost} elapsed={row.elapsed_sec:.2f}s error={row.error}",
                    flush=True,
                )

    summary = _make_summary(config=config, rows=list(rows_by_index.values()))
    _write_outputs(output_dir, summary)
    print(
        f"[done] successful={summary['num_successful']}/{count} "
        f"mean_cost={summary['mean_cost']} output_dir={output_dir}",
        flush=True,
    )
    return 0 if summary["num_successful"] == count else 1


if __name__ == "__main__":
    raise SystemExit(main())
