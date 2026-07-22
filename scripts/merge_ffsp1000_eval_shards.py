from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge disjoint FFSP1000 evaluation shards.")
    parser.add_argument("--shard-dir", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite merged output: {args.output_dir}")

    summaries: list[dict[str, object]] = []
    rows: list[dict[str, object]] = []
    for shard_dir in args.shard_dir:
        summaries.append(json.loads((shard_dir / "summary.json").read_text(encoding="utf-8")))
        with (shard_dir / "per_instance.csv").open(newline="", encoding="utf-8") as handle:
            rows.extend(csv.DictReader(handle))

    invariant_keys = (
        "protocol",
        "label",
        "method",
        "checkpoint",
        "checkpoint_sha256",
        "optimizer_step",
        "test_file",
        "test_file_sha256",
        "num_starts",
        "augmentation_factor",
    )
    reference = summaries[0]
    for summary in summaries[1:]:
        for key in invariant_keys:
            if summary[key] != reference[key]:
                raise ValueError(f"Shard invariant mismatch for {key}")

    rows.sort(key=lambda row: int(row["instance_index"]))
    indices = [int(row["instance_index"]) for row in rows]
    if indices != list(range(100)):
        raise ValueError(f"Expected exact instance indices 0..99, observed {indices}")
    costs = [float(row["cost"]) for row in rows]
    if not all(math.isfinite(cost) for cost in costs):
        raise FloatingPointError("Non-finite cost in FFSP1000 shards")

    args.output_dir.mkdir(parents=True)
    with (args.output_dir / "per_instance.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    merged = {key: reference[key] for key in invariant_keys}
    merged.update(
        {
            "instance_count": 100,
            "start_index": 0,
            "end_index": 100,
            "mean_cost": float(np.mean(costs)),
            "total_elapsed_sec_across_shards": float(
                sum(float(summary["total_elapsed_sec"]) for summary in summaries)
            ),
            "mean_elapsed_sec_per_instance": float(
                np.mean([float(row["batch_time_sec_per_instance"]) for row in rows])
            ),
            "shard_directories": [str(path) for path in args.shard_dir],
            "test_only_no_selection": True,
        }
    )
    (args.output_dir / "summary.json").write_text(
        json.dumps(merged, indent=2), encoding="utf-8"
    )
    print(json.dumps({"event": "merged_summary", **merged}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
