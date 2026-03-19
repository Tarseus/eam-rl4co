import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


DEFAULT_VARIANTS = ["resample", "random_only", "ls_only", "eam"]
DEFAULT_METRICS = ["diversity", "gain", "delta_nll"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot mechanism curves from outputs/mechanism/{task}{size}_{backbone}/seed*/{variant}.csv"
    )
    parser.add_argument("--root", type=Path, default=Path("outputs/mechanism"))
    parser.add_argument("--task", required=True)
    parser.add_argument("--size", type=int, required=True)
    parser.add_argument("--backbone", required=True)
    parser.add_argument("--variants", nargs="+", default=DEFAULT_VARIANTS)
    parser.add_argument("--metrics", nargs="+", default=DEFAULT_METRICS)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def load_rows(csv_path: Path) -> list[dict[str, float]]:
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = []
        for row in reader:
            rows.append(
                {
                    "step": float(row["step"]),
                    "epoch": float(row["epoch"]),
                    "diversity": float(row["diversity"]),
                    "gain": float(row["gain"]),
                    "delta_nll": float(row["delta_nll"]),
                }
            )
    return rows


def aggregate_by_step(rows: list[dict[str, float]], metrics: list[str]) -> dict[float, dict[str, float]]:
    buckets: dict[float, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        step = row["step"]
        for metric in metrics:
            buckets[step][metric].append(row[metric])

    aggregated = {}
    for step, metric_values in buckets.items():
        aggregated[step] = {
            metric: sum(values) / len(values) for metric, values in metric_values.items()
        }
    return aggregated


def main() -> int:
    args = parse_args()
    run_dir = args.root / f"{args.task}{args.size}_{args.backbone}"
    if not run_dir.exists():
        raise SystemExit(f"Run directory not found: {run_dir}")

    output_path = args.output or (run_dir / "mechanism_plot.png")
    figure, axes = plt.subplots(len(args.metrics), 1, figsize=(10, 3 * len(args.metrics)), sharex=True)
    if len(args.metrics) == 1:
        axes = [axes]

    for variant in args.variants:
        all_rows = []
        for seed_dir in sorted(run_dir.glob("seed*")):
            csv_path = seed_dir / f"{variant}.csv"
            if csv_path.exists():
                all_rows.extend(load_rows(csv_path))
        if not all_rows:
            continue
        aggregated = aggregate_by_step(all_rows, args.metrics)
        steps = sorted(aggregated.keys())
        for axis, metric in zip(axes, args.metrics):
            axis.plot(steps, [aggregated[step][metric] for step in steps], label=variant)
            axis.set_ylabel(metric)
            axis.grid(True, alpha=0.3)

    axes[-1].set_xlabel("step")
    axes[0].set_title(f"{args.task}{args.size}_{args.backbone}")
    axes[0].legend()
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=200)
    print(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
