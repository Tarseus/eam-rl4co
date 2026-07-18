from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


METHODS = ("rl", "po", "sll", "bopo", "loss_only", "weighting")
SHAPES = ("jssp10x10", "jssp15x15")


def holm_adjust(p_values: list[float]) -> list[float]:
    order = np.argsort(np.asarray(p_values, dtype=float))
    adjusted = np.empty(len(p_values), dtype=float)
    running = 0.0
    count = len(p_values)
    for rank, idx in enumerate(order):
        candidate = min(1.0, (count - rank) * float(p_values[idx]))
        running = max(running, candidate)
        adjusted[idx] = running
    return adjusted.tolist()


def bootstrap_mean_ci(values: np.ndarray, *, seed: int, draws: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    n = len(values)
    means = np.empty(draws, dtype=float)
    chunk = 2_000
    for offset in range(0, draws, chunk):
        size = min(chunk, draws - offset)
        indices = rng.integers(0, n, size=(size, n))
        means[offset : offset + size] = values[indices].mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return float(low), float(high)


def rank_biserial(delta: np.ndarray) -> float:
    nonzero = delta[delta != 0]
    if len(nonzero) == 0:
        return 0.0
    ranks = stats.rankdata(np.abs(nonzero), method="average")
    positive = float(ranks[nonzero > 0].sum())
    negative = float(ranks[nonzero < 0].sum())
    return (positive - negative) / float(ranks.sum())


def load_rows(root: Path, shape: str, method: str) -> pd.DataFrame:
    path = root / shape / method / "generated.csv"
    frame = pd.read_csv(path).sort_values("instance").reset_index(drop=True)
    required = {"instance", "ref_makespan", "pred_makespan", "gap"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{path} missing columns: {sorted(missing)}")
    if len(frame) != 100 or frame["instance"].duplicated().any():
        raise ValueError(f"{path} must contain 100 unique instances")
    numeric = frame[["ref_makespan", "pred_makespan", "gap"]].to_numpy(dtype=float)
    if not np.isfinite(numeric).all():
        raise ValueError(f"{path} contains non-finite values")
    return frame


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-draws", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=20_260_718)
    args = parser.parse_args()

    comparisons: list[dict[str, object]] = []
    rankings: dict[str, list[dict[str, object]]] = {}
    dataset_digests: dict[str, str] = {}

    for shape_idx, shape in enumerate(SHAPES):
        frames = {method: load_rows(args.root, shape, method) for method in ("slim", *METHODS)}
        slim = frames["slim"]
        order_blob = "\n".join(
            f"{row.instance}|{float(row.ref_makespan):.9f}" for row in slim.itertuples()
        ).encode("utf-8")
        dataset_digests[shape] = hashlib.sha256(order_blob).hexdigest()

        for method, frame in frames.items():
            if not frame["instance"].equals(slim["instance"]):
                raise ValueError(f"Instance order mismatch: {shape}/{method}")
            if not np.array_equal(
                frame["ref_makespan"].to_numpy(), slim["ref_makespan"].to_numpy()
            ):
                raise ValueError(f"Reference makespan mismatch: {shape}/{method}")

        rankings[shape] = sorted(
            (
                {
                    "method": method,
                    "mean_makespan": float(frame["pred_makespan"].mean()),
                    "sd_makespan": float(frame["pred_makespan"].std(ddof=1)),
                    "mean_gap_pct": float(frame["gap"].mean()),
                }
                for method, frame in frames.items()
            ),
            key=lambda item: float(item["mean_makespan"]),
        )

        slim_values = slim["pred_makespan"].to_numpy(dtype=float)
        for method_idx, method in enumerate(METHODS):
            other_values = frames[method]["pred_makespan"].to_numpy(dtype=float)
            delta = slim_values - other_values
            nonzero = delta[delta != 0]
            if len(nonzero):
                wilcoxon = stats.wilcoxon(
                    delta,
                    zero_method="wilcox",
                    correction=False,
                    alternative="two-sided",
                    method="auto",
                )
                wilcoxon_stat = float(wilcoxon.statistic)
                wilcoxon_p = float(wilcoxon.pvalue)
            else:
                wilcoxon_stat = 0.0
                wilcoxon_p = 1.0
            t_result = stats.ttest_rel(slim_values, other_values)
            ci_low, ci_high = bootstrap_mean_ci(
                delta,
                seed=int(args.seed) + shape_idx * 100 + method_idx,
                draws=int(args.bootstrap_draws),
            )
            wins = int((delta < 0).sum())
            ties = int((delta == 0).sum())
            losses = int((delta > 0).sum())
            comparisons.append(
                {
                    "shape": shape,
                    "comparison": f"slim_vs_{method}",
                    "method": method,
                    "n": int(len(delta)),
                    "slim_mean_makespan": float(slim_values.mean()),
                    "method_mean_makespan": float(other_values.mean()),
                    "slim_minus_method_mean": float(delta.mean()),
                    "slim_minus_method_ci95_low": ci_low,
                    "slim_minus_method_ci95_high": ci_high,
                    "slim_relative_change_pct": float(delta.mean() / other_values.mean() * 100.0),
                    "slim_wins": wins,
                    "ties": ties,
                    "slim_losses": losses,
                    "slim_probability_superiority": float((wins + 0.5 * ties) / len(delta)),
                    "wilcoxon_statistic": wilcoxon_stat,
                    "wilcoxon_p_raw": wilcoxon_p,
                    "paired_t_p_raw": float(t_result.pvalue),
                    "rank_biserial_slim_minus_method": rank_biserial(delta),
                }
            )

    adjusted = holm_adjust([float(row["wilcoxon_p_raw"]) for row in comparisons])
    for row, p_adjusted in zip(comparisons, adjusted, strict=True):
        row["wilcoxon_p_holm12"] = p_adjusted
        row["significant_holm12_0.05"] = bool(p_adjusted < 0.05)

    payload = {
        "protocol": {
            "test_instances_per_shape": 100,
            "B": 128,
            "greedy": 0,
            "augmentation_factor": 1,
            "sampling_seed_base": 12_345_678,
            "pairing": "same instance and same rollout-sampling seed per checkpoint",
            "primary_test": "two-sided paired Wilcoxon signed-rank",
            "multiplicity": "Holm correction over 12 SLIM-vs-method comparisons",
            "bootstrap": {
                "draws": int(args.bootstrap_draws),
                "seed_base": int(args.seed),
                "unit": "instance",
            },
            "delta_sign": "SLIM minus comparator; positive means SLIM is worse",
        },
        "dataset_order_sha256": dataset_digests,
        "rankings": rankings,
        "comparisons": comparisons,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    pd.DataFrame(comparisons).to_csv(args.output.with_suffix(".csv"), index=False)
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
