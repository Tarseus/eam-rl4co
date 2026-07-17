from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.stats import wilcoxon


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.train_jssp_large_objectives import _build_model, _evaluate, _resolve


EXPECTED_INITIAL_SHA256 = (
    "5a90dc4027c08ef7bb2192a84a752a638bd4c59c46a24bf2a57c3721baaf8584"
)


def _checkpoint_payload(path: Path) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    config = payload.get("large_scale_config")
    if not isinstance(config, dict):
        raise ValueError(f"Checkpoint has no large_scale_config: {path}")
    return payload


def _validate_protocol(
    *, label: str, method: str, payload: dict[str, Any], expected_step: int | None
) -> dict[str, Any]:
    config = payload["large_scale_config"]
    expected = {
        "method": method,
        "initial_state_sha256": EXPECTED_INITIAL_SHA256,
        "shape": "50x20",
        "physical_batch_size": 1,
        "rollouts_per_instance": 128,
        "select_k": 16,
        "seed": 12345678,
        "pair_scope": "strictly within instance",
    }
    mismatches = {
        key: {"expected": value, "observed": config.get(key)}
        for key, value in expected.items()
        if config.get(key) != value
    }
    step = int(payload.get("optimizer_step", -1))
    if expected_step is not None and step != expected_step:
        mismatches["optimizer_step"] = {
            "expected": expected_step,
            "observed": step,
        }
    if mismatches:
        raise ValueError(f"Protocol mismatch for {label}: {mismatches}")
    return {
        "method": method,
        "optimizer_step": step,
        "best_cost": float(payload["best_cost"]),
        "learning_rate": float(config["learning_rate"]),
        "weight_decay": float(config["weight_decay"]),
        "alpha": float(config["alpha"]),
    }


def _evaluate_checkpoint(
    *,
    args: argparse.Namespace,
    label: str,
    method: str,
    checkpoint: Path,
    expected_step: int | None,
    device: torch.device,
) -> dict[str, Any]:
    payload = _checkpoint_payload(checkpoint)
    protocol = _validate_protocol(
        label=label, method=method, payload=payload, expected_step=expected_step
    )
    model_args = argparse.Namespace(
        method=method,
        num_jobs=50,
        num_machines=20,
        rollouts=128,
        select_k=16,
        po_alpha=0.25,
        alpha=protocol["alpha"],
        eval_rollouts=128,
        greedy=0,
        usw_pair=args.usw_pair,
        asw_pair=args.asw_pair,
    )
    model, _ = _build_model(model_args, checkpoint)
    model = model.to(device)
    result = _evaluate(
        model,
        jobs=50,
        machines=20,
        count=args.validation_count,
        rollouts=128,
        batch_size=args.validation_batch_size,
        seed=args.seed + args.validation_seed_offset,
        device=device,
    )
    costs = np.asarray(result.pop("per_instance_cost"), dtype=np.float64)
    if costs.shape != (args.validation_count,) or not np.isfinite(costs).all():
        raise FloatingPointError(f"Invalid per-instance costs for {label}: {costs}")
    return {
        "label": label,
        "checkpoint": str(checkpoint),
        "protocol": protocol,
        **result,
        "per_instance_cost": costs.tolist(),
    }


def _bootstrap_mean_ci(
    differences: np.ndarray, *, samples: int, seed: int
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    means = np.empty(samples, dtype=np.float64)
    chunk = 2_000
    for start in range(0, samples, chunk):
        stop = min(start + chunk, samples)
        indices = rng.integers(
            0, differences.size, size=(stop - start, differences.size)
        )
        means[start:stop] = differences[indices].mean(axis=1)
    low, high = np.percentile(means, [2.5, 97.5])
    return float(low), float(high)


def _holm_adjust(raw_p_values: list[float]) -> list[float]:
    order = sorted(range(len(raw_p_values)), key=raw_p_values.__getitem__)
    adjusted = [math.nan] * len(raw_p_values)
    running = 0.0
    family_size = len(raw_p_values)
    for rank, index in enumerate(order):
        running = max(running, (family_size - rank) * raw_p_values[index])
        adjusted[index] = min(1.0, running)
    return adjusted


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Paired validation32 inference for the locked JSSP50x20 H4 gate."
    )
    parser.add_argument("--bopo-checkpoint", type=Path, required=True)
    parser.add_argument("--usw-checkpoint", type=Path, required=True)
    parser.add_argument("--asw-checkpoint", type=Path, required=True)
    parser.add_argument("--bopo-step", type=int)
    parser.add_argument("--usw-step", type=int, default=300)
    parser.add_argument("--asw-step", type=int, default=200)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=12345678)
    parser.add_argument("--validation-seed-offset", type=int, default=80_000_003)
    parser.add_argument("--validation-count", type=int, default=32)
    parser.add_argument("--validation-batch-size", type=int, default=8)
    parser.add_argument("--bootstrap-samples", type=int, default=20_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260718)
    parser.add_argument(
        "--usw-pair",
        type=Path,
        default=Path(
            "runs/pref_loss_jssp10x10_from_ffsp100_elite/"
            "20260416-113409/best_pair.json"
        ),
    )
    parser.add_argument(
        "--asw-pair",
        type=Path,
        default=Path(
            "runs/pref_builder_weight_search_jssp10x10_from_best_loss/"
            "20260417-123033/best_pair.json"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.validation_count != 32 or args.validation_batch_size != 8:
        raise ValueError("H4 requires validation count 32 and batch size 8")
    if args.validation_seed_offset != 80_000_003:
        raise ValueError("H4 requires the locked validation seed offset 80000003")
    if args.bootstrap_samples < 1:
        raise ValueError("bootstrap-samples must be positive")

    output = _resolve(args.output)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite paired result: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    specs = [
        ("bopo", "bopo", _resolve(args.bopo_checkpoint), args.bopo_step),
        ("usw", "usw", _resolve(args.usw_checkpoint), args.usw_step),
        ("asw", "asw", _resolve(args.asw_checkpoint), args.asw_step),
    ]
    evaluations = {
        label: _evaluate_checkpoint(
            args=args,
            label=label,
            method=method,
            checkpoint=checkpoint,
            expected_step=expected_step,
            device=device,
        )
        for label, method, checkpoint, expected_step in specs
    }

    baseline = np.asarray(evaluations["bopo"]["per_instance_cost"], dtype=np.float64)
    comparisons: list[dict[str, Any]] = []
    raw_p_values: list[float] = []
    for index, label in enumerate(("usw", "asw")):
        candidate = np.asarray(
            evaluations[label]["per_instance_cost"], dtype=np.float64
        )
        differences = candidate - baseline
        low, high = _bootstrap_mean_ci(
            differences,
            samples=args.bootstrap_samples,
            seed=args.bootstrap_seed + index,
        )
        test = wilcoxon(
            differences,
            alternative="two-sided",
            zero_method="wilcox",
            method="auto",
        )
        raw_p = float(test.pvalue)
        raw_p_values.append(raw_p)
        comparisons.append(
            {
                "candidate": label,
                "baseline": "bopo",
                "count": int(differences.size),
                "candidate_minus_baseline_mean": float(differences.mean()),
                "candidate_wins": int((differences < 0).sum()),
                "ties": int((differences == 0).sum()),
                "candidate_losses": int((differences > 0).sum()),
                "bootstrap_95ci_mean_difference": [low, high],
                "wilcoxon_statistic": float(test.statistic),
                "wilcoxon_raw_p": raw_p,
                "per_instance_difference": differences.tolist(),
            }
        )

    adjusted = _holm_adjust(raw_p_values)
    for comparison, adjusted_p in zip(comparisons, adjusted, strict=True):
        comparison["wilcoxon_holm_p"] = adjusted_p
        comparison["passes_locked_inference"] = (
            comparison["bootstrap_95ci_mean_difference"][1] < 0.0
            and adjusted_p < 0.05
        )

    result = {
        "protocol": "jssp50x20_h4_paired_validation32_v1",
        "selection_data_only": True,
        "ta_dmu_final_data_read": False,
        "seed": args.seed,
        "validation_seed_offset": args.validation_seed_offset,
        "validation_count": args.validation_count,
        "validation_batch_size": args.validation_batch_size,
        "rollouts_per_instance": 128,
        "bootstrap_samples": args.bootstrap_samples,
        "bootstrap_seed": args.bootstrap_seed,
        "holm_family": ["usw_vs_bopo", "asw_vs_bopo"],
        "evaluations": evaluations,
        "comparisons": comparisons,
    }
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
