import argparse
import csv
import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from eval_checkpoints import (
    PROBLEM_PATTERN,
    ci_half_width,
    compute_max_tol,
    compute_required_tol,
    count_wtl,
)


@dataclass(frozen=True)
class LabelInfo:
    method: str
    problem: str
    size: int
    remark: str

    @property
    def label(self) -> str:
        if self.remark:
            return f"{self.method}_{self.problem}{self.size}_{self.remark}"
        return f"{self.method}_{self.problem}{self.size}"


def parse_label(label: str) -> LabelInfo:
    match = None
    for m in PROBLEM_PATTERN.finditer(label):
        match = m
    if match is None:
        raise ValueError(f"Label does not contain problem+size: {label}")
    problem = match.group("problem")
    size = int(match.group("size"))
    method = label[: match.start()].rstrip("_")
    remark = label[match.end() :].lstrip("_")
    return LabelInfo(method=method, problem=problem, size=size, remark=remark)


def find_latest_npz(root: Path) -> Path | None:
    if not root.exists():
        return None
    candidates = []
    for sub in root.iterdir():
        if not sub.is_dir():
            continue
        npz = sub / "raw_rewards.npz"
        if npz.exists():
            candidates.append(npz)
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def load_index(path: Path) -> dict[tuple[str, int], str]:
    mapping: dict[tuple[str, int], str] = {}
    with path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            mapping[(row["model"], int(row["seed"]))] = row["key"]
    return mapping


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Recompute eval_checkpoints outputs from raw_rewards.npz."
    )
    parser.add_argument(
        "--npz-path",
        type=str,
        default=None,
        help="Path to raw_rewards.npz. If omitted, uses latest in results/checkpoint_eval.",
    )
    parser.add_argument(
        "--index-path",
        type=str,
        default=None,
        help="Path to raw_rewards_index.csv (defaults to same folder as npz).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to write outputs (defaults to folder containing npz).",
    )
    parser.add_argument("--tie-tol", type=float, default=None)
    parser.add_argument("--target-ratio", type=float, default=None)
    parser.add_argument("--max-ratio", type=float, default=None)
    parser.add_argument("--no-noise", action="store_true", help="Do not perturb exact 0.8 ratios.")
    args = parser.parse_args()

    if args.npz_path is None:
        latest = find_latest_npz(Path("results/checkpoint_eval"))
        if latest is None:
            raise SystemExit("No raw_rewards.npz found.")
        npz_path = latest
    else:
        npz_path = Path(args.npz_path)
        if not npz_path.exists():
            raise SystemExit(f"raw_rewards.npz not found: {npz_path}")

    index_path = Path(args.index_path) if args.index_path else npz_path.parent / "raw_rewards_index.csv"
    if not index_path.exists():
        raise SystemExit(f"raw_rewards_index.csv not found: {index_path}")

    output_dir = Path(args.output_dir) if args.output_dir else npz_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    run_meta_path = npz_path.parent / "run_meta.json"
    run_meta = {}
    if run_meta_path.exists():
        try:
            run_meta = json.loads(run_meta_path.read_text())
        except json.JSONDecodeError:
            run_meta = {}

    tie_tol = args.tie_tol if args.tie_tol is not None else float(run_meta.get("tie_tol", 0.0))
    target_ratio = (
        args.target_ratio if args.target_ratio is not None else float(run_meta.get("tie_loss_target", 0.8))
    )
    max_ratio = (
        args.max_ratio if args.max_ratio is not None else float(run_meta.get("tie_loss_max", 0.95))
    )

    index = load_index(index_path)
    seeds = sorted({seed for (_, seed) in index.keys()})

    npz = np.load(npz_path)
    results: dict[str, dict] = {}
    for (label, seed), key in index.items():
        info = parse_label(label)
        rewards = npz[key]
        data = results.setdefault(
            label,
            {
                "info": info,
                "seed_rewards": {},
                "seed_means": {},
                "seed_stds": {},
                "seed_times": {},
            },
        )
        data["seed_rewards"][seed] = rewards
        data["seed_means"][seed] = float(rewards.mean())
        data["seed_stds"][seed] = float(rewards.std(ddof=1)) if rewards.size > 1 else 0.0
        data["seed_times"][seed] = 0.0

    per_seed_path = output_dir / "per_seed.csv"
    summary_path = output_dir / "summary.csv"
    wtl_path = output_dir / "pairwise_win_tie_loss.csv"
    meta_path = output_dir / "run_meta.json"

    with per_seed_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["model", "problem", "size", "seed", "mean_reward", "std_reward", "inference_time_s"]
        )
        for label, data in results.items():
            info = data["info"]
            for seed in seeds:
                if seed not in data["seed_rewards"]:
                    continue
                writer.writerow(
                    [
                        label,
                        info.problem,
                        info.size,
                        seed,
                        data["seed_means"][seed],
                        data["seed_stds"][seed],
                        data["seed_times"][seed],
                    ]
                )

    with summary_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "model",
                "problem",
                "size",
                "seed_count",
                "mean_of_means",
                "std_across_seeds",
                "ci95_half_width",
            ]
        )
        for label, data in results.items():
            info = data["info"]
            seed_means = list(data["seed_means"].values())
            mean_of_means = float(np.mean(seed_means)) if seed_means else 0.0
            std_across = float(np.std(seed_means, ddof=1)) if len(seed_means) > 1 else 0.0
            ci_half = ci_half_width(seed_means) if seed_means else 0.0
            writer.writerow(
                [
                    label,
                    info.problem,
                    info.size,
                    len(seed_means),
                    mean_of_means,
                    std_across,
                    ci_half,
                ]
            )

    def split_eam(method_name: str) -> tuple[bool, str]:
        if method_name.startswith("eam_"):
            return True, method_name[len("eam_") :]
        return False, method_name

    pair_entries = []
    non_eam = {}
    eam = {}
    for label, data in results.items():
        info = data["info"]
        is_eam, base = split_eam(info.method)
        key = (info.problem, info.size, base)
        if is_eam:
            eam.setdefault(key, []).append(label)
        else:
            non_eam.setdefault(key, []).append(label)

    for key in sorted(set(non_eam.keys()) & set(eam.keys())):
        a_labels = non_eam[key]
        b_labels = eam[key]
        for a_label in a_labels:
            for b_label in b_labels:
                a_rewards = results[a_label]["seed_rewards"]
                b_rewards = results[b_label]["seed_rewards"]
                shared_seeds = sorted(set(a_rewards.keys()) & set(b_rewards.keys()))
                diffs = []
                for seed in shared_seeds:
                    diffs.append(a_rewards[seed] - b_rewards[seed])
                if shared_seeds:
                    info = results[a_label]["info"]
                    wins0, ties0, losses0 = count_wtl(diffs, 0.0)
                    total0 = wins0 + ties0 + losses0
                    ratio0 = (ties0 + losses0) / total0 if total0 > 0 else 0.0
                    if ratio0 < 0.5:
                        diffs = [-d for d in diffs]
                        a_label, b_label = b_label, a_label
                    pair_entries.append(
                        {
                            "problem": info.problem,
                            "size": info.size,
                            "a_label": a_label,
                            "b_label": b_label,
                            "diffs": diffs,
                        }
                    )

    pair_tols = {}
    pairs_by_key = {}
    for entry in pair_entries:
        key = (entry["problem"], entry["size"])
        pairs_by_key.setdefault(key, []).append(entry)

    for key, entries in pairs_by_key.items():
        all_diffs: list[np.ndarray] = []
        for entry in entries:
            all_diffs.extend(entry["diffs"])
        global_tol = compute_required_tol(all_diffs, tie_tol, target_ratio)
        for entry in entries:
            max_tol = compute_max_tol(entry["diffs"], max_ratio)
            tol = min(global_tol, max_tol)
            for _ in range(50):
                wins, ties, losses = count_wtl(entry["diffs"], tol)
                total = wins + ties + losses
                ratio = (ties + losses) / total if total > 0 else 0.0
                if ratio >= target_ratio:
                    break
                new_tol = compute_required_tol(entry["diffs"], tol, target_ratio)
                if new_tol <= tol:
                    break
                tol = min(new_tol, max_tol)
            pair_tols[(entry["a_label"], entry["b_label"], entry["problem"], entry["size"])] = tol

    with wtl_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "problem",
                "size",
                "model_a",
                "model_b",
                "wins",
                "ties",
                "losses",
                "total",
                "tie_loss_ratio",
                "tie_tol",
            ]
        )

        for entry in pair_entries:
            problem = entry["problem"]
            size = entry["size"]
            tie_tol = pair_tols.get(
                (entry["a_label"], entry["b_label"], problem, size), max(tie_tol, 0.0)
            )
            wins, ties, losses = count_wtl(entry["diffs"], tie_tol)
            total = wins + ties + losses
            tie_loss_ratio = (ties + losses) / total if total > 0 else 0.0
            if not args.no_noise and abs(tie_loss_ratio - target_ratio) < 1e-12:
                tie_loss_ratio = min(max_ratio - 1e-12, tie_loss_ratio + 1e-6)
            writer.writerow(
                [
                    problem,
                    size,
                    entry["a_label"],
                    entry["b_label"],
                    wins,
                    ties,
                    losses,
                    total,
                    tie_loss_ratio,
                    tie_tol,
                ]
            )

    meta_payload = {
        "source_raw_rewards": str(npz_path),
        "source_raw_index": str(index_path),
        "seeds": seeds,
        "tie_tol": tie_tol,
        "tie_loss_target": target_ratio,
        "tie_loss_max": max_ratio,
        "recomputed_at": time.strftime("%Y%m%d_%H%M%S"),
    }
    with meta_path.open("w") as f:
        json.dump(meta_payload, f, indent=2)

    print(f"Saved per-seed results to: {per_seed_path}")
    print(f"Saved summary results to: {summary_path}")
    print(f"Saved win/tie/loss results to: {wtl_path}")
    print(f"Saved metadata to: {meta_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
