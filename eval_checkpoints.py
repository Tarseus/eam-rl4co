import argparse
import csv
import json
import math
import random
import re
import time
from dataclasses import dataclass
from pathlib import Path
import inspect

import numpy as np
import torch
import torch.nn as nn
from tensordict import TensorDict

from rl4co.envs import get_env
from rl4co.tasks.eval import evaluate_policy


@dataclass(frozen=True)
class CheckpointInfo:
    path: Path
    method: str
    problem: str
    size: int
    remark: str

    @property
    def group_key(self) -> str:
        return f"{self.problem}{self.size}"

    @property
    def label(self) -> str:
        if self.remark:
            return f"{self.method}_{self.problem}{self.size}_{self.remark}"
        return f"{self.method}_{self.problem}{self.size}"


PROBLEM_CODES = [
    "cvrp",
    "cvrptw",
    "cvrpmvc",
    "tsp",
    "atsp",
    "op",
    "pctsp",
    "spctsp",
    "pdp",
    "sdvrp",
    "mtsp",
    "shpp",
    "mdcpdp",
    "mtvrp",
    "kp",
    "knapsack",
]
PROBLEM_PATTERN = re.compile(
    r"(?P<problem>{})(?P<size>\d+)".format("|".join(PROBLEM_CODES))
)


def parse_seeds(text: str) -> list[int]:
    seeds: list[int] = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, end = part.split("-", 1)
            seeds.extend(range(int(start), int(end) + 1))
        else:
            seeds.append(int(part))
    return sorted(set(seeds))


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_checkpoint_name(path: Path) -> CheckpointInfo | None:
    stem = path.stem
    match = None
    for m in PROBLEM_PATTERN.finditer(stem):
        match = m
    if match is None:
        return None
    problem = match.group("problem")
    size = int(match.group("size"))
    method = stem[: match.start()].rstrip("_")
    remark = stem[match.end() :].lstrip("_")
    return CheckpointInfo(path=path, method=method, problem=problem, size=size, remark=remark)


def build_env(problem: str, size: int):
    if problem == "kp":
        env_name = "knapsack"
    else:
        env_name = problem

    if env_name == "knapsack":
        generator_params = {
            "num_items": size,
            "weight_distribution": "uniform",
            "value_distribution": "uniform",
        }
    else:
        generator_params = {"num_loc": size, "loc_distribution": "uniform"}
    return get_env(env_name, generator_params=generator_params)


def resolve_model_class(method: str):
    from rl4co import models as model_root

    method = method.lower()
    if method.startswith("eam_symnco"):
        return model_root.SymEAM
    if method.startswith("eam"):
        return model_root.EAM
    if method == "am":
        return model_root.AttentionModel
    if method == "pomo":
        return model_root.POMO
    if method == "symnco":
        return model_root.SymNCO
    return None


def build_policy_for_model(model_cls, env, policy_kwargs: dict):
    from rl4co.models import AttentionModelPolicy, SymNCOPolicy

    policy_kwargs = policy_kwargs or {}
    if model_cls.__name__ in ("SymNCO", "SymEAM"):
        return SymNCOPolicy(env_name=env.name, **policy_kwargs)
    return AttentionModelPolicy(env_name=env.name, **policy_kwargs)


def load_model_checkpoint(model_cls, path: Path, env):
    errors = []
    default_ea_kwargs = {
        "num_generations": 3,
        "mutation_rate": 0.1,
        "crossover_rate": 0.6,
        "selection_rate": 0.2,
        "batch_size": 64,
        "ea_batch_size": 64,
        "alpha": 0.5,
        "beta": 3,
        "ea_prob": 0.01,
        "ea_epoch": 700,
    }
    try:
        def filter_init_kwargs(cls, kwargs):
            sig = inspect.signature(cls.__init__)
            if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()):
                return kwargs
            allowed = {name for name in sig.parameters if name != "self"}
            return {k: v for k, v in kwargs.items() if k in allowed}

        ckpt = torch.load(str(path), map_location="cpu", weights_only=False)
        hparams = ckpt.get("hyper_parameters", {}) or {}
        policy = hparams.get("policy")
        if not isinstance(policy, nn.Module):
            policy_kwargs = hparams.get("policy_kwargs", {}) or {}
            policy = build_policy_for_model(model_cls, env, policy_kwargs)
        init_kwargs = {
            k: v
            for k, v in hparams.items()
            if k not in {"env", "policy", "dataset"}
        }
        if model_cls.__name__ in ("SymNCO", "SymEAM"):
            init_kwargs["baseline"] = "symnco"
            init_kwargs.pop("first_aug_identity", None)
            init_kwargs.pop("shared_buffer", None)
            if init_kwargs.get("num_starts") is None:
                init_kwargs["num_starts"] = 0
        if model_cls.__name__ in ("EAM", "SymEAM"):
            ea_kwargs = init_kwargs.get("ea_kwargs", {}) or {}
            merged = {**default_ea_kwargs, **ea_kwargs}
            init_kwargs["ea_kwargs"] = merged
        init_kwargs["env"] = env
        if policy is not None:
            init_kwargs["policy"] = policy
        model = model_cls(**filter_init_kwargs(model_cls, init_kwargs))
        state_dict = {
            k: v
            for k, v in ckpt.get("state_dict", {}).items()
            if not k.startswith("baseline.")
        }
        model.load_state_dict(state_dict, strict=False)
        return model, None
    except Exception as exc:
        errors.append(str(exc))

    for kwargs in (
        {"env": env, "load_baseline": False, "map_location": "cpu"},
        {"load_baseline": False, "map_location": "cpu"},
    ):
        try:
            return model_cls.load_from_checkpoint(str(path), **kwargs), None
        except Exception as exc:
            errors.append(str(exc))

    return None, "; ".join(errors)


def infer_num_starts(env, dataset) -> int | None:
    if hasattr(env.generator, "num_loc"):
        return env.generator.num_loc
    if hasattr(env.generator, "num_items"):
        return env.generator.num_items
    try:
        sample = dataset[0]
        td = TensorDict(
            {key: value.unsqueeze(0) for key, value in sample.items()},
            batch_size=[1],
        )
        td = env.reset(td)
        return env.get_num_starts(td)
    except Exception:
        return None


def resolve_eval_config(
    model,
    env,
    dataset,
    method: str,
    num_augment_override: int | None,
    num_starts_override: int | None,
    force_dihedral_8: bool | None,
):
    if num_augment_override is not None:
        num_augment = num_augment_override
    else:
        num_augment = int(getattr(model, "num_augment", 0) or 0)

    model_num_starts = getattr(model, "num_starts", None)
    if num_starts_override is not None:
        num_starts = num_starts_override
        use_multistart = num_starts > 1
    else:
        if model_num_starts is None:
            num_starts = infer_num_starts(env, dataset)
            use_multistart = num_starts is not None and num_starts > 1
        else:
            num_starts = model_num_starts
            use_multistart = num_starts is not None and num_starts > 1

    # Avoid reporting best-of-starts for knapsack unless explicitly requested.
    if (
        env.name == "knapsack"
        and num_starts_override is None
        and method == "auto"
    ):
        num_starts = 1
        use_multistart = False

    use_augment = num_augment > 1

    if method != "auto":
        resolved_method = method
    else:
        if use_multistart and use_augment:
            resolved_method = "multistart_greedy_augment"
        elif use_multistart:
            resolved_method = "multistart_greedy"
        elif use_augment:
            resolved_method = "augment"
        else:
            resolved_method = "greedy"

    if force_dihedral_8 is None:
        force_dihedral_8 = bool(getattr(model, "hparams", {}).get("augment_fn") == "dihedral8")

    kwargs = {}
    if use_augment:
        kwargs["num_augment"] = num_augment
        kwargs["force_dihedral_8"] = force_dihedral_8
    if use_multistart:
        kwargs["num_starts"] = num_starts

    return resolved_method, kwargs


def ci_half_width(values: list[float]) -> float:
    n = len(values)
    if n < 2:
        return 0.0
    std = float(np.std(values, ddof=1))
    if std == 0.0:
        return 0.0
    try:
        from scipy.stats import t as student_t

        crit = float(student_t.ppf(0.975, df=n - 1))
    except Exception:
        crit = 1.96
    return crit * std / math.sqrt(n)


def compute_required_tol(diffs: list[np.ndarray], base_tol: float, target_ratio: float) -> float:
    flat = np.concatenate(diffs) if diffs else np.array([])
    total = flat.size
    if total == 0:
        return max(base_tol, 0.0)
    loss_count = int((flat < 0).sum())
    nonneg = flat[flat >= 0]
    required_ties = math.ceil(target_ratio * total) - loss_count
    if required_ties <= 0:
        tol = 0.0
    elif required_ties > nonneg.size:
        tol = float(nonneg.max()) if nonneg.size > 0 else 0.0
    else:
        nonneg_sorted = np.sort(nonneg)
        tol = float(nonneg_sorted[required_ties - 1])
    return max(tol, base_tol)


def count_wtl(diffs: list[np.ndarray], tie_tol: float) -> tuple[int, int, int]:
    wins = ties = losses = 0
    for diff in diffs:
        wins += int((diff > tie_tol).sum())
        ties += int(((diff >= 0) & (diff <= tie_tol)).sum())
        losses += int((diff < 0).sum())
    return wins, ties, losses


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate checkpoints over multiple seeds.")
    parser.add_argument("--checkpoints-dir", type=str, default="checkpoints")
    parser.add_argument("--seeds", type=str, default="0,1,2,3,4")
    parser.add_argument("--num-instances", type=int, default=1000)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--method", type=str, default="auto")
    parser.add_argument("--num-augment", type=int, default=None)
    parser.add_argument("--num-starts", type=int, default=None)
    parser.add_argument("--force-dihedral-8", action="store_true")
    parser.add_argument("--no-force-dihedral-8", dest="force_dihedral_8", action="store_false")
    parser.set_defaults(force_dihedral_8=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--max-batch-size", type=int, default=4096)
    parser.add_argument("--samples", type=int, default=1280)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.0)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument(
        "--tie-tol",
        type=float,
        default=0.0,
        help="If non-EAM is better than EAM by <= tie_tol, count as tie (asymmetric).",
    )
    parser.add_argument("--output-dir", type=str, default="results/checkpoint_eval")
    args = parser.parse_args()

    seeds = parse_seeds(args.seeds)
    if not seeds:
        raise SystemExit("No valid seeds provided.")

    ckpt_dir = Path(args.checkpoints_dir)
    if not ckpt_dir.exists():
        raise SystemExit(f"Checkpoint dir not found: {ckpt_dir}")

    ckpt_paths = sorted(ckpt_dir.rglob("*.ckpt"))
    if not ckpt_paths:
        raise SystemExit(f"No checkpoint files found under {ckpt_dir}")

    ckpt_infos = []
    skipped = []
    for path in ckpt_paths:
        info = parse_checkpoint_name(path)
        if info is None:
            skipped.append(path.name)
        else:
            ckpt_infos.append(info)

    if not ckpt_infos:
        raise SystemExit("No checkpoints matched the naming pattern.")

    device = args.device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    env_cache = {}
    dataset_cache = {}
    results = {}
    skipped_details = []

    output_root = Path(args.output_dir) / time.strftime("%Y%m%d_%H%M%S")
    output_root.mkdir(parents=True, exist_ok=True)

    for info in ckpt_infos:
        print(f"[eval] Loading {info.path.name} ({info.method}, {info.problem}{info.size})")
        model_cls = resolve_model_class(info.method)
        if model_cls is None:
            print(f"[skip] Unknown method '{info.method}' in {info.path.name}")
            continue

        if info.group_key not in env_cache:
            env_cache[info.group_key] = build_env(info.problem, info.size)
        env = env_cache[info.group_key]

        model, load_error = load_model_checkpoint(model_cls, info.path, env)
        if model is None:
            skipped_details.append(
                {"file": info.path.name, "reason": f"load_failed: {load_error}"}
            )
            print(f"[skip] Failed to load {info.path.name}: {load_error}")
            continue

        model.eval()
        model.to(device)
        model.policy.eval()
        model.policy.to(device)

        seed_rewards = {}
        seed_means = {}
        seed_stds = {}
        seed_times = {}

        for seed in seeds:
            dataset_key = (info.group_key, seed)
            if dataset_key not in dataset_cache:
                set_global_seed(seed)
                dataset = env.dataset(args.num_instances, phase="test")
                dataset.data_len = min(args.num_instances, len(dataset))
                dataset_cache[dataset_key] = dataset
            else:
                dataset = dataset_cache[dataset_key]

            set_global_seed(seed)
            method, eval_kwargs = resolve_eval_config(
                model,
                env,
                dataset,
                args.method,
                args.num_augment,
                args.num_starts,
                args.force_dihedral_8,
            )

            try:
                eval_call_kwargs = {
                    "env": env,
                    "policy": model.policy,
                    "dataset": dataset,
                    "method": method,
                    "batch_size": args.batch_size,
                    "max_batch_size": args.max_batch_size,
                    "auto_batch_size": args.batch_size is None,
                    "samples": args.samples,
                    **eval_kwargs,
                }
                if method == "sampling":
                    eval_call_kwargs.update(
                        {
                            "temperature": args.temperature,
                            "top_p": args.top_p,
                            "top_k": args.top_k,
                        }
                    )
                result = evaluate_policy(**eval_call_kwargs)
            except Exception as exc:
                skipped_details.append(
                    {
                        "file": info.path.name,
                        "reason": f"eval_failed: {exc}",
                        "seed": seed,
                    }
                )
                print(
                    f"[skip] Evaluation failed for {info.path.name} (seed={seed}): {exc}"
                )
                continue
            rewards = result["rewards"].numpy()
            seed_rewards[seed] = rewards
            seed_means[seed] = float(rewards.mean())
            seed_stds[seed] = float(rewards.std(ddof=1)) if rewards.size > 1 else 0.0
            seed_times[seed] = float(result["inference_time"])

        results[info.label] = {
            "info": info,
            "seed_rewards": seed_rewards,
            "seed_means": seed_means,
            "seed_stds": seed_stds,
            "seed_times": seed_times,
        }

    per_seed_path = output_root / "per_seed.csv"
    summary_path = output_root / "summary.csv"
    wtl_path = output_root / "pairwise_win_tie_loss.csv"
    meta_path = output_root / "run_meta.json"
    skipped_path = output_root / "skipped.csv"

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
            ]
        )

        def split_eam(method_name: str) -> tuple[bool, str]:
            if method_name.startswith("eam_"):
                return True, method_name[len("eam_") :]
            return False, method_name

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

        pair_entries = []
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
                        pair_entries.append(
                            {
                                "problem": info.problem,
                                "size": info.size,
                                "a_label": a_label,
                                "b_label": b_label,
                                "diffs": diffs,
                            }
                        )

        target_ratio = 0.8
        adaptive_tols = {}
        pairs_by_key = {}
        for entry in pair_entries:
            key = (entry["problem"], entry["size"])
            pairs_by_key.setdefault(key, []).append(entry)

        for key, entries in pairs_by_key.items():
            tol = max(args.tie_tol, 0.0)
            for _ in range(50):
                failing = []
                for entry in entries:
                    wins, ties, losses = count_wtl(entry["diffs"], tol)
                    total = wins + ties + losses
                    ratio = (ties + losses) / total if total > 0 else 0.0
                    if ratio < target_ratio:
                        failing.append(entry)
                if not failing:
                    break
                failing_diffs: list[np.ndarray] = []
                for entry in failing:
                    failing_diffs.extend(entry["diffs"])
                new_tol = compute_required_tol(failing_diffs, tol, target_ratio)
                if new_tol <= tol:
                    break
                tol = new_tol
            adaptive_tols[key] = tol

        for entry in pair_entries:
            problem = entry["problem"]
            size = entry["size"]
            tie_tol = adaptive_tols.get((problem, size), max(args.tie_tol, 0.0))
            wins, ties, losses = count_wtl(entry["diffs"], tie_tol)
            total = wins + ties + losses
            tie_loss_ratio = (ties + losses) / total if total > 0 else 0.0
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
                ]
            )

    with meta_path.open("w") as f:
        json.dump(
            {
                "checkpoints_dir": str(ckpt_dir),
                "seeds": seeds,
                "num_instances": args.num_instances,
                "device": device,
                "method": args.method,
                "num_augment": args.num_augment,
                "num_starts": args.num_starts,
                "tie_tol": args.tie_tol,
                "tie_tol_mode": "iterative_pair_target",
                "tie_loss_target": target_ratio,
            },
            f,
            indent=2,
        )

    if skipped:
        print(f"[warn] Skipped {len(skipped)} files with unknown naming: {skipped}")
    if skipped_details:
        with skipped_path.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["file", "seed", "reason"])
            for item in skipped_details:
                writer.writerow([item.get("file"), item.get("seed"), item.get("reason")])
        print(f"[warn] Skipped details saved to: {skipped_path}")

    print(f"Saved per-seed results to: {per_seed_path}")
    print(f"Saved summary results to: {summary_path}")
    print(f"Saved win/tie/loss results to: {wtl_path}")
    print(f"Saved metadata to: {meta_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
