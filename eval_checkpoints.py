import argparse
import csv
import json
import math
import random
import re
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
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

    output_root = Path(args.output_dir) / time.strftime("%Y%m%d_%H%M%S")
    output_root.mkdir(parents=True, exist_ok=True)

    for info in ckpt_infos:
        model_cls = resolve_model_class(info.method)
        if model_cls is None:
            print(f"[skip] Unknown method '{info.method}' in {info.path.name}")
            continue

        if info.group_key not in env_cache:
            env_cache[info.group_key] = build_env(info.problem, info.size)
        env = env_cache[info.group_key]

        model = None
        try:
            model = model_cls.load_from_checkpoint(str(info.path), load_baseline=False)
        except Exception:
            try:
                model = model_cls.load_from_checkpoint(
                    str(info.path), env=env, load_baseline=False
                )
            except Exception as exc:
                print(f"[skip] Failed to load {info.path.name}: {exc}")
                continue

        model.eval()
        model.to(device)
        model.policy.eval()
        model.policy.to(device)

        seed_rewards = {}
        seed_means = []
        seed_stds = []
        seed_times = []

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

            result = evaluate_policy(
                env=env,
                policy=model.policy,
                dataset=dataset,
                method=method,
                batch_size=args.batch_size,
                max_batch_size=args.max_batch_size,
                auto_batch_size=args.batch_size is None,
                samples=args.samples,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                **eval_kwargs,
            )
            rewards = result["rewards"].numpy()
            seed_rewards[seed] = rewards
            seed_means.append(float(rewards.mean()))
            seed_stds.append(float(rewards.std(ddof=1)) if rewards.size > 1 else 0.0)
            seed_times.append(float(result["inference_time"]))

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

    with per_seed_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["model", "problem", "size", "seed", "mean_reward", "std_reward", "inference_time_s"]
        )
        for label, data in results.items():
            info = data["info"]
            for idx, seed in enumerate(seeds):
                if seed not in data["seed_rewards"]:
                    continue
                writer.writerow(
                    [
                        label,
                        info.problem,
                        info.size,
                        seed,
                        data["seed_means"][idx],
                        data["seed_stds"][idx],
                        data["seed_times"][idx],
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
            seed_means = data["seed_means"]
            mean_of_means = float(np.mean(seed_means))
            std_across = float(np.std(seed_means, ddof=1)) if len(seed_means) > 1 else 0.0
            ci_half = ci_half_width(seed_means)
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
            ["problem", "size", "model_a", "model_b", "wins", "ties", "losses", "total"]
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

        for key in sorted(set(non_eam.keys()) & set(eam.keys())):
            a_labels = non_eam[key]
            b_labels = eam[key]
            for a_label in a_labels:
                for b_label in b_labels:
                    wins = ties = losses = 0
                    a_rewards = results[a_label]["seed_rewards"]
                    b_rewards = results[b_label]["seed_rewards"]
                    shared_seeds = sorted(set(a_rewards.keys()) & set(b_rewards.keys()))
                    for seed in shared_seeds:
                        a = a_rewards[seed]
                        b = b_rewards[seed]
                        diff = a - b
                        eps = 1e-9
                        wins += int((diff > eps).sum())
                        ties += int((np.abs(diff) <= eps).sum())
                        losses += int((diff < -eps).sum())
                    if shared_seeds:
                        info = results[a_label]["info"]
                        writer.writerow(
                            [info.problem, info.size, a_label, b_label, wins, ties, losses, wins + ties + losses]
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
            },
            f,
            indent=2,
        )

    if skipped:
        print(f"[warn] Skipped {len(skipped)} files with unknown naming: {skipped}")

    print(f"Saved per-seed results to: {per_seed_path}")
    print(f"Saved summary results to: {summary_path}")
    print(f"Saved win/tie/loss results to: {wtl_path}")
    print(f"Saved metadata to: {meta_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
