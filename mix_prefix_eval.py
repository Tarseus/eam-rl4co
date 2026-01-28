import argparse
import csv
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from eval_checkpoints import (
    DEFAULT_EA_KWARGS,
    build_env,
    ci_half_width,
    load_model_checkpoint,
    parse_seeds,
    run_ga_search,
    set_global_seed,
)
from rl4co.models import POMO
from rl4co.models.zoo.earl.evolution import EA
from rl4co.tasks.eval import GreedyMultiStartEval, get_automatic_batch_size, evaluate_policy


def parse_t_list(text: str) -> list[float]:
    values = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        if part.endswith("%"):
            val = float(part[:-1]) / 100.0
        else:
            val = float(part)
            if val > 1.0:
                val = val / 100.0
        if not (0.0 <= val <= 1.0):
            raise ValueError(f"t value out of range: {part}")
        values.append(val)
    if not values:
        raise ValueError("No valid t values provided.")
    return values


def resolve_device(device: str | None) -> str:
    if device:
        return device
    return "cuda" if torch.cuda.is_available() else "cpu"


def resolve_batch_size(env, num_starts: int, batch_size: int | None, max_batch_size: int) -> int:
    if batch_size is not None:
        return int(batch_size)
    eval_fn = GreedyMultiStartEval(env, num_starts=num_starts, progress=False)
    return get_automatic_batch_size(
        eval_fn, max_batch_size=max_batch_size, start_batch_size=8192
    )


def mix_prefix_rewards(
    env,
    full_policy,
    dataset,
    prefix_actions: torch.Tensor,
    prefix_len: int,
    batch_size: int,
    device: str,
) -> torch.Tensor:
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=dataset.collate_fn,
    )
    rewards_list = []
    offset = 0
    num_steps = prefix_actions.shape[1]

    with torch.inference_mode():
        for batch in dataloader:
            bs = int(batch.batch_size[0])
            prefix = prefix_actions[offset : offset + bs].to(device)
            offset += bs

            td = batch.to(device)
            td = env.reset(td)
            td_init = td.clone()

            if prefix_len > 0:
                for step in range(prefix_len):
                    td.set("action", prefix[:, step])
                    td = env.step(td)["next"]

            if prefix_len < num_steps:
                out = full_policy(
                    td,
                    env,
                    phase="test",
                    decode_type="greedy",
                    num_starts=0,
                    calc_reward=False,
                )
                remaining_actions = out["actions"]
                actions = torch.cat([prefix[:, :prefix_len], remaining_actions], dim=1)
            else:
                actions = prefix[:, :prefix_len]

            rewards = env.get_reward(td_init, actions)
            rewards_list.append(rewards.cpu())

    return torch.cat(rewards_list)


def write_rows(path: Path, header: list[str], rows: list[list]):
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


def summarize_rows(
    rows: list[list],
    key_indices: list[int],
    value_index: int,
) -> list[list]:
    grouped = {}
    for row in rows:
        key = tuple(row[idx] for idx in key_indices)
        grouped.setdefault(key, []).append(float(row[value_index]))

    summary = []
    for key, values in grouped.items():
        mean_of_means = float(np.mean(values))
        std_across = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
        ci_half = ci_half_width(values)
        summary.append([*key, len(values), mean_of_means, std_across, ci_half])
    return summary


def tensor_std(value: torch.Tensor) -> float:
    if value.numel() <= 1:
        return 0.0
    return float(value.std(unbiased=True))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate prefix-mixed TSP100: first t% steps from mid ckpt, rest from full ckpt."
    )
    parser.add_argument("--mid-ckpt", type=str, required=True)
    parser.add_argument("--full-ckpt", type=str, required=True)
    parser.add_argument("--problem", type=str, default="tsp")
    parser.add_argument("--size", type=int, default=100)
    parser.add_argument(
        "--t-list",
        type=str,
        default="0.1,0.3,0.5,0.7,0.9",
        help="Comma-separated list, supports percent, e.g. 10%,0.3",
    )
    parser.add_argument("--seeds", type=str, default="0,1,2,3,4")
    parser.add_argument("--num-instances", type=int, default=1000)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--max-batch-size", type=int, default=4096)
    parser.add_argument("--output-dir", type=str, default="results/mix_prefix")
    args = parser.parse_args()

    t_values = parse_t_list(args.t_list)
    seeds = parse_seeds(args.seeds)
    if not seeds:
        raise SystemExit("No valid seeds provided.")

    mid_ckpt = Path(args.mid_ckpt)
    full_ckpt = Path(args.full_ckpt)
    if not mid_ckpt.exists():
        raise SystemExit(f"Checkpoint not found: {mid_ckpt}")
    if not full_ckpt.exists():
        raise SystemExit(f"Checkpoint not found: {full_ckpt}")

    device = resolve_device(args.device)
    env = build_env(args.problem, args.size)

    mid_model, err = load_model_checkpoint(POMO, mid_ckpt, env)
    if mid_model is None:
        raise SystemExit(f"Failed to load mid checkpoint: {err}")
    full_model, err = load_model_checkpoint(POMO, full_ckpt, env)
    if full_model is None:
        raise SystemExit(f"Failed to load full checkpoint: {err}")

    mid_model.eval()
    full_model.eval()
    mid_model.policy.eval()
    full_model.policy.eval()
    mid_model.to(device)
    full_model.to(device)
    mid_model.policy.to(device)
    full_model.policy.to(device)

    num_starts = getattr(env.generator, "num_loc", None)
    if num_starts is None:
        raise SystemExit("Cannot infer num_starts from env.")

    batch_size = resolve_batch_size(env, num_starts, args.batch_size, args.max_batch_size)

    output_root = Path(args.output_dir) / time.strftime("%Y%m%d_%H%M%S")
    output_root.mkdir(parents=True, exist_ok=True)

    mix_rows = []
    baseline_rows = []

    for seed in seeds:
        set_global_seed(seed)
        dataset = env.dataset(args.num_instances, phase="test")
        dataset.data_len = min(args.num_instances, len(dataset))

        print(f"[seed {seed}] mid multistart eval")
        mid_eval = evaluate_policy(
            env,
            mid_model.policy,
            dataset,
            method="multistart_greedy",
            batch_size=batch_size,
            max_batch_size=args.max_batch_size,
            auto_batch_size=False,
            num_starts=num_starts,
        )
        mid_actions = mid_eval["actions"]
        mid_rewards = mid_eval["rewards"]

        print(f"[seed {seed}] full multistart eval")
        full_eval = evaluate_policy(
            env,
            full_model.policy,
            dataset,
            method="multistart_greedy",
            batch_size=batch_size,
            max_batch_size=args.max_batch_size,
            auto_batch_size=False,
            num_starts=num_starts,
        )
        full_rewards = full_eval["rewards"]

        # GA improvement on mid checkpoint full solutions
        ea = EA(env, DEFAULT_EA_KWARGS)
        ga_batch_size = batch_size
        print(f"[seed {seed}] mid + GA eval")
        ga_rewards, ga_time, _ = run_ga_search(
            env,
            dataset,
            mid_actions,
            ea,
            ga_batch_size,
            return_actions=False,
        )
        ga_rewards = torch.as_tensor(ga_rewards)

        baseline_rows.append(
            [
                seed,
                float(mid_rewards.mean()),
                tensor_std(mid_rewards),
                float(full_rewards.mean()),
                tensor_std(full_rewards),
                float(ga_rewards.mean()),
                tensor_std(ga_rewards),
            ]
        )

        for t in t_values:
            prefix_len = int(round(t * args.size))
            prefix_len = max(0, min(prefix_len, args.size))
            print(f"[seed {seed}] mix t={t:.3f} (prefix_len={prefix_len})")
            mix_rewards = mix_prefix_rewards(
                env,
                full_model.policy,
                dataset,
                mid_actions,
                prefix_len,
                batch_size,
                device,
            )
            mix_rows.append(
                [
                    seed,
                    t,
                    prefix_len,
                    float(mix_rewards.mean()),
                    tensor_std(mix_rewards),
                ]
            )

    per_seed_mix_path = output_root / "per_seed_mix.csv"
    per_seed_baselines_path = output_root / "per_seed_baselines.csv"
    summary_mix_path = output_root / "summary_mix.csv"
    summary_baselines_path = output_root / "summary_baselines.csv"

    write_rows(
        per_seed_mix_path,
        ["seed", "t", "prefix_len", "mean_reward", "std_reward"],
        mix_rows,
    )
    write_rows(
        per_seed_baselines_path,
        [
            "seed",
            "mid_mean",
            "mid_std",
            "full_mean",
            "full_std",
            "mid_ga_mean",
            "mid_ga_std",
        ],
        baseline_rows,
    )

    mix_summary = summarize_rows(mix_rows, [1, 2], 3)
    write_rows(
        summary_mix_path,
        ["t", "prefix_len", "seed_count", "mean_of_means", "std_across_seeds", "ci95_half_width"],
        mix_summary,
    )

    base_summary = summarize_rows(baseline_rows, [], 1)
    if base_summary:
        base_summary = [
            [
                "mid",
                len(seeds),
                float(np.mean([row[1] for row in baseline_rows])),
                float(np.std([row[1] for row in baseline_rows], ddof=1)) if len(seeds) > 1 else 0.0,
                ci_half_width([row[1] for row in baseline_rows]),
            ],
            [
                "full",
                len(seeds),
                float(np.mean([row[3] for row in baseline_rows])),
                float(np.std([row[3] for row in baseline_rows], ddof=1)) if len(seeds) > 1 else 0.0,
                ci_half_width([row[3] for row in baseline_rows]),
            ],
            [
                "mid_ga",
                len(seeds),
                float(np.mean([row[5] for row in baseline_rows])),
                float(np.std([row[5] for row in baseline_rows], ddof=1)) if len(seeds) > 1 else 0.0,
                ci_half_width([row[5] for row in baseline_rows]),
            ],
        ]
    write_rows(
        summary_baselines_path,
        ["label", "seed_count", "mean_of_means", "std_across_seeds", "ci95_half_width"],
        base_summary,
    )

    meta_path = output_root / "run_meta.txt"
    meta_path.write_text(
        "\n".join(
            [
                f"mid_ckpt={mid_ckpt}",
                f"full_ckpt={full_ckpt}",
                f"problem={args.problem}",
                f"size={args.size}",
                f"t_values={t_values}",
                f"seeds={seeds}",
                f"num_instances={args.num_instances}",
                f"batch_size={batch_size}",
                f"device={device}",
            ]
        )
    )

    print(f"Saved per-seed mix results to: {per_seed_mix_path}")
    print(f"Saved per-seed baselines to: {per_seed_baselines_path}")
    print(f"Saved mix summary to: {summary_mix_path}")
    print(f"Saved baseline summary to: {summary_baselines_path}")
    print(f"Saved run meta to: {meta_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
