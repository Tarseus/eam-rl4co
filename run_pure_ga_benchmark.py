from __future__ import annotations

import argparse
import csv
import os
import platform
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from tensordict import TensorDict

from rl4co.envs.routing.cvrp.env import CVRPEnv
from rl4co.envs.routing.cvrp.generator import CVRPGenerator
from rl4co.envs.routing.knapsack.env import KnapsackEnv
from rl4co.envs.routing.knapsack.generator import KnapsackGenerator
from rl4co.envs.routing.op.env import OPEnv
from rl4co.envs.routing.op.generator import OPGenerator
from rl4co.envs.routing.pctsp.env import PCTSPEnv
from rl4co.envs.routing.pctsp.generator import PCTSPGenerator
from rl4co.envs.routing.tsp.env import TSPEnv
from rl4co.envs.routing.tsp.generator import TSPGenerator
from rl4co.models.zoo.earl.evolution import EA, evolution_worker


@dataclass(frozen=True)
class TaskSpec:
    problem: str
    size: int

    @property
    def key(self) -> str:
        return f"{self.problem}{self.size}"


DEFAULT_TASKS = [
    TaskSpec("tsp", 50),
    TaskSpec("tsp", 100),
    TaskSpec("cvrp", 50),
    TaskSpec("cvrp", 100),
    TaskSpec("pctsp", 100),
    TaskSpec("op", 100),
    TaskSpec("knapsack", 50),
    TaskSpec("knapsack", 100),
]


def _normalize_problem(name: str) -> str:
    name = name.strip().lower()
    if name == "kp":
        return "knapsack"
    if name == "cvrp100" or name == "cvrp50":
        return "cvrp"
    return name


def parse_tasks(text: str | None) -> list[TaskSpec]:
    if not text:
        return list(DEFAULT_TASKS)
    tasks: list[TaskSpec] = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        # Accept tsp50 / tsp_50 / tsp:50
        for sep in ("_", ":"):
            part = part.replace(sep, "")
        # split alpha prefix + numeric suffix
        i = 0
        while i < len(part) and not part[i].isdigit():
            i += 1
        if i == 0 or i == len(part):
            raise SystemExit(f"Invalid task spec: {part!r}. Expected e.g. tsp50,cvrp100,kp50.")
        problem = _normalize_problem(part[:i])
        size = int(part[i:])
        tasks.append(TaskSpec(problem, size))
    return tasks


def build_env(problem: str, size: int):
    if problem == "tsp":
        return TSPEnv(
            generator=TSPGenerator(num_loc=size, loc_distribution="uniform"),
            check_solution=False,
        )
    if problem == "cvrp":
        return CVRPEnv(
            generator=CVRPGenerator(num_loc=size, loc_distribution="uniform"),
            check_solution=False,
        )
    if problem == "pctsp":
        return PCTSPEnv(
            generator=PCTSPGenerator(num_loc=size, loc_distribution="uniform"),
            check_solution=False,
        )
    if problem == "op":
        return OPEnv(
            generator=OPGenerator(num_loc=size, loc_distribution="uniform"),
            check_solution=False,
        )
    if problem == "knapsack":
        return KnapsackEnv(
            generator=KnapsackGenerator(
                num_items=size,
                weight_distribution="uniform",
                value_distribution="uniform",
            ),
            check_solution=False,
        )
    raise ValueError(f"Unsupported problem: {problem}")


def _action_mask_2d(td: TensorDict) -> torch.Tensor:
    mask = td["action_mask"]
    if mask.dim() == 3 and mask.size(-1) == 1:
        mask = mask.squeeze(-1)
    return mask


def _rollout_random_actions(
    env,
    td_reset: TensorDict,
    *,
    max_steps: int,
    seed: int,
    prefer_non_depot_steps: int = 0,
) -> torch.Tensor:
    rng = np.random.default_rng(seed)
    td = td_reset.clone()
    batch = int(td.batch_size[0])

    actions = torch.zeros((batch, max_steps), dtype=torch.int64)
    done = td.get("done", torch.zeros((batch,), dtype=torch.bool))
    if done.dim() == 2:
        done = done.squeeze(-1)

    for t in range(max_steps):
        mask = _action_mask_2d(td).to(torch.bool)
        if mask.ndim != 2:
            raise RuntimeError(f"Unexpected action_mask shape: {tuple(mask.shape)}")

        probs = mask.float()

        if prefer_non_depot_steps > 0 and t < prefer_non_depot_steps and probs.size(-1) > 1:
            non_depot = probs[:, 1:].clone()
            row_sum = non_depot.sum(-1)
            use_non_depot = row_sum > 0
            if use_non_depot.any():
                non_depot[~use_non_depot] = 0.0
                non_depot = non_depot / non_depot.sum(-1, keepdim=True).clamp_min(1.0)
                sampled = torch.multinomial(non_depot, 1).squeeze(-1) + 1
                fallback = torch.multinomial(
                    (probs / probs.sum(-1, keepdim=True).clamp_min(1.0)), 1
                ).squeeze(-1)
                action = torch.where(use_non_depot, sampled, fallback)
            else:
                action = torch.multinomial(
                    (probs / probs.sum(-1, keepdim=True).clamp_min(1.0)), 1
                ).squeeze(-1)
        else:
            denom = probs.sum(-1, keepdim=True)
            safe = denom.squeeze(-1) > 0
            probs = torch.where(safe[:, None], probs / denom.clamp_min(1.0), probs)
            # If no feasible action (should not happen), pick depot / finish (0).
            action = torch.zeros((batch,), dtype=torch.int64)
            if safe.any():
                action[safe] = torch.multinomial(probs[safe], 1).squeeze(-1)

        action = torch.where(done, torch.zeros_like(action), action)
        actions[:, t] = action.cpu()

        td.set("action", action)
        td = env.step(td)["next"]
        done = td.get("done", done)
        if done.dim() == 2:
            done = done.squeeze(-1)
        if bool(done.all()):
            break

    return actions


def _infer_max_steps(env_name: str, size: int) -> int:
    if env_name == "tsp":
        return int(size)
    if env_name == "cvrp":
        return int(2 * size)
    if env_name in {"op", "pctsp"}:
        return int(size + 1)
    if env_name == "knapsack":
        return int(size + 1)
    raise ValueError(f"Unsupported env_name: {env_name}")


def _default_ea_kwargs(*, generations: int, mutation: float, crossover: float, selection: float) -> dict:
    return {
        "num_generations": int(generations),
        "mutation_rate": float(mutation),
        "crossover_rate": float(crossover),
        "selection_rate": float(selection),
        # kept for compatibility with EAM configs; unused by EA directly
        "batch_size": 64,
        "ea_batch_size": 64,
        "alpha": 0.5,
        "beta": 3,
        "ea_prob": 0.01,
        "ea_epoch": 700,
        "method": "ga",
    }


def run_task(
    *,
    task: TaskSpec,
    instances: int,
    batch_size: int,
    seed: int,
    ea_kwargs: dict,
    warmup: bool,
) -> dict:
    env = build_env(task.problem, task.size)
    ea = EA(env, ea_kwargs)

    max_steps = _infer_max_steps(env.name, task.size)
    data = env.generator(instances)

    if warmup:
        warm_data = env.generator(1)
        warm_reset = env.reset(warm_data)
        warm_actions = _rollout_random_actions(
            env,
            warm_reset,
            max_steps=max_steps,
            seed=seed + 99991,
            prefer_non_depot_steps=2 if env.name == "op" else 0,
        )
        _ = evolution_worker(warm_actions, warm_reset, ea, env)

    init_time_s = 0.0
    ga_time_s = 0.0
    rewards: list[torch.Tensor] = []

    for offset in range(0, instances, batch_size):
        bs = min(batch_size, instances - offset)
        batch_data = data[offset : offset + bs]
        td_reset = env.reset(batch_data)

        t0 = time.perf_counter()
        prefer_non_depot = 2 if env.name == "op" else 0
        init_actions = _rollout_random_actions(
            env,
            td_reset,
            max_steps=max_steps,
            seed=seed + offset,
            prefer_non_depot_steps=prefer_non_depot,
        )
        init_time_s += time.perf_counter() - t0

        t0 = time.perf_counter()
        improved_actions, init_td = evolution_worker(init_actions, td_reset, ea, env)
        rewards.append(env.get_reward(init_td, improved_actions).cpu())
        ga_time_s += time.perf_counter() - t0

    rewards_t = torch.cat(rewards, dim=0)
    return {
        "task": task.key,
        "problem": task.problem,
        "size": task.size,
        "instances": instances,
        "max_steps": max_steps,
        "init_time_s": init_time_s,
        "ga_time_s": ga_time_s,
        "total_time_s": init_time_s + ga_time_s,
        "ga_time_per_inst_ms": 1000.0 * ga_time_s / max(instances, 1),
        "reward_mean": float(rewards_t.mean().item()) if rewards_t.numel() else float("nan"),
        "reward_std": float(rewards_t.std(unbiased=False).item()) if rewards_t.numel() else float("nan"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Pure GA benchmark on RL4CO environments.")
    parser.add_argument(
        "--tasks",
        type=str,
        default=None,
        help="Comma-separated list like tsp50,tsp100,cvrp50,cvrp100,pctsp100,op100,kp50,kp100",
    )
    parser.add_argument("--instances", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--generations", type=int, default=3)
    parser.add_argument("--mutation", type=float, default=0.1)
    parser.add_argument("--crossover", type=float, default=0.6)
    parser.add_argument("--selection", type=float, default=0.2)
    parser.add_argument("--no-warmup", action="store_true", help="Disable JIT warmup (includes first-call overhead in timings).")
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="CSV output path (default: results/pure_ga_timings_YYYYmmdd_HHMMSS.csv)",
    )
    args = parser.parse_args()

    tasks = parse_tasks(args.tasks)
    default_name = f"pure_ga_timings_{time.strftime('%Y%m%d_%H%M%S')}.csv"
    out_path = Path(args.out or (Path("results") / default_name))
    out_path.parent.mkdir(parents=True, exist_ok=True)

    torch.set_num_threads(max(1, (os.cpu_count() or 1) // 2))
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    ea_kwargs = _default_ea_kwargs(
        generations=args.generations,
        mutation=args.mutation,
        crossover=args.crossover,
        selection=args.selection,
    )

    rows: list[dict] = []
    meta = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cpu_count": os.cpu_count() or 0,
        "instances": args.instances,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "generations": args.generations,
        "mutation": args.mutation,
        "crossover": args.crossover,
        "selection": args.selection,
    }

    print("Meta:", meta)
    for task in tasks:
        print(f"Running {task.key} ...")
        row = run_task(
            task=task,
            instances=args.instances,
            batch_size=args.batch_size,
            seed=args.seed,
            ea_kwargs=ea_kwargs,
            warmup=not args.no_warmup,
        )
        rows.append(row)
        print(
            f"  ga_time={row['ga_time_s']:.3f}s total={row['total_time_s']:.3f}s "
            f"({row['ga_time_per_inst_ms']:.2f} ms/inst)"
        )

    fieldnames = sorted({k for r in rows for k in r.keys()})
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["_meta"] + fieldnames)
        writer.writeheader()
        writer.writerow({"_meta": meta})
        for r in rows:
            writer.writerow({"_meta": "", **r})

    print(f"Saved: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
