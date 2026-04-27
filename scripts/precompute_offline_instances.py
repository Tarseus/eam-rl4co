from __future__ import annotations

import argparse
import os
import random
from typing import Any, Dict, Sequence

import numpy as np
import torch


def _set_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _size_key(env_name: str) -> str:
    name = str(env_name).strip().lower()
    if name in {"tsp", "cvrp"}:
        return "num_loc"
    if name in {"jssp", "fjsp"}:
        return "num_jobs"
    if name == "ffsp":
        return "num_job"
    raise ValueError(f"Unsupported env for offline instance generation: {env_name}")


def _build_env(env_name: str, *, size: int, generator_params: Dict[str, Any] | None = None):
    from rl4co.envs import CVRPEnv, FJSPEnv, JSSPEnv, TSPEnv
    from rl4co.envs.scheduling.ffsp.env import FFSPEnv

    name = str(env_name).strip().lower()
    gp: Dict[str, Any] = dict(generator_params or {})
    gp[_size_key(name)] = int(size)

    env_map = {
        "tsp": TSPEnv,
        "cvrp": CVRPEnv,
        "jssp": JSSPEnv,
        "fjsp": FJSPEnv,
        "ffsp": FFSPEnv,
    }
    if name not in env_map:
        raise ValueError(f"Unsupported env: {env_name}")
    return env_map[name](generator_params=gp)


def _sample_instances(*, env_name: str, size: int, n: int, seed: int) -> Any:
    _set_seed(int(seed))
    env = _build_env(env_name, size=int(size))
    td = env.generator(int(n))
    return td.to("cpu")


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Precompute RL4CO offline instances to .pt (CPU)")
    p.add_argument("--env", required=True, type=str, help="Environment name (e.g., tsp)")
    p.add_argument("--sizes", required=True, nargs="+", type=int, help="Problem sizes (e.g., 20 100)")
    p.add_argument("--train_size", required=True, type=int, help="Number of train instances per size")
    p.add_argument("--val_size", required=True, type=int, help="Number of val instances per size")
    p.add_argument("--seed", required=True, type=int, help="Base seed for deterministic sampling")
    p.add_argument("--out_dir", required=True, type=str, help="Output directory (e.g., offline_data)")
    return p.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    env_name = str(args.env).strip().lower()
    out_dir = str(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    for size in [int(s) for s in args.sizes]:
        train_seed = int(args.seed) + int(size) * 100_000 + 0
        val_seed = int(args.seed) + int(size) * 100_000 + 1

        td_train = _sample_instances(
            env_name=env_name,
            size=int(size),
            n=int(args.train_size),
            seed=int(train_seed),
        )
        td_val = _sample_instances(
            env_name=env_name,
            size=int(size),
            n=int(args.val_size),
            seed=int(val_seed),
        )

        train_path = os.path.join(out_dir, f"{env_name}{int(size)}_train.pt")
        val_path = os.path.join(out_dir, f"{env_name}{int(size)}_val.pt")
        torch.save(td_train, train_path)
        torch.save(td_val, val_path)
        print(f"[offline] saved size={int(size)} train={train_path} val={val_path}", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

