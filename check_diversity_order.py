import argparse
import json
from pathlib import Path

import numpy as np
import torch

from eval_checkpoints import (
    build_env,
    load_model_checkpoint,
    parse_checkpoint_name,
    resolve_model_class,
    set_global_seed,
)
from rl4co.models.zoo.earl.evolution import EA, evolution_worker
from rl4co.models.zoo.earl.model import (
    DEPOT_ENVS,
    _actions_to_numpy,
    _edge_diversity,
    _edge_edit_diversity,
    _edge_usage_diversity,
    _route_assignment_diversity,
)
from rl4co.utils.ops import batchify


def _align_improved_actions(
    improved_actions: torch.Tensor | None, original_actions: torch.Tensor | None
) -> torch.Tensor | None:
    if improved_actions is None or original_actions is None:
        return improved_actions
    if improved_actions.dim() != original_actions.dim():
        return improved_actions
    if improved_actions.shape[-1] == original_actions.shape[-1]:
        return improved_actions
    if improved_actions.shape[-1] + 1 == original_actions.shape[-1]:
        return torch.cat([original_actions[..., :1], improved_actions], dim=-1)
    return improved_actions


def _infer_num_traj(actions: torch.Tensor | None, batch_size: int) -> int:
    if actions is None or batch_size <= 0:
        return 1
    if actions.dim() == 3 and actions.shape[0] == batch_size:
        return int(actions.shape[1])
    if actions.dim() != 2:
        return 1
    total = actions.shape[0]
    if total % batch_size != 0:
        return 1
    return int(total // batch_size)


def sample_policy_actions(
    policy,
    env,
    td,
    num_samples: int,
    temperature: float,
    top_p: float,
    top_k: int,
):
    with torch.no_grad():
        out = policy(
            td,
            env,
            phase="test",
            decode_type="sampling",
            num_samples=num_samples,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            calc_reward=False,
            return_actions=True,
        )
    return out["actions"]


def run_local_search(env, td, actions, max_iters: int, num_threads: int | None):
    td_cpu = td.cpu()
    actions_cpu = actions.detach().cpu()
    n_traj = _infer_num_traj(actions_cpu, td.batch_size[0])
    if n_traj > 1:
        td_cpu = batchify(td_cpu, n_traj)
    kwargs = {"max_iterations": max_iters}
    if num_threads is not None:
        kwargs["num_threads"] = num_threads
    try:
        improved = env.local_search(td_cpu, actions_cpu, **kwargs)
    except TypeError:
        kwargs.pop("num_threads", None)
        improved = env.local_search(td_cpu, actions_cpu, **kwargs)
    return improved


def compute_diversity(actions, batch_size: int, env_name: str):
    actions_np = _actions_to_numpy(actions, batch_size)
    if actions_np is None:
        return None
    close_tour = env_name == "tsp"
    ignore_zero = env_name in DEPOT_ENVS
    edge_div = _edge_diversity(actions_np, close_tour=close_tour, ignore_zero=ignore_zero)
    edge_entropy, edge_simpson = _edge_usage_diversity(
        actions_np, close_tour=close_tour, ignore_zero=ignore_zero
    )
    route_div = _route_assignment_diversity(actions_np) if ignore_zero else 0.0
    return {
        "edge_div": edge_div,
        "edge_entropy": edge_entropy,
        "edge_simpson": edge_simpson,
        "route_div": route_div,
    }


def compute_edit_diversity(original, improved, batch_size: int, env_name: str):
    original_np = _actions_to_numpy(original, batch_size)
    improved_np = _actions_to_numpy(improved, batch_size)
    if original_np is None or improved_np is None:
        return None
    close_tour = env_name == "tsp"
    ignore_zero = env_name in DEPOT_ENVS
    return _edge_edit_diversity(
        original_np, improved_np, close_tour=close_tour, ignore_zero=ignore_zero
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Check LS < EAM < random diversity ordering for checkpoints."
    )
    parser.add_argument("--ckpt", action="append", default=[], help="Checkpoint path")
    parser.add_argument("--checkpoints-dir", type=str, default="checkpoints")
    parser.add_argument("--filter", type=str, default=None, help="Substring filter")
    parser.add_argument("--problem", type=str, default=None)
    parser.add_argument("--size", type=int, default=None)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--num-instances", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-samples", type=int, default=20)
    parser.add_argument("--temperature", type=float, default=1.5)
    parser.add_argument("--top-p", type=float, default=0.0)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--ls-iters", type=int, default=50)
    parser.add_argument("--ls-threads", type=int, default=None)
    parser.add_argument("--ga-gens", type=int, default=3)
    parser.add_argument("--mutation-rate", type=float, default=0.1)
    parser.add_argument("--crossover-rate", type=float, default=0.6)
    parser.add_argument("--selection-rate", type=float, default=0.2)
    parser.add_argument(
        "--metric",
        type=str,
        default="edge_entropy",
        choices=("edge_div", "edge_entropy", "edge_simpson", "route_div"),
    )
    parser.add_argument("--tol", type=float, default=1e-4)
    return parser.parse_args()


def collect_checkpoints(args):
    paths = [Path(p) for p in args.ckpt]
    if not paths:
        ckpt_dir = Path(args.checkpoints_dir)
        if ckpt_dir.exists():
            paths = sorted(ckpt_dir.rglob("*.ckpt"))
        if args.filter:
            paths = [p for p in paths if args.filter in p.name]
    return paths


def build_ea(env, args):
    ea_kwargs = {
        "num_generations": args.ga_gens,
        "mutation_rate": args.mutation_rate,
        "crossover_rate": args.crossover_rate,
        "selection_rate": args.selection_rate,
        "batch_size": args.batch_size,
        "ea_batch_size": args.batch_size,
    }
    return EA(env, ea_kwargs)


def main() -> int:
    args = parse_args()
    set_global_seed(args.seed)

    ckpt_paths = collect_checkpoints(args)
    if not ckpt_paths:
        raise SystemExit("No checkpoints found.")

    results = []
    for path in ckpt_paths:
        info = parse_checkpoint_name(path)
        if info is None:
            if args.problem is None or args.size is None:
                print(f"[skip] {path.name}: cannot infer problem/size.")
                continue
            problem = args.problem
            size = args.size
            method = "unknown"
        else:
            problem = info.problem
            size = info.size
            method = info.method

        model_cls = resolve_model_class(method) if info is not None else None
        if model_cls is None:
            print(f"[skip] {path.name}: unknown model class.")
            continue

        env = build_env(problem, size)
        model, err = load_model_checkpoint(model_cls, path, env)
        if model is None:
            print(f"[skip] {path.name}: {err}")
            continue

        device = torch.device(args.device)
        model.to(device)
        model.eval()
        model.policy.eval()
        env.to(device)

        ea = getattr(model, "ea", None)
        if ea is None:
            ea = build_ea(env, args)

        totals = {
            "random": {"edge_div": 0.0, "edge_entropy": 0.0, "edge_simpson": 0.0, "route_div": 0.0},
            "ls": {"edge_div": 0.0, "edge_entropy": 0.0, "edge_simpson": 0.0, "route_div": 0.0},
            "eam": {"edge_div": 0.0, "edge_entropy": 0.0, "edge_simpson": 0.0, "route_div": 0.0},
        }
        edit_totals = {"ls": 0.0, "eam": 0.0}
        total_instances = 0

        remaining = args.num_instances
        while remaining > 0:
            batch_size = min(args.batch_size, remaining)
            td = env.reset(batch_size=batch_size)
            td = td.to(device)

            random_actions = sample_policy_actions(
                model.policy,
                env,
                td,
                num_samples=args.num_samples,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
            ).detach().cpu()

            ls_actions = run_local_search(
                env, td, random_actions, args.ls_iters, args.ls_threads
            )
            ls_actions = _align_improved_actions(ls_actions, random_actions)

            eam_actions, _, pop_actions = evolution_worker(
                random_actions, td, ea, env, return_population=True
            )
            eam_actions = _align_improved_actions(eam_actions, random_actions)
            eam_div_actions = (
                pop_actions if pop_actions is not None else eam_actions
            )

            random_metrics = compute_diversity(random_actions, batch_size, env.name)
            ls_metrics = compute_diversity(ls_actions, batch_size, env.name)
            eam_metrics = compute_diversity(eam_div_actions, batch_size, env.name)
            ls_edit = compute_edit_diversity(
                random_actions, ls_actions, batch_size, env.name
            )
            eam_edit = compute_edit_diversity(
                random_actions, eam_actions, batch_size, env.name
            )

            for metric in totals["random"]:
                totals["random"][metric] += random_metrics[metric] * batch_size
                totals["ls"][metric] += ls_metrics[metric] * batch_size
                totals["eam"][metric] += eam_metrics[metric] * batch_size
            if ls_edit is not None:
                edit_totals["ls"] += ls_edit * batch_size
            if eam_edit is not None:
                edit_totals["eam"] += eam_edit * batch_size

            total_instances += batch_size
            remaining -= batch_size

        for method_name in totals:
            for metric in totals[method_name]:
                totals[method_name][metric] /= max(total_instances, 1)
        for method_name in edit_totals:
            edit_totals[method_name] /= max(total_instances, 1)

        metric = args.metric
        ls_val = totals["ls"][metric]
        eam_val = totals["eam"][metric]
        rnd_val = totals["random"][metric]
        ok = (ls_val + args.tol) < eam_val and (eam_val + args.tol) < rnd_val

        summary = {
            "checkpoint": str(path),
            "problem": problem,
            "size": size,
            "metric": metric,
            "ok": ok,
            "random": totals["random"],
            "ls": totals["ls"],
            "eam": totals["eam"],
            "edit_edge": edit_totals,
        }
        results.append(summary)

        status = "OK" if ok else "FAIL"
        print(f"[{status}] {path.name} {metric}: LS={ls_val:.4f} EAM={eam_val:.4f} RAND={rnd_val:.4f}")

    print(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
