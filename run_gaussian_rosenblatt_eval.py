import argparse
import csv
import math
import time
from pathlib import Path

import torch
from tensordict import TensorDict

from rl4co.data.dataset import TensorDictDataset
from rl4co.envs.routing.tsp.env import TSPEnv
from rl4co.envs.routing.tsp.generator import TSPGenerator
from rl4co.tasks.eval import evaluate_policy
from rl4co.utils.ops import gather_by_index, get_tour_length

from eval_checkpoints import load_model_checkpoint, resolve_eval_config, set_global_seed


def rosenblatt_normal(locs: torch.Tensor, mean: float, std: float, eps: float) -> torch.Tensor:
    if std <= 0:
        raise ValueError("loc_std must be > 0")
    z = (locs - mean) / std
    u = 0.5 * (1.0 + torch.erf(z / math.sqrt(2.0)))
    if eps is not None and eps > 0:
        u = u.clamp(min=eps, max=1.0 - eps)
    return u


def build_dataset(locs: torch.Tensor) -> TensorDictDataset:
    td = TensorDict({"locs": locs}, batch_size=torch.Size([locs.shape[0]]))
    return TensorDictDataset(td)


def eval_policy_with_dataset(
    env,
    model,
    dataset,
    method: str,
    num_augment: int | None,
    num_starts: int | None,
    force_dihedral_8: bool | None,
    batch_size: int | None,
    max_batch_size: int,
    samples: int,
    temperature: float,
    top_p: float,
    top_k: int,
):
    resolved_method, eval_kwargs = resolve_eval_config(
        model,
        env,
        dataset,
        method,
        num_augment,
        num_starts,
        force_dihedral_8,
    )

    eval_call_kwargs = {
        "env": env,
        "policy": model.policy,
        "dataset": dataset,
        "method": resolved_method,
        "batch_size": batch_size,
        "max_batch_size": max_batch_size,
        "auto_batch_size": batch_size is None,
        "samples": samples,
        **eval_kwargs,
    }
    if resolved_method == "sampling":
        eval_call_kwargs.update(
            {
                "temperature": temperature,
                "top_p": top_p,
                "top_k": top_k,
            }
        )

    t0 = time.perf_counter()
    result = evaluate_policy(**eval_call_kwargs)
    wall_time = time.perf_counter() - t0

    return result, resolved_method, eval_kwargs, wall_time


def summarize_lengths(locs: torch.Tensor, actions: torch.Tensor) -> tuple[float, float]:
    ordered = gather_by_index(locs, actions)
    lengths = get_tour_length(ordered).cpu()
    return float(lengths.mean()), float(lengths.std(unbiased=True))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate POMO checkpoint on Gaussian TSP100 and Rosenblatt-uniformized inputs."
    )
    parser.add_argument(
        "--ckpt-path", type=str, default="checkpoints/pomo_tsp100.ckpt"
    )
    parser.add_argument("--num-instances", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--max-batch-size", type=int, default=4096)
    parser.add_argument("--method", type=str, default="auto")
    parser.add_argument("--num-augment", type=int, default=None)
    parser.add_argument("--num-starts", type=int, default=None)
    parser.add_argument("--force-dihedral-8", action="store_true")
    parser.add_argument("--no-force-dihedral-8", dest="force_dihedral_8", action="store_false")
    parser.set_defaults(force_dihedral_8=None)
    parser.add_argument("--samples", type=int, default=1280)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.0)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--loc-mean", type=float, default=0.0)
    parser.add_argument("--loc-std", type=float, default=1.0)
    parser.add_argument("--rosenblatt-eps", type=float, default=1e-6)
    parser.add_argument(
        "--output",
        type=str,
        default="results/gaussian_rosenblatt_eval.csv",
    )
    args = parser.parse_args()

    set_global_seed(args.seed)

    device = args.device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    env = TSPEnv(
        TSPGenerator(
            num_loc=100,
            loc_distribution="normal",
            loc_mean=args.loc_mean,
            loc_std=args.loc_std,
        )
    )

    ckpt_path = Path(args.ckpt_path)
    if not ckpt_path.exists():
        raise SystemExit(f"Checkpoint not found: {ckpt_path}")

    from rl4co.models import POMO

    model, load_error = load_model_checkpoint(POMO, ckpt_path, env)
    if model is None:
        raise SystemExit(f"Failed to load checkpoint: {load_error}")

    model.eval()
    model.to(device)
    model.policy.eval()
    model.policy.to(device)

    td_gauss = env.generator(batch_size=args.num_instances)
    locs_gauss = td_gauss["locs"].cpu()
    dataset_gauss = build_dataset(locs_gauss)

    print("=== Gaussian TSP100 ===")
    result_gauss, method_gauss, eval_kwargs_gauss, wall_gauss = eval_policy_with_dataset(
        env,
        model,
        dataset_gauss,
        args.method,
        args.num_augment,
        args.num_starts,
        args.force_dihedral_8,
        args.batch_size,
        args.max_batch_size,
        args.samples,
        args.temperature,
        args.top_p,
        args.top_k,
    )
    mean_len_gauss, std_len_gauss = summarize_lengths(locs_gauss, result_gauss["actions"])

    locs_uniform = rosenblatt_normal(
        locs_gauss, args.loc_mean, args.loc_std, args.rosenblatt_eps
    )
    dataset_uniform = build_dataset(locs_uniform)

    print("=== Rosenblatt -> Uniform (evaluate on original Gaussian) ===")
    result_uniform, method_uniform, eval_kwargs_uniform, wall_uniform = eval_policy_with_dataset(
        env,
        model,
        dataset_uniform,
        args.method,
        args.num_augment,
        args.num_starts,
        args.force_dihedral_8,
        args.batch_size,
        args.max_batch_size,
        args.samples,
        args.temperature,
        args.top_p,
        args.top_k,
    )
    mean_len_uniform, std_len_uniform = summarize_lengths(
        locs_gauss, result_uniform["actions"]
    )

    total_wall = wall_gauss + wall_uniform

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not output_path.exists()
    with output_path.open("a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(
                [
                    "mode",
                    "mean_length",
                    "std_length",
                    "inference_time_s",
                    "wall_time_s",
                    "num_instances",
                    "seed",
                    "method",
                    "num_augment",
                    "num_starts",
                    "ckpt_path",
                    "loc_mean",
                    "loc_std",
                ]
            )
        writer.writerow(
            [
                "gaussian",
                mean_len_gauss,
                std_len_gauss,
                float(result_gauss["inference_time"]),
                wall_gauss,
                args.num_instances,
                args.seed,
                method_gauss,
                eval_kwargs_gauss.get("num_augment"),
                eval_kwargs_gauss.get("num_starts"),
                str(ckpt_path),
                args.loc_mean,
                args.loc_std,
            ]
        )
        writer.writerow(
            [
                "rosenblatt_uniform",
                mean_len_uniform,
                std_len_uniform,
                float(result_uniform["inference_time"]),
                wall_uniform,
                args.num_instances,
                args.seed,
                method_uniform,
                eval_kwargs_uniform.get("num_augment"),
                eval_kwargs_uniform.get("num_starts"),
                str(ckpt_path),
                args.loc_mean,
                args.loc_std,
            ]
        )

    print("")
    print("Results:")
    print(
        f"  Gaussian mean length: {mean_len_gauss:.6f} (std {std_len_gauss:.6f})"
    )
    print(
        f"  Rosenblatt->Uniform mean length (on Gaussian coords): {mean_len_uniform:.6f} "
        f"(std {std_len_uniform:.6f})"
    )
    print("")
    print("Timing:")
    print(
        f"  Gaussian test wall time: {wall_gauss:.2f}s (inference {result_gauss['inference_time']:.2f}s)"
    )
    print(
        f"  Rosenblatt test wall time: {wall_uniform:.2f}s (inference {result_uniform['inference_time']:.2f}s)"
    )
    print(f"  Total wall time: {total_wall:.2f}s")
    print(f"Saved results to: {output_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
