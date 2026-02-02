import argparse
import os
import time

import lightning as L
import torch
from lightning.pytorch.loggers import CSVLogger
from tensordict.tensordict import TensorDict

from rl4co.envs.routing import (
    CVRPEnv,
    CVRPGenerator,
    KnapsackEnv,
    KnapsackGenerator,
    TSPEnv,
    TSPGenerator,
)
from rl4co.models import AttentionModelPolicy, EAM, POMO
from rl4co.utils import RL4COTrainer


def _parse_milestones(value: str | None) -> list[int] | None:
    if value is None:
        return None
    if isinstance(value, list):
        return [int(v) for v in value]
    parts = [item.strip() for item in value.split(",") if item.strip()]
    return [int(item) for item in parts] if parts else None


def _default_lr_milestones(epochs: int) -> list[int]:
    if epochs <= 1:
        return []
    first = max(1, int(epochs * 0.8))
    second = max(first + 1, int(epochs * 0.95))
    second = min(second, max(1, epochs - 1))
    if first >= epochs:
        return []
    if second <= first or second >= epochs:
        return [first]
    return [first, second]


def _normalize_problem(name: str) -> str:
    key = name.lower().strip()
    if key == "knapsack":
        key = "kp"
    return key


def _parse_bool(value: str | None, *, default: bool) -> bool:
    if value is None:
        return default
    lowered = value.strip().lower()
    if lowered in {"1", "true", "yes", "y"}:
        return True
    if lowered in {"0", "false", "no", "n"}:
        return False
    raise SystemExit(f"Invalid boolean value: {value!r}. Use true/false.")


class CorrelatedKnapsackGenerator(KnapsackGenerator):
    """Knapsack generator with mixed correlated values (harder but similar scale)."""

    def __init__(self, *args, value_noise: float = 0.05, corr: float = 0.7, **kwargs):
        super().__init__(*args, **kwargs)
        self.value_noise = value_noise
        if not 0.0 <= corr <= 1.0:
            raise ValueError(f"corr must be in [0, 1], got {corr}")
        self.corr = corr

    def _generate(self, batch_size) -> TensorDict:
        weights = self.weight_sampler.sample((*batch_size, self.num_items))
        base_values = self.value_sampler.sample((*batch_size, self.num_items))
        noise = torch.empty_like(weights).uniform_(-self.value_noise, self.value_noise)
        values = (
            self.corr * weights + (1.0 - self.corr) * base_values + noise
        ).clamp(self.min_value, self.max_value)

        items = torch.stack((weights, values), dim=-1)
        depot = torch.zeros(*batch_size, 1, 2, device=items.device, dtype=items.dtype)
        locs = torch.cat((depot, items), dim=-2)
        capacity = torch.full((*batch_size, 1), self.capacity)

        return TensorDict(
            {
                "weights": weights,
                "demand": weights,
                "values": values,
                "locs": locs,
                "vehicle_capacity": capacity,
            },
            batch_size=batch_size,
        )


def _require_eam():
    if EAM is not None:
        return EAM
    try:
        from rl4co.models.zoo.earl.model import EAM as EAM_cls
    except Exception as exc:
        raise SystemExit(
            "EAM is unavailable. Install optional dependencies (e.g., numba) "
            "and ensure rl4co.models.zoo.earl is importable. "
            f"Original error: {exc}"
        ) from exc
    return EAM_cls


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Train POMO or EAM-POMO on CVRP/TSP/KP (any size)."
    )
    parser.add_argument(
        "--model",
        type=str,
        default="pomo",
        help="Model to train: pomo or eam-pomo",
    )
    parser.add_argument(
        "--problem",
        type=str,
        default="cvrp",
        help="Problem type: cvrp, tsp, kp (or knapsack).",
    )
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--device", type=int, default=0, help="Physical GPU id")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--problem-size",
        type=int,
        default=100,
        help="Problem size: #customers for CVRP, #nodes for TSP, #items for KP.",
    )
    parser.add_argument(
        "--capacity",
        type=float,
        default=None,
        help="Override capacity for CVRP/KP (defaults to generator table/heuristic).",
    )
    parser.add_argument("--num-augment", type=int, default=8)
    parser.add_argument(
        "--num-starts",
        type=int,
        default=None,
        help="Number of multistart rollouts. For KP, recommend 5 or 10.",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--train-data-size", type=int, default=160_000)
    parser.add_argument("--val-data-size", type=int, default=10_000)
    parser.add_argument("--test-data-size", type=int, default=10_000)
    parser.add_argument("--log-dir", type=str, default="logs")
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--version", type=str, default=None)
    parser.add_argument(
        "--lr-milestones",
        type=str,
        default=None,
        help="Comma-separated epochs for MultiStepLR (default: 80% and 95% of --epochs).",
    )
    parser.add_argument(
        "--kp-correlated",
        type=str,
        default="true",
        help="Use correlated KP values (true/false). Default: true.",
    )
    parser.add_argument("--kp-min-weight", type=float, default=0.4)
    parser.add_argument("--kp-max-weight", type=float, default=0.6)
    parser.add_argument("--kp-min-value", type=float, default=0.4)
    parser.add_argument("--kp-max-value", type=float, default=0.6)
    parser.add_argument(
        "--kp-value-noise",
        type=float,
        default=0.05,
        help="Noise added to correlated KP values (default: 0.05).",
    )
    parser.add_argument(
        "--kp-corr",
        type=float,
        default=0.7,
        help="Mixing factor between weights and independent values (0..1).",
    )
    args = parser.parse_args()

    model_key = args.model.lower().replace("_", "-")
    if model_key not in {"pomo", "eam-pomo"}:
        raise SystemExit("Unknown model. Use --model pomo or --model eam-pomo.")

    os.environ.setdefault("WANDB_MODE", "offline")
    L.seed_everything(args.seed, workers=True)

    problem_key = _normalize_problem(args.problem)
    if problem_key not in {"cvrp", "tsp", "kp"}:
        raise SystemExit("Unknown problem. Use --problem cvrp, tsp, or kp.")
    if problem_key == "kp" and args.num_starts is None:
        args.num_starts = 5

    check_solution = not (problem_key == "kp" and model_key == "pomo")
    kp_correlated = _parse_bool(args.kp_correlated, default=True)
    if problem_key == "cvrp":
        env = CVRPEnv(
            CVRPGenerator(
                num_loc=args.problem_size,
                loc_distribution="uniform",
                num_depots=1,
                capacity=args.capacity,
            )
        )
    elif problem_key == "tsp":
        env = TSPEnv(
            TSPGenerator(
                num_loc=args.problem_size,
                loc_distribution="uniform",
            )
        )
    else:
        kp_gen_cls = CorrelatedKnapsackGenerator if kp_correlated else KnapsackGenerator
        kp_gen_kwargs = {
            "num_items": args.problem_size,
            "min_weight": args.kp_min_weight,
            "max_weight": args.kp_max_weight,
            "min_value": args.kp_min_value,
            "max_value": args.kp_max_value,
            "weight_distribution": "uniform",
            "value_distribution": "uniform",
            "capacity": args.capacity,
        }
        if kp_correlated:
            kp_gen_kwargs["value_noise"] = args.kp_value_noise
            kp_gen_kwargs["corr"] = args.kp_corr
        env = KnapsackEnv(kp_gen_cls(**kp_gen_kwargs), check_solution=check_solution)
    policy = AttentionModelPolicy(
        env_name=env.name,
        embed_dim=128,
        num_encoder_layers=6,
        num_heads=8,
        normalization="instance",
        use_graph_context=False,
    )

    metrics = {
        "train": [
            "loss",
            "reward",
            "max_reward",
            "delta_nll",
            "diversity_edge",
            "diversity_edge_entropy",
            "diversity_edge_simpson",
            "diversity_route",
            "diversity_edit_edge",
            "ga_cost_gain",
            "ga_cost_gain_rel",
            "t_decode",
            "t_ga",
            "t_diag",
        ],
        "val": [
            "reward",
            "reward_no_ls",
            "max_reward",
            "max_reward_no_ls",
            "max_aug_reward",
            "max_aug_reward_no_ls",
        ],
        "test": ["reward", "max_reward", "max_aug_reward"],
    }

    lr_milestones = _parse_milestones(args.lr_milestones)
    if lr_milestones is None:
        lr_milestones = _default_lr_milestones(args.epochs)

    if model_key == "pomo":
        model = POMO(
            env,
            policy,
            num_starts=args.num_starts,
            reward_scale="norm",
            batch_size=args.batch_size,
            optimizer_kwargs={"lr": 1e-4, "weight_decay": 1e-6},
            lr_scheduler="MultiStepLR",
            lr_scheduler_kwargs={"milestones": lr_milestones, "gamma": 0.1},
            train_data_size=args.train_data_size,
            val_data_size=args.val_data_size,
            test_data_size=args.test_data_size,
            num_augment=args.num_augment,
            metrics=metrics,
        )
        model_tag = "pomo"
    else:
        ea_kwargs = {
            "num_generations": 3,
            "mutation_rate": 0.1,
            "crossover_rate": 0.6,
            "selection_rate": 0.2,
            "batch_size": args.batch_size,
            "ea_batch_size": args.batch_size,
            "alpha": 0.5,
            "beta": 3,
            "ea_prob": 0.01,
            "ea_epoch": 700,
            "improve_mode": "ga",
            "val_improve": False,
            "val_improve_mode": "ga",
            "val_num_generations": 0,
        }
        eam_cls = _require_eam()
        model = eam_cls(
            env,
            policy,
            baseline="shared",
            num_starts=args.num_starts,
            reward_scale="norm",
            batch_size=args.batch_size,
            optimizer_kwargs={"lr": 1e-4, "weight_decay": 1e-6},
            lr_scheduler="MultiStepLR",
            lr_scheduler_kwargs={"milestones": lr_milestones, "gamma": 0.1},
            train_data_size=args.train_data_size,
            val_data_size=args.val_data_size,
            test_data_size=args.test_data_size,
            num_augment=args.num_augment,
            metrics=metrics,
            ea_kwargs=ea_kwargs,
        )
        model_tag = "eam_pomo"

    run_name = args.run_name or f"{model_tag}_{problem_key}{args.problem_size}"
    version = args.version or (
        f"{time.strftime('%Y%m%d_%H%M%S')}_seed{args.seed}_gpu{args.device}"
    )
    logger = CSVLogger(save_dir=args.log_dir, name=run_name, version=version)

    trainer = RL4COTrainer(
        max_epochs=args.epochs,
        accelerator="gpu",
        devices=[args.device],
        precision=32,
        logger=logger,
        enable_checkpointing=False,
    )
    trainer.fit(model)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
