import argparse
import os
import time

import lightning as L
from lightning.pytorch.loggers import CSVLogger

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
    args = parser.parse_args()

    model_key = args.model.lower().replace("_", "-")
    if model_key not in {"pomo", "eam-pomo"}:
        raise SystemExit("Unknown model. Use --model pomo or --model eam-pomo.")

    os.environ.setdefault("WANDB_MODE", "offline")
    L.seed_everything(args.seed, workers=True)

    problem_key = _normalize_problem(args.problem)
    if problem_key not in {"cvrp", "tsp", "kp"}:
        raise SystemExit("Unknown problem. Use --problem cvrp, tsp, or kp.")

    check_solution = not (problem_key == "kp" and model_key == "pomo")
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
        env = KnapsackEnv(
            KnapsackGenerator(
                num_items=args.problem_size,
                weight_distribution="uniform",
                value_distribution="uniform",
                capacity=args.capacity,
            ),
            check_solution=check_solution,
        )
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
