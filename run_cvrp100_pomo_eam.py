import argparse
import os
import time

import lightning as L
from lightning.pytorch.loggers import CSVLogger

from rl4co.envs.routing import CVRPEnv, CVRPGenerator
from rl4co.models import AttentionModelPolicy, EAM, POMO
from rl4co.utils import RL4COTrainer


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Train POMO or EAM-POMO on CVRP100."
    )
    parser.add_argument(
        "--model",
        type=str,
        default="pomo",
        help="Model to train: pomo or eam-pomo",
    )
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--device", type=int, default=0, help="Physical GPU id")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--problem-size", type=int, default=100)
    parser.add_argument("--num-augment", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--train-data-size", type=int, default=160_000)
    parser.add_argument("--val-data-size", type=int, default=10_000)
    parser.add_argument("--test-data-size", type=int, default=10_000)
    parser.add_argument("--log-dir", type=str, default="logs")
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--version", type=str, default=None)
    args = parser.parse_args()

    model_key = args.model.lower().replace("_", "-")
    if model_key not in {"pomo", "eam-pomo"}:
        raise SystemExit("Unknown model. Use --model pomo or --model eam-pomo.")

    os.environ.setdefault("WANDB_MODE", "offline")
    L.seed_everything(args.seed, workers=True)

    env = CVRPEnv(
        CVRPGenerator(
            num_loc=args.problem_size,
            loc_distribution="uniform",
            num_depots=1,
        )
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
            "diversity_node",
            "ga_cost_gain",
            "ga_cost_gain_rel",
            "t_decode",
            "t_ga",
            "t_diag",
        ],
        "val": ["reward", "max_reward", "max_aug_reward"],
        "test": ["reward", "max_reward", "max_aug_reward"],
    }

    if model_key == "pomo":
        model = POMO(
            env,
            policy,
            batch_size=args.batch_size,
            optimizer_kwargs={"lr": 1e-4, "weight_decay": 1e-6},
            lr_scheduler="MultiStepLR",
            lr_scheduler_kwargs={"milestones": [160, 190], "gamma": 0.1},
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
        }
        model = EAM(
            env,
            policy,
            baseline="shared",
            batch_size=args.batch_size,
            optimizer_kwargs={"lr": 1e-4, "weight_decay": 1e-6},
            lr_scheduler="MultiStepLR",
            lr_scheduler_kwargs={"milestones": [160, 190], "gamma": 0.1},
            train_data_size=args.train_data_size,
            val_data_size=args.val_data_size,
            test_data_size=args.test_data_size,
            num_augment=args.num_augment,
            metrics=metrics,
            ea_kwargs=ea_kwargs,
        )
        model_tag = "eam_pomo"

    run_name = args.run_name or f"{model_tag}_cvrp{args.problem_size}"
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
