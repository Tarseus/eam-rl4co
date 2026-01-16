import argparse
import os
import time

from rl4co.envs.routing import TSPEnv, TSPGenerator
from rl4co.models import AttentionModelPolicy, EAM
from rl4co.utils import RL4COTrainer
from lightning.pytorch.loggers import CSVLogger


def main() -> int:
    parser = argparse.ArgumentParser(description="Train EAM (local search) on TSP100.")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--cuda", type=int, default=0, help="Physical GPU id")
    parser.add_argument("--baseline", choices=["shared", "rollout"], default="shared")
    parser.add_argument("--num-augment", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--train-data-size", type=int, default=160_000)
    parser.add_argument("--val-data-size", type=int, default=10_000)
    parser.add_argument("--test-data-size", type=int, default=10_000)
    parser.add_argument("--log-dir", type=str, default="logs")
    parser.add_argument("--run-name", type=str, default="eam_tsp100_localsearch")
    parser.add_argument("--version", type=str, default=None)
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda)
    os.environ.setdefault("WANDB_MODE", "offline")

    env = TSPEnv(TSPGenerator(num_loc=100, loc_distribution="uniform"))
    policy = AttentionModelPolicy(
        env_name=env.name,
        embed_dim=128,
        num_encoder_layers=6,
        num_heads=8,
        normalization="instance",
        use_graph_context=False,
    )

    num_generations = 3
    ea_kwargs = {
        "num_generations": num_generations,
        "mutation_rate": 0.1,
        "crossover_rate": 0.6,
        "selection_rate": 0.2,
        "batch_size": args.batch_size,
        "ea_batch_size": args.batch_size,
        "alpha": 0.5,
        "beta": 3,
        "ea_prob": 0.01,
        "ea_epoch": 700,
        "improve_mode": "local_search",
        "local_search_max_iterations": num_generations,
    }

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

    model = EAM(
        env,
        policy,
        baseline=args.baseline,
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

    version = args.version or time.strftime("%Y%m%d_%H%M%S")
    logger = CSVLogger(save_dir=args.log_dir, name=args.run_name, version=version)

    trainer = RL4COTrainer(
        max_epochs=args.epochs,
        accelerator="gpu",
        devices=[0],
        precision=32,
        logger=logger,
        enable_checkpointing=False,
    )
    trainer.fit(model)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
