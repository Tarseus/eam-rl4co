import argparse
import os
from typing import Iterable


def _as_int_list(xs: Iterable[str]) -> list[int]:
    return [int(x) for x in xs]


def _train_eam_kp(
    *,
    problem_sizes: list[int],
    epochs: int,
    baseline: str,
    device_index: int = 0,
    ea_kwargs: dict | None = None,
):
    os.environ.setdefault("WANDB_MODE", "offline")

    from lightning.pytorch.loggers import WandbLogger

    from rl4co.envs.routing import KnapsackEnv, KnapsackGenerator
    from rl4co.models import AttentionModelPolicy, EAM
    from rl4co.utils import RL4COTrainer

    ea_kwargs = ea_kwargs or {
        "num_generations": 3,
        "mutation_rate": 0.1,
        "crossover_rate": 0.6,
        "selection_rate": 0.2,
        "batch_size": 64,
        "ea_batch_size": 64,
        "alpha": 0.5,
        "beta": 3,
        "ea_prob": 0.01,
        "ea_epoch": 700,
    }

    metrics = {
        "train": ["reward", "loss"],
        "val": ["reward", "max_reward", "max_aug_reward"],
        "test": ["reward", "max_reward", "max_aug_reward"],
    }

    for problem_size in problem_sizes:
        generator = KnapsackGenerator(
            num_items=problem_size, weight_distribution="uniform", value_distribution="uniform"
        )
        env = KnapsackEnv(generator)

        policy = AttentionModelPolicy(
            env_name=env.name,
            embed_dim=128,
            num_encoder_layers=6,
            num_heads=8,
            normalization="instance",
            use_graph_context=False,
        )

        model = EAM(
            env,
            policy,
            batch_size=ea_kwargs["batch_size"],
            optimizer_kwargs={"lr": 1e-4, "weight_decay": 1e-6},
            lr_scheduler="MultiStepLR",
            lr_scheduler_kwargs={"milestones": [80, 95], "gamma": 0.1},
            train_data_size=100_000,
            val_data_size=10_000,
            test_data_size=10_000,
            baseline=baseline,
            num_augment=0,
            num_starts=1 if baseline == "rollout" else None,
            ea_kwargs=ea_kwargs,
            metrics=metrics,
        )

        name = f"eam_{'am' if baseline == 'rollout' else 'pomo'}_kp{problem_size}"
        logger = WandbLogger(project="rl4co_kp_eam", name=name)

        trainer = RL4COTrainer(
            max_epochs=epochs,
            accelerator="gpu",
            devices=[device_index],
            precision=32,
            logger=logger,
        )
        trainer.fit(model)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--cuda", type=int, default=4, help="Physical GPU id (run sequentially on this GPU only)")
    args = parser.parse_args()

    # Pin to a single GPU. Lightning will see this as CUDA device 0 inside the process.
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda)
    os.environ.setdefault("WANDB_MODE", "offline")

    # Sequential runs on the same GPU:
    # 1) KP50 + KP100: EAM with AM(rollout) baseline
    _train_eam_kp(problem_sizes=[50, 100], epochs=args.epochs, baseline="rollout", device_index=0)
    # 2) KP100: EAM with POMO(shared) baseline
    _train_eam_kp(problem_sizes=[100], epochs=args.epochs, baseline="shared", device_index=0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
