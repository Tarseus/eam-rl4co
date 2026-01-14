import argparse
import os
import subprocess
import sys
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


def _launch_subprocesses(args: argparse.Namespace) -> int:
    # Launch 2 worker processes in parallel:
    # - worker A: KP50 + KP100 EAM-attentionmodel on CUDA:4
    # - worker B: KP100 EAM-POMO on CUDA:5
    python = sys.executable
    script = os.path.abspath(__file__)

    base_env = os.environ.copy()
    base_env.setdefault("WANDB_MODE", "offline")

    cmd_a = [
        python,
        script,
        "--worker",
        "eam-am",
        "--epochs",
        str(args.epochs),
        "--problem-sizes",
        "50",
        "100",
    ]
    env_a = base_env.copy()
    env_a["CUDA_VISIBLE_DEVICES"] = str(args.cuda4)

    cmd_b = [
        python,
        script,
        "--worker",
        "eam-pomo",
        "--epochs",
        str(args.epochs),
        "--problem-sizes",
        "100",
    ]
    env_b = base_env.copy()
    env_b["CUDA_VISIBLE_DEVICES"] = str(args.cuda5)

    p_a = subprocess.Popen(cmd_a, env=env_a)
    p_b = subprocess.Popen(cmd_b, env=env_b)

    rc_a = p_a.wait()
    rc_b = p_b.wait()
    return rc_a or rc_b


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--cuda4", type=int, default=4, help="Physical GPU id for the EAM-AM worker")
    parser.add_argument("--cuda5", type=int, default=5, help="Physical GPU id for the EAM-POMO worker")
    parser.add_argument(
        "--worker",
        type=str,
        default=None,
        choices=["eam-am", "eam-pomo"],
        help="Internal flag (used by the launcher)",
    )
    parser.add_argument("--problem-sizes", nargs="+", type=int, default=None)
    args = parser.parse_args()

    if args.worker is None:
        return _launch_subprocesses(args)

    if not args.problem_sizes:
        raise SystemExit("--problem-sizes is required in worker mode")

    if args.worker == "eam-am":
        _train_eam_kp(problem_sizes=_as_int_list(args.problem_sizes), epochs=args.epochs, baseline="rollout")
        return 0
    if args.worker == "eam-pomo":
        _train_eam_kp(problem_sizes=_as_int_list(args.problem_sizes), epochs=args.epochs, baseline="shared")
        return 0

    raise SystemExit(f"Unknown worker: {args.worker}")


if __name__ == "__main__":
    raise SystemExit(main())

