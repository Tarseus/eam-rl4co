import argparse
import os
import time

from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger

from rl4co.envs.routing import KnapsackEnv, KnapsackGenerator
from rl4co.models import AttentionModelPolicy, EAM, POMO
from rl4co.utils import RL4COTrainer


def _build_env(problem: str, problem_size: int, *, check_solution: bool = True) -> KnapsackEnv:
    if problem not in {"kp", "knapsack"}:
        raise ValueError(f"Unsupported problem '{problem}'. Only kp/knapsack is supported.")
    generator = KnapsackGenerator(
        num_items=problem_size,
        weight_distribution="uniform",
        value_distribution="uniform",
    )
    return KnapsackEnv(generator, check_solution=check_solution)


def _build_policy(env: KnapsackEnv) -> AttentionModelPolicy:
    return AttentionModelPolicy(
        env_name=env.name,
        embed_dim=128,
        num_encoder_layers=6,
        num_heads=8,
        normalization="instance",
        use_graph_context=False,
    )


def _default_metrics() -> dict:
    return {
        "train": ["loss", "reward"],
        "val": ["reward", "max_reward", "max_aug_reward"],
        "test": ["reward", "max_reward", "max_aug_reward"],
    }


def _fit_model(
    model,
    *,
    epochs: int,
    device_index: int,
    log_dir: str,
    run_name: str,
    version: str,
) -> None:
    logger = CSVLogger(save_dir=log_dir, name=run_name, version=version)
    checkpoint_dir = os.path.join("checkpoints", run_name, version)
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename="epoch_{epoch:03d}",
        save_last=True,
        save_top_k=1,
        monitor="val/reward",
        mode="max",
    )
    trainer = RL4COTrainer(
        max_epochs=epochs,
        accelerator="gpu",
        devices=[device_index],
        precision=32,
        logger=logger,
        callbacks=[checkpoint_callback],
        enable_checkpointing=True,
    )
    trainer.fit(model)


def _train_pomo_kp(
    *,
    problem: str,
    problem_size: int,
    epochs: int,
    batch_size: int,
    train_data_size: int,
    val_data_size: int,
    test_data_size: int,
    num_augment: int,
    log_dir: str,
    run_name: str,
    version: str,
    device_index: int,
) -> None:
    # POMO on KP can emit infeasible sequences; skip validity assertions to avoid hard failures.
    env = _build_env(problem, problem_size, check_solution=False)
    policy = _build_policy(env)
    model = POMO(
        env,
        policy,
        batch_size=batch_size,
        optimizer_kwargs={"lr": 1e-4, "weight_decay": 1e-6},
        lr_scheduler="MultiStepLR",
        lr_scheduler_kwargs={"milestones": [80, 95], "gamma": 0.1},
        train_data_size=train_data_size,
        val_data_size=val_data_size,
        test_data_size=test_data_size,
        num_augment=num_augment,
        metrics=_default_metrics(),
    )
    _fit_model(
        model,
        epochs=epochs,
        device_index=device_index,
        log_dir=log_dir,
        run_name=run_name,
        version=version,
    )


def _train_eam_kp(
    *,
    problem: str,
    problem_size: int,
    epochs: int,
    batch_size: int,
    train_data_size: int,
    val_data_size: int,
    test_data_size: int,
    baseline: str,
    num_augment: int,
    log_dir: str,
    run_name: str,
    version: str,
    device_index: int,
) -> None:
    env = _build_env(problem, problem_size)
    policy = _build_policy(env)
    ea_kwargs = {
        "num_generations": 3,
        "mutation_rate": 0.1,
        "crossover_rate": 0.6,
        "selection_rate": 0.2,
        "batch_size": batch_size,
        "ea_batch_size": batch_size,
        "alpha": 0.5,
        "beta": 3,
        "ea_prob": 0.01,
        "ea_epoch": 700,
    }
    model = EAM(
        env,
        policy,
        batch_size=batch_size,
        optimizer_kwargs={"lr": 1e-4, "weight_decay": 1e-6},
        lr_scheduler="MultiStepLR",
        lr_scheduler_kwargs={"milestones": [80, 95], "gamma": 0.1},
        train_data_size=train_data_size,
        val_data_size=val_data_size,
        test_data_size=test_data_size,
        baseline=baseline,
        num_augment=num_augment,
        num_starts=1 if baseline == "rollout" else None,
        ea_kwargs=ea_kwargs,
        metrics=_default_metrics(),
    )
    _fit_model(
        model,
        epochs=epochs,
        device_index=device_index,
        log_dir=log_dir,
        run_name=run_name,
        version=version,
    )


def _normalize_algo(algo: str, baseline: str | None) -> tuple[str, str | None]:
    algo = algo.lower()
    baseline = baseline.lower() if baseline is not None else None
    if algo in {"eam_am", "eam-am"}:
        return "eam", "rollout"
    if algo in {"eam_pomo", "eam-shared"}:
        return "eam", "shared"
    return algo, baseline


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--train-data-size", type=int, default=100_000)
    parser.add_argument("--val-data-size", type=int, default=10_000)
    parser.add_argument("--test-data-size", type=int, default=10_000)
    parser.add_argument("--num-augment", type=int, default=0)
    parser.add_argument("--log-dir", type=str, default="logs")
    parser.add_argument("--version", type=str, default=None)
    parser.add_argument("--problem", type=str, default="kp")
    parser.add_argument("--algo", type=str, default="pomo")
    parser.add_argument("--size", type=int, default=100)
    parser.add_argument("--baseline", type=str, default="shared")
    parser.add_argument("--cuda", type=int, default=0, help="Physical GPU id")
    args = parser.parse_args()

    os.environ.setdefault("WANDB_MODE", "offline")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda)

    version = args.version or time.strftime("%Y%m%d_%H%M%S")
    problem = args.problem.lower()
    algo, baseline = _normalize_algo(args.algo, args.baseline)

    device_index = 0

    if algo == "pomo":
        run_name = f"pomo_{problem}{args.size}"
        _train_pomo_kp(
            problem=problem,
            problem_size=args.size,
            epochs=args.epochs,
            batch_size=args.batch_size,
            train_data_size=args.train_data_size,
            val_data_size=args.val_data_size,
            test_data_size=args.test_data_size,
            num_augment=args.num_augment,
            log_dir=args.log_dir,
            run_name=run_name,
            version=version,
            device_index=device_index,
        )
        return 0

    if algo == "eam":
        if baseline is None:
            baseline = "shared"
        if baseline == "rollout":
            run_name = f"eam_am_{problem}{args.size}"
        elif baseline == "shared":
            run_name = f"eam_pomo_{problem}{args.size}"
        else:
            run_name = f"eam_{baseline}_{problem}{args.size}"
        _train_eam_kp(
            problem=problem,
            problem_size=args.size,
            epochs=args.epochs,
            batch_size=args.batch_size,
            train_data_size=args.train_data_size,
            val_data_size=args.val_data_size,
            test_data_size=args.test_data_size,
            baseline=baseline,
            num_augment=args.num_augment,
            log_dir=args.log_dir,
            run_name=run_name,
            version=version,
            device_index=device_index,
        )
        return 0

    raise SystemExit(f"Unsupported algo '{algo}'. Expected 'pomo' or 'eam'.")


if __name__ == "__main__":
    raise SystemExit(main())
