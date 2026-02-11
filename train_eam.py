import argparse
import os
import time
from pathlib import Path

import torch
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger

from rl4co.envs.routing import CVRPEnv, CVRPGenerator, TSPEnv, TSPGenerator
from rl4co.models import AttentionModelPolicy, EAM
from rl4co.utils import RL4COTrainer


def _build_env(problem: str, size: int):
    problem = problem.lower().strip()
    if problem == "tsp":
        return TSPEnv(TSPGenerator(num_loc=size, loc_distribution="uniform"))
    if problem == "cvrp":
        return CVRPEnv(CVRPGenerator(num_loc=size, loc_distribution="uniform", num_depots=1))
    raise SystemExit(f"Unsupported --problem {problem!r}. Use tsp/cvrp.")


def _resolve_resume_checkpoint(run_name: str, version: str | None) -> tuple[str | None, str | None]:
    root = Path("checkpoints") / run_name
    if version:
        vdir = root / version
        if not vdir.exists():
            return None, None
        last_ckpt = vdir / "last.ckpt"
        if last_ckpt.exists():
            return str(last_ckpt), version
        ckpts = sorted(vdir.rglob("*.ckpt"), key=lambda p: p.stat().st_mtime)
        if ckpts:
            return str(ckpts[-1]), version
        return None, None

    if not root.exists():
        return None, None
    version_dirs = sorted(
        [d for d in root.iterdir() if d.is_dir()], key=lambda p: p.stat().st_mtime
    )
    for vdir in reversed(version_dirs):
        last_ckpt = vdir / "last.ckpt"
        if last_ckpt.exists():
            return str(last_ckpt), vdir.name
        ckpts = sorted(vdir.rglob("*.ckpt"), key=lambda p: p.stat().st_mtime)
        if ckpts:
            return str(ckpts[-1]), vdir.name
    return None, None


def _default_lr_milestones(epochs: int) -> list[int]:
    if epochs <= 1:
        return []
    first = max(1, int(epochs * 0.8))
    second = max(first + 1, int(epochs * 0.95))
    if first >= epochs:
        return []
    if second >= epochs:
        return [first]
    return [first, second]


def main() -> int:
    parser = argparse.ArgumentParser(description="Minimal EAM training entrypoint.")
    parser.add_argument("--problem", choices=["tsp", "cvrp"], default="tsp")
    parser.add_argument("--size", type=int, default=100)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--cuda", type=int, default=0, help="Physical GPU id")
    parser.add_argument("--baseline", choices=["shared", "rollout"], default="shared")
    parser.add_argument("--num-augment", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--train-data-size", type=int, default=160_000)
    parser.add_argument("--val-data-size", type=int, default=10_000)
    parser.add_argument("--test-data-size", type=int, default=10_000)
    parser.add_argument("--log-dir", type=str, default="logs")
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--version", type=str, default=None)
    parser.add_argument("--resume", action="store_true", help="Resume from latest checkpoint.")
    parser.add_argument("--ckpt-path", type=str, default=None, help="Resume from a specific checkpoint.")
    parser.add_argument("--precision", type=int, default=32, choices=[16, 32])
    # EA hyperparameters (keep minimal; override from CLI if needed)
    parser.add_argument("--ea-generations", type=int, default=3)
    parser.add_argument("--ea-mutation", type=float, default=0.1)
    parser.add_argument("--ea-crossover", type=float, default=0.6)
    parser.add_argument("--ea-selection", type=float, default=0.2)
    parser.add_argument("--ea-alpha", type=float, default=0.5)
    parser.add_argument("--ea-beta", type=float, default=3.0)
    parser.add_argument("--ea-prob", type=float, default=0.01)
    parser.add_argument("--ea-epoch", type=int, default=700)
    args = parser.parse_args()

    if args.size <= 1:
        raise SystemExit("--size must be > 1.")

    run_name = args.run_name or f"eam_{args.problem}{args.size}"

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda)
    os.environ.setdefault("WANDB_MODE", "offline")

    env = _build_env(args.problem, args.size)
    policy = AttentionModelPolicy(
        env_name=env.name,
        embed_dim=128,
        num_encoder_layers=6,
        num_heads=8,
        normalization="instance",
        use_graph_context=False,
    )

    ea_kwargs = {
        "num_generations": args.ea_generations,
        "mutation_rate": args.ea_mutation,
        "crossover_rate": args.ea_crossover,
        "selection_rate": args.ea_selection,
        "batch_size": args.batch_size,
        "ea_batch_size": args.batch_size,
        "alpha": args.ea_alpha,
        "beta": args.ea_beta,
        "ea_prob": args.ea_prob,
        "ea_epoch": args.ea_epoch,
    }

    metrics = {
        "train": ["loss", "reward", "max_reward"],
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
        lr_scheduler_kwargs={"milestones": _default_lr_milestones(args.epochs), "gamma": 0.1},
        train_data_size=args.train_data_size,
        val_data_size=args.val_data_size,
        test_data_size=args.test_data_size,
        num_augment=args.num_augment,
        metrics=metrics,
        ea_kwargs=ea_kwargs,
    )

    version = args.version or time.strftime("%Y%m%d_%H%M%S")
    ckpt_path = None
    if args.ckpt_path:
        ckpt_candidate = Path(args.ckpt_path)
        if not ckpt_candidate.exists():
            raise SystemExit(f"Checkpoint not found: {ckpt_candidate}")
        ckpt_path = str(ckpt_candidate)
        if args.version is None:
            version = ckpt_candidate.parent.name
    elif args.resume:
        ckpt_path, resume_version = _resolve_resume_checkpoint(run_name, args.version)
        if ckpt_path is None:
            raise SystemExit("No checkpoint found to resume from.")
        if args.version is None and resume_version is not None:
            version = resume_version

    logger = CSVLogger(save_dir=args.log_dir, name=run_name, version=version)
    checkpoint_dir = os.path.join("checkpoints", run_name, version)
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename="epoch_{epoch:03d}",
        save_last=True,
        save_top_k=1,
        monitor="val/reward",
        mode="max",
    )

    accelerator = "gpu" if torch.cuda.is_available() else "cpu"
    devices = [0] if accelerator == "gpu" else 1
    trainer = RL4COTrainer(
        max_epochs=args.epochs,
        accelerator=accelerator,
        devices=devices,
        precision=args.precision,
        logger=logger,
        callbacks=[checkpoint_callback],
        enable_checkpointing=True,
    )
    trainer.fit(model, ckpt_path=ckpt_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
