import argparse
import os
import time
from pathlib import Path

from lightning.pytorch.callbacks import ModelCheckpoint
from rl4co.envs.routing import TSPEnv, TSPGenerator
from rl4co.models import AttentionModelPolicy, EAM
from rl4co.utils import RL4COTrainer
from lightning.pytorch.loggers import CSVLogger


def main() -> int:
    parser = argparse.ArgumentParser(description="Train EAM on TSP100.")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--cuda", type=int, default=0, help="Physical GPU id")
    parser.add_argument("--baseline", choices=["shared", "rollout"], default="shared")
    parser.add_argument("--num-augment", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--train-data-size", type=int, default=160_000)
    parser.add_argument("--val-data-size", type=int, default=10_000)
    parser.add_argument("--test-data-size", type=int, default=10_000)
    parser.add_argument("--log-dir", type=str, default="logs")
    parser.add_argument("--run-name", type=str, default="eam_tsp100")
    parser.add_argument("--version", type=str, default=None)
    parser.add_argument("--resume", action="store_true", help="Resume from latest checkpoint.")
    parser.add_argument("--ckpt-path", type=str, default=None, help="Resume from a specific checkpoint.")
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

    def resolve_resume_checkpoint(run_name: str, version: str | None) -> tuple[str | None, str | None]:
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
        version_dirs = sorted([d for d in root.iterdir() if d.is_dir()], key=lambda p: p.stat().st_mtime)
        for vdir in reversed(version_dirs):
            last_ckpt = vdir / "last.ckpt"
            if last_ckpt.exists():
                return str(last_ckpt), vdir.name
            ckpts = sorted(vdir.rglob("*.ckpt"), key=lambda p: p.stat().st_mtime)
            if ckpts:
                return str(ckpts[-1]), vdir.name
        return None, None

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
        ckpt_path, resume_version = resolve_resume_checkpoint(args.run_name, args.version)
        if ckpt_path is None:
            raise SystemExit("No checkpoint found to resume from.")
        if args.version is None and resume_version is not None:
            version = resume_version

    logger = CSVLogger(save_dir=args.log_dir, name=args.run_name, version=version)
    checkpoint_dir = os.path.join("checkpoints", args.run_name, version)
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename="epoch_{epoch:03d}",
        save_last=True,
        save_top_k=1,
        monitor="val/reward",
        mode="max",
    )

    trainer = RL4COTrainer(
        max_epochs=args.epochs,
        accelerator="gpu",
        devices=[0],
        precision=32,
        logger=logger,
        callbacks=[checkpoint_callback],
        enable_checkpointing=True,
    )
    trainer.fit(model, ckpt_path=ckpt_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
