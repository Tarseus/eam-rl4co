import argparse
import csv
import gc
import os
import time
from dataclasses import dataclass

import torch
from lightning.pytorch.callbacks import Callback

from rl4co.envs.routing import (
    CVRPEnv,
    CVRPGenerator,
    KnapsackEnv,
    KnapsackGenerator,
    OPEnv,
    OPGenerator,
    PCTSPEnv,
    PCTSPGenerator,
    TSPEnv,
    TSPGenerator,
)
from rl4co.models import (
    AttentionModel,
    AttentionModelPolicy,
    EAM,
    POMO,
    SymNCO,
    SymNCOPolicy,
)
from rl4co.utils import RL4COTrainer


@dataclass(frozen=True)
class TaskSpec:
    model_key: str
    problem: str
    size: int


class StepTimer(Callback):
    def __init__(self) -> None:
        self.cpu_time_s: float | None = None
        self.gpu_time_s: float | None = None
        self._cpu_start: float | None = None
        self._start_event: torch.cuda.Event | None = None
        self._end_event: torch.cuda.Event | None = None

    def on_train_epoch_start(self, trainer, pl_module) -> None:
        self.cpu_time_s = 0.0
        if pl_module.device.type == "cuda":
            self.gpu_time_s = 0.0
        else:
            self.gpu_time_s = None

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx) -> None:
        device = pl_module.device
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            with torch.cuda.device(device):
                self._start_event = torch.cuda.Event(enable_timing=True)
                self._end_event = torch.cuda.Event(enable_timing=True)
                self._start_event.record()
        self._cpu_start = time.process_time()

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx) -> None:
        if self._cpu_start is not None:
            self.cpu_time_s += time.process_time() - self._cpu_start
            self._cpu_start = None
        device = pl_module.device
        if self._start_event is not None and device.type == "cuda":
            with torch.cuda.device(device):
                self._end_event.record()
                torch.cuda.synchronize(device)
                self.gpu_time_s += self._start_event.elapsed_time(self._end_event) / 1000.0
            self._start_event = None
            self._end_event = None


def _default_metrics() -> dict:
    return {
        "train": ["loss", "reward"],
        "val": ["reward"],
        "test": ["reward"],
    }


def _build_env(problem: str, size: int):
    if problem == "tsp":
        generator = TSPGenerator(num_loc=size, loc_distribution="uniform")
        return TSPEnv(generator)
    if problem == "cvrp":
        generator = CVRPGenerator(num_loc=size, loc_distribution="uniform")
        return CVRPEnv(generator)
    if problem == "kp":
        generator = KnapsackGenerator(
            num_items=size, weight_distribution="uniform", value_distribution="uniform"
        )
        return KnapsackEnv(generator)
    if problem == "pctsp":
        generator = PCTSPGenerator(num_loc=size, loc_distribution="uniform")
        return PCTSPEnv(generator)
    if problem == "op":
        generator = OPGenerator(num_loc=size, loc_distribution="uniform")
        return OPEnv(generator)
    raise ValueError(f"Unsupported problem: {problem}")


def _build_am_policy(env, args):
    return AttentionModelPolicy(
        env_name=env.name,
        embed_dim=args.embed_dim,
        num_encoder_layers=args.num_encoder_layers,
        num_heads=args.num_heads,
        normalization=args.normalization,
        use_graph_context=args.use_graph_context,
    )


def _build_symnco_policy(env, args):
    return SymNCOPolicy(
        env_name=env.name,
        embed_dim=args.embed_dim,
        num_encoder_layers=args.num_encoder_layers,
        num_heads=args.num_heads,
        normalization=args.normalization,
        use_graph_context=args.use_graph_context,
    )


def _build_model(spec: TaskSpec, args):
    env = _build_env(spec.problem, spec.size)
    common_kwargs = {
        "batch_size": args.batch_size,
        "optimizer_kwargs": {"lr": args.lr, "weight_decay": args.weight_decay},
        "train_data_size": args.train_data_size,
        "val_data_size": args.val_data_size,
        "test_data_size": args.test_data_size,
        "metrics": _default_metrics(),
    }
    if spec.model_key == "am":
        policy = _build_am_policy(env, args)
        return AttentionModel(env, policy, baseline="rollout", **common_kwargs)
    if spec.model_key == "pomo":
        policy = _build_am_policy(env, args)
        return POMO(env, policy, num_augment=args.pomo_num_augment, **common_kwargs)
    if spec.model_key == "symnco":
        policy = _build_symnco_policy(env, args)
        return SymNCO(
            env,
            policy,
            baseline="symnco",
            num_augment=args.symnco_num_augment,
            num_starts=None,
            **common_kwargs,
        )
    if spec.model_key == "eam_pomo":
        policy = _build_am_policy(env, args)
        ea_kwargs = {
            "num_generations": args.ea_num_generations,
            "mutation_rate": args.ea_mutation_rate,
            "crossover_rate": args.ea_crossover_rate,
            "selection_rate": args.ea_selection_rate,
            "batch_size": args.batch_size,
            "ea_batch_size": args.batch_size,
            "alpha": args.ea_alpha,
            "beta": args.ea_beta,
            "ea_prob": args.ea_prob,
            "ea_epoch": args.ea_epoch,
        }
        return EAM(
            env,
            policy,
            baseline="shared",
            num_augment=args.pomo_num_augment,
            ea_kwargs=ea_kwargs,
            **common_kwargs,
        )
    raise ValueError(f"Unsupported model key: {spec.model_key}")


def _build_trainer(args, timer: StepTimer):
    return RL4COTrainer(
        max_epochs=1,
        accelerator="gpu",
        devices=[args.cuda],
        precision=args.precision,
        logger=False,
        enable_checkpointing=False,
        enable_model_summary=False,
        enable_progress_bar=False,
        num_sanity_val_steps=0,
        limit_val_batches=0,
        callbacks=[timer],
    )


def _format_problem(problem: str, size: int) -> str:
    return f"{problem.upper()}{size}"


def _model_label(model_key: str) -> str:
    labels = {
        "am": "AttentionModel",
        "pomo": "POMO",
        "symnco": "SymNCO",
        "eam_pomo": "EAM-POMO",
    }
    return labels.get(model_key, model_key)


def _append_result(path: str, row: dict) -> None:
    dir_name = os.path.dirname(path)
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)
    write_header = not os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def _task_list() -> list[TaskSpec]:
    tasks = []
    for problem, size in [
        ("tsp", 50),
        ("tsp", 100),
        ("cvrp", 50),
        ("cvrp", 100),
        ("kp", 50),
        ("kp", 100),
        ("pctsp", 100),
        ("op", 100),
    ]:
        tasks.append(TaskSpec("am", problem, size))
    for problem, size in [
        ("tsp", 50),
        ("tsp", 100),
        ("cvrp", 50),
        ("cvrp", 100),
        ("kp", 50),
        ("kp", 100),
    ]:
        tasks.append(TaskSpec("pomo", problem, size))
    for problem, size in [
        ("tsp", 50),
        ("tsp", 100),
        ("cvrp", 50),
        ("cvrp", 100),
    ]:
        tasks.append(TaskSpec("symnco", problem, size))
    tasks.append(TaskSpec("eam_pomo", "tsp", 100))
    return tasks


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Measure CPU/GPU compute time for one training epoch across models."
    )
    parser.add_argument("--cuda", type=int, default=0, help="Physical GPU id")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--train-data-size", type=int, default=160_000)
    parser.add_argument("--val-data-size", type=int, default=10_000)
    parser.add_argument("--test-data-size", type=int, default=10_000)
    parser.add_argument("--precision", type=str, default="32")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-6)
    parser.add_argument("--embed-dim", type=int, default=128)
    parser.add_argument("--num-encoder-layers", type=int, default=6)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--normalization", type=str, default="instance")
    parser.add_argument("--use-graph-context", action="store_true", default=False)
    parser.add_argument("--pomo-num-augment", type=int, default=8)
    parser.add_argument("--symnco-num-augment", type=int, default=4)
    parser.add_argument("--ea-num-generations", type=int, default=5)
    parser.add_argument("--ea-mutation-rate", type=float, default=0.05)
    parser.add_argument("--ea-crossover-rate", type=float, default=0.6)
    parser.add_argument("--ea-selection-rate", type=float, default=0.2)
    parser.add_argument("--ea-alpha", type=float, default=0.5)
    parser.add_argument("--ea-beta", type=float, default=3.0)
    parser.add_argument("--ea-prob", type=float, default=0.01)
    parser.add_argument("--ea-epoch", type=int, default=700)
    parser.add_argument(
        "--output",
        type=str,
        default=os.path.join("results", "epoch_timing.csv"),
    )
    args = parser.parse_args()

    os.environ.setdefault("WANDB_MODE", "offline")
    if args.precision.isdigit():
        args.precision = int(args.precision)

    results = []
    for spec in _task_list():
        model_name = _model_label(spec.model_key)
        problem_name = _format_problem(spec.problem, spec.size)
        print(f"Running {model_name} on {problem_name}...")

        timer = StepTimer()
        model = _build_model(spec, args)
        trainer = _build_trainer(args, timer)
        trainer.fit(model)

        row = {
            "model": model_name,
            "problem": problem_name,
            "cpu_time_s": timer.cpu_time_s,
            "gpu_time_s": timer.gpu_time_s,
            "batch_size": args.batch_size,
            "train_data_size": args.train_data_size,
            "precision": args.precision,
        }
        results.append(row)
        _append_result(args.output, row)

        print(
            f"  CPU {timer.cpu_time_s:.3f}s | GPU {timer.gpu_time_s:.3f}s"
            if timer.cpu_time_s is not None and timer.gpu_time_s is not None
            else f"  CPU {timer.cpu_time_s}s | GPU {timer.gpu_time_s}s"
        )

        del model, trainer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print(f"Saved results to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
