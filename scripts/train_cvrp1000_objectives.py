from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch
from tensordict import TensorDict


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rl4co.envs import get_env
from rl4co.models import POMO
from rl4co.models.zoo.pomo.po4cops_cvrp_policy import PO4COPsCVRPPolicy
from rl4co.utils.ops import unbatchify
from scripts.eval_downloaded_routing_checkpoints import (
    _patch_legacy_policy_object,
    checkpoint_hparams,
)


METHODS = ("po", "bopo")


def _resolve(path: str | Path) -> Path:
    value = Path(path).expanduser()
    return (value if value.is_absolute() else REPO_ROOT / value).resolve()


def _autocast(device: torch.device, precision: str):
    value = str(precision).lower()
    if value in {"32", "32-true", "fp32", "float32"}:
        return nullcontext()
    if device.type != "cuda":
        raise ValueError(f"precision={precision} requires CUDA")
    if value in {"bf16", "bf16-mixed", "bfloat16"}:
        return torch.autocast("cuda", dtype=torch.bfloat16, cache_enabled=False)
    if value in {"16", "16-mixed", "fp16", "float16"}:
        return torch.autocast("cuda", dtype=torch.float16, cache_enabled=False)
    raise ValueError(f"Unsupported precision: {precision}")


def _gradient_norm(parameters) -> float:
    total = 0.0
    for parameter in parameters:
        if parameter.grad is not None:
            total += float(parameter.grad.detach().float().square().sum().cpu())
    return math.sqrt(total)


def _dynamic_batch(
    *,
    target_size: int,
    capacity: int,
    seed: int,
    instance_index: int,
    batch_size: int,
) -> TensorDict:
    depots = []
    locs = []
    demands = []
    for offset in range(batch_size):
        generator = torch.Generator(device="cpu")
        generator.manual_seed(
            int(seed) + int(target_size) * 1_000_003 + (instance_index + offset) * 104_729
        )
        coordinates = torch.rand((target_size + 1, 2), generator=generator)
        demand = torch.randint(1, 10, (target_size,), generator=generator)
        depots.append(coordinates[0])
        locs.append(coordinates[1:])
        demands.append(demand.float() / float(capacity))
    return TensorDict(
        {
            "depot": torch.stack(depots),
            "locs": torch.stack(locs),
            "demand": torch.stack(demands),
            "capacity": torch.full((batch_size,), float(capacity)),
        },
        batch_size=[batch_size],
    )


def _load_fixed(path: Path, env) -> TensorDict:
    data = env.load_data(path)
    if len(data.batch_size) != 1:
        raise ValueError(f"Expected a rank-1 instance batch, got {data.batch_size}")
    return data


def _build_model(
    *,
    checkpoint_path: Path,
    method: str,
    target_size: int,
    capacity: int,
    num_starts: int,
    bopo_select_k: int,
    alpha: float,
    seed: int,
) -> tuple[POMO, Any, dict[str, Any]]:
    hparams = checkpoint_hparams(checkpoint_path)
    payload = hparams.pop("_checkpoint_payload")
    hparams["policy"] = _patch_legacy_policy_object(hparams.get("policy"))
    env = get_env(
        "cvrp",
        generator_params={"num_loc": target_size, "capacity": float(capacity)},
        seed=seed,
    )
    hparams.update(
        {
            "env": env,
            "num_starts": num_starts,
            "num_augment": 8,
            "loss_type": "po_loss" if method == "po" else "bopo_loss",
            "alpha": alpha,
            "po_impl": "bt",
            "bopo_pair_mode": "anchor_best",
            "bopo_select_strategy": "paper",
            "bopo_select_k": bopo_select_k,
            "generate_default_data": True,
            "batch_size": 1,
            "train_data_size": 1,
            "val_data_size": 1,
            "test_data_size": 1,
        }
    )
    model = POMO(**hparams)
    missing, unexpected = model.load_state_dict(payload["state_dict"], strict=False)
    if missing or unexpected:
        raise RuntimeError(f"Checkpoint mismatch: missing={missing}, unexpected={unexpected}")
    if not isinstance(model.policy, PO4COPsCVRPPolicy):
        raise TypeError(f"Expected PO4COPsCVRPPolicy, got {type(model.policy).__name__}")
    return model, env, payload


def _clear_policy_cache(model: POMO) -> None:
    model.policy.encoded_nodes = None
    for layer in model.policy.decoder.feature_layers:
        layer.k = None
        layer.v = None
    model.policy.decoder.logit_layer.logitk = None


def _checkpoint_payload(
    *,
    model: POMO,
    optimizer: torch.optim.Optimizer,
    optimizer_step: int,
    best_cost: float,
    best_step: int,
    config: dict[str, Any],
) -> dict[str, Any]:
    _clear_policy_cache(model)
    hyper_parameters = dict(model.hparams)
    hyper_parameters.update(
        {
            "env": model.env,
            "policy": model.policy,
            "num_starts": model.num_starts,
            "loss_type": model.loss_type,
            "po_impl": model.po_impl,
            "bopo_pair_mode": model.bopo_pair_mode,
            "bopo_select_strategy": model.bopo_select_strategy,
            "bopo_select_k": model.bopo_select_k,
        }
    )
    return {
        "state_dict": model.state_dict(),
        "hyper_parameters": hyper_parameters,
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": optimizer_step,
        "global_step": optimizer_step,
        "optimizer_step": optimizer_step,
        "best_cost": best_cost,
        "best_step": best_step,
        "method": config["method"],
        "dynamic_training_config": config,
    }


def _atomic_save(payload: dict[str, Any], path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def _evaluate(
    *,
    model: POMO,
    env,
    dataset: TensorDict,
    num_instances: int,
    num_starts: int,
    num_augment: int,
    device: torch.device,
    precision: str,
) -> dict[str, Any]:
    model.eval()
    count = min(int(num_instances), int(dataset.batch_size[0]))
    costs = []
    elapsed = []
    for index in range(count):
        raw = dataset[index : index + 1].to(device)
        td = env.reset(raw).to(device)
        if num_augment > 1:
            td = model.augment(td)
        started = time.perf_counter()
        with torch.inference_mode(), _autocast(device, precision):
            out = model.policy(
                td,
                env,
                phase="test",
                num_starts=num_starts,
                return_actions=False,
                return_entropy=False,
                return_sum_log_likelihood=True,
            )
        reward = unbatchify(out["reward"], (num_augment, num_starts))
        costs.append(float(-reward.max().detach().cpu()))
        elapsed.append(float(time.perf_counter() - started))
    values = torch.tensor(costs, dtype=torch.float64)
    return {
        "num_instances": count,
        "num_starts": num_starts,
        "num_augment": num_augment,
        "precision": precision,
        "mean_cost": float(values.mean()),
        "std_cost": float(values.std(unbiased=False)),
        "min_cost": float(values.min()),
        "max_cost": float(values.max()),
        "mean_elapsed_sec": float(sum(elapsed) / len(elapsed)),
        "per_instance_cost": costs,
    }


def _write_jsonl(path: Path, record: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=METHODS, default="po")
    parser.add_argument("--checkpoint", default="downloads/cvrp100/po/checkpoint.ckpt")
    parser.add_argument("--target-size", type=int, default=1000)
    parser.add_argument("--capacity", type=int, default=150)
    parser.add_argument("--num-starts", type=int, default=20)
    parser.add_argument("--bopo-select-k", type=int, default=10)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--train-batch-size", type=int, default=1)
    parser.add_argument("--data-start-index", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-6)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--precision", default="bf16-mixed")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--validation-file", default="data/vrp/vrp1000_val_seed4321.npz")
    parser.add_argument("--validation-size", type=int, default=32)
    parser.add_argument("--validation-every", type=int, default=100)
    parser.add_argument("--validation-starts", type=int, default=100)
    parser.add_argument("--validation-augment", type=int, default=1)
    parser.add_argument("--log-every", type=int, default=5)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--evaluate-only", action="store_true")
    parser.add_argument("--evaluation-file", default="data/vrp/vrp1000_test_seed1234.npz")
    parser.add_argument("--evaluation-instances", type=int, default=100)
    parser.add_argument("--evaluation-starts", type=int, default=100)
    parser.add_argument("--evaluation-augment", type=int, default=8)
    parser.add_argument("--evaluation-precision", default="32-true")
    parser.add_argument("--evaluation-output", default=None)
    args = parser.parse_args()

    if args.train_batch_size != 1:
        raise ValueError("Initial CVRP1000 implementation requires train-batch-size=1")
    if args.num_starts < 2 or args.num_starts > args.target_size:
        raise ValueError("num-starts must be in [2, target-size]")
    if args.method == "bopo" and args.num_starts % args.bopo_select_k != 0:
        raise ValueError("BOPO num-starts must be divisible by bopo-select-k")

    checkpoint_path = _resolve(args.checkpoint)
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
    model, env, _ = _build_model(
        checkpoint_path=checkpoint_path,
        method=args.method,
        target_size=args.target_size,
        capacity=args.capacity,
        num_starts=args.num_starts,
        bopo_select_k=args.bopo_select_k,
        alpha=args.alpha,
        seed=args.seed,
    )
    model = model.to(device)

    resume_path = _resolve(args.resume) if args.resume else None
    resume_payload = None
    if resume_path is not None:
        resume_payload = torch.load(resume_path, map_location="cpu", weights_only=False)
        missing, unexpected = model.load_state_dict(resume_payload["state_dict"], strict=False)
        if missing or unexpected:
            raise RuntimeError(f"Resume mismatch: missing={missing}, unexpected={unexpected}")

    if args.evaluate_only:
        dataset = _load_fixed(_resolve(args.evaluation_file), env)
        result = _evaluate(
            model=model,
            env=env,
            dataset=dataset,
            num_instances=args.evaluation_instances,
            num_starts=args.evaluation_starts,
            num_augment=args.evaluation_augment,
            device=device,
            precision=args.evaluation_precision,
        )
        result.update(
            {
                "method": args.method,
                "checkpoint": str(resume_path or checkpoint_path),
                "dataset": str(_resolve(args.evaluation_file)),
                "target_size": args.target_size,
                "capacity": args.capacity,
            }
        )
        output = _resolve(
            args.evaluation_output
            or f"logs/cvrp1000_baselines/eval_{args.method}_fixed100.json"
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(json.dumps(result, ensure_ascii=False), flush=True)
        return

    output_dir = _resolve(
        args.output_dir or f"logs/cvrp1000_baselines/{args.method}_from_cvrp100_1000steps"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    optimizer_step = 0
    best_cost = float("inf")
    best_step = 0
    if resume_payload is not None:
        optimizer.load_state_dict(resume_payload["optimizer_state_dict"])
        optimizer_step = int(resume_payload["optimizer_step"])
        best_cost = float(resume_payload["best_cost"])
        best_step = int(resume_payload["best_step"])
    target_step = optimizer_step + args.steps
    config = {
        "protocol": "cvrp1000_po_bopo_baseline_v1",
        "method": args.method,
        "common_initialization": str(checkpoint_path),
        "target_size": args.target_size,
        "capacity": args.capacity,
        "num_starts": args.num_starts,
        "bopo_select_k": args.bopo_select_k,
        "continuation_steps": args.steps,
        "resume_optimizer_step": optimizer_step,
        "target_optimizer_step": target_step,
        "train_batch_size": args.train_batch_size,
        "data_start_index": args.data_start_index,
        "seed": args.seed,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "alpha": args.alpha,
        "precision": args.precision,
        "validation_file": str(_resolve(args.validation_file)),
        "validation_size": args.validation_size,
        "validation_every": args.validation_every,
        "validation_starts": args.validation_starts,
        "validation_augment": args.validation_augment,
        "resume": str(resume_path) if resume_path else None,
    }
    (output_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    history_path = output_dir / "history.jsonl"
    validation = _load_fixed(_resolve(args.validation_file), env)

    def validate(step: int) -> None:
        nonlocal best_cost, best_step
        metrics = _evaluate(
            model=model,
            env=env,
            dataset=validation,
            num_instances=args.validation_size,
            num_starts=args.validation_starts,
            num_augment=args.validation_augment,
            device=device,
            precision=args.precision,
        )
        metrics.pop("per_instance_cost")
        record = {
            "event": "validation",
            "optimizer_step": step,
            "continuation_step": step - config["resume_optimizer_step"],
            **metrics,
        }
        improved = record["mean_cost"] < best_cost
        if improved:
            best_cost = record["mean_cost"]
            best_step = step
        record.update({"best_mean_cost": best_cost, "best_step": best_step})
        _write_jsonl(history_path, record)
        print(json.dumps(record), flush=True)
        payload = _checkpoint_payload(
            model=model,
            optimizer=optimizer,
            optimizer_step=step,
            best_cost=best_cost,
            best_step=best_step,
            config=config,
        )
        _atomic_save(payload, output_dir / "last.ckpt")
        if improved:
            _atomic_save(payload, output_dir / "best.ckpt")

    if optimizer_step == 0:
        validate(0)
    started = time.perf_counter()
    while optimizer_step < target_step:
        model.train()
        optimizer.zero_grad(set_to_none=True)
        instance_index = args.data_start_index + (optimizer_step - config["resume_optimizer_step"])
        batch = _dynamic_batch(
            target_size=args.target_size,
            capacity=args.capacity,
            seed=args.seed,
            instance_index=instance_index,
            batch_size=1,
        ).to(device)
        action_seed = args.seed + instance_index * 1_000_003
        torch.manual_seed(action_seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(action_seed)
            torch.cuda.reset_peak_memory_stats(device)
        step_started = time.perf_counter()
        with _autocast(device, args.precision):
            out = model.shared_step(batch, optimizer_step, "train")
            loss = out["loss"]
        loss.backward()
        grad_norm = _gradient_norm(model.parameters())
        optimizer.step()
        optimizer_step += 1
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        record = {
            "event": "train",
            "optimizer_step": optimizer_step,
            "continuation_step": optimizer_step - config["resume_optimizer_step"],
            "loss": float(loss.detach().cpu()),
            "grad_norm": grad_norm,
            "elapsed_sec": float(time.perf_counter() - step_started),
        }
        if device.type == "cuda":
            record.update(
                {
                    "peak_memory_allocated_gib": torch.cuda.max_memory_allocated(device)
                    / 1024**3,
                    "peak_memory_reserved_gib": torch.cuda.max_memory_reserved(device)
                    / 1024**3,
                }
            )
        _write_jsonl(history_path, record)
        if record["continuation_step"] % args.log_every == 0:
            print(json.dumps(record), flush=True)
        if record["continuation_step"] % args.validation_every == 0 or optimizer_step == target_step:
            validate(optimizer_step)

    summary = {
        **config,
        "completed_optimizer_step": optimizer_step,
        "best_mean_cost": best_cost,
        "best_step": best_step,
        "elapsed_sec": time.perf_counter() - started,
        "best_checkpoint": str(output_dir / "best.ckpt"),
        "last_checkpoint": str(output_dir / "last.ckpt"),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"event": "complete", **summary}), flush=True)


if __name__ == "__main__":
    main()
