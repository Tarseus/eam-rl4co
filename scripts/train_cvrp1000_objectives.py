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


METHODS = ("po", "bopo", "usw", "asw")
DEFAULT_PAIR_PATHS = {
    "usw": REPO_ROOT
    / "runs"
    / "pref_loss_cvrp100_from_tsp100_elite"
    / "20260320-224008"
    / "best_pair.json",
    "asw": REPO_ROOT
    / "runs"
    / "pref_builder_weight_search_cvrp100"
    / "20260403-132739"
    / "best_pair.json",
}


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
    usw_pair_path: Path,
    asw_pair_path: Path,
    require_po4cops_policy: bool = True,
) -> tuple[POMO, Any, dict[str, Any]]:
    hparams = checkpoint_hparams(checkpoint_path)
    payload = hparams.pop("_checkpoint_payload")
    hparams["policy"] = _patch_legacy_policy_object(hparams.get("policy"))
    env = get_env(
        "cvrp",
        generator_params={"num_loc": target_size, "capacity": float(capacity)},
        seed=seed,
    )
    loss_type = {
        "po": "po_loss",
        "bopo": "bopo_loss",
        "usw": "free_loss",
        "asw": "free_loss",
    }[method]
    hparams.update(
        {
            "env": env,
            "num_starts": num_starts,
            "num_augment": 8,
            "loss_type": loss_type,
            "alpha": alpha,
            "po_impl": "exponential",
            "bopo_pair_mode": "anchor_best",
            "bopo_select_strategy": "paper",
            "bopo_select_k": bopo_select_k,
            "free_loss_ir_json_path": None,
            "pref_builder_ir_json_path": None,
            "pref_pair_json_path": None,
            "memory_efficient_preference": True,
            "memory_efficient_checkpoint_encoder": True,
            "memory_efficient_checkpoint_decoder": True,
            "memory_efficient_verify_replay": True,
            "generate_default_data": True,
            "batch_size": 1,
            "train_data_size": 1,
            "val_data_size": 1,
            "test_data_size": 1,
        }
    )
    if method in {"usw", "asw"}:
        hparams["pref_pair_json_path"] = str(
            usw_pair_path if method == "usw" else asw_pair_path
        )
    model = POMO(**hparams)
    missing, unexpected = model.load_state_dict(payload["state_dict"], strict=False)
    if missing or unexpected:
        raise RuntimeError(f"Checkpoint mismatch: missing={missing}, unexpected={unexpected}")
    if require_po4cops_policy and not isinstance(model.policy, PO4COPsCVRPPolicy):
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
            "pref_pair_json_path": model.pref_pair_json_path,
            "preference_po_anchor_weight": float(
                getattr(model, "preference_po_anchor_weight", 0.0)
            ),
            "preference_po_anchor_alpha": float(
                getattr(model, "preference_po_anchor_alpha", 0.05)
            ),
            "memory_efficient_preference": True,
            "memory_efficient_checkpoint_encoder": True,
            "memory_efficient_checkpoint_decoder": True,
            "memory_efficient_verify_replay": True,
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
    parser.add_argument("--capacity", type=int, default=50)
    parser.add_argument("--num-starts", type=int, default=20)
    parser.add_argument("--bopo-select-k", type=int, default=10)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--accumulate", type=int, default=1)
    parser.add_argument("--train-batch-size", type=int, default=1)
    parser.add_argument("--data-start-index", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-6)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--po-anchor-weight", type=float, default=0.0)
    parser.add_argument("--po-anchor-alpha", type=float, default=0.05)
    parser.add_argument(
        "--detach-pref-weights",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--precision", default="bf16-mixed")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--validation-file", default="data/vrp/vrp1000_val_seed4321.npz")
    parser.add_argument("--validation-size", type=int, default=32)
    parser.add_argument("--validation-every", type=int, default=100)
    parser.add_argument("--validation-starts", type=int, default=100)
    parser.add_argument("--validation-augment", type=int, default=1)
    parser.add_argument("--log-every", type=int, default=5)
    parser.add_argument(
        "--usw-pair-path",
        default=str(DEFAULT_PAIR_PATHS["usw"].relative_to(REPO_ROOT)),
    )
    parser.add_argument(
        "--asw-pair-path",
        default=str(DEFAULT_PAIR_PATHS["asw"].relative_to(REPO_ROOT)),
    )
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--resume", default=None)
    parser.add_argument(
        "--restore-checkpoint-optimizer",
        action="store_true",
        help="Restore optimizer/global step from the initial Lightning checkpoint.",
    )
    parser.add_argument("--evaluate-only", action="store_true")
    parser.add_argument("--evaluation-file", default="data/vrp/vrp1000_test_seed1234.npz")
    parser.add_argument("--evaluation-instances", type=int, default=100)
    parser.add_argument("--evaluation-starts", type=int, default=100)
    parser.add_argument("--evaluation-augment", type=int, default=8)
    parser.add_argument("--evaluation-precision", default="32-true")
    parser.add_argument("--evaluation-output", default=None)
    args = parser.parse_args()

    if args.steps < 1 or args.accumulate < 1 or args.train_batch_size < 1:
        raise ValueError("steps, accumulate, and train-batch-size must be >= 1")
    if args.data_start_index < 0:
        raise ValueError("data-start-index must be >= 0")
    if args.capacity <= 0:
        raise ValueError("capacity must be > 0")
    if not 0.0 <= args.po_anchor_weight <= 1.0:
        raise ValueError("po-anchor-weight must be in [0, 1]")
    if args.po_anchor_alpha <= 0.0:
        raise ValueError("po-anchor-alpha must be > 0")
    if args.num_starts < 2 or args.num_starts > args.target_size:
        raise ValueError("num-starts must be in [2, target-size]")
    if args.method == "bopo" and args.num_starts % args.bopo_select_k != 0:
        raise ValueError("BOPO num-starts must be divisible by bopo-select-k")
    if args.validation_augment not in {1, 8} or args.evaluation_augment not in {1, 8}:
        raise ValueError("validation-augment and evaluation-augment must be 1 or 8")

    checkpoint_path = _resolve(args.checkpoint)
    usw_pair_path = _resolve(args.usw_pair_path)
    asw_pair_path = _resolve(args.asw_pair_path)
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
    model, env, initial_payload = _build_model(
        checkpoint_path=checkpoint_path,
        method=args.method,
        target_size=args.target_size,
        capacity=args.capacity,
        num_starts=args.num_starts,
        bopo_select_k=args.bopo_select_k,
        alpha=args.alpha,
        seed=args.seed,
        usw_pair_path=usw_pair_path,
        asw_pair_path=asw_pair_path,
    )
    model.bopo_select_k = int(args.bopo_select_k)
    model.alpha = float(args.alpha)
    model.detach_pref_weights = bool(args.detach_pref_weights)
    model.preference_po_anchor_weight = float(args.po_anchor_weight)
    model.preference_po_anchor_alpha = float(args.po_anchor_alpha)
    model = model.to(device)

    resume_path = _resolve(args.resume) if args.resume else None
    if resume_path is not None and args.restore_checkpoint_optimizer:
        raise ValueError("Use either --resume or --restore-checkpoint-optimizer, not both")
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
    elif args.restore_checkpoint_optimizer:
        optimizer_states = initial_payload.get("optimizer_states") or []
        if len(optimizer_states) != 1:
            raise ValueError(
                "Expected exactly one optimizer state in the initial checkpoint, "
                f"got {len(optimizer_states)}"
            )
        optimizer.load_state_dict(optimizer_states[0])
        optimizer_step = int(initial_payload.get("global_step", 0))
    target_step = optimizer_step + args.steps
    config = {
        "protocol": "cvrp1000_tsp_success_route_v3_batched",
        "method": args.method,
        "common_initialization": str(checkpoint_path),
        "target_size": args.target_size,
        "capacity": args.capacity,
        "num_starts": args.num_starts,
        "bopo_select_k": args.bopo_select_k,
        "continuation_steps": args.steps,
        "resume_optimizer_step": optimizer_step,
        "target_optimizer_step": target_step,
        "accumulate": args.accumulate,
        "train_batch_size": args.train_batch_size,
        "effective_batch_size": args.accumulate * args.train_batch_size,
        "data_start_index": args.data_start_index,
        "seed": args.seed,
        "learning_rate": float(optimizer.param_groups[0]["lr"]),
        "weight_decay": float(optimizer.param_groups[0]["weight_decay"]),
        "alpha": args.alpha,
        "po_impl": "exponential",
        "po_anchor_weight": args.po_anchor_weight,
        "po_anchor_alpha": args.po_anchor_alpha,
        "detach_pref_weights": bool(args.detach_pref_weights),
        "precision": args.precision,
        "validation_file": str(_resolve(args.validation_file)),
        "validation_size": args.validation_size,
        "validation_every": args.validation_every,
        "validation_starts": args.validation_starts,
        "validation_augment": args.validation_augment,
        "resume": str(resume_path) if resume_path else None,
        "restored_initial_optimizer": bool(args.restore_checkpoint_optimizer),
        "fresh_optimizer": not bool(args.restore_checkpoint_optimizer or resume_path),
        "memory_efficient_forced_replay": True,
        "usw_pair_path": str(usw_pair_path),
        "asw_pair_path": str(asw_pair_path),
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

    if resume_payload is None:
        validate(optimizer_step)
    started = time.perf_counter()
    while optimizer_step < target_step:
        model.train()
        optimizer.zero_grad(set_to_none=True)
        micro_losses = []
        replay_errors = []
        coefficient_means = []
        step_started = time.perf_counter()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        continuation_step = optimizer_step - config["resume_optimizer_step"]
        for accumulation_index in range(args.accumulate):
            micro_batch_index = continuation_step * args.accumulate + accumulation_index
            instance_index = (
                args.data_start_index + micro_batch_index * args.train_batch_size
            )
            batch = _dynamic_batch(
                target_size=args.target_size,
                capacity=args.capacity,
                seed=args.seed,
                instance_index=instance_index,
                batch_size=args.train_batch_size,
            ).to(device)
            action_seed = args.seed + instance_index * 1_000_003
            torch.manual_seed(action_seed)
            if device.type == "cuda":
                torch.cuda.manual_seed_all(action_seed)
            with _autocast(device, args.precision):
                # The replay-based helper is currently implemented only for
                # PO4COPsTSPPolicy. CVRP uses the ordinary POMO preference
                # forward, which is still small enough for the short
                # continuation protocol used here.
                if model.policy.__class__.__name__ == "PO4COPsTSPPolicy":
                    td = env.reset(batch).to(device)
                    out = model._memory_efficient_preference_step(
                        td=td,
                        batch=batch,
                        n_start=args.num_starts,
                        dataloader_idx=None,
                        log_metrics=False,
                    )
                else:
                    previous_memory_efficient = model.memory_efficient_preference
                    model.memory_efficient_preference = False
                    try:
                        out = model.shared_step(
                            batch=batch,
                            batch_idx=0,
                            phase="train",
                            dataloader_idx=None,
                        )
                    finally:
                        model.memory_efficient_preference = previous_memory_efficient
                    zero = out["loss"].detach().new_zeros(())
                    out["memory_efficient_replay_error"] = zero
                    out["memory_efficient_coefficient_abs_mean"] = zero
                scaled_loss = out["loss"] / float(args.accumulate)
            scaled_loss.backward()
            micro_losses.append(float(out["loss"].detach().cpu()))
            replay_errors.append(
                float(out["memory_efficient_replay_error"].detach().cpu())
            )
            coefficient_means.append(
                float(
                    out["memory_efficient_coefficient_abs_mean"].detach().cpu()
                )
            )
        grad_norm = _gradient_norm(model.parameters())
        optimizer.step()
        optimizer_step += 1
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        record = {
            "event": "train",
            "optimizer_step": optimizer_step,
            "continuation_step": optimizer_step - config["resume_optimizer_step"],
            "train_batch_size": args.train_batch_size,
            "effective_batch_size": args.accumulate * args.train_batch_size,
            "instances_processed_this_run": (
                (optimizer_step - config["resume_optimizer_step"])
                * args.accumulate
                * args.train_batch_size
            ),
            "loss": float(sum(micro_losses) / len(micro_losses)),
            "grad_norm": grad_norm,
            "replay_error": float(max(replay_errors)),
            "coefficient_abs_mean": float(
                sum(coefficient_means) / len(coefficient_means)
            ),
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
        "instances_processed_this_run": (
            (optimizer_step - config["resume_optimizer_step"])
            * args.accumulate
            * args.train_batch_size
        ),
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
