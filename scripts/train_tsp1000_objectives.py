from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

import torch
from tensordict import TensorDict


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rl4co.utils.ops import unbatchify
from scripts.probe_tsp1000_dynamic_training import (
    DEFAULT_PAIR_PATHS,
    METHODS,
    _autocast_context,
    _build_model,
    _gradient_norm,
    _resolve_path,
)


def _dynamic_locs(
    *,
    target_size: int,
    seed: int,
    instance_index: int,
    batch_size: int = 1,
) -> torch.Tensor:
    instances = []
    for batch_offset in range(int(batch_size)):
        generator = torch.Generator(device="cpu")
        generator.manual_seed(
            int(seed)
            + int(target_size) * 1_000_003
            + (int(instance_index) + batch_offset) * 104_729
        )
        instances.append(
            torch.rand(
                (int(target_size), 2),
                generator=generator,
                dtype=torch.float32,
            )
        )
    return torch.stack(instances, dim=0)


def _validation_instances(
    *,
    target_size: int,
    seed: int,
    count: int,
) -> list[torch.Tensor]:
    return [
        _dynamic_locs(
            target_size=target_size,
            seed=int(seed) + 91_000_003,
            instance_index=index,
        )
        for index in range(int(count))
    ]


def _clear_policy_cache(model) -> None:
    model.policy.encoded_nodes = None
    for layer in model.policy.decoder.layers:
        layer.k = None
        layer.v = None
        layer.logitk = None
        layer.q_first = None
        layer.first = False


def _checkpoint_payload(
    *,
    model,
    optimizer,
    method: str,
    optimizer_step: int,
    best_tour_length: float,
    best_step: int,
    config: dict[str, Any],
) -> dict[str, Any]:
    _clear_policy_cache(model)
    hyper_parameters = dict(model.hparams)
    hyper_parameters.update(
        {
            "env": model.env,
            "policy": model.policy,
            "num_starts": int(config["num_starts"]),
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
        }
    )
    return {
        "state_dict": model.state_dict(),
        "hyper_parameters": hyper_parameters,
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": int(optimizer_step),
        "global_step": int(optimizer_step),
        "method": str(method),
        "optimizer_step": int(optimizer_step),
        "best_tour_length": float(best_tour_length),
        "best_step": int(best_step),
        "dynamic_training_config": config,
    }


def _atomic_torch_save(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary_path)
    os.replace(temporary_path, path)


def _validate(
    *,
    model,
    env,
    instances: list[torch.Tensor],
    num_starts: int,
    num_augment: int,
    device: torch.device,
    precision: str,
    eval_start_node: str,
) -> dict[str, float]:
    model.eval()
    previous_start_node = model.policy.start_node
    previous_eval_type = model.policy.eval_type
    model.policy.start_node = str(eval_start_node)
    model.policy.eval_type = "argmax"
    tour_lengths = []
    started_at = time.perf_counter()
    try:
        for locs in instances:
            batch = TensorDict({"locs": locs.to(device)}, batch_size=[1]).to(device)
            td = env.reset(batch).to(device)
            if int(num_augment) > 1:
                td = model.augment(td)
            with torch.inference_mode(), _autocast_context(device, precision):
                out = model.policy(
                    td,
                    env,
                    phase="test",
                    num_starts=int(num_starts),
                    return_actions=False,
                    return_entropy=False,
                    return_sum_log_likelihood=True,
                )
            reward = unbatchify(
                out["reward"],
                (int(num_augment), int(num_starts)),
            )
            tour_lengths.append(float(-reward.max().detach().cpu()))
    finally:
        model.policy.start_node = previous_start_node
        model.policy.eval_type = previous_eval_type
    values = torch.tensor(tour_lengths, dtype=torch.float64)
    return {
        "mean_tour_length": float(values.mean()),
        "std_tour_length": float(values.std(unbiased=False)),
        "min_tour_length": float(values.min()),
        "max_tour_length": float(values.max()),
        "elapsed_sec": float(time.perf_counter() - started_at),
    }


def _write_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", required=True, choices=METHODS)
    parser.add_argument("--checkpoint", default="downloads/tsp100/po/checkpoint.ckpt")
    parser.add_argument("--target-size", type=int, default=1000)
    parser.add_argument("--num-starts", type=int, default=20)
    parser.add_argument("--bopo-select-k", type=int, default=10)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--additional-steps", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--instances-per-epoch", type=int, default=100_000)
    parser.add_argument("--accumulate", type=int, default=1)
    parser.add_argument("--train-batch-size", type=int, default=1)
    parser.add_argument("--data-start-index", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-6)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--logit-clipping", type=float, default=None)
    parser.add_argument(
        "--train-start-node",
        choices=("same", "random", "pomo"),
        default="pomo",
    )
    parser.add_argument(
        "--train-decode-type",
        choices=("sampling", "hybrid"),
        default="sampling",
    )
    parser.add_argument(
        "--eval-start-node",
        choices=("same", "random", "pomo"),
        default="pomo",
    )
    parser.add_argument("--po-anchor-weight", type=float, default=0.0)
    parser.add_argument("--po-anchor-alpha", type=float, default=0.05)
    parser.add_argument(
        "--detach-pref-weights",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--precision", default="bf16-mixed")
    parser.add_argument(
        "--memory-efficient",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--validation-size", type=int, default=8)
    parser.add_argument("--validation-every", type=int, default=25)
    parser.add_argument("--validation-every-epochs", type=int, default=None)
    parser.add_argument("--validation-starts", type=int, default=1000)
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
    args = parser.parse_args()

    if args.steps < 1 or args.accumulate < 1 or args.train_batch_size < 1:
        raise ValueError("steps, accumulate, and train-batch-size must be >= 1")
    if args.additional_steps is not None and args.additional_steps < 1:
        raise ValueError("additional-steps must be >= 1")
    if args.epochs is not None and args.epochs < 1:
        raise ValueError("epochs must be >= 1")
    if args.additional_steps is not None and args.epochs is not None:
        raise ValueError("additional-steps and epochs are mutually exclusive")
    if args.instances_per_epoch < 1:
        raise ValueError("instances-per-epoch must be >= 1")
    if args.validation_every_epochs is not None and args.validation_every_epochs < 1:
        raise ValueError("validation-every-epochs must be >= 1")
    if args.data_start_index < 0:
        raise ValueError("data-start-index must be >= 0")
    if not 0.0 <= args.po_anchor_weight <= 1.0:
        raise ValueError("po-anchor-weight must be in [0, 1]")
    if args.po_anchor_alpha <= 0.0:
        raise ValueError("po-anchor-alpha must be > 0")
    if args.num_starts % args.bopo_select_k != 0:
        raise ValueError("num-starts must be divisible by bopo-select-k")
    if args.validation_augment not in {1, 8}:
        raise ValueError("validation-augment must be 1 or 8")
    checkpoint_path = _resolve_path(args.checkpoint)
    usw_pair_path = _resolve_path(args.usw_pair_path)
    asw_pair_path = _resolve_path(args.asw_pair_path)
    device = torch.device(args.device)
    torch.manual_seed(int(args.seed))
    if device.type == "cuda":
        torch.cuda.manual_seed_all(int(args.seed))
    output_dir = _resolve_path(
        args.output_dir
        or f"logs/tsp1000_dynamic/train_{args.method}_seed{args.seed}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    model, env = _build_model(
        method=args.method,
        checkpoint_path=checkpoint_path,
        target_size=args.target_size,
        num_starts=args.num_starts,
        seed=args.seed,
        usw_pair_path=usw_pair_path,
        asw_pair_path=asw_pair_path,
    )
    model.bopo_select_k = int(args.bopo_select_k)
    model.alpha = float(args.alpha)
    model.policy.start_node = str(args.train_start_node)
    model.policy.train_decode_type = str(args.train_decode_type)
    # Keep hybrid behavior scoped to the training decode type. Validation and
    # test use greedy decoding with their own start-node policy.
    model.policy.eval_type = "argmax"
    if args.logit_clipping is not None:
        for decoder_layer in model.policy.decoder.layers:
            decoder_layer.logit_clipping = float(args.logit_clipping)
    model.detach_pref_weights = bool(args.detach_pref_weights)
    model.preference_po_anchor_weight = float(args.po_anchor_weight)
    model.preference_po_anchor_alpha = float(args.po_anchor_alpha)
    model.memory_efficient_preference = bool(args.memory_efficient)
    model = model.to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(args.learning_rate),
        weight_decay=float(args.weight_decay),
    )
    optimizer_step = 0
    best_tour_length = math.inf
    best_step = 0
    resume_path = None
    if args.resume:
        resume_path = _resolve_path(args.resume)
        resume_payload = torch.load(resume_path, map_location="cpu", weights_only=False)
        if resume_payload.get("method") != args.method:
            raise ValueError(
                f"Resume method={resume_payload.get('method')} does not match {args.method}"
            )
        model.load_state_dict(resume_payload["state_dict"], strict=True)
        optimizer.load_state_dict(resume_payload["optimizer_state_dict"])
        optimizer_step = int(resume_payload["optimizer_step"])
        best_tour_length = float(resume_payload["best_tour_length"])
        best_step = int(resume_payload["best_step"])
    resume_optimizer_step = int(optimizer_step)
    epoch_steps = math.ceil(
        args.instances_per_epoch / (args.train_batch_size * args.accumulate)
    )
    requested_additional_steps = (
        int(args.epochs) * epoch_steps
        if args.epochs is not None
        else args.additional_steps
    )
    target_optimizer_step = (
        resume_optimizer_step + int(requested_additional_steps)
        if requested_additional_steps is not None
        else int(args.steps)
    )
    if target_optimizer_step <= resume_optimizer_step:
        raise ValueError(
            f"Target optimizer step {target_optimizer_step} must be greater than "
            f"resume step {resume_optimizer_step}"
        )
    validation_every_steps = (
        int(args.validation_every_epochs) * epoch_steps
        if args.validation_every_epochs is not None
        else int(args.validation_every)
    )

    config = {
        "protocol": "tsp1000_dynamic_finetune_v2_batched",
        "method": args.method,
        "common_initialization": str(checkpoint_path),
        "target_size": int(args.target_size),
        "num_starts": int(args.num_starts),
        "bopo_select_k": int(args.bopo_select_k),
        "steps": int(target_optimizer_step),
        "additional_steps": int(target_optimizer_step - resume_optimizer_step),
        "requested_epochs": int(args.epochs) if args.epochs is not None else None,
        "instances_per_epoch": int(args.instances_per_epoch),
        "updates_per_epoch": int(epoch_steps),
        "accumulate": int(args.accumulate),
        "train_batch_size": int(args.train_batch_size),
        "data_start_index": int(args.data_start_index),
        "seed": int(args.seed),
        "learning_rate": float(args.learning_rate),
        "weight_decay": float(args.weight_decay),
        "alpha": float(args.alpha),
        "logit_clipping": (
            float(args.logit_clipping) if args.logit_clipping is not None else None
        ),
        "train_start_node": str(args.train_start_node),
        "train_decode_type": str(args.train_decode_type),
        "eval_start_node": str(args.eval_start_node),
        "po_anchor_weight": float(args.po_anchor_weight),
        "po_anchor_alpha": float(args.po_anchor_alpha),
        "detach_pref_weights": bool(args.detach_pref_weights),
        "precision": str(args.precision),
        "memory_efficient": bool(args.memory_efficient),
        "validation_size": int(args.validation_size),
        "validation_every": int(args.validation_every),
        "validation_every_epochs": (
            int(args.validation_every_epochs)
            if args.validation_every_epochs is not None
            else None
        ),
        "validation_every_steps": int(validation_every_steps),
        "validation_starts": int(args.validation_starts),
        "validation_augment": int(args.validation_augment),
        "dynamic_instances": True,
        "resume_path": str(resume_path) if resume_path is not None else None,
        "resume_optimizer_step": int(resume_optimizer_step),
        "usw_pair_path": str(usw_pair_path),
        "asw_pair_path": str(asw_pair_path),
    }
    (output_dir / "config.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    history_path = output_dir / "history.jsonl"
    if resume_path is not None:
        resume_record = {
            "event": "resume",
            "optimizer_step": int(optimizer_step),
            "best_mean_tour_length": float(best_tour_length),
            "best_step": int(best_step),
            "resume_path": str(resume_path),
        }
        if not history_path.exists():
            _write_jsonl(history_path, resume_record)
        resume_payload = _checkpoint_payload(
            model=model,
            optimizer=optimizer,
            method=args.method,
            optimizer_step=optimizer_step,
            best_tour_length=best_tour_length,
            best_step=best_step,
            config=config,
        )
        for checkpoint_name in ("best.ckpt", "last.ckpt"):
            checkpoint_file = output_dir / checkpoint_name
            if not checkpoint_file.exists():
                _atomic_torch_save(resume_payload, checkpoint_file)
    validation_instances = _validation_instances(
        target_size=args.target_size,
        seed=args.seed,
        count=args.validation_size,
    )

    def run_validation(step: int) -> None:
        nonlocal best_tour_length, best_step
        metrics = _validate(
            model=model,
            env=env,
            instances=validation_instances,
            num_starts=args.validation_starts,
            num_augment=args.validation_augment,
            device=device,
            precision=args.precision,
            eval_start_node=args.eval_start_node,
        )
        record = {
            "event": "validation",
            "optimizer_step": int(step),
            "continuation_step": int(step - resume_optimizer_step),
            "virtual_epoch": float(
                (step - resume_optimizer_step)
                * args.accumulate
                * args.train_batch_size
                / args.instances_per_epoch
            ),
            **metrics,
        }
        improved = metrics["mean_tour_length"] < best_tour_length
        if improved:
            best_tour_length = metrics["mean_tour_length"]
            best_step = int(step)
        record["best_mean_tour_length"] = float(best_tour_length)
        record["best_step"] = int(best_step)
        _write_jsonl(history_path, record)
        print(json.dumps(record, ensure_ascii=False), flush=True)
        payload = _checkpoint_payload(
            model=model,
            optimizer=optimizer,
            method=args.method,
            optimizer_step=step,
            best_tour_length=best_tour_length,
            best_step=best_step,
            config=config,
        )
        _atomic_torch_save(payload, output_dir / "last.ckpt")
        if improved:
            _atomic_torch_save(payload, output_dir / "best.ckpt")

    if optimizer_step == 0:
        run_validation(0)

    train_started_at = time.perf_counter()
    while optimizer_step < target_optimizer_step:
        model.train()
        optimizer.zero_grad(set_to_none=True)
        micro_losses = []
        replay_errors = []
        coefficient_means = []
        step_started_at = time.perf_counter()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        for accumulation_index in range(args.accumulate):
            continuation_step = optimizer_step - resume_optimizer_step
            micro_batch_index = continuation_step * args.accumulate + accumulation_index
            instance_index = (
                args.data_start_index
                + micro_batch_index * args.train_batch_size
            )
            locs = _dynamic_locs(
                target_size=args.target_size,
                seed=args.seed,
                instance_index=instance_index,
                batch_size=args.train_batch_size,
            )
            batch = TensorDict(
                {"locs": locs.to(device)},
                batch_size=[args.train_batch_size],
            ).to(device)
            td = env.reset(batch).to(device)
            action_seed = (
                int(args.seed)
                + (int(args.data_start_index) + int(micro_batch_index)) * 1_000_003
            )
            torch.manual_seed(action_seed)
            if device.type == "cuda":
                torch.cuda.manual_seed_all(action_seed)
            with _autocast_context(device, args.precision):
                if args.memory_efficient:
                    step_out = model._memory_efficient_preference_step(
                        td=td,
                        batch=batch,
                        n_start=args.num_starts,
                        dataloader_idx=None,
                        log_metrics=False,
                    )
                else:
                    step_out = model.shared_step(batch, 0, "train")
                scaled_loss = step_out["loss"] / float(args.accumulate)
            scaled_loss.backward()
            micro_losses.append(float(step_out["loss"].detach().cpu()))
            if args.memory_efficient:
                replay_errors.append(
                    float(step_out["memory_efficient_replay_error"].detach().cpu())
                )
                coefficient_means.append(
                    float(
                        step_out["memory_efficient_coefficient_abs_mean"].detach().cpu()
                    )
                )
        grad_norm = _gradient_norm(model.parameters())
        optimizer.step()
        optimizer_step += 1
        continuation_steps_completed = optimizer_step - resume_optimizer_step
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        train_record = {
            "event": "train",
            "optimizer_step": int(optimizer_step),
            "continuation_step": int(continuation_steps_completed),
            "virtual_epoch": float(
                continuation_steps_completed
                * args.accumulate
                * args.train_batch_size
                / args.instances_per_epoch
            ),
            "train_batch_size": int(args.train_batch_size),
            "instances_processed_this_run": int(
                (optimizer_step - config["resume_optimizer_step"])
                * args.accumulate
                * args.train_batch_size
            ),
            "loss": float(sum(micro_losses) / len(micro_losses)),
            "grad_norm": float(grad_norm),
            "replay_error": (
                float(max(replay_errors)) if replay_errors else None
            ),
            "coefficient_abs_mean": (
                float(sum(coefficient_means) / len(coefficient_means))
                if coefficient_means
                else None
            ),
            "elapsed_sec": float(time.perf_counter() - step_started_at),
        }
        if device.type == "cuda":
            train_record.update(
                {
                    "peak_memory_allocated_gib": float(
                        torch.cuda.max_memory_allocated(device) / (1024**3)
                    ),
                    "peak_memory_reserved_gib": float(
                        torch.cuda.max_memory_reserved(device) / (1024**3)
                    ),
                }
            )
        _write_jsonl(history_path, train_record)
        if continuation_steps_completed % args.log_every == 0:
            print(json.dumps(train_record, ensure_ascii=False), flush=True)
        if (
            continuation_steps_completed % validation_every_steps == 0
            or optimizer_step == target_optimizer_step
        ):
            run_validation(optimizer_step)

    summary = {
        **config,
        "completed_steps": int(optimizer_step),
        "instances_processed_this_run": int(
            (optimizer_step - config["resume_optimizer_step"])
            * args.accumulate
            * args.train_batch_size
        ),
        "virtual_epochs_completed": float(
            (optimizer_step - config["resume_optimizer_step"])
            * args.accumulate
            * args.train_batch_size
            / args.instances_per_epoch
        ),
        "best_mean_tour_length": float(best_tour_length),
        "best_step": int(best_step),
        "elapsed_sec": float(time.perf_counter() - train_started_at),
        "best_checkpoint": str(output_dir / "best.ckpt"),
        "last_checkpoint": str(output_dir / "last.ckpt"),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps({"event": "complete", **summary}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
