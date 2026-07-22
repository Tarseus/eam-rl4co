from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import torch
from tensordict import TensorDict


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.probe_tsp1000_dynamic_training import (
    DEFAULT_PAIR_PATHS,
    _autocast_context,
    _build_model,
    _resolve_path,
)


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _mean(records: list[dict[str, float]], key: str) -> float:
    return statistics.mean(record[key] for record in records)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="downloads/tsp100/po/checkpoint.ckpt")
    parser.add_argument("--method", choices=("po", "bopo", "usw", "asw"), default="usw")
    parser.add_argument("--target-size", type=int, default=1000)
    parser.add_argument("--num-starts", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--precision", default="bf16-mixed")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=2)
    parser.add_argument(
        "--checkpoint-encoder",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--checkpoint-decoder",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--usw-pair-path",
        default=str(DEFAULT_PAIR_PATHS["usw"].relative_to(REPO_ROOT)),
    )
    parser.add_argument(
        "--asw-pair-path",
        default=str(DEFAULT_PAIR_PATHS["asw"].relative_to(REPO_ROOT)),
    )
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    if args.batch_size < 1 or args.num_starts < 2:
        raise ValueError("batch-size must be positive and num-starts must be at least 2")
    if args.warmup < 0 or args.iterations < 1:
        raise ValueError("warmup must be nonnegative and iterations must be positive")

    device = torch.device(args.device)
    checkpoint = _resolve_path(args.checkpoint)
    model, env = _build_model(
        method=args.method,
        checkpoint_path=checkpoint,
        target_size=args.target_size,
        num_starts=args.num_starts,
        seed=args.seed,
        usw_pair_path=_resolve_path(args.usw_pair_path),
        asw_pair_path=_resolve_path(args.asw_pair_path),
    )
    model.memory_efficient_checkpoint_encoder = bool(args.checkpoint_encoder)
    model.memory_efficient_checkpoint_decoder = bool(args.checkpoint_decoder)
    model = model.to(device).train()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(args.learning_rate),
        weight_decay=1e-6,
    )

    generator = torch.Generator(device="cpu")
    generator.manual_seed(args.seed + args.target_size * 1_000_003)
    locs = torch.rand(
        (args.batch_size, args.target_size, 2),
        generator=generator,
        dtype=torch.float32,
    ).to(device)
    batch = TensorDict({"locs": locs}, batch_size=[args.batch_size]).to(device)

    policy_call_starts: list[float] = []
    policy_call_times: list[float] = []

    def before_policy(*_args) -> None:
        _sync(device)
        policy_call_starts.append(time.perf_counter())

    def after_policy(*_args) -> None:
        _sync(device)
        policy_call_times.append(time.perf_counter() - policy_call_starts[-1])

    pre_handle = model.policy.register_forward_pre_hook(before_policy)
    post_handle = model.policy.register_forward_hook(after_policy)
    records: list[dict[str, float]] = []
    total_iterations = args.warmup + args.iterations
    try:
        for iteration in range(total_iterations):
            optimizer.zero_grad(set_to_none=True)
            torch.manual_seed(args.seed + iteration)
            if device.type == "cuda":
                torch.cuda.manual_seed_all(args.seed + iteration)
                torch.cuda.reset_peak_memory_stats(device)
            policy_call_times.clear()

            _sync(device)
            total_started = time.perf_counter()
            reset_started = total_started
            td = env.reset(batch).to(device)
            _sync(device)
            reset_sec = time.perf_counter() - reset_started

            objective_started = time.perf_counter()
            with _autocast_context(device, args.precision):
                out = model._memory_efficient_preference_step(
                    td=td,
                    batch=batch,
                    n_start=args.num_starts,
                    dataloader_idx=None,
                    log_metrics=False,
                )
            _sync(device)
            objective_sec = time.perf_counter() - objective_started

            backward_started = time.perf_counter()
            out["loss"].backward()
            _sync(device)
            backward_sec = time.perf_counter() - backward_started

            optimizer_started = time.perf_counter()
            optimizer.step()
            _sync(device)
            optimizer_sec = time.perf_counter() - optimizer_started
            total_sec = time.perf_counter() - total_started

            if len(policy_call_times) != 2:
                raise RuntimeError(
                    f"Expected rollout and replay policy calls, got {len(policy_call_times)}"
                )
            gib = float(1024**3)
            record = {
                "total_sec": total_sec,
                "reset_sec": reset_sec,
                "rollout_sec": policy_call_times[0],
                "replay_sec": policy_call_times[1],
                "objective_other_sec": max(
                    0.0, objective_sec - sum(policy_call_times)
                ),
                "backward_sec": backward_sec,
                "optimizer_sec": optimizer_sec,
                "peak_allocated_gib": (
                    torch.cuda.max_memory_allocated(device) / gib
                    if device.type == "cuda"
                    else 0.0
                ),
                "peak_reserved_gib": (
                    torch.cuda.max_memory_reserved(device) / gib
                    if device.type == "cuda"
                    else 0.0
                ),
            }
            print(json.dumps({"iteration": iteration, **record}), flush=True)
            if iteration >= args.warmup:
                records.append(record)
    finally:
        pre_handle.remove()
        post_handle.remove()

    summary = {
        "protocol": "tsp1000_training_speed_profile_v1",
        "method": args.method,
        "target_size": args.target_size,
        "num_starts": args.num_starts,
        "batch_size": args.batch_size,
        "precision": args.precision,
        "checkpoint_encoder": args.checkpoint_encoder,
        "checkpoint_decoder": args.checkpoint_decoder,
        "iterations": args.iterations,
        "mean": {key: _mean(records, key) for key in records[0]},
    }
    summary["mean"].update(
        {
            "instances_per_sec": args.batch_size / summary["mean"]["total_sec"],
            "trajectories_per_sec": (
                args.batch_size * args.num_starts / summary["mean"]["total_sec"]
            ),
        }
    )
    print(json.dumps(summary, indent=2), flush=True)
    if args.output:
        output = _resolve_path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
