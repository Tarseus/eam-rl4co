from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import random
import sys
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tensordict import TensorDict


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rl4co.data.transforms import StateAugmentation
from rl4co.envs import get_env
from rl4co.models.zoo.symnco.losses import (
    invariance_loss,
    problem_symmetricity_loss,
    solution_symmetricity_loss,
)
from rl4co.utils.ops import unbatchify
from scripts.standard_routing_policy import METHODS, PROBLEMS, build_policy


def _resolve(path: str | Path | None) -> Path | None:
    if path is None:
        return None
    value = Path(path).expanduser()
    return (value if value.is_absolute() else REPO_ROOT / value).resolve()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _autocast(device: torch.device, precision: str):
    value = precision.lower()
    if value in {"32", "32-true", "fp32", "float32"}:
        return nullcontext()
    if device.type != "cuda":
        raise ValueError(f"precision={precision} requires CUDA")
    if value in {"bf16", "bf16-mixed", "bfloat16"}:
        return torch.autocast("cuda", dtype=torch.bfloat16, cache_enabled=False)
    if value in {"16", "16-mixed", "fp16", "float16"}:
        return torch.autocast("cuda", dtype=torch.float16, cache_enabled=False)
    raise ValueError(f"Unsupported precision={precision}")


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _dynamic_batch(
    *,
    problem: str,
    target_size: int,
    capacity: int,
    seed: int,
    instance_index: int,
    batch_size: int,
) -> TensorDict:
    rows: list[TensorDict] = []
    for offset in range(batch_size):
        generator = torch.Generator(device="cpu")
        generator.manual_seed(
            int(seed)
            + int(target_size) * 1_000_003
            + (int(instance_index) + offset) * 104_729
        )
        if problem == "tsp":
            rows.append(
                TensorDict(
                    {"locs": torch.rand((target_size, 2), generator=generator)},
                    batch_size=[],
                )
            )
        else:
            coordinates = torch.rand((target_size + 1, 2), generator=generator)
            demand = torch.randint(1, 10, (target_size,), generator=generator)
            rows.append(
                TensorDict(
                    {
                        "depot": coordinates[0],
                        "locs": coordinates[1:],
                        "demand": demand.float() / float(capacity),
                        "capacity": torch.tensor([float(capacity)]),
                    },
                    batch_size=[],
                )
            )
    return torch.stack(rows, dim=0)


def _symnco_loss(
    *, policy, env, batch: TensorDict, num_starts: int, num_augment: int, alpha: float, beta: float
) -> tuple[torch.Tensor, dict[str, float]]:
    td = env.reset(batch)
    if num_augment > 1:
        td = StateAugmentation(num_augment=num_augment, augment_fn="symmetric")(td)
    out = policy(
        td,
        env,
        phase="train",
        num_starts=num_starts,
        return_actions=False,
        return_init_embeds=True,
    )
    reward = unbatchify(out["reward"], (num_augment, num_starts))
    log_likelihood = unbatchify(out["log_likelihood"], (num_augment, num_starts))
    loss_ps = problem_symmetricity_loss(reward, log_likelihood) if num_starts > 1 else 0
    loss_ss = solution_symmetricity_loss(reward, log_likelihood) if num_augment > 1 else 0
    loss_inv = invariance_loss(out["proj_embeddings"], num_augment) if num_augment > 1 else 0
    loss = loss_ps + beta * loss_ss + alpha * loss_inv
    return loss, {
        "reward_mean": float(reward.detach().mean().cpu()),
        "loss_ps": float(torch.as_tensor(loss_ps).detach().cpu()),
        "loss_ss": float(torch.as_tensor(loss_ss).detach().cpu()),
        "loss_inv": float(torch.as_tensor(loss_inv).detach().cpu()),
    }


def _am_loss(*, policy, baseline_policy, env, batch: TensorDict) -> tuple[torch.Tensor, dict[str, float]]:
    # CVRP reset/rollout mutates parts of its TensorDict; the sampled policy and
    # frozen rollout baseline must receive independent copies of the instance.
    td = env.reset(batch.clone())
    out = policy(td, env, phase="train", return_actions=False)
    with torch.no_grad():
        baseline_td = env.reset(batch.clone())
        baseline_out = baseline_policy(
            baseline_td,
            env,
            phase="test",
            decode_type="greedy",
            return_actions=False,
        )
    advantage = out["reward"] - baseline_out["reward"]
    loss = -(advantage.detach() * out["log_likelihood"]).mean()
    return loss, {
        "reward_mean": float(out["reward"].detach().mean().cpu()),
        "baseline_reward_mean": float(baseline_out["reward"].detach().mean().cpu()),
        "advantage_mean": float(advantage.detach().mean().cpu()),
    }


def _evaluate(
    *,
    policy,
    env,
    problem: str,
    target_size: int,
    capacity: int,
    seed: int,
    num_instances: int,
    num_starts: int,
    device: torch.device,
    precision: str,
) -> float:
    policy.eval()
    values: list[float] = []
    with torch.inference_mode():
        for index in range(num_instances):
            batch = _dynamic_batch(
                problem=problem,
                target_size=target_size,
                capacity=capacity,
                seed=seed,
                instance_index=index,
                batch_size=1,
            ).to(device)
            td = env.reset(batch).to(device)
            with _autocast(device, precision):
                out = policy(
                    td,
                    env,
                    phase="test",
                    num_starts=num_starts,
                    return_actions=False,
                )
            values.append(float(-out["reward"].max().detach().cpu()))
    policy.train()
    return float(np.mean(values))


def _atomic_save(payload: dict[str, Any], path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Continue AM or SymNCO training on dynamic TSP1000/CVRP1000 instances."
    )
    parser.add_argument("--method", required=True, choices=METHODS)
    parser.add_argument("--problem", required=True, choices=PROBLEMS)
    parser.add_argument("--init-checkpoint", required=True)
    parser.add_argument("--target-size", type=int, default=1000)
    parser.add_argument("--capacity", type=int, default=50)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--train-batch-size", type=int, default=1)
    parser.add_argument("--accumulate", type=int, default=1)
    parser.add_argument("--num-starts", type=int, default=20)
    parser.add_argument("--num-augment", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-6)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--symnco-alpha", type=float, default=0.1)
    parser.add_argument("--symnco-beta", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--data-start-index", type=int, default=0)
    parser.add_argument("--precision", default="bf16-mixed")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--validation-size", type=int, default=8)
    parser.add_argument("--validation-every", type=int, default=100)
    parser.add_argument("--validation-starts", type=int, default=None)
    parser.add_argument("--log-every", type=int, default=5)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if min(args.steps, args.train_batch_size, args.accumulate) < 1:
        raise ValueError("steps, train-batch-size, and accumulate must be >= 1")
    if args.method == "symnco" and (args.num_starts < 2 or args.num_augment < 2):
        raise ValueError("SymNCO requires at least two starts and two augmentations")
    if args.problem == "tsp" and args.capacity != 50:
        print("[note] --capacity is ignored for TSP", flush=True)
    validation_starts = args.validation_starts or (
        args.target_size if args.problem == "tsp" else 100
    )
    if validation_starts > args.target_size:
        raise ValueError("validation-starts cannot exceed target-size")

    _seed_everything(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if device.type == "cuda":
        # cuDNN SDPA has no valid execution plan for some long-sequence
        # autoregressive attention shapes on RTX 3090. Keep PyTorch's flash,
        # memory-efficient, and math fallbacks enabled instead.
        torch.backends.cuda.enable_cudnn_sdp(False)
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        torch.backends.cuda.enable_math_sdp(True)
    output_dir = _resolve(args.output_dir)
    assert output_dir is not None
    output_dir.mkdir(parents=True, exist_ok=True)
    init_checkpoint = _resolve(args.init_checkpoint)
    assert init_checkpoint is not None and init_checkpoint.exists()

    env = get_env(
        args.problem,
        generator_params={
            "num_loc": args.target_size,
            **({"capacity": float(args.capacity)} if args.problem == "cvrp" else {}),
        },
        seed=args.seed,
    )
    policy, init_metadata = build_policy(
        method=args.method,
        problem=args.problem,
        init_checkpoint=init_checkpoint,
    )
    policy = policy.to(device)
    baseline_policy = copy.deepcopy(policy).eval() if args.method == "am" else None
    if baseline_policy is not None:
        for parameter in baseline_policy.parameters():
            parameter.requires_grad_(False)
    optimizer = torch.optim.Adam(
        policy.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )

    start_step = 0
    best_cost = math.inf
    best_step = 0
    resume_path = _resolve(args.resume)
    if resume_path is not None:
        payload = torch.load(resume_path, map_location="cpu", weights_only=False)
        if payload.get("method") != args.method or payload.get("problem") != args.problem:
            raise ValueError("Resume method/problem does not match requested run")
        policy.load_state_dict(payload["policy_state_dict"], strict=True)
        optimizer.load_state_dict(payload["optimizer_state_dict"])
        start_step = int(payload["optimizer_step"])
        best_cost = float(payload["best_cost"])
        best_step = int(payload["best_step"])
        if baseline_policy is not None and payload.get("baseline_policy_state_dict"):
            baseline_policy.load_state_dict(payload["baseline_policy_state_dict"], strict=True)

    config = {
        "protocol": "scale1000_standard_baseline_finetune_v1",
        "method": args.method,
        "problem": args.problem,
        "target_size": args.target_size,
        "capacity": args.capacity if args.problem == "cvrp" else None,
        "optimizer_steps": args.steps,
        "resume_optimizer_step": start_step,
        "train_batch_size": args.train_batch_size,
        "accumulate": args.accumulate,
        "effective_batch_size": args.train_batch_size * args.accumulate,
        "num_starts": 1 if args.method == "am" else args.num_starts,
        "num_augment": 1 if args.method == "am" else args.num_augment,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "max_grad_norm": args.max_grad_norm,
        "symnco_alpha": args.symnco_alpha if args.method == "symnco" else None,
        "symnco_beta": args.symnco_beta if args.method == "symnco" else None,
        "seed": args.seed,
        "data_start_index": args.data_start_index,
        "precision": args.precision,
        "validation_size": args.validation_size,
        "validation_every": args.validation_every,
        "validation_starts": validation_starts,
        "dynamic_instances": True,
        "am_baseline": "frozen_greedy_rollout_updated_on_validation_improvement"
        if args.method == "am"
        else None,
        **init_metadata,
        "init_checkpoint_sha256": _sha256(init_checkpoint),
        "resume": str(resume_path) if resume_path is not None else None,
    }
    (output_dir / "config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    history_path = output_dir / "history.jsonl"

    policy.train()
    for optimizer_step in range(start_step + 1, args.steps + 1):
        started = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        micro_metrics: list[dict[str, float]] = []
        for micro_index in range(args.accumulate):
            instance_index = (
                args.data_start_index
                + ((optimizer_step - 1) * args.accumulate + micro_index)
                * args.train_batch_size
            )
            batch = _dynamic_batch(
                problem=args.problem,
                target_size=args.target_size,
                capacity=args.capacity,
                seed=args.seed,
                instance_index=instance_index,
                batch_size=args.train_batch_size,
            ).to(device)
            with _autocast(device, args.precision):
                if args.method == "am":
                    assert baseline_policy is not None
                    loss, metrics = _am_loss(
                        policy=policy,
                        baseline_policy=baseline_policy,
                        env=env,
                        batch=batch,
                    )
                else:
                    loss, metrics = _symnco_loss(
                        policy=policy,
                        env=env,
                        batch=batch,
                        num_starts=args.num_starts,
                        num_augment=args.num_augment,
                        alpha=args.symnco_alpha,
                        beta=args.symnco_beta,
                    )
                (loss / args.accumulate).backward()
            metrics["loss"] = float(loss.detach().cpu())
            micro_metrics.append(metrics)

        grad_norm = float(
            torch.nn.utils.clip_grad_norm_(policy.parameters(), args.max_grad_norm)
        )
        optimizer.step()
        elapsed = time.perf_counter() - started
        record: dict[str, Any] = {
            "optimizer_step": optimizer_step,
            "elapsed_sec": elapsed,
            "grad_norm": grad_norm,
        }
        for key in micro_metrics[0]:
            record[key] = float(np.mean([item[key] for item in micro_metrics]))

        should_validate = (
            optimizer_step == start_step + 1
            or optimizer_step % args.validation_every == 0
            or optimizer_step == args.steps
        )
        if should_validate:
            record["validation_cost"] = _evaluate(
                policy=policy,
                env=env,
                problem=args.problem,
                target_size=args.target_size,
                capacity=args.capacity,
                seed=args.seed + 50_000_000,
                num_instances=args.validation_size,
                num_starts=validation_starts,
                device=device,
                precision=args.precision,
            )
            if record["validation_cost"] < best_cost:
                best_cost = float(record["validation_cost"])
                best_step = optimizer_step
                if baseline_policy is not None:
                    baseline_policy.load_state_dict(policy.state_dict(), strict=True)

        checkpoint_payload = {
            "format": "scale1000_standard_policy_v1",
            "method": args.method,
            "problem": args.problem,
            "target_size": args.target_size,
            "capacity": args.capacity if args.problem == "cvrp" else None,
            "policy_state_dict": policy.state_dict(),
            "baseline_policy_state_dict": (
                baseline_policy.state_dict() if baseline_policy is not None else None
            ),
            "optimizer_state_dict": optimizer.state_dict(),
            "optimizer_step": optimizer_step,
            "best_cost": best_cost,
            "best_step": best_step,
            "config": config,
        }
        _atomic_save(checkpoint_payload, output_dir / "last.ckpt")
        if should_validate and best_step == optimizer_step:
            _atomic_save(checkpoint_payload, output_dir / "best.ckpt")
        with history_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        if optimizer_step % args.log_every == 0 or should_validate:
            print("[train] " + json.dumps(record, ensure_ascii=False), flush=True)

    summary = {
        "completed_steps": args.steps,
        "best_step": best_step,
        "best_cost": best_cost,
        "output_dir": str(output_dir),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("[done] " + json.dumps(summary, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
