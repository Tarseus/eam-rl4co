from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rl4co.envs import JSSPEnv
from rl4co.models.zoo.mgl_jssp.data import cluster_edges, extract_features
from rl4co.models.zoo.mgl_jssp.model import MGLJSSPModel
from rl4co.models.zoo.mgl_jssp.sampling import sampling
from scripts.eval_downloaded_routing_checkpoints import checkpoint_hparams


METHODS = ("rl", "po", "sll", "bopo", "usw", "asw")
DEFAULT_PAIR_PATHS = {
    "usw": REPO_ROOT
    / "runs/pref_loss_jssp10x10_from_ffsp100_elite/20260416-113409/best_pair.json",
    "asw": REPO_ROOT
    / "runs/pref_builder_weight_search_jssp10x10_from_best_loss/20260417-123033/best_pair.json",
}


def _resolve(path: str | Path) -> Path:
    value = Path(path).expanduser()
    return (value if value.is_absolute() else REPO_ROOT / value).resolve()


def _seed(seed: int, device: torch.device) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed) % (2**32))
    torch.manual_seed(int(seed))
    if device.type == "cuda":
        torch.cuda.manual_seed_all(int(seed))


def _gradient_norm(parameters) -> float:
    total = 0.0
    for parameter in parameters:
        if parameter.grad is not None:
            total += float(parameter.grad.detach().float().square().sum().cpu())
    return math.sqrt(total)


def _state_sha256(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        digest.update(name.encode("utf-8"))
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _dynamic_instance(
    *, jobs: int, machines: int, seed: int, instance_index: int
) -> dict[str, Any]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed) + jobs * 1_000_003 + machines * 10_007 + instance_index * 104_729)
    machine_rows = [torch.randperm(machines, generator=generator) for _ in range(jobs)]
    machine_tensor = torch.stack(machine_rows).long()
    costs = torch.randint(1, 100, (jobs, machines), generator=generator).float()
    job_edges, mac_edges = cluster_edges(jobs, machines, machine_tensor, device="cpu")
    x = extract_features(jobs, machines, costs, machine_tensor, device="cpu")
    return {
        "name": f"random_{jobs}x{machines}_{instance_index:08d}",
        "path": "<dynamic>",
        "j": jobs,
        "m": machines,
        "shape": f"{jobs}x{machines}",
        "x": x,
        "job_edges": job_edges,
        "mac_edges": mac_edges,
        "costs": costs,
        "machines": machine_tensor,
        "makespan": 0.0,
    }


def _build_model(args: argparse.Namespace, checkpoint: Path):
    hparams = checkpoint_hparams(checkpoint)
    payload = hparams.pop("_checkpoint_payload")
    hparams.pop("generate_default_data", None)
    env = JSSPEnv(
        generator_params={"num_jobs": args.num_jobs, "num_machines": args.num_machines}
    )
    hparams.update(
        {
            "env": env,
            "baseline": (
                args.method if args.method in {"rl", "po", "sll", "bopo"} else "bopo"
            ),
            "B": args.rollouts,
            "K": args.select_k,
            "D": 1,
            "pair_mode": "anchor_best",
            "po_impl": "bt",
            "po_alpha": args.po_alpha,
            "alpha": args.alpha,
            "val_B": args.eval_rollouts,
            "test_B": args.eval_rollouts,
            "greedy": int(args.greedy),
            "batch_size": 1,
            "val_batch_size": 1,
            "test_batch_size": 1,
            "allowed_shapes": [[args.num_jobs, args.num_machines]],
            "required_allowed_shapes": [[args.num_jobs, args.num_machines]],
            "expected_train_dataset_size": None,
            "expected_val_dataset_size": None,
            "expected_test_dataset_size": None,
            "init_external_checkpoint_path": None,
            "free_loss_ir_json_path": None,
            "pref_builder_ir_json_path": None,
            "pref_pair_json_path": None,
        }
    )
    if args.method in {"usw", "asw"}:
        hparams["pref_pair_json_path"] = str(
            _resolve(args.usw_pair if args.method == "usw" else args.asw_pair)
        )
    model = MGLJSSPModel(**hparams)
    missing, unexpected = model.load_state_dict(payload["state_dict"], strict=False)
    if missing or unexpected:
        raise RuntimeError(f"Checkpoint mismatch: missing={missing}, unexpected={unexpected}")
    return model, payload


def _evaluate(
    model: MGLJSSPModel,
    *,
    jobs: int,
    machines: int,
    count: int,
    rollouts: int,
    batch_size: int,
    seed: int,
    device: torch.device,
) -> dict[str, Any]:
    model.eval()
    costs: list[float] = []
    started = time.perf_counter()
    for offset in range(0, int(count), int(batch_size)):
        indices = list(range(offset, min(offset + int(batch_size), int(count))))
        instances = [
            _dynamic_instance(
                jobs=jobs,
                machines=machines,
                seed=seed,
                instance_index=index,
            )
            for index in indices
        ]
        _seed(seed + offset * 1_000_003, device)
        makespans, _, _ = sampling(
            instances,
            model.encoder,
            model.decoder,
            bs=rollouts,
            use_greedy=model.use_greedy,
            device=str(device),
        )
        best = makespans.view(len(instances), rollouts).min(dim=1).values
        costs.extend(float(value) for value in best.detach().cpu())
    elapsed = time.perf_counter() - started
    return {
        "count": len(costs),
        "mean_cost": sum(costs) / len(costs),
        "per_instance_cost": costs,
        "elapsed_sec": elapsed,
        "seconds_per_instance": elapsed / len(costs),
        "rollouts_per_instance": rollouts,
    }


def _write_jsonl(path: Path, record: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def _atomic_save(payload: dict[str, Any], path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def _save_payload(model, optimizer, step: int, best: float, config: dict[str, Any]):
    return {
        "state_dict": model.state_dict(),
        "hyper_parameters": dict(model.hparams),
        "optimizer_state_dict": optimizer.state_dict(),
        "optimizer_step": step,
        "global_step": step,
        "best_cost": best,
        "method": config["method"],
        "large_scale_config": config,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Common-checkpoint continuation for large JSSP RL/PO/SLL/BOPO/USW/ASW."
    )
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--num-jobs", type=int, default=50)
    parser.add_argument("--num-machines", type=int, default=20)
    parser.add_argument("--rollouts", type=int, default=128)
    parser.add_argument("--select-k", type=int, default=16)
    parser.add_argument("--po-alpha", type=float, default=0.25)
    parser.add_argument(
        "--alpha",
        type=float,
        default=1.0,
        help="Free-loss/preference-builder calibration alpha (used by ASW artifacts).",
    )
    parser.add_argument("--greedy", type=int, choices=(0, 1), default=0)
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--accumulate", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-6)
    parser.add_argument(
        "--max-grad-norm",
        type=float,
        default=None,
        help="Optional per-update gradient-norm cap; disabled by default.",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=12345678)
    parser.add_argument("--data-start-index", type=int, default=0)
    parser.add_argument("--eval-rollouts", type=int, default=128)
    parser.add_argument("--validation-count", type=int, default=8)
    parser.add_argument("--validation-batch-size", type=int, default=8)
    parser.add_argument("--validation-every", type=int, default=100)
    parser.add_argument("--evaluate-only", action="store_true")
    parser.add_argument("--evaluation-count", type=int, default=16)
    parser.add_argument("--evaluation-batch-size", type=int, default=8)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--usw-pair", type=Path, default=DEFAULT_PAIR_PATHS["usw"])
    parser.add_argument("--asw-pair", type=Path, default=DEFAULT_PAIR_PATHS["asw"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.rollouts <= 1 or args.select_k <= 0 or args.rollouts % args.select_k:
        raise ValueError("JSSP requires rollouts > 1 and rollouts % select_k == 0")
    if args.accumulate < 1 or args.steps < 0:
        raise ValueError("steps must be nonnegative and accumulate must be positive")
    if args.max_grad_norm is not None and args.max_grad_norm <= 0:
        raise ValueError("max_grad_norm must be positive when provided")
    checkpoint = _resolve(args.checkpoint)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    model, _ = _build_model(args, checkpoint)
    model = model.to(device)
    initial_sha = _state_sha256(model)

    if args.evaluate_only:
        result = _evaluate(
            model,
            jobs=args.num_jobs,
            machines=args.num_machines,
            count=args.evaluation_count,
            rollouts=args.eval_rollouts,
            batch_size=args.evaluation_batch_size,
            seed=args.seed + 90_000_001,
            device=device,
        )
        print(json.dumps({"method": args.method, "checkpoint": str(checkpoint), "initial_state_sha256": initial_sha, **result}), flush=True)
        return

    output_dir = _resolve(
        args.output_dir
        or f"logs/scheduling_large_scale/jssp{args.num_jobs}x{args.num_machines}/{args.method}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    step = 0
    best_cost = float("inf")
    if args.resume:
        resume = torch.load(_resolve(args.resume), map_location="cpu", weights_only=False)
        model.load_state_dict(resume["state_dict"], strict=True)
        optimizer.load_state_dict(resume["optimizer_state_dict"])
        step = int(resume["optimizer_step"])
        best_cost = float(resume["best_cost"])
    target_step = step + args.steps
    config = {
        "protocol": "jssp_large_common_checkpoint_continuation_v1",
        "method": args.method,
        "common_initialization": str(checkpoint),
        "initial_state_sha256": initial_sha,
        "fresh_optimizer": args.resume is None,
        "shape": f"{args.num_jobs}x{args.num_machines}",
        "physical_batch_size": 1,
        "rollouts_per_instance": args.rollouts,
        "select_k": args.select_k,
        "alpha": args.alpha,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "max_grad_norm": args.max_grad_norm,
        "accumulate": args.accumulate,
        "seed": args.seed,
        "pair_scope": "strictly within instance",
    }
    (output_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    history = output_dir / "history.jsonl"

    def validate() -> None:
        nonlocal best_cost
        result = _evaluate(
            model,
            jobs=args.num_jobs,
            machines=args.num_machines,
            count=args.validation_count,
            rollouts=args.eval_rollouts,
            batch_size=args.validation_batch_size,
            seed=args.seed + 80_000_003,
            device=device,
        )
        result.pop("per_instance_cost")
        improved = result["mean_cost"] < best_cost
        best_cost = min(best_cost, result["mean_cost"])
        record = {"event": "validation", "optimizer_step": step, "best_cost": best_cost, **result}
        _write_jsonl(history, record)
        payload = _save_payload(model, optimizer, step, best_cost, config)
        _atomic_save(payload, output_dir / "last.ckpt")
        if improved:
            _atomic_save(payload, output_dir / "best.ckpt")
        print(json.dumps(record), flush=True)

    if not args.resume:
        validate()
    while step < target_step:
        model.train()
        optimizer.zero_grad(set_to_none=True)
        losses: list[float] = []
        pair_counts: list[float] = []
        started = time.perf_counter()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        for micro in range(args.accumulate):
            index = args.data_start_index + step * args.accumulate + micro
            instance = _dynamic_instance(
                jobs=args.num_jobs,
                machines=args.num_machines,
                seed=args.seed,
                instance_index=index,
            )
            if (instance["j"], instance["m"]) != (args.num_jobs, args.num_machines):
                raise RuntimeError("Dynamic JSSP generator returned an unsupported shape")
            _seed(args.seed + index * 1_000_003, device)
            loss, _, _, pair_count = model._training_rollout([instance])
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite JSSP loss: {loss}")
            (loss / args.accumulate).backward()
            losses.append(float(loss.detach().cpu()))
            if pair_count is not None:
                pair_counts.append(float(pair_count.detach().cpu()))
        grad_norm = _gradient_norm(model.parameters())
        if not math.isfinite(grad_norm) or grad_norm == 0.0:
            raise FloatingPointError(f"Invalid gradient norm: {grad_norm}")
        grad_norm_after_clip = grad_norm
        if args.max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            grad_norm_after_clip = _gradient_norm(model.parameters())
            if not math.isfinite(grad_norm_after_clip) or grad_norm_after_clip == 0.0:
                raise FloatingPointError(
                    f"Invalid gradient norm after clipping: {grad_norm_after_clip}"
                )
        optimizer.step()
        step += 1
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        record = {
            "event": "train",
            "optimizer_step": step,
            "loss": sum(losses) / len(losses),
            "grad_norm": grad_norm,
            "grad_norm_after_clip": grad_norm_after_clip,
            "max_grad_norm": args.max_grad_norm,
            "elapsed_sec": time.perf_counter() - started,
            "candidate_count_per_instance": args.rollouts,
            "select_k_per_instance": args.select_k,
            "physical_instance_batch": 1,
            "pair_count": (sum(pair_counts) / len(pair_counts) if pair_counts else None),
        }
        if device.type == "cuda":
            record.update(
                peak_memory_allocated_gib=torch.cuda.max_memory_allocated(device) / 1024**3,
                peak_memory_reserved_gib=torch.cuda.max_memory_reserved(device) / 1024**3,
            )
        _write_jsonl(history, record)
        print(json.dumps(record), flush=True)
        if step % args.validation_every == 0 or step == target_step:
            validate()


if __name__ == "__main__":
    main()
