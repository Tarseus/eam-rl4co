from __future__ import annotations

import argparse
import hashlib
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

from rl4co.envs import FFSPEnv
from rl4co.models import MatNet
from rl4co.utils.ops import unbatchify
from scripts.eval_downloaded_routing_checkpoints import (
    _patch_legacy_policy_object,
    checkpoint_hparams,
)


METHODS = ("po", "bopo", "usw", "asw")
DEFAULT_PAIR_PATHS = {
    "usw": REPO_ROOT
    / "runs/pref_loss_ffsp100_discovery/20260403-142801/best_pair.json",
    "asw": REPO_ROOT
    / "runs/pref_builder_weight_search_ffsp100/20260416-111514/best_pair.json",
}


def _resolve(path: str | Path) -> Path:
    value = Path(path).expanduser()
    return (value if value.is_absolute() else REPO_ROOT / value).resolve()


def _autocast(device: torch.device, precision: str):
    value = str(precision).lower()
    if value in {"32", "fp32", "float32", "32-true"}:
        return nullcontext()
    if device.type != "cuda":
        raise ValueError(f"precision={precision} requires CUDA")
    if value in {"bf16", "bfloat16", "bf16-mixed"}:
        return torch.autocast("cuda", dtype=torch.bfloat16, cache_enabled=False)
    if value in {"16", "fp16", "float16", "16-mixed"}:
        return torch.autocast("cuda", dtype=torch.float16, cache_enabled=False)
    raise ValueError(f"Unsupported precision: {precision}")


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


def _seed(seed: int, device: torch.device) -> None:
    torch.manual_seed(int(seed))
    if device.type == "cuda":
        torch.cuda.manual_seed_all(int(seed))


def _dynamic_batch(
    *, jobs: int, stages: int, machines: int, seed: int, instance_index: int
) -> TensorDict:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed) + jobs * 1_000_003 + instance_index * 104_729)
    run_time = torch.randint(
        2,
        10,
        (1, jobs, stages * machines),
        generator=generator,
        dtype=torch.int64,
    )
    return TensorDict({"run_time": run_time}, batch_size=[1])


def _build_model(args: argparse.Namespace, checkpoint: Path):
    hparams = checkpoint_hparams(checkpoint)
    payload = hparams.pop("_checkpoint_payload")
    # Keep the entry point compatible with remote revisions that predate the
    # optional routing-only forced-replay flags. FFSP always uses full graph.
    hparams.pop("memory_efficient_preference", None)
    hparams["policy"] = _patch_legacy_policy_object(hparams.get("policy"))
    env = FFSPEnv(
        generator_params={
            "num_stage": args.num_stages,
            "num_machine": args.num_machines,
            "num_job": args.target_jobs,
            "min_time": 2,
            "max_time": 10,
            "flatten_stages": False,
        },
        seed=args.seed,
    )
    hparams.update(
        {
            "env": env,
            "num_starts": args.num_starts,
            "num_augment": 1,
            "loss_type": {
                "po": "po_loss",
                "bopo": "bopo_loss",
                "usw": "free_loss",
                "asw": "free_loss",
            }[args.method],
            "alpha": args.alpha,
            "po_impl": "exponential",
            "bopo_pair_mode": "anchor_best",
            "bopo_select_strategy": "paper",
            "bopo_select_k": args.bopo_select_k,
            "free_loss_ir_json_path": None,
            "pref_builder_ir_json_path": None,
            "pref_pair_json_path": None,
            "generate_default_data": True,
            "batch_size": 1,
            "train_data_size": 1,
            "val_data_size": 1,
            "test_data_size": 1,
        }
    )
    if args.method in {"usw", "asw"}:
        hparams["pref_pair_json_path"] = str(
            _resolve(args.usw_pair if args.method == "usw" else args.asw_pair)
        )
    model = MatNet(**hparams)
    missing, unexpected = model.load_state_dict(payload["state_dict"], strict=False)
    if missing or unexpected:
        raise RuntimeError(f"Checkpoint mismatch: missing={missing}, unexpected={unexpected}")
    return model, env, payload


def _evaluate(
    model,
    env,
    *,
    count: int,
    seed: int,
    starts: int,
    augment: int,
    batch_size: int,
    device: torch.device,
    precision: str,
) -> dict[str, Any]:
    model.eval()
    costs: list[float] = []
    started = time.perf_counter()
    for offset in range(0, int(count), int(batch_size)):
        indices = list(range(offset, min(offset + int(batch_size), int(count))))
        raw = TensorDict(
            {
                "run_time": torch.cat(
                    [
                        _dynamic_batch(
                            jobs=env.num_job,
                            stages=env.num_stage,
                            machines=env.num_machine,
                            seed=seed,
                            instance_index=index,
                        )["run_time"]
                        for index in indices
                    ],
                    dim=0,
                )
            },
            batch_size=[len(indices)],
        ).to(device)
        best = torch.full((len(indices),), float("inf"), device=device)
        for aug_index in range(int(augment)):
            _seed(seed + offset * 1_000_003 + aug_index, device)
            td = env.reset(raw.clone()).to(device)
            with torch.inference_mode(), _autocast(device, precision):
                out = model.policy(
                    td,
                    env,
                    phase="test",
                    num_starts=starts,
                    return_actions=False,
                )
            reward = unbatchify(out["reward"], (0, starts))
            best = torch.minimum(best, -reward.max(dim=-1).values)
        costs.extend(float(value) for value in best.detach().cpu())
    elapsed = time.perf_counter() - started
    return {
        "count": len(costs),
        "mean_cost": sum(costs) / len(costs),
        "per_instance_cost": costs,
        "elapsed_sec": elapsed,
        "seconds_per_instance": elapsed / len(costs),
        "num_starts": starts,
        "num_augment": augment,
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
        description="Common-checkpoint continuation for FFSP1000 PO/BOPO/USW/ASW."
    )
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--target-jobs", type=int, default=1000)
    parser.add_argument("--num-stages", type=int, default=3)
    parser.add_argument("--num-machines", type=int, default=4)
    parser.add_argument("--num-starts", type=int, default=24)
    parser.add_argument("--bopo-select-k", type=int, default=6)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--accumulate", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-6)
    parser.add_argument("--precision", default="fp32")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=12345678)
    parser.add_argument("--data-start-index", type=int, default=0)
    parser.add_argument("--validation-count", type=int, default=8)
    parser.add_argument("--validation-every", type=int, default=100)
    parser.add_argument("--validation-augment", type=int, default=1)
    parser.add_argument("--validation-batch-size", type=int, default=8)
    parser.add_argument("--evaluate-only", action="store_true")
    parser.add_argument("--evaluation-count", type=int, default=16)
    parser.add_argument("--evaluation-augment", type=int, default=1)
    parser.add_argument("--evaluation-batch-size", type=int, default=8)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--usw-pair", type=Path, default=DEFAULT_PAIR_PATHS["usw"])
    parser.add_argument("--asw-pair", type=Path, default=DEFAULT_PAIR_PATHS["asw"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.num_starts != math.factorial(args.num_machines):
        raise ValueError("FFSP uses one start per machine order; require starts == machines!")
    if args.accumulate < 1 or args.steps < 0:
        raise ValueError("steps must be nonnegative and accumulate must be positive")
    checkpoint = _resolve(args.checkpoint)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    model, env, _ = _build_model(args, checkpoint)
    model = model.to(device)
    initial_sha = _state_sha256(model)

    if args.evaluate_only:
        result = _evaluate(
            model,
            env,
            count=args.evaluation_count,
            seed=args.seed + 90_000_001,
            starts=args.num_starts,
            augment=args.evaluation_augment,
            batch_size=args.evaluation_batch_size,
            device=device,
            precision=args.precision,
        )
        print(json.dumps({"method": args.method, "checkpoint": str(checkpoint), "initial_state_sha256": initial_sha, **result}), flush=True)
        return

    output_dir = _resolve(
        args.output_dir
        or f"logs/scheduling_large_scale/ffsp{args.target_jobs}/{args.method}"
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
        "protocol": "ffsp1000_common_checkpoint_continuation_v1",
        "method": args.method,
        "common_initialization": str(checkpoint),
        "initial_state_sha256": initial_sha,
        "fresh_optimizer": args.resume is None,
        "target_jobs": args.target_jobs,
        "num_starts": args.num_starts,
        "bopo_select_k": args.bopo_select_k,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "accumulate": args.accumulate,
        "precision": args.precision,
        "seed": args.seed,
    }
    (output_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    history = output_dir / "history.jsonl"

    def validate() -> None:
        nonlocal best_cost
        result = _evaluate(
            model,
            env,
            count=args.validation_count,
            seed=args.seed + 80_000_003,
            starts=args.num_starts,
            augment=args.validation_augment,
            batch_size=args.validation_batch_size,
            device=device,
            precision=args.precision,
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
        started = time.perf_counter()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        for micro in range(args.accumulate):
            index = args.data_start_index + step * args.accumulate + micro
            batch = _dynamic_batch(
                jobs=args.target_jobs,
                stages=args.num_stages,
                machines=args.num_machines,
                seed=args.seed,
                instance_index=index,
            ).to(device)
            _seed(args.seed + index * 1_000_003, device)
            with _autocast(device, args.precision):
                out = model.shared_step(batch, 0, "train")
                loss = out["loss"]
            if loss is None or not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite FFSP loss: {loss}")
            (loss / args.accumulate).backward()
            losses.append(float(loss.detach().cpu()))
        grad_norm = _gradient_norm(model.parameters())
        if not math.isfinite(grad_norm) or grad_norm == 0.0:
            raise FloatingPointError(f"Invalid gradient norm: {grad_norm}")
        optimizer.step()
        step += 1
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        record = {
            "event": "train",
            "optimizer_step": step,
            "loss": sum(losses) / len(losses),
            "grad_norm": grad_norm,
            "elapsed_sec": time.perf_counter() - started,
            "candidate_count_per_instance": args.num_starts,
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
