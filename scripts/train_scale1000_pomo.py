from __future__ import annotations

import argparse
import csv
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

from rl4co.envs import get_env
from rl4co.models import POMO
from rl4co.utils.ops import unbatchify
from scripts.eval_downloaded_routing_checkpoints import (
    _finalize_model_from_payload,
    _patch_legacy_policy_object,
    checkpoint_hparams,
)
from scripts.train_scale1000_am_symnco import _dynamic_batch


PROBLEMS = ("tsp", "cvrp")


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


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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


def _build_model(
    *, checkpoint: Path, problem: str, target_size: int, capacity: int, num_starts: int
) -> tuple[POMO, Any]:
    hparams = checkpoint_hparams(checkpoint)
    payload = hparams.pop("_checkpoint_payload")
    hparams.pop("ea_kwargs", None)
    hparams["policy"] = _patch_legacy_policy_object(hparams.get("policy"))
    env = get_env(
        problem,
        generator_params={
            "num_loc": target_size,
            **({"capacity": float(capacity)} if problem == "cvrp" else {}),
        },
        seed=1234,
    )
    hparams.update(
        {
            "env": env,
            "num_starts": num_starts,
            "num_augment": 8,
            "generate_default_data": True,
            "batch_size": 1,
            "train_data_size": 1,
            "val_data_size": 1,
            "test_data_size": 1,
        }
    )
    model = POMO(**hparams)
    model = _finalize_model_from_payload(model=model, payload=payload)
    return model, env


def _pomo_loss(*, model: POMO, env, batch: TensorDict, num_starts: int):
    td = env.reset(batch)
    out = model.policy(
        td,
        env,
        phase="train",
        num_starts=num_starts,
        return_actions=False,
        return_entropy=False,
        return_sum_log_likelihood=True,
    )
    reward = unbatchify(out["reward"], num_starts)
    log_likelihood = unbatchify(out["log_likelihood"], num_starts)
    advantage = reward - reward.mean(dim=-1, keepdim=True)
    loss = -(advantage.detach() * log_likelihood).mean()
    return loss, reward


def _validate(
    *,
    model: POMO,
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
    model.eval()
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
                out = model.policy(
                    td,
                    env,
                    phase="test",
                    num_starts=num_starts,
                    return_actions=False,
                )
            values.append(float(-out["reward"].max().detach().cpu()))
    model.train()
    return float(np.mean(values))


def _checkpoint_payload(
    *,
    model: POMO,
    optimizer: torch.optim.Optimizer,
    optimizer_step: int,
    best_cost: float,
    best_step: int,
    config: dict[str, Any],
) -> dict[str, Any]:
    return {
        "format": "scale1000_pomo_continuation_v1",
        "state_dict": model.state_dict(),
        "hyper_parameters": dict(model.hparams),
        "optimizer_state_dict": optimizer.state_dict(),
        "optimizer_step": optimizer_step,
        "epoch": optimizer_step,
        "global_step": optimizer_step,
        "best_cost": best_cost,
        "best_step": best_step,
        "config": config,
    }


def _atomic_save(payload: dict[str, Any], path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def _load_eval_dataset(path: Path, *, problem: str, env) -> TensorDict:
    if problem == "cvrp":
        return env.load_data(path)
    with np.load(path) as payload:
        locs = torch.from_numpy(np.asarray(payload["locs"], dtype=np.float32))
    return TensorDict({"locs": locs}, batch_size=[locs.shape[0]])


def _evaluate_locked(
    *,
    model: POMO,
    env,
    problem: str,
    target_size: int,
    capacity: int,
    dataset_path: Path,
    num_instances: int,
    num_starts: int,
    num_augment: int,
    precision: str,
    device: torch.device,
    checkpoint: Path,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset = _load_eval_dataset(dataset_path, problem=problem, env=env)
    if int(dataset.batch_size[0]) < num_instances:
        raise ValueError("Evaluation dataset has too few rows")
    model.eval()
    rows: list[dict[str, Any]] = []
    for index in range(num_instances):
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
        started = time.perf_counter()
        td = env.reset(dataset[index : index + 1].to(device)).to(device)
        if num_augment > 1:
            td = model.augment(td)
        with torch.inference_mode(), _autocast(device, precision):
            out = model.policy(
                td,
                env,
                phase="test",
                num_starts=num_starts,
                return_actions=False,
            )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            peak = float(torch.cuda.max_memory_allocated(device)) / float(1024**3)
        else:
            peak = None
        row = {
            "method": "pomo",
            "problem": problem,
            "instance_index": index,
            "target_size": target_size,
            "capacity": capacity if problem == "cvrp" else None,
            "num_starts": num_starts,
            "num_augment": num_augment,
            "cost": float(-out["reward"].max().detach().cpu()),
            "elapsed_sec": time.perf_counter() - started,
            "peak_memory_allocated_gib": peak,
        }
        rows.append(row)
        with (output_dir / "per_instance.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(row))
            writer.writeheader()
            writer.writerows(rows)
        print(f"[result] pomo {problem} row={index} cost={row['cost']:.6f}", flush=True)
    costs = np.asarray([row["cost"] for row in rows], dtype=np.float64)
    summary = {
        "protocol": "scale1000_pomo_continuation_locked_eval_v1",
        "method": "pomo",
        "problem": problem,
        "target_size": target_size,
        "capacity": capacity if problem == "cvrp" else None,
        "num_instances": num_instances,
        "num_starts": num_starts,
        "num_augment": num_augment,
        "precision": precision,
        "dataset": str(dataset_path),
        "dataset_sha256": _sha256(dataset_path),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": _sha256(checkpoint),
        "mean_cost": float(costs.mean()),
        "std_cost": float(costs.std(ddof=1)),
        "rows": rows,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"[done-eval] mean_cost={summary['mean_cost']}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Short POMO continuation at routing scale 1000")
    parser.add_argument("--problem", required=True, choices=PROBLEMS)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--target-size", type=int, default=1000)
    parser.add_argument("--capacity", type=int, default=50)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--train-batch-size", type=int, default=1)
    parser.add_argument("--num-starts", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-6)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--precision", default="bf16-mixed")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--validation-size", type=int, default=8)
    parser.add_argument("--validation-every", type=int, default=100)
    parser.add_argument("--validation-starts", type=int, default=None)
    parser.add_argument("--log-every", type=int, default=5)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--evaluate-only", action="store_true")
    parser.add_argument("--evaluation-file", default=None)
    parser.add_argument("--evaluation-instances", type=int, default=None)
    parser.add_argument("--evaluation-starts", type=int, default=None)
    parser.add_argument("--evaluation-augment", type=int, default=8)
    parser.add_argument("--evaluation-output", default=None)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.num_starts < 2 or args.num_starts > args.target_size:
        raise ValueError("num-starts must be in [2, target-size]")
    _seed_everything(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.backends.cuda.enable_cudnn_sdp(False)
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        torch.backends.cuda.enable_math_sdp(True)
    checkpoint = _resolve(args.checkpoint)
    output_dir = _resolve(args.output_dir)
    resume = _resolve(args.resume)
    assert checkpoint is not None and checkpoint.exists()
    assert output_dir is not None
    output_dir.mkdir(parents=True, exist_ok=True)
    model, env = _build_model(
        checkpoint=checkpoint,
        problem=args.problem,
        target_size=args.target_size,
        capacity=args.capacity,
        num_starts=args.num_starts,
    )
    if resume is not None:
        payload = torch.load(resume, map_location="cpu", weights_only=False)
        model.load_state_dict(payload["state_dict"], strict=True)
    model = model.to(device)

    if args.evaluate_only:
        if resume is None or args.evaluation_file is None or args.evaluation_output is None:
            raise ValueError("evaluate-only requires --resume, --evaluation-file and --evaluation-output")
        _evaluate_locked(
            model=model,
            env=env,
            problem=args.problem,
            target_size=args.target_size,
            capacity=args.capacity,
            dataset_path=_resolve(args.evaluation_file),
            num_instances=args.evaluation_instances or (100 if args.problem == "tsp" else 128),
            num_starts=args.evaluation_starts or (args.target_size if args.problem == "tsp" else 100),
            num_augment=args.evaluation_augment,
            precision="32-true",
            device=device,
            checkpoint=resume,
            output_dir=_resolve(args.evaluation_output),
        )
        return 0

    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    start_step = 0
    best_cost = math.inf
    best_step = 0
    if resume is not None:
        payload = torch.load(resume, map_location="cpu", weights_only=False)
        optimizer.load_state_dict(payload["optimizer_state_dict"])
        start_step = int(payload["optimizer_step"])
        best_cost = float(payload["best_cost"])
        best_step = int(payload["best_step"])
    validation_starts = args.validation_starts or (
        args.target_size if args.problem == "tsp" else 100
    )
    config = {
        "protocol": "scale1000_pomo_continuation_v1",
        "problem": args.problem,
        "target_size": args.target_size,
        "capacity": args.capacity if args.problem == "cvrp" else None,
        "optimizer_steps": args.steps,
        "train_batch_size": args.train_batch_size,
        "num_starts": args.num_starts,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "precision": args.precision,
        "seed": args.seed,
        "validation_size": args.validation_size,
        "validation_every": args.validation_every,
        "validation_starts": validation_starts,
        "initial_checkpoint": str(checkpoint),
        "initial_checkpoint_sha256": _sha256(checkpoint),
    }
    (output_dir / "config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    history_path = output_dir / "history.jsonl"

    def validate_and_save(step: int) -> None:
        nonlocal best_cost, best_step
        cost = _validate(
            model=model,
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
        if cost < best_cost:
            best_cost, best_step = cost, step
            payload = _checkpoint_payload(
                model=model,
                optimizer=optimizer,
                optimizer_step=step,
                best_cost=best_cost,
                best_step=best_step,
                config=config,
            )
            _atomic_save(payload, output_dir / "best.ckpt")
        record = {"event": "validation", "optimizer_step": step, "validation_cost": cost}
        with history_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")
        print("[validation] " + json.dumps(record), flush=True)

    if start_step == 0:
        validate_and_save(0)
    model.train()
    for step in range(start_step + 1, args.steps + 1):
        started = time.perf_counter()
        batch = _dynamic_batch(
            problem=args.problem,
            target_size=args.target_size,
            capacity=args.capacity,
            seed=args.seed,
            instance_index=(step - 1) * args.train_batch_size,
            batch_size=args.train_batch_size,
        ).to(device)
        optimizer.zero_grad(set_to_none=True)
        with _autocast(device, args.precision):
            loss, reward = _pomo_loss(
                model=model, env=env, batch=batch, num_starts=args.num_starts
            )
        loss.backward()
        grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm))
        optimizer.step()
        record = {
            "event": "train",
            "optimizer_step": step,
            "elapsed_sec": time.perf_counter() - started,
            "loss": float(loss.detach().cpu()),
            "reward_mean": float(reward.detach().mean().cpu()),
            "grad_norm": grad_norm,
        }
        with history_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")
        if step % args.log_every == 0:
            print("[train] " + json.dumps(record), flush=True)
        if step % args.validation_every == 0 or step == args.steps:
            validate_and_save(step)
        if step % args.log_every == 0 or step == args.steps:
            payload = _checkpoint_payload(
                model=model,
                optimizer=optimizer,
                optimizer_step=step,
                best_cost=best_cost,
                best_step=best_step,
                config=config,
            )
            _atomic_save(payload, output_dir / "last.ckpt")
    summary = {
        "completed_steps": args.steps,
        "best_step": best_step,
        "best_cost": best_cost,
        "output_dir": str(output_dir),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("[done] " + json.dumps(summary), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
