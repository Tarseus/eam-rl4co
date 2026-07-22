from __future__ import annotations

import argparse
import gc
import json
import math
import sys
import time
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from tensordict import TensorDict


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rl4co.envs import get_env
from rl4co.models import POMO
from rl4co.models.zoo.pomo.po4cops_tsp_policy import PO4COPsTSPPolicy
from scripts.eval_downloaded_routing_checkpoints import (
    _patch_legacy_policy_object,
    checkpoint_hparams,
)


METHODS = ("po", "bopo", "usw", "asw")
DEFAULT_PAIR_PATHS = {
    "usw": REPO_ROOT
    / "runs"
    / "pref_loss_tsp100_discovery"
    / "20260317-131507"
    / "best_pair.json",
    "asw": REPO_ROOT
    / "runs"
    / "pref_builder_weight_search_tsp100"
    / "20260414-113757"
    / "best_pair.json",
}


@dataclass(frozen=True)
class ProbeResult:
    method: str
    loss: float
    grad_norm: float
    parameter_delta: float
    elapsed_sec: float
    peak_memory_allocated_gib: float
    peak_memory_reserved_gib: float
    replay_error: float | None
    finite: bool


def _parse_methods(raw: str) -> list[str]:
    methods = [item.strip().lower() for item in str(raw).split(",") if item.strip()]
    unknown = sorted(set(methods) - set(METHODS))
    if unknown:
        raise ValueError(f"Unknown methods: {unknown}")
    if not methods:
        raise ValueError("At least one method is required")
    return methods


def _autocast_context(device: torch.device, precision: str):
    normalized = str(precision).strip().lower()
    if normalized in {"32", "fp32", "float32", "32-true"}:
        return nullcontext()
    if device.type != "cuda":
        raise ValueError(f"precision={precision} requires CUDA")
    if normalized in {"16", "fp16", "float16", "16-mixed"}:
        return torch.autocast(
            device_type="cuda",
            dtype=torch.float16,
            cache_enabled=False,
        )
    if normalized in {"bf16", "bfloat16", "bf16-mixed"}:
        return torch.autocast(
            device_type="cuda",
            dtype=torch.bfloat16,
            cache_enabled=False,
        )
    raise ValueError(f"Unsupported precision: {precision}")


def _resolve_path(path: str | Path) -> Path:
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = REPO_ROOT / candidate
    return candidate.resolve()


def _build_model(
    *,
    method: str,
    checkpoint_path: Path,
    target_size: int,
    num_starts: int,
    seed: int,
    usw_pair_path: Path,
    asw_pair_path: Path,
) -> tuple[POMO, Any]:
    hparams = checkpoint_hparams(checkpoint_path)
    payload = hparams.pop("_checkpoint_payload")
    hparams["policy"] = _patch_legacy_policy_object(hparams.get("policy"))
    env = get_env(
        "tsp",
        generator_params={"num_loc": int(target_size)},
        seed=int(seed),
    )
    hparams.update(
        {
            "env": env,
            "num_starts": int(num_starts),
            "num_augment": 8,
            "loss_type": "po_loss" if method == "po" else "bopo_loss",
            "alpha": 0.05,
            "po_impl": "exponential",
            "bopo_pair_mode": "anchor_best",
            "bopo_select_strategy": "paper",
            "bopo_select_k": int(num_starts),
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
        hparams["loss_type"] = "free_loss"
        hparams["pref_pair_json_path"] = str(
            usw_pair_path if method == "usw" else asw_pair_path
        )
    model = POMO(**hparams)
    missing, unexpected = model.load_state_dict(payload["state_dict"], strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f"Checkpoint state mismatch: missing={missing}, unexpected={unexpected}"
        )
    if not isinstance(model.policy, PO4COPsTSPPolicy):
        raise TypeError(f"Expected PO4COPsTSPPolicy, got {type(model.policy).__name__}")
    return model, env


def _gradient_norm(parameters) -> float:
    squared_norm = 0.0
    for parameter in parameters:
        if parameter.grad is None:
            continue
        squared_norm += float(parameter.grad.detach().float().square().sum().cpu())
    return math.sqrt(squared_norm)


def _probe_method(
    *,
    method: str,
    checkpoint_path: Path,
    locs: torch.Tensor,
    target_size: int,
    num_starts: int,
    seed: int,
    learning_rate: float,
    device: torch.device,
    precision: str,
    usw_pair_path: Path,
    asw_pair_path: Path,
) -> ProbeResult:
    model, _ = _build_model(
        method=method,
        checkpoint_path=checkpoint_path,
        target_size=target_size,
        num_starts=num_starts,
        seed=seed,
        usw_pair_path=usw_pair_path,
        asw_pair_path=asw_pair_path,
    )
    model = model.to(device)
    model.train()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(learning_rate),
        weight_decay=1e-6,
    )
    first_parameter = next(parameter for parameter in model.parameters() if parameter.requires_grad)
    parameter_before = first_parameter.detach().cpu().clone()
    batch = TensorDict(
        {"locs": locs.to(device)},
        batch_size=[locs.shape[0]],
    ).to(device)

    optimizer.zero_grad(set_to_none=True)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
    torch.manual_seed(int(seed))
    if device.type == "cuda":
        torch.cuda.manual_seed_all(int(seed))
    started_at = time.perf_counter()
    with _autocast_context(device, precision):
        step_out = model.shared_step(batch, 0, "train")
        loss = step_out["loss"]
    loss.backward()
    grad_norm = _gradient_norm(model.parameters())
    optimizer.step()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed_sec = time.perf_counter() - started_at
    parameter_delta = float(
        (first_parameter.detach().cpu() - parameter_before).float().norm()
    )
    if device.type == "cuda":
        gib = float(1024**3)
        peak_allocated = float(torch.cuda.max_memory_allocated(device)) / gib
        peak_reserved = float(torch.cuda.max_memory_reserved(device)) / gib
    else:
        peak_allocated = 0.0
        peak_reserved = 0.0
    replay_error_value = step_out.get("memory_efficient_replay_error")
    if isinstance(replay_error_value, torch.Tensor):
        replay_error = float(replay_error_value.detach().cpu())
    else:
        replay_error = None
    finite = bool(
        torch.isfinite(loss.detach()).all()
        and math.isfinite(grad_norm)
        and math.isfinite(parameter_delta)
    )
    result = ProbeResult(
        method=method,
        loss=float(loss.detach().cpu()),
        grad_norm=grad_norm,
        parameter_delta=parameter_delta,
        elapsed_sec=elapsed_sec,
        peak_memory_allocated_gib=peak_allocated,
        peak_memory_reserved_gib=peak_reserved,
        replay_error=replay_error,
        finite=finite,
    )
    del optimizer, model, batch, loss, step_out
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="downloads/tsp100/po/checkpoint.ckpt")
    parser.add_argument("--methods", default=",".join(METHODS))
    parser.add_argument("--target-size", type=int, default=1000)
    parser.add_argument("--num-starts", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--precision", default="bf16-mixed")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--usw-pair-path",
        default=str(DEFAULT_PAIR_PATHS["usw"].relative_to(REPO_ROOT)),
    )
    parser.add_argument(
        "--asw-pair-path",
        default=str(DEFAULT_PAIR_PATHS["asw"].relative_to(REPO_ROOT)),
    )
    parser.add_argument(
        "--output",
        default="logs/tsp1000_dynamic/probe_tsp1000_dynamic_training.json",
    )
    args = parser.parse_args()

    methods = _parse_methods(args.methods)
    checkpoint_path = _resolve_path(args.checkpoint)
    usw_pair_path = _resolve_path(args.usw_pair_path)
    asw_pair_path = _resolve_path(args.asw_pair_path)
    for required_path in (checkpoint_path, usw_pair_path, asw_pair_path):
        if not required_path.is_file():
            raise FileNotFoundError(required_path)
    if args.target_size < 2 or args.num_starts < 2 or args.batch_size < 1:
        raise ValueError("target-size, num-starts, and batch-size must be positive")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    instance_generator = torch.Generator(device="cpu")
    instance_generator.manual_seed(int(args.seed) + int(args.target_size) * 1_000_003)
    locs = torch.rand(
        (int(args.batch_size), int(args.target_size), 2),
        generator=instance_generator,
        dtype=torch.float32,
    )

    results = []
    for method in methods:
        print(f"[probe] method={method} size={args.target_size} starts={args.num_starts}", flush=True)
        result = _probe_method(
            method=method,
            checkpoint_path=checkpoint_path,
            locs=locs,
            target_size=args.target_size,
            num_starts=args.num_starts,
            seed=args.seed,
            learning_rate=args.learning_rate,
            device=device,
            precision=args.precision,
            usw_pair_path=usw_pair_path,
            asw_pair_path=asw_pair_path,
        )
        results.append(result)
        print(json.dumps(asdict(result), ensure_ascii=False), flush=True)

    output_path = _resolve_path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "protocol": "tsp1000_dynamic_one_step_v1",
        "checkpoint": str(checkpoint_path),
        "target_size": int(args.target_size),
        "num_starts": int(args.num_starts),
        "batch_size": int(args.batch_size),
        "seed": int(args.seed),
        "learning_rate": float(args.learning_rate),
        "precision": str(args.precision),
        "device": str(device),
        "usw_pair_path": str(usw_pair_path),
        "asw_pair_path": str(asw_pair_path),
        "results": [asdict(result) for result in results],
    }
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[probe] wrote {output_path}", flush=True)
    if not all(result.finite and result.parameter_delta > 0 for result in results):
        raise RuntimeError("At least one method failed the trainability probe")


if __name__ == "__main__":
    main()
