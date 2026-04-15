from __future__ import annotations

import argparse
import gc
import json
import math
import os
import sys
import traceback
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
PTP_ROOT = REPO_ROOT / "PTP"
for _path in (REPO_ROOT, PTP_ROOT):
    path_s = str(_path)
    if path_s not in sys.path:
        sys.path.insert(0, path_s)

from rl4co.envs import FFSPEnv
from rl4co.models import MatNet
from rl4co.models.rl.reinforce.preference_losses import bopo_loss, po_loss, sll_loss
from rl4co.utils.ops import unbatchify


@dataclass
class MemoryPoint:
    allocated_mb: float
    reserved_mb: float
    peak_allocated_mb: float
    peak_reserved_mb: float


def _cuda_mem(device: torch.device) -> MemoryPoint:
    if device.type != "cuda":
        return MemoryPoint(0.0, 0.0, 0.0, 0.0)
    return MemoryPoint(
        allocated_mb=float(torch.cuda.memory_allocated(device)) / (1024.0**2),
        reserved_mb=float(torch.cuda.memory_reserved(device)) / (1024.0**2),
        peak_allocated_mb=float(torch.cuda.max_memory_allocated(device)) / (1024.0**2),
        peak_reserved_mb=float(torch.cuda.max_memory_reserved(device)) / (1024.0**2),
    )


def _reset_cuda_stats(device: torch.device) -> None:
    if device.type != "cuda":
        return
    with torch.cuda.device(device):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)


def _cleanup_cuda(device: torch.device) -> None:
    gc.collect()
    if device.type != "cuda":
        return
    with torch.cuda.device(device):
        torch.cuda.synchronize(device)
        torch.cuda.empty_cache()


def _autocast_context(device: torch.device, precision: str):
    mode = str(precision or "32-true").strip().lower()
    if device.type != "cuda":
        return nullcontext()
    if mode == "16-mixed":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    if mode == "bf16-mixed":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def _make_env(device: torch.device) -> FFSPEnv:
    env = FFSPEnv(
        generator_params={
            "num_stage": 3,
            "num_machine": 4,
            "num_job": 100,
            "flatten_stages": False,
        }
    )
    return env.to(device)


def _make_model(
    *,
    env: FFSPEnv,
    loss_type: str,
    batch_size: int,
    num_starts: int,
) -> MatNet:
    baseline = "no" if loss_type == "rl_loss" else "shared"
    return MatNet(
        env=env,
        baseline=baseline,
        loss_type=loss_type,
        po_impl="exponential",
        alpha=1.0,
        sll_impl="sll",
        sll_temperature=1.0,
        bopo_pair_mode="anchor_best",
        bopo_select_strategy="paper",
        bopo_select_k=6,
        bopo_select_quantile=0.5,
        num_starts=num_starts,
        batch_size=batch_size,
        train_data_size=batch_size,
        val_data_size=batch_size,
        test_data_size=batch_size,
        generate_default_data=False,
        log_on_step=False,
        policy_params={
            "embed_dim": 256,
            "num_encoder_layers": 3,
            "num_heads": 16,
            "normalization": "instance",
            "use_graph_context": False,
            "feedforward_hidden": 512,
            "train_decode_type": "sampling",
            "val_decode_type": "greedy",
            "test_decode_type": "greedy",
        },
        metrics={
            "train": ["loss", "reward", "max_reward"],
            "val": ["reward", "max_reward"],
            "test": ["reward", "max_reward"],
            "log_on_step": False,
        },
    )


def _summarize_exception(exc: BaseException) -> dict[str, Any]:
    return {
        "type": type(exc).__name__,
        "message": str(exc),
        "traceback": traceback.format_exc(limit=12),
        "is_oom": "out of memory" in str(exc).lower(),
    }


def _make_batch(env: FFSPEnv, batch_size: int, device: torch.device):
    batch = env.generator([int(batch_size)])
    try:
        return batch.to(device)
    except Exception:
        return batch


def _forward_and_loss(
    *,
    model: MatNet,
    env: FFSPEnv,
    batch: Any,
    num_starts: int,
) -> tuple[torch.Tensor, dict[str, Any]]:
    td = env.reset(batch)
    policy_kwargs: dict[str, Any] = {"phase": "train", "num_starts": int(num_starts)}
    if model.loss_type == "bopo_loss":
        policy_kwargs["return_actions"] = True
    out = model.policy(td, env, **policy_kwargs)

    reward = unbatchify(out["reward"], (0, num_starts))
    raw_log_likelihood = out["log_likelihood"]
    log_likelihood = unbatchify(raw_log_likelihood, (0, num_starts))
    out["log_likelihood"] = log_likelihood

    if model.loss_type == "bopo_loss" and isinstance(out.get("actions"), torch.Tensor):
        out["actions"] = unbatchify(out["actions"], (0, num_starts))

    model.calculate_loss(td, batch, out, reward, log_likelihood)
    return out["loss"], out


def _run_full_step(
    *,
    device: torch.device,
    loss_type: str,
    batch_size: int,
    num_starts: int,
    precision: str,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "mode": "full_step",
        "loss_type": loss_type,
        "batch_size": int(batch_size),
        "num_starts": int(num_starts),
        "precision": str(precision),
        "status": "ok",
        "stages": {},
    }

    env = None
    model = None
    optimizer = None
    batch = None
    loss = None
    out = None

    try:
        _cleanup_cuda(device)
        env = _make_env(device)
        model = _make_model(env=env, loss_type=loss_type, batch_size=batch_size, num_starts=num_starts).to(device)
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-6)
        result["stages"]["after_model_init"] = asdict(_cuda_mem(device))

        _reset_cuda_stats(device)
        batch = _make_batch(env, batch_size, device)
        result["stages"]["after_batch_materialize"] = asdict(_cuda_mem(device))

        optimizer.zero_grad(set_to_none=True)
        with _autocast_context(device, precision):
            loss, out = _forward_and_loss(model=model, env=env, batch=batch, num_starts=num_starts)
        result["loss_value"] = float(loss.detach().float().item())
        result["stages"]["after_forward_and_loss"] = asdict(_cuda_mem(device))

        loss.backward()
        result["stages"]["after_backward"] = asdict(_cuda_mem(device))

        optimizer.step()
        result["stages"]["after_optimizer_step"] = asdict(_cuda_mem(device))
        result["peak"] = asdict(_cuda_mem(device))
        if isinstance(out, dict) and "reward" in out:
            try:
                result["reward_mean"] = float(out["reward"].detach().float().mean().item())
            except Exception:
                pass
    except RuntimeError as exc:
        result["status"] = "runtime_error"
        result["error"] = _summarize_exception(exc)
        result["peak"] = asdict(_cuda_mem(device))
    except Exception as exc:  # noqa: BLE001
        result["status"] = "error"
        result["error"] = _summarize_exception(exc)
        result["peak"] = asdict(_cuda_mem(device))
    finally:
        del out
        del loss
        del batch
        del optimizer
        del model
        del env
        _cleanup_cuda(device)
    return result


def _run_multi_step(
    *,
    device: torch.device,
    loss_type: str,
    batch_size: int,
    num_starts: int,
    precision: str,
    steps: int,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "mode": "multi_step",
        "loss_type": loss_type,
        "batch_size": int(batch_size),
        "num_starts": int(num_starts),
        "precision": str(precision),
        "steps": int(steps),
        "status": "ok",
        "per_step": [],
    }

    env = None
    model = None
    optimizer = None

    try:
        _cleanup_cuda(device)
        env = _make_env(device)
        model = _make_model(env=env, loss_type=loss_type, batch_size=batch_size, num_starts=num_starts).to(device)
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-6)
        result["stages"] = {"after_model_init": asdict(_cuda_mem(device))}

        for step_idx in range(int(steps)):
            batch = None
            out = None
            loss = None
            try:
                _reset_cuda_stats(device)
                batch = _make_batch(env, batch_size, device)
                optimizer.zero_grad(set_to_none=True)
                with _autocast_context(device, precision):
                    loss, out = _forward_and_loss(model=model, env=env, batch=batch, num_starts=num_starts)
                loss.backward()
                optimizer.step()

                mem = _cuda_mem(device)
                result["per_step"].append(
                    {
                        "step": int(step_idx),
                        "loss_value": float(loss.detach().float().item()),
                        "allocated_mb": float(mem.allocated_mb),
                        "reserved_mb": float(mem.reserved_mb),
                        "peak_allocated_mb": float(mem.peak_allocated_mb),
                        "peak_reserved_mb": float(mem.peak_reserved_mb),
                    }
                )
            except RuntimeError as exc:
                result["status"] = "runtime_error"
                result["error"] = _summarize_exception(exc)
                result["failed_step"] = int(step_idx)
                result["peak"] = asdict(_cuda_mem(device))
                break
            finally:
                del out
                del loss
                del batch
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
        if result["status"] == "ok":
            result["peak"] = asdict(_cuda_mem(device))
            reserved_values = [float(item["reserved_mb"]) for item in result["per_step"]]
            if reserved_values:
                result["reserved_growth_mb"] = float(max(reserved_values) - min(reserved_values))
    except Exception as exc:  # noqa: BLE001
        result["status"] = "error"
        result["error"] = _summarize_exception(exc)
        result["peak"] = asdict(_cuda_mem(device))
    finally:
        del optimizer
        del model
        del env
        _cleanup_cuda(device)
    return result


def _synthetic_rl_loss(
    reward: torch.Tensor,
    log_likelihood: torch.Tensor,
) -> torch.Tensor:
    baseline = reward.mean(dim=1, keepdim=True)
    advantage = reward - baseline
    return -(advantage * log_likelihood).mean()


def _run_loss_only(
    *,
    device: torch.device,
    loss_type: str,
    batch_size: int,
    num_starts: int,
    precision: str,
    sequence_length: int,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "mode": "loss_only",
        "loss_type": loss_type,
        "batch_size": int(batch_size),
        "num_starts": int(num_starts),
        "precision": str(precision),
        "status": "ok",
        "stages": {},
    }

    reward = None
    log_likelihood = None
    loss = None

    try:
        _cleanup_cuda(device)
        _reset_cuda_stats(device)

        reward = torch.randn(batch_size, num_starts, device=device, dtype=torch.float32)
        log_likelihood = torch.randn(
            batch_size,
            num_starts,
            device=device,
            dtype=torch.float32,
            requires_grad=True,
        )
        result["stages"]["after_inputs"] = asdict(_cuda_mem(device))

        with _autocast_context(device, precision):
            if loss_type == "rl_loss":
                loss = _synthetic_rl_loss(reward, log_likelihood)
            elif loss_type == "po_loss":
                loss, pref_rate = po_loss(reward, log_likelihood, alpha=1.0, impl="exponential")
                result["pref_rate"] = float(pref_rate.detach().float().item())
            elif loss_type == "sll_loss":
                loss = sll_loss(reward, log_likelihood, alpha=1.0, impl="sll", temperature=1.0)
            elif loss_type == "bopo_loss":
                seq_len = torch.full_like(log_likelihood, float(sequence_length))
                loss, pair_count = bopo_loss(
                    reward,
                    log_likelihood,
                    alpha=1.0,
                    pair_mode="anchor_best",
                    select_strategy="paper",
                    select_k=6,
                    select_quantile=0.5,
                    sequence_length=seq_len,
                )
                result["pair_count"] = float(pair_count.detach().float().item())
            else:
                raise ValueError(f"Unsupported loss_type for loss-only probe: {loss_type}")
        result["loss_value"] = float(loss.detach().float().item())
        result["stages"]["after_loss"] = asdict(_cuda_mem(device))

        loss.backward()
        result["stages"]["after_backward"] = asdict(_cuda_mem(device))
        result["peak"] = asdict(_cuda_mem(device))
    except RuntimeError as exc:
        result["status"] = "runtime_error"
        result["error"] = _summarize_exception(exc)
        result["peak"] = asdict(_cuda_mem(device))
    except Exception as exc:  # noqa: BLE001
        result["status"] = "error"
        result["error"] = _summarize_exception(exc)
        result["peak"] = asdict(_cuda_mem(device))
    finally:
        del loss
        del log_likelihood
        del reward
        _cleanup_cuda(device)
    return result


def _parse_csv_ints(raw: str) -> list[int]:
    out: list[int] = []
    for token in str(raw or "").split(","):
        token = token.strip()
        if token:
            out.append(int(token))
    if not out:
        raise ValueError("Expected at least one integer.")
    return out


def _parse_csv_strs(raw: str) -> list[str]:
    out = [str(token).strip() for token in str(raw or "").split(",") if str(token).strip()]
    if not out:
        raise ValueError("Expected at least one token.")
    return out


def _best_stage_peak(record: dict[str, Any]) -> float:
    peak = ((record.get("peak") or {}).get("peak_allocated_mb"))
    if peak is None:
        return math.nan
    return float(peak)


def _stage_alloc(record: dict[str, Any], stage_name: str) -> float | None:
    stage = (record.get("stages") or {}).get(stage_name) or {}
    value = stage.get("allocated_mb")
    if value is None:
        return None
    return float(value)


def _largest_stage_jump(record: dict[str, Any]) -> tuple[str, float | None]:
    order = [
        "after_model_init",
        "after_batch_materialize",
        "after_forward_and_loss",
        "after_backward",
        "after_optimizer_step",
    ]
    prev_name = None
    prev_value = None
    best_name = "-"
    best_delta = None
    for name in order:
        value = _stage_alloc(record, name)
        if value is None:
            continue
        if prev_value is not None:
            delta = value - prev_value
            if best_delta is None or delta > best_delta:
                best_delta = delta
                best_name = f"{prev_name}->{name}"
        prev_name = name
        prev_value = value
    return best_name, best_delta


def _print_summary(records: list[dict[str, Any]]) -> None:
    print()
    print("Summary")
    print("-" * 120)
    header = (
        f"{'mode':<10} {'loss':<10} {'bs':>4} {'starts':>6} {'status':<14} "
        f"{'fwd_mb':>10} {'bwd_mb':>10} {'peak_alloc_mb':>14} {'peak_res_mb':>12} "
        f"{'max_jump':>26} {'loss':>12}"
    )
    print(header)
    print("-" * len(header))
    for rec in records:
        peak = rec.get("peak") or {}
        peak_alloc = peak.get("peak_allocated_mb")
        peak_res = peak.get("peak_reserved_mb")
        loss_value = rec.get("loss_value")
        fwd_mb = _stage_alloc(rec, "after_forward_and_loss")
        bwd_mb = _stage_alloc(rec, "after_backward")
        jump_name, jump_delta = _largest_stage_jump(rec)
        growth_mb = rec.get("reserved_growth_mb")
        print(
            f"{str(rec.get('mode','')):<10} "
            f"{str(rec.get('loss_type','')):<10} "
            f"{int(rec.get('batch_size',0)):>4} "
            f"{int(rec.get('num_starts',0)):>6} "
            f"{str(rec.get('status','')):<14} "
            f"{(f'{float(fwd_mb):.1f}' if fwd_mb is not None else '-'):>10} "
            f"{(f'{float(bwd_mb):.1f}' if bwd_mb is not None else '-'):>10} "
            f"{(f'{float(peak_alloc):.1f}' if peak_alloc is not None else '-'):>14} "
            f"{(f'{float(peak_res):.1f}' if peak_res is not None else '-'):>12} "
            f"{(f'{jump_name}:{float(jump_delta):.1f}' if jump_delta is not None else '-'):>26} "
            f"{(f'{float(loss_value):.6f}' if loss_value is not None else '-'):>12}"
        )
        if growth_mb is not None:
            print(f"{'':<56} reserved_growth_mb={float(growth_mb):.1f}")
    print("-" * len(header))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Diagnose why FFSP100 MatNet SLL training OOMs by comparing memory across losses and phases."
    )
    parser.add_argument("--gpu", type=int, default=0, help="CUDA device index.")
    parser.add_argument("--batch-sizes", type=str, default="32", help="Comma-separated batch sizes to probe.")
    parser.add_argument("--losses", type=str, default="rl_loss,po_loss,sll_loss,bopo_loss", help="Comma-separated losses to probe.")
    parser.add_argument("--num-starts", type=int, default=24, help="POMO/MatNet multistart count.")
    parser.add_argument("--precision", type=str, default="32-true", help="32-true | 16-mixed | bf16-mixed")
    parser.add_argument("--sequence-length", type=int, default=300, help="Approx rollout length for FFSP100 BOPO loss-only probe.")
    parser.add_argument("--loop-steps", type=int, default=0, help="If > 0, run a consecutive multi-step train probe to detect memory creep.")
    parser.add_argument("--skip-loss-only", action="store_true", help="Skip synthetic loss-only probe.")
    parser.add_argument("--skip-full-step", action="store_true", help="Skip full train-step probe.")
    parser.add_argument("--json-out", type=str, default="", help="Optional JSON output path.")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("CUDA is not available; this script is intended for GPU OOM diagnosis.", file=sys.stderr)
        return 2

    device = torch.device(f"cuda:{int(args.gpu)}")
    torch.cuda.set_device(device)

    losses = _parse_csv_strs(args.losses)
    batch_sizes = _parse_csv_ints(args.batch_sizes)

    print(f"Using device: {device}")
    props = torch.cuda.get_device_properties(device)
    total_mem_gb = float(props.total_memory) / (1024.0**3)
    print(f"GPU: {props.name} | total_memory_gb={total_mem_gb:.2f}")
    print(f"Losses: {losses}")
    print(f"Batch sizes: {batch_sizes}")
    print(f"Num starts: {args.num_starts}")
    print(f"Precision: {args.precision}")

    all_records: list[dict[str, Any]] = []
    for batch_size in batch_sizes:
        for loss_type in losses:
            print()
            print(f"[probe] loss={loss_type} batch_size={batch_size} mode=loss_only")
            if not args.skip_loss_only:
                rec = _run_loss_only(
                    device=device,
                    loss_type=loss_type,
                    batch_size=batch_size,
                    num_starts=args.num_starts,
                    precision=args.precision,
                    sequence_length=args.sequence_length,
                )
                all_records.append(rec)
                print(json.dumps(rec, indent=2))

            print()
            print(f"[probe] loss={loss_type} batch_size={batch_size} mode=full_step")
            if not args.skip_full_step:
                rec = _run_full_step(
                    device=device,
                    loss_type=loss_type,
                    batch_size=batch_size,
                    num_starts=args.num_starts,
                    precision=args.precision,
                )
                all_records.append(rec)
                print(json.dumps(rec, indent=2))

            if int(args.loop_steps) > 0:
                print()
                print(f"[probe] loss={loss_type} batch_size={batch_size} mode=multi_step steps={int(args.loop_steps)}")
                rec = _run_multi_step(
                    device=device,
                    loss_type=loss_type,
                    batch_size=batch_size,
                    num_starts=args.num_starts,
                    precision=args.precision,
                    steps=int(args.loop_steps),
                )
                all_records.append(rec)
                print(json.dumps(rec, indent=2))

    _print_summary(all_records)

    if args.json_out:
        out_path = Path(args.json_out).expanduser()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(all_records, indent=2), encoding="utf-8")
        print(f"Wrote JSON report to {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
