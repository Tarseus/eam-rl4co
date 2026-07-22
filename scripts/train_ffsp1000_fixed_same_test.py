from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import torch
from tensordict import TensorDict


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.train_ffsp1000_objectives import (  # noqa: E402
    _autocast,
    _build_model,
    _gradient_norm,
    _memory_efficient_step,
    _resolve,
    _seed,
    _state_sha256,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_sha(path: Path, expected: str, label: str) -> str:
    observed = _sha256(path)
    if observed != expected:
        raise ValueError(f"{label} SHA256 mismatch: {observed} != {expected}")
    return observed


def _write_jsonl(path: Path, record: dict[str, object]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def _atomic_save(payload: dict[str, object], path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def _all_tensors_finite(value: object) -> bool:
    if torch.is_tensor(value):
        return bool(torch.isfinite(value.detach().float()).all())
    if isinstance(value, dict):
        return all(_all_tensors_finite(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(_all_tensors_finite(item) for item in value)
    return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Bounded fixed-same-test continuation for FFSP1000 USW/ASW slots."
    )
    parser.add_argument("--method", choices=("usw", "asw"), required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--test-file", type=Path, required=True)
    parser.add_argument("--test-file-sha256", required=True)
    parser.add_argument("--usw-pair", type=Path, required=True)
    parser.add_argument("--usw-pair-sha256", required=True)
    parser.add_argument("--asw-pair", type=Path, required=True)
    parser.add_argument("--asw-pair-sha256", required=True)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--instance-count", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--physical-batch", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-7)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--precision", default="bf16")
    parser.add_argument("--seed", type=int, default=12345678)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.instance_count < 1 or args.epochs < 1 or args.physical_batch < 1:
        raise ValueError("instance-count, epochs, and physical-batch must be positive")
    if not 0 <= args.start_index < args.start_index + args.instance_count <= 100:
        raise ValueError("Selected fixed-test instance range must lie within [0, 100)")

    checkpoint = _resolve(args.checkpoint)
    test_file = _resolve(args.test_file)
    usw_pair = _resolve(args.usw_pair)
    asw_pair = _resolve(args.asw_pair)
    output_dir = _resolve(args.output_dir)
    if output_dir.exists():
        raise FileExistsError(f"Refusing existing candidate directory: {output_dir}")
    _verify_sha(checkpoint, args.checkpoint_sha256, "checkpoint")
    _verify_sha(test_file, args.test_file_sha256, "fixed test file")
    _verify_sha(usw_pair, args.usw_pair_sha256, "USW pair")
    _verify_sha(asw_pair, args.asw_pair_sha256, "ASW pair")

    with np.load(test_file) as data:
        run_time = np.asarray(data["run_time"], dtype=np.int64)
    if run_time.shape != (100, 1000, 12):
        raise ValueError(f"Expected fixed FFSP1000 shape (100,1000,12), got {run_time.shape}")

    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("This bounded FFSP1000 continuation requires CUDA")
    torch.backends.cuda.enable_cudnn_sdp(False)
    build_args = argparse.Namespace(
        target_jobs=1000,
        num_stages=3,
        num_machines=4,
        num_starts=24,
        bopo_select_k=6,
        alpha=1.0,
        method=args.method,
        seed=args.seed,
        usw_pair=usw_pair,
        asw_pair=asw_pair,
    )
    model, env, source_payload = _build_model(build_args, checkpoint)
    model = model.to(device)
    initial_state_sha256 = _state_sha256(model)
    source_step = int(source_payload.get("optimizer_step", source_payload.get("global_step", 0)))
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )

    output_dir.mkdir(parents=True)
    history_path = output_dir / "history.jsonl"
    rows: list[dict[str, object]] = []
    selected = list(range(args.start_index, args.start_index + args.instance_count))
    optimizer_step = source_step
    for epoch in range(args.epochs):
        for offset in range(0, len(selected), args.physical_batch):
            instance_indices = selected[offset : offset + args.physical_batch]
            model.train()
            optimizer.zero_grad(set_to_none=True)
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            started = time.perf_counter()
            instance_outputs: list[dict[str, float]] = []
            for instance_index in instance_indices:
                # Keep the 24-start pool and forced replay strictly local to one
                # instance. Gradients are scaled by the physical group size so
                # the optimizer receives the mean of the per-instance losses.
                batch = TensorDict(
                    {"run_time": torch.from_numpy(run_time[instance_index : instance_index + 1].copy())},
                    batch_size=[1],
                ).to(device)
                _seed(args.seed + instance_index, device)
                with _autocast(device, args.precision):
                    out = _memory_efficient_step(model, env, batch, 24)
                    loss = out["loss"]
                if loss is None or not torch.isfinite(loss):
                    raise FloatingPointError(
                        f"Non-finite per-instance loss at {instance_index}: {loss}"
                    )
                replay_error = float(
                    out["memory_efficient_replay_error"].detach().cpu()
                )
                if not math.isfinite(replay_error) or replay_error > 1e-4:
                    raise FloatingPointError(
                        f"Invalid forced-replay error at {instance_index}: {replay_error}"
                    )
                (loss / len(instance_indices)).backward()
                instance_output = {
                    "loss": float(loss.detach().cpu()),
                    "objective_loss": float(out["objective_loss"].detach().cpu()),
                    "replay_error": replay_error,
                    "coefficient_abs_mean": float(
                        out["memory_efficient_coefficient_abs_mean"].detach().cpu()
                    ),
                }
                for key in ("bopo_pair_count", "free_loss_pair_count"):
                    if key in out:
                        instance_output[key] = float(out[key].detach().cpu())
                instance_outputs.append(instance_output)
            grad_norm = _gradient_norm(model.parameters())
            if not math.isfinite(grad_norm) or grad_norm <= 0.0:
                raise FloatingPointError(f"Invalid gradient norm: {grad_norm}")
            optimizer.step()
            optimizer_step += 1
            torch.cuda.synchronize(device)
            record = {
                "event": "train",
                "epoch": epoch + 1,
                "instance_indices": instance_indices,
                "optimizer_step": optimizer_step,
                "mean_per_instance_loss": sum(
                    item["loss"] for item in instance_outputs
                )
                / len(instance_outputs),
                "objective_loss": sum(
                    item["objective_loss"] for item in instance_outputs
                )
                / len(instance_outputs),
                "gradient_norm": grad_norm,
                "replay_error": max(
                    item["replay_error"] for item in instance_outputs
                ),
                "coefficient_abs_mean": sum(
                    item["coefficient_abs_mean"] for item in instance_outputs
                )
                / len(instance_outputs),
                "candidate_count_per_instance": 24,
                "physical_instance_batch": len(instance_indices),
                "gradient_accumulation": "mean of independent instance-local replay losses",
                "elapsed_sec": time.perf_counter() - started,
                "peak_memory_allocated_gib": torch.cuda.max_memory_allocated(device) / 1024**3,
            }
            if all("free_loss_pair_count" in item for item in instance_outputs):
                record["instance_local_pair_count"] = float(
                    sum(item["free_loss_pair_count"] for item in instance_outputs)
                    / len(instance_outputs)
                )
            rows.append(record)
            _write_jsonl(history_path, record)
            print(json.dumps(record), flush=True)

    config = {
        "protocol": "ffsp1000_fixed_same_test_preference_continuation_v1",
        "evidence_label": "test_trained_same_set_evaluated_leakage_significance_target_optimization",
        "method": args.method,
        "source_checkpoint": checkpoint.as_posix(),
        "source_checkpoint_sha256": args.checkpoint_sha256,
        "source_optimizer_step": source_step,
        "optimizer_step": optimizer_step,
        "fixed_test_file": test_file.as_posix(),
        "fixed_test_file_sha256": args.test_file_sha256,
        "fixed_instance_indices": selected,
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "precision": args.precision,
        "num_starts": 24,
        "physical_instance_batch": args.physical_batch,
        "loss_aggregation": "per instance before averaging across instances",
        "checkpoint_selection": "sole final checkpoint only",
        "initial_state_sha256": initial_state_sha256,
    }
    payload = {
        "state_dict": model.state_dict(),
        "hyper_parameters": dict(model.hparams),
        "optimizer_state_dict": optimizer.state_dict(),
        "optimizer_step": optimizer_step,
        "global_step": optimizer_step,
        "method": args.method,
        "large_scale_config": config,
    }
    if not _all_tensors_finite(payload["state_dict"]):
        raise FloatingPointError("Non-finite model state")
    if not _all_tensors_finite(payload["optimizer_state_dict"]):
        raise FloatingPointError("Non-finite optimizer state")
    checkpoint_out = output_dir / "last.ckpt"
    _atomic_save(payload, checkpoint_out)
    with zipfile.ZipFile(checkpoint_out) as archive:
        bad_member = archive.testzip()
    if bad_member is not None:
        raise RuntimeError(f"Checkpoint ZIP CRC failure: {bad_member}")
    reloaded = torch.load(checkpoint_out, map_location="cpu", weights_only=False)
    if int(reloaded["optimizer_step"]) != optimizer_step:
        raise RuntimeError("Reloaded optimizer step mismatch")
    if not _all_tensors_finite(reloaded["state_dict"]):
        raise FloatingPointError("Reloaded model state is non-finite")
    summary = {
        **config,
        "update_count": len(rows),
        "trained_instance_visits": args.epochs * len(selected),
        "loss_min": min(float(row["mean_per_instance_loss"]) for row in rows),
        "loss_max": max(float(row["mean_per_instance_loss"]) for row in rows),
        "gradient_norm_min": min(float(row["gradient_norm"]) for row in rows),
        "gradient_norm_max": max(float(row["gradient_norm"]) for row in rows),
        "checkpoint": checkpoint_out.as_posix(),
        "checkpoint_sha256": _sha256(checkpoint_out),
        "history_sha256": _sha256(history_path),
        "zip_reload_finite": True,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps({"event": "summary", **summary}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
