from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import sys
import time
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rl4co.models.zoo.mgl_jssp.data import load_instance
from rl4co.models.zoo.mgl_jssp.sampling import JobShopStates

try:
    from scripts.train_jssp_large_objectives_complete_rows import _build_model, _resolve
except ImportError:
    from scripts.train_jssp_large_objectives import _build_model, _resolve


EVIDENCE_LABEL = (
    "test_trained_same_set_evaluated_leakage_significance_target_optimization"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _state_sha256(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        digest.update(name.encode("utf-8"))
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _ordered_files_sha256(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(bytes.fromhex(_sha256(path)))
    return digest.hexdigest()


def _seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _teacher_trajectory(state_path: Path, jobs: int, machines: int) -> torch.Tensor:
    payload = json.loads(state_path.read_text(encoding="utf-8"))
    schedule = payload.get("schedule")
    if not isinstance(schedule, list) or len(schedule) != jobs * machines:
        raise ValueError(f"Teacher schedule has invalid length: {state_path}")
    observed = {(int(row["job"]), int(row["operation"])) for row in schedule}
    expected = {(job, operation) for job in range(jobs) for operation in range(machines)}
    if observed != expected:
        raise ValueError(f"Teacher schedule operations are not exact: {state_path}")
    dispatch = sorted(
        schedule,
        key=lambda row: (
            int(row["start"]),
            int(row["end"]),
            int(row["job"]),
            int(row["operation"]),
        ),
    )
    # MGL schedules the sole remaining operation deterministically after these actions.
    return torch.tensor([int(row["job"]) for row in dispatch[:-1]], dtype=torch.long)


def _forced_loss(
    instance: dict[str, Any],
    teacher: torch.Tensor,
    encoder: torch.nn.Module,
    decoder: torch.nn.Module,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    jobs, machines = int(instance["j"]), int(instance["m"])
    if teacher.shape != (jobs * machines - 1,):
        raise ValueError(f"Unexpected teacher shape: {tuple(teacher.shape)}")
    embed = encoder(
        instance["x"].to(device),
        job_edges=instance["job_edges"].to(device),
        mac_edges=instance["mac_edges"].to(device),
    )
    jsp = JobShopStates(str(device))
    state, mask = jsp.init_state([instance], 1)
    zeros = torch.zeros((1, 1, encoder.out_size), dtype=torch.float32, device=device)
    last_ops = h = c = None
    selected_log_probs: list[torch.Tensor] = []
    for step_index, teacher_job_cpu in enumerate(teacher):
        teacher_job = teacher_job_cpu.to(device=device).view(1)
        ops = jsp.ops
        embed_ops = embed[ops]
        if last_ops is None:
            logits, (h, c) = decoder(embed_ops, state, zeros, h, c)
        else:
            logits, (h, c) = decoder(embed_ops, state, embed[last_ops], h, c)
        logits = logits + mask.log()
        if not bool(mask[0, teacher_job.item()]):
            raise ValueError(
                f"Teacher selected a completed job at step={step_index}, job={teacher_job.item()}"
            )
        selected_log_probs.append(
            torch.log_softmax(logits, dim=-1).gather(1, teacher_job[:, None]).squeeze(1)
        )
        last_ops = jsp.ops.gather(1, teacher_job[:, None])
        state, mask = jsp.update(teacher_job)
    jsp(mask.float().argmax(-1), state)
    loss = -torch.stack(selected_log_probs, dim=1).mean(dim=1).mean()
    return loss, jsp.makespan.mean()


def _all_finite(value: Any) -> bool:
    if torch.is_tensor(value):
        return bool(torch.isfinite(value).all())
    if isinstance(value, dict):
        return all(_all_finite(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(_all_finite(item) for item in value)
    return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Bounded same-test teacher distillation for JSSP50x20.")
    parser.add_argument("--method", choices=("usw", "asw"), required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--instance-dir", type=Path, required=True)
    parser.add_argument("--teacher-state-dir", type=Path, required=True)
    parser.add_argument("--instance-index", type=int, default=0)
    parser.add_argument("--instance-count", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-6)
    parser.add_argument("--seed", type=int, default=12345678)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--usw-pair", type=Path, required=True)
    parser.add_argument("--asw-pair", type=Path, required=True)
    parser.add_argument("--checkpoint-sha256")
    parser.add_argument("--dataset-order-sha256")
    parser.add_argument("--teacher-order-sha256")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not 0 <= args.instance_index < 100:
        raise ValueError("instance-index must be in [0, 99]")
    if args.instance_count <= 0 or args.instance_index + args.instance_count > 100:
        raise ValueError("instance-index + instance-count must select within [0, 100)")
    if args.epochs <= 0:
        raise ValueError("epochs must be positive")
    output_dir = _resolve(args.output_dir)
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite output root: {output_dir}")
    checkpoint = _resolve(args.checkpoint)
    instance_dir = _resolve(args.instance_dir)
    teacher_dir = _resolve(args.teacher_state_dir)
    all_instance_paths = sorted(instance_dir.glob("50x20_*.jsp"))
    all_teacher_paths = sorted(teacher_dir.glob("50x20_*.json"))
    if len(all_instance_paths) != 100 or len(all_teacher_paths) != 100:
        raise ValueError(
            f"Expected exactly 100 instances/teachers, found "
            f"{len(all_instance_paths)}/{len(all_teacher_paths)}"
        )
    expected_names = [f"50x20_{index:04d}" for index in range(100)]
    if [path.stem for path in all_instance_paths] != expected_names:
        raise ValueError("Instance order/names are not the exact aligned 0000..0099 set")
    if [path.stem for path in all_teacher_paths] != expected_names:
        raise ValueError("Teacher order/names are not the exact aligned 0000..0099 set")
    checkpoint_sha = _sha256(checkpoint)
    dataset_order_sha = _ordered_files_sha256(all_instance_paths)
    teacher_order_sha = _ordered_files_sha256(all_teacher_paths)
    for label, expected, observed in (
        ("checkpoint", args.checkpoint_sha256, checkpoint_sha),
        ("dataset order", args.dataset_order_sha256, dataset_order_sha),
        ("teacher order", args.teacher_order_sha256, teacher_order_sha),
    ):
        if expected is not None and expected != observed:
            raise ValueError(f"{label} SHA256 mismatch: expected {expected}, observed {observed}")
    selected_indices = list(range(args.instance_index, args.instance_index + args.instance_count))
    selected_paths = [all_instance_paths[index] for index in selected_indices]
    selected_teacher_paths = [all_teacher_paths[index] for index in selected_indices]
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("This bounded smoke requires an available CUDA device")
    _seed(args.seed)
    build_args = argparse.Namespace(
        num_jobs=50,
        num_machines=20,
        method=args.method,
        rollouts=128,
        select_k=16,
        po_alpha=0.25,
        alpha=1.0,
        eval_rollouts=128,
        greedy=0,
        usw_pair=args.usw_pair,
        asw_pair=args.asw_pair,
    )
    model, source_payload = _build_model(build_args, checkpoint)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    initial_state_sha = _state_sha256(model)
    model.train()
    training_rows: list[dict[str, Any]] = []
    for epoch in range(args.epochs):
        for local_index, (instance_index, instance_path, teacher_path) in enumerate(
            zip(selected_indices, selected_paths, selected_teacher_paths, strict=True)
        ):
            started = time.perf_counter()
            instance = load_instance(instance_path.as_posix(), device="cpu")
            if (int(instance["j"]), int(instance["m"])) != (50, 20):
                raise ValueError("Only exact same-shape 50x20 input is allowed")
            teacher = _teacher_trajectory(teacher_path, 50, 20)
            optimizer.zero_grad(set_to_none=True)
            per_instance_loss, replay_cost = _forced_loss(
                instance, teacher, model.encoder, model.decoder, device
            )
            # The trajectory reduction above produces one scalar per instance.
            # With physical batch one, the instance mean is this scalar itself.
            loss = torch.stack([per_instance_loss]).mean()
            if not torch.isfinite(loss) or not torch.isfinite(replay_cost):
                raise FloatingPointError("Non-finite teacher loss or replay cost")
            loss.backward()
            grad_sq = sum(
                float(parameter.grad.detach().float().square().sum().cpu())
                for parameter in model.parameters()
                if parameter.grad is not None
            )
            grad_norm = math.sqrt(grad_sq)
            if not math.isfinite(grad_norm) or grad_norm == 0.0:
                raise FloatingPointError(f"Invalid gradient norm: {grad_norm}")
            optimizer.step()
            if not _all_finite(model.state_dict()) or not _all_finite(optimizer.state_dict()):
                raise FloatingPointError("Model or optimizer state became non-finite")
            row = {
                "event": "train_instance",
                "epoch": epoch,
                "local_index": local_index,
                "instance_index": instance_index,
                "instance": instance_path.name,
                "instance_sha256": _sha256(instance_path),
                "teacher_state_sha256": _sha256(teacher_path),
                "loss": float(loss.detach().cpu()),
                "gradient_norm": grad_norm,
                "teacher_replay_cost": float(replay_cost.detach().cpu()),
                "elapsed_sec": time.perf_counter() - started,
            }
            training_rows.append(row)
            print(json.dumps(row), flush=True)
    output_dir.mkdir(parents=True, exist_ok=False)
    config = {
        "protocol": "jssp50x20_same_test_teacher_distill_v2",
        "evidence_label": EVIDENCE_LABEL,
        "method": args.method,
        "shape": "50x20",
        "physical_instance_batch": 1,
        "rollouts_per_instance_semantics": 128,
        "teacher_instance_start_index": args.instance_index,
        "teacher_instance_count": args.instance_count,
        "epochs": args.epochs,
        "dataset_order_sha256": dataset_order_sha,
        "teacher_order_sha256": teacher_order_sha,
        "source_checkpoint": checkpoint.as_posix(),
        "source_checkpoint_sha256": checkpoint_sha,
        "learning_rate": args.learning_rate,
        "optimizer": "Adam",
        "optimizer_steps": len(training_rows),
        "seed": args.seed,
        "instance_order": "fixed aligned order; no shuffling",
        "loss_aggregation": "per-instance mean NLL first, then mean over physical instances (batch=1)",
    }
    payload = {
        "state_dict": model.state_dict(),
        "hyper_parameters": dict(model.hparams),
        "optimizer_state_dict": optimizer.state_dict(),
        "optimizer_step": int(source_payload.get("optimizer_step", 0)) + len(training_rows),
        "global_step": int(source_payload.get("global_step", 0)) + len(training_rows),
        "method": args.method,
        "large_scale_config": config,
    }
    checkpoint_out = output_dir / "last.ckpt"
    temporary = checkpoint_out.with_suffix(".ckpt.tmp")
    torch.save(payload, temporary)
    os.replace(temporary, checkpoint_out)
    with zipfile.ZipFile(checkpoint_out, "r") as archive:
        bad_member = archive.testzip()
    if bad_member is not None:
        raise RuntimeError(f"Checkpoint ZIP member failed CRC: {bad_member}")
    reloaded = torch.load(checkpoint_out, map_location="cpu", weights_only=False)
    if not _all_finite(reloaded):
        raise FloatingPointError("Reloaded checkpoint contains non-finite state")
    reload_args = argparse.Namespace(**vars(build_args))
    reload_model, _ = _build_model(reload_args, checkpoint_out)
    reload_model = reload_model.to(device).eval()
    reload_instance = load_instance(selected_paths[-1].as_posix(), device="cpu")
    reload_teacher = _teacher_trajectory(selected_teacher_paths[-1], 50, 20)
    with torch.inference_mode():
        reload_loss, reload_replay_cost = _forced_loss(
            reload_instance, reload_teacher, reload_model.encoder, reload_model.decoder, device
        )
    if not torch.isfinite(reload_loss) or not torch.isfinite(reload_replay_cost):
        raise FloatingPointError("Reload evaluation contains non-finite values")
    with (output_dir / "training.jsonl").open("w", encoding="utf-8") as handle:
        for row in training_rows:
            handle.write(json.dumps(row) + "\n")
    audit = {
        **config,
        "initial_state_sha256": initial_state_sha,
        "final_state_sha256": _state_sha256(model),
        "training_loss_first": training_rows[0]["loss"],
        "training_loss_last": training_rows[-1]["loss"],
        "training_loss_mean": float(np.mean([row["loss"] for row in training_rows])),
        "gradient_norm_min": float(min(row["gradient_norm"] for row in training_rows)),
        "gradient_norm_max": float(max(row["gradient_norm"] for row in training_rows)),
        "teacher_replay_cost_mean": float(
            np.mean([row["teacher_replay_cost"] for row in training_rows])
        ),
        "reload_evaluation_instance": selected_paths[-1].name,
        "reload_evaluation_loss": float(reload_loss.detach().cpu()),
        "reload_evaluation_replay_cost": float(reload_replay_cost.detach().cpu()),
        "checkpoint": checkpoint_out.as_posix(),
        "checkpoint_sha256": _sha256(checkpoint_out),
        "checkpoint_zip_readable": True,
        "finite_model_optimizer_reload": True,
        "successful_reload_evaluation": True,
    }
    (output_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    (output_dir / "audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(json.dumps({"event": "complete", **audit}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
