from __future__ import annotations

import argparse
import os
import re
from pathlib import Path

import torch


def checkpoint_with_learning_rate(
    *,
    repo: Path,
    checkpoint: Path,
    label: str,
    learning_rate: float,
) -> Path:
    source_mtime = checkpoint.stat().st_mtime_ns
    safe_label = re.sub(r"[^A-Za-z0-9_.-]+", "_", label)
    safe_lr = f"{learning_rate:.8g}".replace(".", "p").replace("-", "m")
    output_dir = repo / "logs" / "codex_remote" / "resume_checkpoints"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / (
        f"{safe_label}__{checkpoint.stem}__lr_{safe_lr}__mtime_{source_mtime}.ckpt"
    )
    if output_path.exists():
        return output_path

    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    optimizer_states = payload.get("optimizer_states", [])
    if not optimizer_states:
        raise RuntimeError(f"Checkpoint has no optimizer state: {checkpoint}")
    for optimizer_state in optimizer_states:
        for param_group in optimizer_state.get("param_groups", []):
            param_group["lr"] = learning_rate
            if "initial_lr" in param_group:
                param_group["initial_lr"] = learning_rate
    for scheduler_state in payload.get("lr_schedulers", []):
        if isinstance(scheduler_state.get("base_lrs"), list):
            scheduler_state["base_lrs"] = [
                learning_rate for _ in scheduler_state["base_lrs"]
            ]
        if isinstance(scheduler_state.get("_last_lr"), list):
            scheduler_state["_last_lr"] = [
                learning_rate for _ in scheduler_state["_last_lr"]
            ]

    temp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    torch.save(payload, temp_path)
    os.replace(temp_path, output_path)
    return output_path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--learning-rate", type=float, required=True)
    args = parser.parse_args()
    path = checkpoint_with_learning_rate(
        repo=args.repo.resolve(),
        checkpoint=args.checkpoint.resolve(),
        label=args.label,
        learning_rate=float(args.learning_rate),
    )
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
