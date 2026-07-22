#!/usr/bin/env python3
"""Resume a CVRP50 preference full-train until it beats SymNCO aug-max."""

from __future__ import annotations

import argparse
import csv
import os
import re
import signal
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


def _float_or_none(raw: str | None) -> float | None:
    if raw in (None, ""):
        return None
    try:
        return float(raw)
    except ValueError:
        return None


@dataclass
class MetricSummary:
    best_value: float | None = None
    best_path: Path | None = None
    best_epoch: int | None = None
    latest_epoch: int | None = None
    latest_value: float | None = None
    latest_value_path: Path | None = None
    latest_value_epoch: int | None = None


def read_metrics(repo: Path, patterns: list[str], metric: str, mode: str) -> MetricSummary:
    summary = MetricSummary()
    best_value: float | None = None

    for pattern in patterns:
        for metrics_path in sorted(repo.glob(pattern)):
            try:
                with metrics_path.open(newline="") as handle:
                    rows = list(csv.DictReader(handle))
            except OSError:
                continue
            for row in rows:
                value = _float_or_none(row.get(metric))
                if value is not None:
                    row_epoch = _float_or_none(row.get("epoch"))
                    metric_epoch = int(row_epoch) if row_epoch is not None else None
                    summary.latest_value = value
                    summary.latest_value_path = metrics_path
                    summary.latest_value_epoch = metric_epoch
                    if best_value is None:
                        best_value = value
                        summary.best_value = value
                        summary.best_path = metrics_path
                        summary.best_epoch = metric_epoch
                    elif mode == "max" and value > best_value:
                        best_value = value
                        summary.best_value = value
                        summary.best_path = metrics_path
                        summary.best_epoch = metric_epoch
                    elif mode == "min" and value < best_value:
                        best_value = value
                        summary.best_value = value
                        summary.best_path = metrics_path
                        summary.best_epoch = metric_epoch

                epoch = _float_or_none(row.get("epoch"))
                if epoch is not None:
                    epoch_int = int(epoch)
                    summary.latest_epoch = (
                        epoch_int
                        if summary.latest_epoch is None
                        else max(summary.latest_epoch, epoch_int)
                    )

    return summary


def latest_checkpoint(repo: Path, patterns: list[str]) -> Path | None:
    candidates: list[tuple[float, Path]] = []
    for pattern in patterns:
        for ckpt_path in repo.glob(pattern):
            try:
                candidates.append((ckpt_path.stat().st_mtime, ckpt_path))
            except OSError:
                continue
    if not candidates:
        return None
    candidates.sort()
    return candidates[-1][1]


def matching_processes(pattern: str) -> list[tuple[int, str]]:
    proc = subprocess.run(
        ["pgrep", "-af", pattern],
        check=False,
        capture_output=True,
        text=True,
    )
    matches: list[tuple[int, str]] = []
    for line in proc.stdout.splitlines():
        if "watch_cvrp50_pref_until_symnco.py" in line or "pgrep -af" in line:
            continue
        parts = line.strip().split(maxsplit=1)
        if len(parts) != 2:
            continue
        try:
            pid = int(parts[0])
        except ValueError:
            continue
        matches.append((pid, parts[1]))
    return matches


def process_running(pattern: str) -> bool:
    return bool(matching_processes(pattern))


def stop_matching_processes(
    pattern: str,
    *,
    wait_seconds: int = 90,
) -> tuple[list[int], list[int]]:
    matches = matching_processes(pattern)
    pids = [pid for pid, _ in matches]
    for pid, command in matches:
        print(f"[stop] sending SIGTERM pid={pid} command={command}", flush=True)
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass

    deadline = time.time() + max(0, wait_seconds)
    remaining = set(pids)
    while remaining and time.time() < deadline:
        live = {pid for pid, _ in matching_processes(pattern)}
        remaining.intersection_update(live)
        if remaining:
            time.sleep(2)

    forced: list[int] = []
    for pid in sorted(remaining):
        print(f"[stop] sending SIGKILL pid={pid}", flush=True)
        try:
            os.kill(pid, signal.SIGKILL)
            forced.append(pid)
        except ProcessLookupError:
            pass
    return pids, forced


def checkpoint_with_learning_rate(
    *,
    repo: Path,
    checkpoint: Path,
    label: str,
    learning_rate: float | None,
) -> Path:
    if learning_rate is None:
        return checkpoint

    import torch

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
    print(
        f"[checkpoint] source={checkpoint} prepared={output_path} "
        f"optimizer_lr={learning_rate}",
        flush=True,
    )
    return output_path


def launch_training(
    *,
    repo: Path,
    python_bin: str,
    label: str,
    best_pair: Path,
    resume_ckpt: Path,
    cuda_visible_devices: str,
    max_epochs: int,
    optimizer_lr: float | None,
    checkpoint_monitor: str,
    checkpoint_mode: str,
    checkpoint_save_top_k: int,
) -> tuple[int, Path]:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = repo / "logs" / "train" / "runs" / f"{label}_{timestamp}"
    ckpt_dir = run_dir / "checkpoints"
    log_dir = repo / "logs" / "codex_remote"
    log_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{label}_train_{timestamp}.log"

    cmd = [
        python_bin,
        "-u",
        "run.py",
        "experiment=routing/pomo-po4cops-cvrp50-po",
        f"hydra.run.dir={run_dir}",
        "model.loss_type=free_loss",
        f"+model.pref_pair_json_path={best_pair}",
        "~callbacks.learning_rate_monitor",
        "~callbacks.rich_progress_bar",
        f"callbacks.model_checkpoint.dirpath={ckpt_dir}",
        "callbacks.model_checkpoint.filename='epoch_{epoch:03d}'",
        "callbacks.model_checkpoint.auto_insert_metric_name=False",
        f"callbacks.model_checkpoint.monitor={checkpoint_monitor}",
        f"callbacks.model_checkpoint.mode={checkpoint_mode}",
        f"callbacks.model_checkpoint.save_top_k={checkpoint_save_top_k}",
        "callbacks.model_checkpoint.save_last=True",
        "trainer.accelerator=gpu",
        "trainer.devices=[0]",
        "trainer.enable_progress_bar=false",
        "logger=csv",
        f"logger.csv.name={label}",
        "test=True",
        f"ckpt_path={resume_ckpt}",
        f"trainer.max_epochs={max_epochs}",
    ]
    if optimizer_lr is not None:
        cmd.append(f"model.optimizer_kwargs.lr={optimizer_lr}")

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices
    env.setdefault("PYTHONPATH", f"{repo}:{repo / 'PTP'}")
    env["PYTHONPATH"] = f"{repo}:{repo / 'PTP'}:{env.get('PYTHONPATH', '')}"
    env.setdefault("LOG_TZ", "Asia/Shanghai")
    env.setdefault("LOG_LEVEL", "INFO")
    env.setdefault("HYDRA_FULL_ERROR", "1")

    with log_path.open("a", encoding="utf-8") as log_handle:
        log_handle.write(f"label={label}\n")
        log_handle.write(f"best_pair={best_pair}\n")
        log_handle.write(f"resume_ckpt={resume_ckpt}\n")
        log_handle.write(f"cuda_visible_devices={cuda_visible_devices}\n")
        log_handle.write(f"max_epochs={max_epochs}\n")
        log_handle.write(f"optimizer_lr={optimizer_lr}\n")
        log_handle.write(f"checkpoint_monitor={checkpoint_monitor}\n")
        log_handle.write(f"checkpoint_mode={checkpoint_mode}\n")
        log_handle.write(f"checkpoint_save_top_k={checkpoint_save_top_k}\n")
        log_handle.write("command=" + " ".join(cmd) + "\n")
        log_handle.flush()
        proc = subprocess.Popen(
            cmd,
            cwd=repo,
            env=env,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
        )
    return int(proc.pid), log_path


def is_better(value: float, baseline: float, mode: str) -> bool:
    return value > baseline if mode == "max" else value < baseline


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default="/data1/gushengda/eam-rl4co")
    parser.add_argument("--python-bin", default="/data1/gushengda/anaconda3/envs/rlco1/bin/python")
    parser.add_argument("--label", required=True)
    parser.add_argument("--best-pair", required=True)
    parser.add_argument("--baseline", type=float, default=-10.43764)
    parser.add_argument("--metric", default="test/max_aug_reward")
    parser.add_argument("--mode", choices=["max", "min"], default="max")
    parser.add_argument("--progress-metric", default="val/max_aug_reward")
    parser.add_argument("--progress-mode", choices=["max", "min"], default="max")
    parser.add_argument("--optimizer-lr", type=float, default=None)
    parser.add_argument("--checkpoint-monitor", default=None)
    parser.add_argument("--checkpoint-mode", choices=["max", "min"], default=None)
    parser.add_argument("--checkpoint-save-top-k", type=int, default=1)
    parser.add_argument("--stop-training-on-target", action="store_true")
    parser.add_argument("--target-stop-delay-seconds", type=int, default=30)
    parser.add_argument("--stop-wait-seconds", type=int, default=90)
    parser.add_argument("--metrics-glob", action="append", required=True)
    parser.add_argument("--checkpoint-glob", action="append", required=True)
    parser.add_argument("--process-pattern", default="")
    parser.add_argument("--cuda-visible-devices", default="0")
    parser.add_argument("--poll-seconds", type=int, default=600)
    parser.add_argument("--epoch-increment", type=int, default=80)
    parser.add_argument("--min-next-max-epochs", type=int, default=180)
    args = parser.parse_args()

    repo = Path(args.repo).resolve()
    best_pair = Path(args.best_pair).resolve()
    process_pattern = args.process_pattern or args.label

    while True:
        target = read_metrics(repo, args.metrics_glob, args.metric, args.mode)
        print(
            f"[watch:{args.label}] metric={args.metric} best={target.best_value} "
            f"best_epoch={target.best_epoch} baseline={args.baseline} "
            f"best_metrics={target.best_path} latest_epoch={target.latest_epoch}",
            flush=True,
        )
        if args.progress_metric:
            progress = read_metrics(repo, args.metrics_glob, args.progress_metric, args.progress_mode)
            print(
                f"[watch:{args.label}] progress_metric={args.progress_metric} "
                f"latest={progress.latest_value} latest_epoch={progress.latest_value_epoch} "
                f"best={progress.best_value} best_epoch={progress.best_epoch} "
                f"best_metrics={progress.best_path}",
                flush=True,
            )
        if target.best_value is not None and is_better(target.best_value, args.baseline, args.mode):
            print(f"[watch:{args.label}] target beaten.", flush=True)
            if args.stop_training_on_target:
                if args.target_stop_delay_seconds > 0:
                    print(
                        f"[watch:{args.label}] waiting "
                        f"{args.target_stop_delay_seconds}s for checkpoint flush before stop.",
                        flush=True,
                    )
                    time.sleep(args.target_stop_delay_seconds)
                stopped, forced = stop_matching_processes(
                    process_pattern,
                    wait_seconds=args.stop_wait_seconds,
                )
                print(
                    f"[watch:{args.label}] stopped_pids={stopped} forced_pids={forced}",
                    flush=True,
                )
            print(f"[watch:{args.label}] exiting.", flush=True)
            return 0

        if process_running(process_pattern):
            print(f"[watch:{args.label}] matching training process is running; sleeping.", flush=True)
            time.sleep(args.poll_seconds)
            continue

        ckpt = latest_checkpoint(repo, args.checkpoint_glob)
        if ckpt is None:
            print(f"[watch:{args.label}] no checkpoint found; sleeping.", flush=True)
            time.sleep(args.poll_seconds)
            continue

        ckpt = checkpoint_with_learning_rate(
            repo=repo,
            checkpoint=ckpt,
            label=args.label,
            learning_rate=args.optimizer_lr,
        )

        next_max_epochs = max(
            int(args.min_next_max_epochs),
            int(target.latest_epoch or 0) + int(args.epoch_increment),
        )
        pid, log_path = launch_training(
            repo=repo,
            python_bin=args.python_bin,
            label=args.label,
            best_pair=best_pair,
            resume_ckpt=ckpt,
            cuda_visible_devices=args.cuda_visible_devices,
            max_epochs=next_max_epochs,
            optimizer_lr=args.optimizer_lr,
            checkpoint_monitor=args.checkpoint_monitor or args.metric,
            checkpoint_mode=args.checkpoint_mode or args.mode,
            checkpoint_save_top_k=args.checkpoint_save_top_k,
        )
        print(
            f"[watch:{args.label}] launched pid={pid} ckpt={ckpt} "
            f"max_epochs={next_max_epochs} log={log_path}",
            flush=True,
        )
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
