#!/usr/bin/env python3
"""Resume a full-train job until a metric beats a baseline."""

from __future__ import annotations

import argparse
import csv
import subprocess
import time
from pathlib import Path


def read_best(repo: Path, run_glob: str, metrics_rel: str, metric: str, mode: str):
    best = None
    latest_epoch = None
    for run_dir in sorted((repo / "logs/train/runs").glob(run_glob)):
        metrics = run_dir / metrics_rel
        if not metrics.exists():
            continue
        with metrics.open(newline="") as f:
            for row in csv.DictReader(f):
                raw = row.get(metric)
                if raw not in (None, ""):
                    try:
                        value = float(raw)
                    except ValueError:
                        continue
                    item = (value, run_dir, metrics, row.get("epoch"), row.get("step"))
                    if best is None:
                        best = item
                    elif mode == "max" and value > best[0]:
                        best = item
                    elif mode == "min" and value < best[0]:
                        best = item
                epoch_raw = row.get("epoch")
                if epoch_raw not in (None, ""):
                    try:
                        epoch = int(float(epoch_raw))
                    except ValueError:
                        continue
                    latest_epoch = epoch if latest_epoch is None else max(latest_epoch, epoch)
    return best, latest_epoch


def latest_checkpoint(repo: Path, run_glob: str) -> Path | None:
    candidates = []
    for run_dir in (repo / "logs/train/runs").glob(run_glob):
        ckpt = run_dir / "checkpoints/last.ckpt"
        if ckpt.exists():
            candidates.append((ckpt.stat().st_mtime, ckpt))
    if not candidates:
        return None
    candidates.sort()
    return candidates[-1][1]


def is_better(value: float, baseline: float, mode: str) -> bool:
    return value > baseline if mode == "max" else value < baseline


def process_running(pattern: str) -> bool:
    out = subprocess.run(
        ["bash", "-lc", f"pgrep -af {pattern!r} || true"],
        check=False,
        capture_output=True,
        text=True,
    ).stdout.strip()
    lines = [
        line
        for line in out.splitlines()
        if "watch_metric_until_better.py" not in line and "pgrep -af" not in line
    ]
    return bool(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default="/data1/gushengda/eam-rl4co")
    parser.add_argument("--run-glob", required=True)
    parser.add_argument("--metrics-rel", required=True)
    parser.add_argument("--metric", required=True)
    parser.add_argument("--mode", choices=["max", "min"], default="max")
    parser.add_argument("--baseline", type=float, required=True)
    parser.add_argument("--process-pattern", required=True)
    parser.add_argument("--problem", required=True)
    parser.add_argument("--best-pair", required=True)
    parser.add_argument("--cuda-visible-devices", required=True)
    parser.add_argument("--poll-seconds", type=int, default=600)
    parser.add_argument("--epoch-increment", type=int, default=80)
    args = parser.parse_args()

    repo = Path(args.repo).resolve()
    while True:
        best, latest_epoch = read_best(repo, args.run_glob, args.metrics_rel, args.metric, args.mode)
        print(
            f"[watch] metric={args.metric} mode={args.mode} best={best} "
            f"baseline={args.baseline} latest_epoch={latest_epoch}",
            flush=True,
        )
        if best is not None and is_better(best[0], args.baseline, args.mode):
            print("[watch] target beaten; exiting watcher.", flush=True)
            return 0

        if process_running(args.process_pattern):
            print("[watch] training is running; sleeping.", flush=True)
            time.sleep(args.poll_seconds)
            continue

        ckpt = latest_checkpoint(repo, args.run_glob)
        if ckpt is None:
            print("[watch] no last.ckpt found; sleeping.", flush=True)
            time.sleep(args.poll_seconds)
            continue

        next_max_epochs = (latest_epoch or 0) + args.epoch_increment
        cmd = [
            "bash",
            "scripts/launch_pref_full_train_on_success.sh",
            args.problem,
            str(Path(args.best_pair).resolve()),
            args.cuda_visible_devices,
            str(ckpt),
            f"trainer.max_epochs={next_max_epochs}",
        ]
        print("[watch] launching: " + " ".join(cmd), flush=True)
        subprocess.run(cmd, cwd=repo, check=False)
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
