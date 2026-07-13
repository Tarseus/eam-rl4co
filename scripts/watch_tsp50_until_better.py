#!/usr/bin/env python3
"""Keep resuming TSP50 weighting full-train until validation beats baseline."""

from __future__ import annotations

import argparse
import csv
import subprocess
import time
from pathlib import Path


def read_best(run_root: Path, metric: str) -> tuple[float | None, Path | None, int | None]:
    best_value: float | None = None
    best_path: Path | None = None
    latest_epoch: int | None = None
    for metrics in sorted(
        run_root.glob("logs/train/runs/tsp50_weighting_pref_target_*/tsp50_weighting_pref_target/version_0/metrics.csv")
    ):
        with metrics.open(newline="") as f:
            for row in csv.DictReader(f):
                raw = row.get(metric)
                if raw in (None, ""):
                    continue
                value = float(raw)
                if best_value is None or value > best_value:
                    best_value = value
                    best_path = metrics
                epoch_raw = row.get("epoch")
                if epoch_raw not in (None, ""):
                    epoch = int(float(epoch_raw))
                    latest_epoch = epoch if latest_epoch is None else max(latest_epoch, epoch)
    return best_value, best_path, latest_epoch


def latest_checkpoint(run_root: Path) -> Path | None:
    candidates = []
    for ckpt in run_root.glob("logs/train/runs/tsp50_weighting_pref_target_*/checkpoints/last.ckpt"):
        try:
            candidates.append((ckpt.stat().st_mtime, ckpt))
        except OSError:
            continue
    if not candidates:
        return None
    candidates.sort()
    return candidates[-1][1]


def tsp_running() -> bool:
    out = subprocess.run(
        ["bash", "-lc", "pgrep -af 'run.py .*tsp50_weighting_pref_target' || true"],
        check=False,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return bool(out)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default="/data1/gushengda/eam-rl4co")
    parser.add_argument("--baseline", type=float, required=True)
    parser.add_argument("--best-pair", required=True)
    parser.add_argument("--cuda-visible-devices", default="0")
    parser.add_argument("--metric", default="val/max_aug_reward")
    parser.add_argument("--poll-seconds", type=int, default=600)
    parser.add_argument("--epoch-increment", type=int, default=80)
    args = parser.parse_args()

    repo = Path(args.repo).resolve()
    best_pair = str(Path(args.best_pair).resolve())

    while True:
        best, best_metrics, latest_epoch = read_best(repo, args.metric)
        print(
            f"[watch] best={best} baseline={args.baseline} best_metrics={best_metrics} latest_epoch={latest_epoch}",
            flush=True,
        )
        if best is not None and best > args.baseline:
            print("[watch] target beaten; exiting watcher.", flush=True)
            return 0

        if tsp_running():
            print("[watch] tsp50 training is running; sleeping.", flush=True)
            time.sleep(args.poll_seconds)
            continue

        ckpt = latest_checkpoint(repo)
        if ckpt is None:
            print("[watch] no last.ckpt found; sleeping.", flush=True)
            time.sleep(args.poll_seconds)
            continue

        next_max_epochs = (latest_epoch or 0) + args.epoch_increment
        cmd = [
            "bash",
            "scripts/launch_pref_full_train_on_success.sh",
            "tsp50",
            best_pair,
            args.cuda_visible_devices,
            str(ckpt),
            f"trainer.max_epochs={next_max_epochs}",
        ]
        print("[watch] launching: " + " ".join(cmd), flush=True)
        subprocess.run(cmd, cwd=repo, check=False)
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
