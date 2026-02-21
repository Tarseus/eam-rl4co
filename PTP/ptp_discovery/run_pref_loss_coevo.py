from __future__ import annotations

import argparse
import logging
import os
import sys
import time

# Make this file usable both as a module (`-m ptp_discovery.run_pref_loss_coevo`)
# and as a direct script (`python ptp_discovery/run_pref_loss_coevo.py`).
if __package__ is None or __package__ == "":
    # This file lives at <repo_root>/PTP/ptp_discovery/run_pref_loss_coevo.py
    ptp_discovery_dir = os.path.dirname(os.path.abspath(__file__))  # .../PTP/ptp_discovery
    ptp_root = os.path.abspath(os.path.join(ptp_discovery_dir, ".."))  # .../PTP
    repo_root = os.path.abspath(os.path.join(ptp_root, ".."))  # .../

    # Ensure local packages (e.g. `rl4co/`) are importable even when invoked as:
    #   python PTP/ptp_discovery/run_pref_loss_coevo.py ...
    for path in (repo_root, ptp_root):
        if os.path.isdir(path) and path not in sys.path:
            sys.path.insert(0, path)

from ptp_discovery.pref_loss_coevo_loop import run_pref_loss_coevo


def _find_latest_run_dir(config_path: str) -> str:
    import yaml

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    out_root = os.path.abspath(str(cfg.get("output_root", "runs/pref_loss_coevo")))
    if not os.path.isdir(out_root):
        raise FileNotFoundError(f"output_root does not exist: {out_root}")

    candidates: list[str] = []
    for name in os.listdir(out_root):
        path = os.path.join(out_root, name)
        if not os.path.isdir(path):
            continue
        if os.path.isfile(os.path.join(path, "checkpoint.json")):
            candidates.append(path)

    if not candidates:
        raise FileNotFoundError(f"No resumable runs (checkpoint.json) found under: {out_root}")

    return sorted(candidates)[-1]


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Co-evolution search over preference builders (g) and losses (f).",
    )
    p.add_argument("--config", type=str, required=True, help="Path to YAML config.")
    p.add_argument("--resume-dir", type=str, default=None, help="Resume from an existing run directory.")
    p.add_argument(
        "--resume-latest",
        action="store_true",
        help="Resume from latest run under output_root (requires checkpoint.json).",
    )
    p.add_argument("--device", type=str, default=None, help="Override device string (e.g., cuda or cpu).")
    return p


def main() -> None:
    # Timezone for `%(asctime)s` in Python logging.
    # - If `LOG_TZ` is set (e.g., "Asia/Shanghai"), we apply it via `TZ` and `tzset` (POSIX).
    # - If not set, logging uses the system local timezone.
    log_tz = os.environ.get("LOG_TZ")
    if log_tz:
        os.environ["TZ"] = str(log_tz)
        try:
            time.tzset()
        except AttributeError:
            # Windows / non-POSIX platforms may not provide tzset.
            pass
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s:%(name)s: %(message)s",
    )
    parser = _build_arg_parser()
    args = parser.parse_args()

    overrides = {}
    if args.device is not None:
        overrides["device"] = args.device

    resume_dir = args.resume_dir
    if args.resume_latest:
        if resume_dir is not None:
            raise SystemExit("Pass only one of --resume-dir or --resume-latest.")
        resume_dir = _find_latest_run_dir(args.config)

    run_pref_loss_coevo(args.config, resume_dir=resume_dir, **overrides)


if __name__ == "__main__":
    main()
