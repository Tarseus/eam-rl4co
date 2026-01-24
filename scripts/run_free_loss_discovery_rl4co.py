from __future__ import annotations

import argparse
import logging
import os
import sys


def _ensure_paths() -> None:
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ptp_root = os.path.join(repo_root, "PTP")
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    if ptp_root not in sys.path:
        sys.path.insert(0, ptp_root)


def _find_latest_run_dir(config_path: str) -> str:
    import yaml

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    out_root = os.path.abspath(str(cfg.get("output_root", "runs/free_loss_discovery")))
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
        raise FileNotFoundError(
            f"No resumable runs (checkpoint.json) found under: {out_root}"
        )

    return sorted(candidates)[-1]


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="RL4CO-backed discovery of free-form preference losses.",
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to a YAML config file for free loss discovery.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Override device string (e.g., cuda or cpu).",
    )
    parser.add_argument(
        "--resume-dir",
        type=str,
        default=None,
        help="Resume from an existing run directory (must contain checkpoint.json).",
    )
    parser.add_argument(
        "--resume-latest",
        action="store_true",
        help="Resume from the latest run under output_root (requires checkpoint.json).",
    )
    return parser


def main() -> None:
    _ensure_paths()
    from ptp_discovery.free_loss_eoh_loop import run_free_loss_eoh

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s:%(name)s: %(message)s",
    )
    logging.getLogger("ptp_discovery.free_loss_ir").setLevel(logging.DEBUG)

    parser = _build_arg_parser()
    args = parser.parse_args()

    overrides = {}
    if args.device is not None:
        overrides["device"] = args.device
    overrides.setdefault("backend", "rl4co")

    resume_dir = args.resume_dir
    if args.resume_latest:
        if resume_dir is not None:
            raise SystemExit("Pass only one of --resume-dir or --resume-latest.")
        resume_dir = _find_latest_run_dir(args.config)

    run_free_loss_eoh(args.config, resume_dir=resume_dir, **overrides)


if __name__ == "__main__":
    main()
