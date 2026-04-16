from __future__ import annotations

import argparse
import json
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
from ptp_discovery.runtime_trace import RuntimeTrace


def _load_config_yaml(config_path: str) -> dict:
    import yaml

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    if not isinstance(cfg, dict):
        raise ValueError(f"Invalid YAML config: {config_path}")
    return dict(cfg)


def _configured_generation_budget(cfg: dict) -> int:
    budgets = cfg.get("budgets", {}) or {}
    if not isinstance(budgets, dict):
        budgets = {}
    raw = budgets.get("generations", cfg.get("generations", 1))
    try:
        return max(1, int(raw))
    except (TypeError, ValueError):
        return 1


def _find_latest_run_dir_from_cfg(cfg: dict, *, require_incomplete: bool = False) -> str | None:
    out_root = os.path.abspath(str(cfg.get("output_root", "runs/pref_loss_coevo")))
    if not os.path.isdir(out_root):
        raise FileNotFoundError(f"output_root does not exist: {out_root}")

    generation_budget = _configured_generation_budget(cfg)
    candidates: list[str] = []
    for name in os.listdir(out_root):
        path = os.path.join(out_root, name)
        if not os.path.isdir(path):
            continue
        if os.path.isfile(os.path.join(path, "checkpoint.json")):
            candidates.append(path)

    if not candidates:
        raise FileNotFoundError(f"No resumable runs (checkpoint.json) found under: {out_root}")

    for path in sorted(candidates, reverse=True):
        if not require_incomplete:
            return path
        ckpt_path = os.path.join(path, "checkpoint.json")
        try:
            with open(ckpt_path, "r", encoding="utf-8") as f:
                checkpoint = json.load(f)
        except Exception:  # noqa: BLE001
            continue
        if not isinstance(checkpoint, dict):
            continue
        try:
            next_generation = int(checkpoint.get("next_generation", 0) or 0)
        except (TypeError, ValueError):
            next_generation = 0
        if next_generation < generation_budget:
            return path
    return None


def _find_latest_run_dir(config_path: str) -> str:
    cfg = _load_config_yaml(config_path)
    path = _find_latest_run_dir_from_cfg(cfg, require_incomplete=False)
    if path is None:
        raise FileNotFoundError(
            f"No resumable runs (checkpoint.json) found under: "
            f"{os.path.abspath(str(cfg.get('output_root', 'runs/pref_loss_coevo')))}"
        )
    return path


def _resolve_resume_dir_from_config(config_path: str) -> tuple[str | None, str | None]:
    cfg = _load_config_yaml(config_path)
    resume_cfg = cfg.get("resume", {}) or {}
    if not isinstance(resume_cfg, dict):
        return None, None
    if not bool(resume_cfg.get("enabled", False)):
        return None, None

    mode = str(resume_cfg.get("mode", "latest_incomplete") or "latest_incomplete").strip().lower()
    strict = bool(resume_cfg.get("strict", False))
    run_dir = None
    try:
        if mode in {"latest", "resume-latest"}:
            run_dir = _find_latest_run_dir_from_cfg(cfg, require_incomplete=False)
        elif mode in {"latest_incomplete", "latest-incomplete", "latest_resumable", "latest-resumable"}:
            run_dir = _find_latest_run_dir_from_cfg(cfg, require_incomplete=True)
        elif mode in {"dir", "resume-dir"}:
            raw_dir = str(resume_cfg.get("dir", "") or "").strip()
            if raw_dir:
                run_dir = os.path.abspath(raw_dir)
        else:
            raise ValueError(f"Unsupported resume.mode={mode!r} in {config_path}")
    except FileNotFoundError:
        if strict:
            raise
        return None, None

    if run_dir and not os.path.isdir(run_dir):
        if strict:
            raise FileNotFoundError(f"Configured resume dir does not exist: {run_dir}")
        return None, None
    if run_dir is None and strict:
        raise FileNotFoundError(
            f"Configured resume.mode={mode!r} could not find a matching run under "
            f"{os.path.abspath(str(cfg.get('output_root', 'runs/pref_loss_coevo')))}"
        )
    return run_dir, mode


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
    level_name = str(os.environ.get("LOG_LEVEL", "INFO") or "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)
    logging.basicConfig(
        level=level,
        format="[%(asctime)s] %(levelname)s:%(name)s: %(message)s",
    )
    # Reduce noisy per-request logs from LLM HTTP clients so progress logs are visible in *.out.
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    parser = _build_arg_parser()
    args = parser.parse_args()
    resume_dir = args.resume_dir
    resume_source = None
    if args.resume_latest:
        if resume_dir is not None:
            raise SystemExit("Pass only one of --resume-dir or --resume-latest.")
        resume_dir = _find_latest_run_dir(args.config)
        resume_source = "cli_latest"
    elif resume_dir is not None:
        resume_source = "cli_dir"
    else:
        resume_dir, resume_mode = _resolve_resume_dir_from_config(args.config)
        if resume_dir is not None:
            resume_source = f"config:{resume_mode}"
            logging.info("Using config-driven resume: mode=%s run_dir=%s", resume_mode, resume_dir)
    trace_dir = os.path.join(os.getcwd(), "logs", "runtime")
    os.makedirs(trace_dir, exist_ok=True)
    trace_path = os.path.join(trace_dir, f"pref_loss_launcher_{os.getpid()}.json")
    try:
        runtime_trace_hb_s = float(os.environ.get("RUNTIME_TRACE_HEARTBEAT_S", "30") or "30")
    except ValueError:
        runtime_trace_hb_s = 30.0
    runtime_trace = RuntimeTrace(
        trace_path,
        role="pref_loss_launcher",
        heartbeat_interval_s=runtime_trace_hb_s,
    )
    runtime_trace.start(
        extra={
            "config_path": os.path.abspath(str(args.config)),
            "resume_dir": (os.path.abspath(str(resume_dir)) if resume_dir else None),
            "resume_latest": bool(args.resume_latest),
            "resume_source": resume_source,
            "device_override": args.device,
        }
    )
    runtime_trace.install_signal_handlers()

    overrides = {}
    if args.device is not None:
        overrides["device"] = args.device

    try:
        run_pref_loss_coevo(args.config, resume_dir=resume_dir, **overrides)
    except KeyboardInterrupt as exc:
        runtime_trace.fail(reason="keyboard_interrupt", exc=exc, exit_code=130)
        raise
    except BaseException as exc:  # noqa: BLE001
        runtime_trace.fail(reason="unhandled_exception", exc=exc, exit_code=1)
        raise
    else:
        runtime_trace.finish(reason="completed", exit_code=0)
    finally:
        runtime_trace.close()


if __name__ == "__main__":
    main()
