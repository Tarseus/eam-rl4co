#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence


TERMINAL_STATUSES = {"finished", "failed", "signal_exit", "exiting_without_final_state"}


@dataclass
class TraceInfo:
    path: Path
    payload: Dict[str, Any]
    status: str
    pid: int | None
    pid_alive: bool | None
    last_hb_epoch: float | None
    hb_age_s: float | None


def _safe_load_json(path: Path) -> Dict[str, Any] | None:
    try:
        with path.open("r", encoding="utf-8") as f:
            raw = json.load(f)
        if isinstance(raw, dict):
            return dict(raw)
    except Exception:
        return None
    return None


def _pid_alive(pid: int | None) -> bool | None:
    if pid is None or pid <= 0:
        return None
    try:
        os.kill(int(pid), 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except Exception:
        return None
    return True


def _trace_info(path: Path, *, now_epoch: float) -> TraceInfo | None:
    payload = _safe_load_json(path)
    if payload is None:
        return None
    status = str(payload.get("status", "unknown") or "unknown")
    pid_raw = payload.get("pid")
    try:
        pid = int(pid_raw) if pid_raw is not None else None
    except Exception:
        pid = None
    last_hb_raw = payload.get("last_heartbeat_epoch_s")
    try:
        last_hb_epoch = float(last_hb_raw) if last_hb_raw is not None else None
    except Exception:
        last_hb_epoch = None
    hb_age_s = (now_epoch - last_hb_epoch) if last_hb_epoch is not None else None
    return TraceInfo(
        path=path,
        payload=payload,
        status=status,
        pid=pid,
        pid_alive=_pid_alive(pid),
        last_hb_epoch=last_hb_epoch,
        hb_age_s=hb_age_s,
    )


def _kernel_oom_hints(max_lines: int = 40) -> List[str]:
    if platform.system().lower() != "linux":
        return []
    cmds: Sequence[Sequence[str]] = (
        ("bash", "-lc", "journalctl -k -b -1 --no-pager 2>/dev/null | grep -Ei 'out of memory|oom-killer|killed process'"),
        ("bash", "-lc", "dmesg -T 2>/dev/null | grep -Ei 'out of memory|oom-killer|killed process'"),
    )
    for cmd in cmds:
        try:
            out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True, timeout=5)
        except Exception:
            continue
        lines = [ln.strip() for ln in out.splitlines() if ln.strip()]
        if lines:
            return lines[-max(1, int(max_lines)) :]
    return []


def _find_run_dirs(runs_root: Path) -> List[Path]:
    if not runs_root.is_dir():
        return []
    out: List[Path] = []
    for checkpoint in runs_root.glob("**/checkpoint.json"):
        run_dir = checkpoint.parent
        if run_dir not in out:
            out.append(run_dir)
    return sorted(out)


def _fmt_age(age_s: float | None) -> str:
    if age_s is None:
        return "n/a"
    if age_s < 60:
        return f"{age_s:.0f}s"
    if age_s < 3600:
        return f"{age_s / 60:.1f}m"
    return f"{age_s / 3600:.1f}h"


def _classify_run(
    run_dir: Path,
    *,
    now_epoch: float,
    stale_s: float,
    flag_missing_trace: bool,
) -> Dict[str, Any]:
    run_trace_path = run_dir / "runtime_status.json"
    run_trace = _trace_info(run_trace_path, now_epoch=now_epoch) if run_trace_path.is_file() else None

    checkpoint = _safe_load_json(run_dir / "checkpoint.json") or {}
    summary = _safe_load_json(run_dir / "summary.json") or {}

    hf_root = run_dir / "hf_subprocess"
    hf_missing_result = 0
    hf_stale_running = 0
    hf_failed = 0
    hf_total = 0
    hf_problem_samples: List[str] = []

    if hf_root.is_dir():
        for task_dir in sorted(p for p in hf_root.iterdir() if p.is_dir()):
            hf_total += 1
            has_result = (task_dir / "result.json").is_file()
            tinfo = _trace_info(task_dir / "runtime_status.json", now_epoch=now_epoch)
            if not has_result:
                hf_missing_result += 1
            if tinfo is not None:
                stale = (tinfo.hb_age_s is not None) and (tinfo.hb_age_s > stale_s)
                if tinfo.status in {"failed", "signal_exit", "exiting_without_final_state"}:
                    hf_failed += 1
                    if len(hf_problem_samples) < 8:
                        hf_problem_samples.append(f"{task_dir.name}:{tinfo.status}")
                elif (tinfo.status == "running") and stale and (tinfo.pid_alive is False):
                    hf_stale_running += 1
                    if len(hf_problem_samples) < 8:
                        hf_problem_samples.append(f"{task_dir.name}:stale_running_no_pid")
                elif (not has_result) and stale and (tinfo.pid_alive is False):
                    if len(hf_problem_samples) < 8:
                        hf_problem_samples.append(f"{task_dir.name}:no_result_stale")
            elif not has_result:
                if len(hf_problem_samples) < 8:
                    hf_problem_samples.append(f"{task_dir.name}:no_runtime_trace_no_result")

    suspicion: List[str] = []
    if run_trace is None:
        if flag_missing_trace:
            suspicion.append("missing_run_runtime_trace")
    else:
        stale_run = (run_trace.hb_age_s is not None) and (run_trace.hb_age_s > stale_s)
        if run_trace.status not in TERMINAL_STATUSES:
            if stale_run and (run_trace.pid_alive is False):
                suspicion.append("run_trace_stale_and_pid_dead")
            elif stale_run:
                suspicion.append("run_trace_stale")
        if run_trace.status == "failed":
            suspicion.append("run_failed")
        if run_trace.status == "signal_exit":
            sig = str(run_trace.payload.get("signal", ""))
            suspicion.append(f"run_signal_exit:{sig or 'unknown'}")
        if run_trace.status == "exiting_without_final_state":
            suspicion.append("run_exit_without_final_state")

    if hf_missing_result > 0:
        suspicion.append(f"hf_missing_result:{hf_missing_result}")
    if hf_stale_running > 0:
        suspicion.append(f"hf_stale_running:{hf_stale_running}")
    if hf_failed > 0:
        suspicion.append(f"hf_failed:{hf_failed}")

    likely_hard_kill = (
        ("run_trace_stale_and_pid_dead" in suspicion or "run_exit_without_final_state" in suspicion)
        and (hf_missing_result > 0 or hf_stale_running > 0)
    )
    likely_oom_or_reboot = likely_hard_kill

    return {
        "run_dir": str(run_dir),
        "trace_present": bool(run_trace is not None),
        "trace_status": (run_trace.status if run_trace else None),
        "trace_reason": (run_trace.payload.get("reason") if run_trace else None),
        "trace_pid": (run_trace.pid if run_trace else None),
        "trace_pid_alive": (run_trace.pid_alive if run_trace else None),
        "trace_heartbeat_age_s": (run_trace.hb_age_s if run_trace else None),
        "checkpoint_next_generation": checkpoint.get("next_generation"),
        "summary_last_generation": summary.get("last_generation"),
        "hf_total": int(hf_total),
        "hf_missing_result": int(hf_missing_result),
        "hf_stale_running": int(hf_stale_running),
        "hf_failed": int(hf_failed),
        "hf_problem_samples": hf_problem_samples,
        "suspicion": suspicion,
        "likely_hard_kill": bool(likely_hard_kill),
        "likely_oom_or_reboot": bool(likely_oom_or_reboot),
    }


def _print_report(report: Dict[str, Any], *, stale_s: float) -> None:
    print(f"stale_threshold_s: {stale_s:.0f}")
    oom_hints = report.get("kernel_oom_hints") or []
    if oom_hints:
        print("kernel_oom_hints:")
        for ln in oom_hints:
            print(f"  - {ln}")
    else:
        print("kernel_oom_hints: none")

    launchers = report.get("launcher_traces", [])
    print(f"launcher_traces: {len(launchers)}")
    for rec in launchers[:12]:
        print(
            "  - "
            + f"{rec.get('path')} status={rec.get('status')} pid={rec.get('pid')} "
            + f"alive={rec.get('pid_alive')} hb_age={_fmt_age(rec.get('hb_age_s'))}"
        )

    runs = report.get("runs", [])
    print(f"runs_scanned: {len(runs)}")
    problematic = [r for r in runs if r.get("suspicion")]
    print(f"runs_problematic: {len(problematic)}")
    for rec in problematic:
        print(
            "  - "
            + f"{rec.get('run_dir')} status={rec.get('trace_status')} hb_age={_fmt_age(rec.get('trace_heartbeat_age_s'))} "
            + f"next_gen={rec.get('checkpoint_next_generation')} summary_last_gen={rec.get('summary_last_generation')} "
            + f"suspicion={rec.get('suspicion')}"
        )
        if rec.get("hf_problem_samples"):
            print(f"    hf_problem_samples={rec.get('hf_problem_samples')}")
        if rec.get("likely_oom_or_reboot"):
            print("    diagnosis=likely_oom_or_hard_kill_or_reboot")


def main() -> int:
    ap = argparse.ArgumentParser(description="Postmortem analyzer for runtime_status traces.")
    ap.add_argument("--runs-root", type=str, default="runs", help="Root directory containing run subdirectories.")
    ap.add_argument("--launcher-dir", type=str, default="logs/runtime", help="Directory of launcher runtime traces.")
    ap.add_argument("--stale-minutes", type=float, default=10.0, help="Heartbeat stale threshold in minutes.")
    ap.add_argument("--json-out", type=str, default="", help="Optional path to write full JSON report.")
    ap.add_argument("--kernel-oom-max-lines", type=int, default=40, help="Max kernel OOM lines to keep in report.")
    ap.add_argument(
        "--flag-missing-trace",
        action="store_true",
        help="Treat missing run-level runtime_status.json as suspicious.",
    )
    args = ap.parse_args()

    now_epoch = float(time.time())
    stale_s = max(60.0, float(args.stale_minutes) * 60.0)

    launcher_dir = Path(args.launcher_dir).resolve()
    launcher_traces: List[Dict[str, Any]] = []
    if launcher_dir.is_dir():
        for p in sorted(launcher_dir.glob("*.json")):
            t = _trace_info(p, now_epoch=now_epoch)
            if t is None:
                continue
            launcher_traces.append(
                {
                    "path": str(p),
                    "status": t.status,
                    "pid": t.pid,
                    "pid_alive": t.pid_alive,
                    "hb_age_s": t.hb_age_s,
                    "reason": t.payload.get("reason"),
                }
            )

    runs_root = Path(args.runs_root).resolve()
    run_dirs = _find_run_dirs(runs_root)
    runs = [
        _classify_run(
            rd,
            now_epoch=now_epoch,
            stale_s=stale_s,
            flag_missing_trace=bool(args.flag_missing_trace),
        )
        for rd in run_dirs
    ]

    report: Dict[str, Any] = {
        "generated_at_epoch_s": now_epoch,
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(now_epoch)),
        "stale_threshold_s": stale_s,
        "runs_root": str(runs_root),
        "launcher_dir": str(launcher_dir),
        "kernel_oom_hints": _kernel_oom_hints(max_lines=int(args.kernel_oom_max_lines)),
        "launcher_traces": launcher_traces,
        "runs": runs,
    }

    _print_report(report, stale_s=stale_s)
    if args.json_out:
        out_path = Path(args.json_out).resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"json_report_written: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
