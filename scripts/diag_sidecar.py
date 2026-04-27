#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import time
from typing import Any, Dict, Tuple


def _now_payload() -> Dict[str, Any]:
    return {
        "ts": float(time.time()),
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
    }


def _safe_read_text(path: str) -> str | None:
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            return f.read()
    except Exception:
        return None


def _safe_read_int(path: str) -> int | None:
    txt = _safe_read_text(path)
    if txt is None:
        return None
    txt = txt.strip()
    if not txt:
        return None
    try:
        return int(txt)
    except Exception:
        return None


def _pid_alive(pid: int) -> bool:
    if pid <= 1:
        return False
    proc_dir = f"/proc/{pid}"
    if os.path.isdir(proc_dir):
        return True
    try:
        os.kill(pid, 0)
        return True
    except Exception:
        return False


def _parse_proc_status(pid: int) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    txt = _safe_read_text(f"/proc/{pid}/status")
    if not txt:
        return out

    def _kb(key: str) -> int | None:
        m = re.search(rf"^{re.escape(key)}:\s+(\d+)\s+kB\s*$", txt, flags=re.MULTILINE)
        if not m:
            return None
        return int(m.group(1)) * 1024

    out["name"] = None
    m = re.search(r"^Name:\s+(.*)$", txt, flags=re.MULTILINE)
    if m:
        out["name"] = m.group(1).strip()
    out["rss_bytes"] = _kb("VmRSS")
    out["hwm_bytes"] = _kb("VmHWM")
    out["vms_bytes"] = _kb("VmSize")
    out["threads"] = None
    m = re.search(r"^Threads:\s+(\d+)\s*$", txt, flags=re.MULTILINE)
    if m:
        out["threads"] = int(m.group(1))
    return out


def _parse_proc_cgroup(pid: int) -> Tuple[str | None, str | None]:
    """Return (v2_path, v1_memory_path)."""
    txt = _safe_read_text(f"/proc/{pid}/cgroup")
    if not txt:
        return None, None
    v2_path: str | None = None
    v1_mem: str | None = None
    for ln in txt.splitlines():
        parts = ln.strip().split(":", 2)
        if len(parts) != 3:
            continue
        controllers = parts[1]
        path = parts[2]
        if controllers == "":
            v2_path = path
        if "memory" in controllers.split(","):
            v1_mem = path
    return v2_path, v1_mem


def _read_cgroup_memory(pid: int) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    v2_path, v1_mem = _parse_proc_cgroup(pid)

    # cgroup v2 (unified)
    if v2_path is not None and os.path.isdir("/sys/fs/cgroup"):
        base = os.path.join("/sys/fs/cgroup", v2_path.lstrip("/"))
        cur = _safe_read_int(os.path.join(base, "memory.current"))
        max_raw = _safe_read_text(os.path.join(base, "memory.max"))
        max_v: int | None = None
        if max_raw is not None:
            max_raw = max_raw.strip()
            if max_raw.isdigit():
                max_v = int(max_raw)
        events_txt = _safe_read_text(os.path.join(base, "memory.events"))
        events: Dict[str, int] = {}
        if events_txt:
            for ln in events_txt.splitlines():
                k, _, v = ln.partition(" ")
                if k and v.strip().isdigit():
                    events[k] = int(v.strip())
        out["cgroup_v2"] = {
            "path": v2_path,
            "memory_current": cur,
            "memory_max": max_v,
            "memory_events": events,
        }

    # cgroup v1 (memory controller)
    if v1_mem is not None and os.path.isdir("/sys/fs/cgroup/memory"):
        base = os.path.join("/sys/fs/cgroup/memory", v1_mem.lstrip("/"))
        usage = _safe_read_int(os.path.join(base, "memory.usage_in_bytes"))
        limit = _safe_read_int(os.path.join(base, "memory.limit_in_bytes"))
        out["cgroup_v1_memory"] = {
            "path": v1_mem,
            "usage_in_bytes": usage,
            "limit_in_bytes": limit,
        }

    return out


def _read_nvidia_smi() -> Dict[str, Any] | None:
    try:
        cmd = [
            "nvidia-smi",
            "--query-gpu=index,name,uuid,memory.used,memory.total,utilization.gpu,utilization.memory",
            "--format=csv,noheader,nounits",
        ]
        raw = subprocess.check_output(cmd, stderr=subprocess.STDOUT, timeout=3, text=True)
    except Exception:
        return None

    gpus = []
    for ln in raw.splitlines():
        ln = ln.strip()
        if not ln:
            continue
        parts = [p.strip() for p in ln.split(",")]
        if len(parts) < 7:
            continue
        try:
            gpus.append(
                {
                    "index": int(parts[0]),
                    "name": parts[1],
                    "uuid": parts[2],
                    "mem_used_mb": int(parts[3]),
                    "mem_total_mb": int(parts[4]),
                    "util_gpu_pct": int(parts[5]),
                    "util_mem_pct": int(parts[6]),
                }
            )
        except Exception:
            continue
    return {"gpus": gpus}


def main() -> int:
    ap = argparse.ArgumentParser(description="Lightweight monitoring sidecar for free_loss_eoh runs.")
    ap.add_argument("--parent-pid", type=int, required=True)
    ap.add_argument("--run-dir", type=str, required=True)
    ap.add_argument("--interval-s", type=float, default=30.0)
    args = ap.parse_args()

    run_dir = os.path.abspath(args.run_dir)
    diag_dir = os.path.join(run_dir, "diagnostics")
    os.makedirs(diag_dir, exist_ok=True)
    out_path = os.path.join(diag_dir, "sidecar_heartbeat.jsonl")
    summary_path = os.path.join(diag_dir, "sidecar_summary.json")

    parent_pid = int(args.parent_pid)
    interval_s = max(float(args.interval_s), 2.0)

    first_events: Dict[str, int] | None = None
    last_events: Dict[str, int] | None = None
    last_sample: Dict[str, Any] | None = None

    with open(out_path, "a", encoding="utf-8") as f:
        f.write(json.dumps({**_now_payload(), "event": "sidecar_start", "parent_pid": parent_pid}) + "\n")
        f.flush()

        while True:
            alive = _pid_alive(parent_pid)
            sample: Dict[str, Any] = {**_now_payload(), "event": "tick", "parent_pid": parent_pid, "parent_alive": alive}
            if alive:
                sample["proc"] = _parse_proc_status(parent_pid)
                sample["cgroup"] = _read_cgroup_memory(parent_pid)
                smi = _read_nvidia_smi()
                if smi is not None:
                    sample["nvidia_smi"] = smi

                # Track cgroup v2 oom signals if available.
                ev = (
                    (sample.get("cgroup") or {})
                    .get("cgroup_v2", {})
                    .get("memory_events", {})
                )
                if isinstance(ev, dict) and ev:
                    last_events = {k: int(v) for k, v in ev.items() if isinstance(v, int)}
                    if first_events is None:
                        first_events = dict(last_events)

                last_sample = sample
                f.write(json.dumps(sample, ensure_ascii=False) + "\n")
                f.flush()
                time.sleep(interval_s)
                continue

            # parent is gone
            f.write(json.dumps({**_now_payload(), "event": "parent_missing", "parent_pid": parent_pid}) + "\n")
            f.flush()
            break

    summary: Dict[str, Any] = {**_now_payload(), "parent_pid": parent_pid, "last_sample": last_sample}
    if first_events is not None or last_events is not None:
        summary["cgroup_v2_memory_events_first"] = first_events
        summary["cgroup_v2_memory_events_last"] = last_events
        if first_events and last_events:
            summary["cgroup_v2_memory_events_delta"] = {
                k: int(last_events.get(k, 0)) - int(first_events.get(k, 0)) for k in sorted(set(first_events) | set(last_events))
            }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

