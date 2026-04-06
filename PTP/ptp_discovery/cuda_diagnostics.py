from __future__ import annotations

import csv
import os
import shutil
import subprocess
import time
from typing import Any, Dict, Iterable, List, Mapping, Sequence

import torch


def _device_labels(devices: Sequence[str] | None) -> List[str]:
    if devices is None:
        if not torch.cuda.is_available():
            return []
        return [f"cuda:{idx}" for idx in range(int(torch.cuda.device_count()))]
    out: List[str] = []
    for raw in devices:
        ds = str(raw or "").strip()
        if ds:
            out.append(ds)
    return out


def _device_index(device_str: str) -> int | None:
    ds = str(device_str or "").strip().lower()
    if ds == "cuda":
        return 0
    if ds.startswith("cuda:"):
        try:
            return int(ds.split(":", 1)[1].strip())
        except (TypeError, ValueError):
            return None
    return None


def _run_nvidia_smi_query(args: Sequence[str], *, timeout_s: float = 3.0) -> subprocess.CompletedProcess[str] | None:
    exe = shutil.which("nvidia-smi")
    if not exe:
        return None
    try:
        return subprocess.run(
            [exe, *list(args)],
            check=False,
            capture_output=True,
            text=True,
            timeout=max(float(timeout_s), 0.5),
        )
    except Exception:
        return None


def _collect_nvidia_smi(*, timeout_s: float = 3.0) -> Dict[str, Any]:
    out: Dict[str, Any] = {"available": False, "gpus": [], "processes": []}
    proc_gpus = _run_nvidia_smi_query(
        [
            "--query-gpu=index,uuid,name,memory.total,memory.used,utilization.gpu,utilization.memory",
            "--format=csv,noheader,nounits",
        ],
        timeout_s=timeout_s,
    )
    if proc_gpus is None or proc_gpus.returncode != 0:
        if proc_gpus is not None:
            out["error"] = (proc_gpus.stderr or proc_gpus.stdout or "").strip()
        return out

    gpus: List[Dict[str, Any]] = []
    reader = csv.reader((proc_gpus.stdout or "").splitlines())
    for row in reader:
        if len(row) < 7:
            continue
        try:
            gpus.append(
                {
                    "index": int(str(row[0]).strip()),
                    "uuid": str(row[1]).strip(),
                    "name": str(row[2]).strip(),
                    "memory_total_mb": int(float(str(row[3]).strip())),
                    "memory_used_mb": int(float(str(row[4]).strip())),
                    "util_gpu_pct": int(float(str(row[5]).strip())),
                    "util_mem_pct": int(float(str(row[6]).strip())),
                }
            )
        except (TypeError, ValueError):
            continue
    out["available"] = True
    out["gpus"] = gpus

    proc_apps = _run_nvidia_smi_query(
        [
            "--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory",
            "--format=csv,noheader,nounits",
        ],
        timeout_s=timeout_s,
    )
    if proc_apps is None or proc_apps.returncode != 0:
        return out

    gpu_index_by_uuid = {str(g.get("uuid")): int(g.get("index")) for g in gpus if g.get("uuid") is not None}
    processes: List[Dict[str, Any]] = []
    reader = csv.reader((proc_apps.stdout or "").splitlines())
    for row in reader:
        if len(row) < 4:
            continue
        try:
            processes.append(
                {
                    "gpu_uuid": str(row[0]).strip(),
                    "gpu_index": gpu_index_by_uuid.get(str(row[0]).strip()),
                    "pid": int(str(row[1]).strip()),
                    "process_name": str(row[2]).strip(),
                    "used_gpu_memory_mb": int(float(str(row[3]).strip())),
                }
            )
        except (TypeError, ValueError):
            continue
    out["processes"] = processes
    return out


def collect_cuda_snapshot(
    *,
    devices: Sequence[str] | None = None,
    include_torch: bool = True,
    include_nvidia_smi: bool = False,
    nvidia_smi_timeout_s: float = 3.0,
) -> Dict[str, Any]:
    snapshot: Dict[str, Any] = {
        "captured_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "torch_cuda_available": bool(torch.cuda.is_available()),
        "torch_device_count": int(torch.cuda.device_count()) if torch.cuda.is_available() else 0,
        "requested_devices": list(_device_labels(devices)),
        "torch_devices": [],
    }

    if include_torch and torch.cuda.is_available():
        for label in snapshot["requested_devices"]:
            if not str(label).startswith("cuda"):
                continue
            idx = _device_index(str(label))
            if idx is None:
                continue
            item: Dict[str, Any] = {"label": str(label), "index": int(idx)}
            try:
                props = torch.cuda.get_device_properties(idx)
                free_b, total_b = torch.cuda.mem_get_info(idx)
                item.update(
                    {
                        "name": str(props.name),
                        "memory_total_mb": int(total_b // (1024**2)),
                        "memory_free_mb": int(free_b // (1024**2)),
                        "memory_used_mb": int((total_b - free_b) // (1024**2)),
                        "memory_allocated_mb": int(torch.cuda.memory_allocated(idx) // (1024**2)),
                        "memory_reserved_mb": int(torch.cuda.memory_reserved(idx) // (1024**2)),
                        "max_memory_allocated_mb": int(torch.cuda.max_memory_allocated(idx) // (1024**2)),
                        "max_memory_reserved_mb": int(torch.cuda.max_memory_reserved(idx) // (1024**2)),
                    }
                )
            except Exception as exc:
                item["error"] = f"{type(exc).__name__}: {exc}"
            snapshot["torch_devices"].append(item)

    if include_nvidia_smi:
        snapshot["nvidia_smi"] = _collect_nvidia_smi(timeout_s=nvidia_smi_timeout_s)

    return snapshot


def format_cuda_snapshot(snapshot: Mapping[str, Any]) -> str:
    if not isinstance(snapshot, Mapping):
        return "cuda_diag=invalid"
    parts: List[str] = []
    parts.append(f"vis={snapshot.get('cuda_visible_devices')!r}")

    torch_items = snapshot.get("torch_devices")
    if isinstance(torch_items, Iterable):
        tparts: List[str] = []
        for item in torch_items:
            if not isinstance(item, Mapping):
                continue
            label = str(item.get("label") or f"cuda:{item.get('index')}")
            if item.get("error"):
                tparts.append(f"{label}(error={item.get('error')})")
                continue
            tparts.append(
                f"{label}(used={int(item.get('memory_used_mb', 0))}M"
                f"/free={int(item.get('memory_free_mb', 0))}M"
                f"/alloc={int(item.get('memory_allocated_mb', 0))}M"
                f"/resv={int(item.get('memory_reserved_mb', 0))}M)"
            )
        if tparts:
            parts.append("torch=" + " ".join(tparts))

    nvsmi = snapshot.get("nvidia_smi")
    if isinstance(nvsmi, Mapping) and bool(nvsmi.get("available")):
        gpu_parts: List[str] = []
        gpu_rows = nvsmi.get("gpus")
        if isinstance(gpu_rows, Iterable):
            for item in gpu_rows:
                if not isinstance(item, Mapping):
                    continue
                gpu_parts.append(
                    f"cuda:{int(item.get('index', -1))}(used={int(item.get('memory_used_mb', 0))}"
                    f"M/{int(item.get('memory_total_mb', 0))}M util={int(item.get('util_gpu_pct', 0))}%)"
                )
        if gpu_parts:
            parts.append("smi=" + " ".join(gpu_parts))

        proc_parts: List[str] = []
        proc_rows = nvsmi.get("processes")
        if isinstance(proc_rows, Iterable):
            for item in list(proc_rows)[:16]:
                if not isinstance(item, Mapping):
                    continue
                proc_parts.append(
                    f"cuda:{item.get('gpu_index')} pid={item.get('pid')} {item.get('process_name')} "
                    f"{int(item.get('used_gpu_memory_mb', 0))}M"
                )
        if proc_parts:
            parts.append("procs=" + "; ".join(proc_parts))

    return "cuda_diag(" + " | ".join(parts) + ")"
