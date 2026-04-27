import argparse
import json
import os
import sys
import time
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Tuple


def _parse_devices(devices_csv: str) -> List[str]:
    devices: List[str] = []
    for part in (devices_csv or "").split(","):
        part = part.strip()
        if part:
            devices.append(part)
    return devices


def _parse_cuda_index(device_str: str) -> Optional[int]:
    s = str(device_str).strip().lower()
    if not s.startswith("cuda"):
        return None
    if ":" not in s:
        return 0
    try:
        return int(s.split(":", 1)[1])
    except Exception:
        return None


def _worker(
    device_str: str,
    task_queue: Any,
    result_queue: Any,
    *,
    use_torch: bool,
) -> None:
    torch_info: Dict[str, Any] = {}
    torch = None
    if use_torch:
        try:
            import torch as _torch  # type: ignore

            torch = _torch
        except Exception as exc:  # noqa: BLE001
            torch_info = {"torch_ok": False, "torch_error": str(exc)}

    if torch is not None:
        try:
            idx = _parse_cuda_index(device_str)
            if idx is not None and torch.cuda.is_available():
                torch.cuda.set_device(idx)
                _ = torch.empty((1,), device="cuda")
                torch_info = {
                    "torch_ok": True,
                    "cuda_available": True,
                    "device_index": int(idx),
                    "device_name": str(torch.cuda.get_device_name(int(idx))),
                }
            else:
                torch_info = {
                    "torch_ok": True,
                    "cuda_available": bool(torch.cuda.is_available()),
                    "device_index": None,
                    "device_name": None,
                }
        except Exception as exc:  # noqa: BLE001
            torch_info = {"torch_ok": False, "torch_error": str(exc)}

    while True:
        payload = task_queue.get()
        if payload is None:
            return

        task_id = int(payload.get("task_id", -1))
        sleep_s = float(payload.get("sleep_s", 0.0) or 0.0)
        t0 = time.time()
        try:
            if torch is not None and bool(torch_info.get("torch_ok")) and bool(torch_info.get("cuda_available")):
                idx = _parse_cuda_index(device_str)
                if idx is not None:
                    # Small allocation to ensure the correct device is usable.
                    _ = torch.empty((256, 256), device=f"cuda:{int(idx)}")
            if sleep_s > 0:
                time.sleep(sleep_s)
            ok = True
            err = None
        except Exception as exc:  # noqa: BLE001
            ok = False
            err = str(exc)

        result_queue.put(
            {
                "task_id": task_id,
                "device_str": str(device_str),
                "ok": bool(ok),
                "error": err,
                "elapsed_s": float(time.time() - t0),
                "torch": dict(torch_info),
                "pid": int(os.getpid()),
            }
        )


def _run_smoke(args: argparse.Namespace) -> int:
    import multiprocessing as mp

    devices = list(args.devices)
    if not devices:
        print("ERROR: --devices is empty", file=sys.stderr)
        return 2

    tasks = int(args.tasks)
    if tasks <= 0:
        print("ERROR: --tasks must be > 0", file=sys.stderr)
        return 2

    procs = min(int(args.processes), len(devices), tasks)
    ctx = mp.get_context(str(args.start_method))

    task_queue: Any = ctx.Queue()
    result_queue: Any = ctx.Queue()

    for t in range(tasks):
        task_queue.put({"task_id": int(t), "sleep_s": float(args.sleep_s)})
    for _ in range(procs):
        task_queue.put(None)

    workers: List[Any] = []
    for w_idx in range(procs):
        dev = devices[w_idx % len(devices)]
        p = ctx.Process(
            target=_worker,
            args=(str(dev), task_queue, result_queue),
            kwargs={"use_torch": bool(args.use_torch)},
        )
        p.daemon = False
        p.start()
        workers.append(p)

    results: List[Dict[str, Any]] = []
    for _ in range(tasks):
        results.append(dict(result_queue.get()))

    for p in workers:
        p.join()

    ok_count = sum(1 for r in results if bool(r.get("ok")))
    by_dev = Counter(str(r.get("device_str")) for r in results)
    by_pid = Counter(int(r.get("pid", -1)) for r in results)
    torch_ok = [r.get("torch", {}).get("torch_ok") for r in results if isinstance(r.get("torch"), dict)]

    print("=== pinned-mp smoke test ===")
    print(f"processes={procs} tasks={tasks} start_method={args.start_method}")
    print("device_task_counts=", dict(by_dev))
    print("pid_task_counts=", dict(by_pid))
    if torch_ok:
        print("torch_ok_counts=", dict(Counter(torch_ok)))
        # Print one example per device.
        seen = set()
        for r in results:
            d = str(r.get("device_str"))
            if d in seen:
                continue
            seen.add(d)
            torch_meta = r.get("torch")
            if isinstance(torch_meta, dict):
                print(f"torch_meta[{d}]=", torch_meta)

    if ok_count != tasks:
        bad = [r for r in results if not bool(r.get("ok"))]
        print(f"FAIL: ok={ok_count}/{tasks}", file=sys.stderr)
        print("first_errors=", bad[:5], file=sys.stderr)
        return 1

    # Heuristic check: distribution shouldn't be pathologically imbalanced.
    # With round-robin pinned workers, the worst-case difference is at most 1.
    if len(by_dev) > 1:
        mx = max(by_dev.values())
        mn = min(by_dev.values())
        if (mx - mn) > 1:
            print(f"WARN: task distribution skewed (max-min={mx-mn})", file=sys.stderr)

    print("PASS")
    return 0


def _analyze_pairs_jsonl(path: str) -> Tuple[Counter, Counter]:
    stage = Counter()
    device = Counter()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            s = str(rec.get("stage", ""))
            stage[s] += 1
            if s == "high_fidelity":
                device[str(rec.get("device", rec.get("device_str", "")))] += 1
    return stage, device


def _run_analyze(args: argparse.Namespace) -> int:
    stage, device = _analyze_pairs_jsonl(str(args.pairs_jsonl))
    print("=== pairs.jsonl analyze ===")
    print("stages=", dict(stage))
    print("hf_device_counts=", dict(device))
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Smoke test for pinned HF multiprocessing + simple run analysis.")
    sub = p.add_subparsers(dest="cmd", required=True)

    ps = sub.add_parser("smoke", help="Spawn pinned workers and execute dummy tasks.")
    ps.add_argument("--devices", type=str, required=True, help='CSV like "cuda:0,cuda:1,cuda:2"')
    ps.add_argument("--processes", type=int, default=5)
    ps.add_argument("--tasks", type=int, default=20)
    ps.add_argument("--sleep-s", type=float, default=0.5)
    ps.add_argument("--start-method", type=str, default="spawn")
    ps.add_argument("--use-torch", action="store_true", help="Try allocating a small tensor on each device.")

    pa = sub.add_parser("analyze", help="Analyze a run's pairs.jsonl for HF stage + device counts.")
    pa.add_argument("--pairs-jsonl", type=str, required=True)

    args = p.parse_args(argv)
    if args.cmd == "smoke":
        args.devices = _parse_devices(args.devices)
        return _run_smoke(args)
    if args.cmd == "analyze":
        return _run_analyze(args)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())

