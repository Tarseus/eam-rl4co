from __future__ import annotations

import atexit
import json
import os
import signal
import socket
import sys
import threading
import time
import traceback
from typing import Any, Dict, Mapping


def _ts(epoch_s: float | None = None) -> str:
    if epoch_s is None:
        epoch_s = time.time()
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(float(epoch_s)))


def _atomic_write_json(path: str, payload: Mapping[str, Any]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(dict(payload), f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)


class RuntimeTrace:
    """Lightweight runtime status + heartbeat sidecar for long-running jobs."""

    _TERMINAL = {"finished", "failed", "signal_exit", "exiting_without_final_state"}

    def __init__(
        self,
        path: str,
        *,
        role: str,
        heartbeat_interval_s: float = 30.0,
    ) -> None:
        self.path = os.path.abspath(str(path))
        self.role = str(role)
        self.heartbeat_interval_s = max(0.0, float(heartbeat_interval_s))
        self._lock = threading.Lock()
        self._stop_evt = threading.Event()
        self._hb_thread: threading.Thread | None = None
        self._started = False
        self._signal_handlers: Dict[int, Any] = {}
        self._state: Dict[str, Any] = {
            "role": self.role,
            "status": "created",
            "pid": int(os.getpid()),
            "ppid": int(os.getppid()),
            "host": socket.gethostname(),
            "argv": list(sys.argv),
            "cwd": os.getcwd(),
            "created_at": _ts(),
            "created_at_epoch_s": float(time.time()),
            "heartbeat_count": 0,
        }
        atexit.register(self._on_atexit)

    def _write_locked(self) -> None:
        _atomic_write_json(self.path, self._state)

    def start(self, *, extra: Mapping[str, Any] | None = None) -> None:
        with self._lock:
            if self._started:
                return
            now = float(time.time())
            self._state["status"] = "running"
            self._state["started_at"] = _ts(now)
            self._state["started_at_epoch_s"] = now
            self._state["last_heartbeat_at"] = _ts(now)
            self._state["last_heartbeat_epoch_s"] = now
            if extra:
                self._state["extra"] = dict(extra)
            self._state.pop("ended_at", None)
            self._state.pop("ended_at_epoch_s", None)
            self._write_locked()
            self._started = True

        if self.heartbeat_interval_s > 0.0:
            self._hb_thread = threading.Thread(
                target=self._heartbeat_loop,
                name=f"runtime-trace-{self.role}",
                daemon=True,
            )
            self._hb_thread.start()

    def _heartbeat_loop(self) -> None:
        while not self._stop_evt.wait(self.heartbeat_interval_s):
            self.heartbeat(min_interval_s=0.0)

    def heartbeat(
        self,
        *,
        extra: Mapping[str, Any] | None = None,
        min_interval_s: float | None = None,
        force: bool = False,
    ) -> None:
        with self._lock:
            if not self._started:
                return
            now = float(time.time())
            last = float(self._state.get("last_heartbeat_epoch_s", 0.0) or 0.0)
            min_gap = 0.0 if min_interval_s is None else max(0.0, float(min_interval_s))
            if (not force) and (min_gap > 0.0) and ((now - last) < min_gap):
                return
            self._state["last_heartbeat_at"] = _ts(now)
            self._state["last_heartbeat_epoch_s"] = now
            self._state["heartbeat_count"] = int(self._state.get("heartbeat_count", 0) or 0) + 1
            if extra:
                merged = dict(self._state.get("progress", {}))
                merged.update(dict(extra))
                self._state["progress"] = merged
            self._write_locked()

    def finish(
        self,
        *,
        reason: str = "completed",
        exit_code: int = 0,
        extra: Mapping[str, Any] | None = None,
    ) -> None:
        self._stop_evt.set()
        with self._lock:
            now = float(time.time())
            self._state["status"] = "finished"
            self._state["reason"] = str(reason)
            self._state["exit_code"] = int(exit_code)
            self._state["ended_at"] = _ts(now)
            self._state["ended_at_epoch_s"] = now
            if extra:
                merged = dict(self._state.get("result", {}))
                merged.update(dict(extra))
                self._state["result"] = merged
            self._write_locked()

    def fail(
        self,
        *,
        reason: str,
        exc: BaseException | None = None,
        exit_code: int = 1,
        extra: Mapping[str, Any] | None = None,
    ) -> None:
        self._stop_evt.set()
        with self._lock:
            now = float(time.time())
            self._state["status"] = "failed"
            self._state["reason"] = str(reason)
            self._state["exit_code"] = int(exit_code)
            self._state["ended_at"] = _ts(now)
            self._state["ended_at_epoch_s"] = now
            if exc is not None:
                self._state["exception_type"] = type(exc).__name__
                self._state["exception_message"] = str(exc)
                self._state["exception_traceback"] = traceback.format_exc()
            if extra:
                merged = dict(self._state.get("result", {}))
                merged.update(dict(extra))
                self._state["result"] = merged
            self._write_locked()

    def mark_signal(self, signum: int) -> None:
        self._stop_evt.set()
        with self._lock:
            now = float(time.time())
            try:
                sig_name = signal.Signals(int(signum)).name
            except Exception:  # noqa: BLE001
                sig_name = f"SIG{int(signum)}"
            self._state["status"] = "signal_exit"
            self._state["reason"] = "signal_received"
            self._state["signal"] = sig_name
            self._state["signal_num"] = int(signum)
            self._state["exit_code"] = int(128 + int(signum))
            self._state["ended_at"] = _ts(now)
            self._state["ended_at_epoch_s"] = now
            self._write_locked()

    def install_signal_handlers(self) -> None:
        sigs = [signal.SIGINT, signal.SIGTERM]
        if hasattr(signal, "SIGBREAK"):
            sigs.append(signal.SIGBREAK)  # type: ignore[attr-defined]
        for sig in sigs:
            if int(sig) in self._signal_handlers:
                continue
            self._signal_handlers[int(sig)] = signal.getsignal(sig)

            def _handler(signum: int, _frame: Any, *, _self: RuntimeTrace = self) -> None:
                _self.mark_signal(signum)
                if int(signum) == int(signal.SIGINT):
                    raise KeyboardInterrupt()
                raise SystemExit(128 + int(signum))

            signal.signal(sig, _handler)

    def restore_signal_handlers(self) -> None:
        for sig_num, old_handler in list(self._signal_handlers.items()):
            try:
                signal.signal(signal.Signals(sig_num), old_handler)
            except Exception:  # noqa: BLE001
                pass
        self._signal_handlers.clear()

    def close(self) -> None:
        self._stop_evt.set()
        self.restore_signal_handlers()
        t = self._hb_thread
        if t is not None and t.is_alive():
            t.join(timeout=1.0)

    def _on_atexit(self) -> None:
        self._stop_evt.set()
        with self._lock:
            status = str(self._state.get("status", ""))
            if status not in self._TERMINAL:
                now = float(time.time())
                self._state["status"] = "exiting_without_final_state"
                self._state["reason"] = "process_exit_without_finish_or_fail"
                self._state["ended_at"] = _ts(now)
                self._state["ended_at_epoch_s"] = now
                self._write_locked()
