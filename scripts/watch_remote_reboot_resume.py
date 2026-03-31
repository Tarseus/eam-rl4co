#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from typing import Any

WATCH_MODE_PREF_LOSS = "pref_loss"
WATCH_MODE_FREE_LOSS = "free_loss"
WATCH_MODE_CHOICES = (WATCH_MODE_PREF_LOSS, WATCH_MODE_FREE_LOSS)


@dataclass
class RemoteResult:
    returncode: int
    stdout: str
    stderr: str


class RemoteSSH:
    def __init__(
        self,
        host: str,
        *,
        ssh_bin: str = "ssh",
        ssh_args: list[str] | None = None,
        connect_timeout_s: float = 8.0,
    ) -> None:
        self.host = str(host)
        self.ssh_bin = str(ssh_bin)
        self.ssh_args = list(ssh_args or [])
        self.connect_timeout_s = max(1.0, float(connect_timeout_s))

    def run(
        self,
        remote_cmd: str,
        *,
        timeout_s: float = 20.0,
        use_bash_lc: bool = False,
    ) -> RemoteResult:
        remote_argv: list[str]
        if use_bash_lc:
            remote_argv = [f"bash -lc {shlex.quote(str(remote_cmd))}"]
        else:
            remote_argv = [str(remote_cmd)]
        cmd = [
            self.ssh_bin,
            *self.ssh_args,
            "-o",
            f"ConnectTimeout={int(self.connect_timeout_s)}",
            self.host,
            *remote_argv,
        ]
        try:
            cp = subprocess.run(
                cmd,
                text=True,
                capture_output=True,
                timeout=max(1.0, float(timeout_s)),
                check=False,
            )
            return RemoteResult(
                returncode=int(cp.returncode),
                stdout=str(cp.stdout or ""),
                stderr=str(cp.stderr or ""),
            )
        except subprocess.TimeoutExpired:
            return RemoteResult(returncode=124, stdout="", stderr="ssh_timeout")
        except Exception as exc:  # noqa: BLE001
            return RemoteResult(returncode=255, stdout="", stderr=f"ssh_error:{type(exc).__name__}:{exc}")


def _ts() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())


def _log(msg: str) -> None:
    sys.stdout.write(f"[{_ts()}] {msg}\n")
    sys.stdout.flush()


def _q(v: str) -> str:
    return shlex.quote(str(v))


def _normalize_watch_mode(mode: str) -> str:
    value = str(mode or WATCH_MODE_PREF_LOSS).strip().lower()
    if value not in WATCH_MODE_CHOICES:
        raise ValueError(f"Unsupported watch mode: {mode}")
    return value


def _watch_mode_defaults(mode: str) -> dict[str, str]:
    mode_norm = _normalize_watch_mode(mode)
    if mode_norm == WATCH_MODE_FREE_LOSS:
        return {
            "default_output_root": "runs/free_loss_discovery",
            "process_needle": "run_free_loss_discovery_rl4co.py",
            "resume_label": "free_loss",
        }
    return {
        "default_output_root": "runs/pref_loss_coevo",
        "process_needle": "run_pref_loss_coevo.py",
        "resume_label": "pref_loss",
    }


def _build_resume_command(*, watch_mode: str, python_bin: str, config_path: str) -> str:
    mode_norm = _normalize_watch_mode(watch_mode)
    if mode_norm == WATCH_MODE_FREE_LOSS:
        return (
            f"{_q(python_bin)} -u scripts/run_free_loss_discovery_rl4co.py "
            + f"--config {_q(config_path)} --resume-latest"
        )
    return (
        f"{_q(python_bin)} -u PTP/ptp_discovery/run_pref_loss_coevo.py "
        + f"--config {_q(config_path)} --resume-latest"
    )


def _safe_json_loads(text: str) -> dict[str, Any] | None:
    raw = str(text or "").strip()
    if not raw:
        return None
    lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
    for candidate in ([raw] + list(reversed(lines))):
        try:
            payload = json.loads(candidate)
            if isinstance(payload, dict):
                return payload
        except Exception:
            continue
    return None


def _read_remote_boot_id(client: RemoteSSH, *, timeout_s: float) -> tuple[str | None, str | None]:
    rs = client.run("cat /proc/sys/kernel/random/boot_id", timeout_s=timeout_s)
    if rs.returncode != 0:
        err = str(rs.stderr or "").strip().replace("\n", " ")
        if len(err) > 200:
            err = err[:200] + "..."
        return None, (err or f"ssh_rc={rs.returncode}")
    boot_id = str(rs.stdout or "").strip()
    if not boot_id:
        return None, "empty_boot_id"
    return boot_id, None


def _probe_latest_run(
    client: RemoteSSH,
    *,
    watch_mode: str,
    remote_workdir: str,
    config_path: str,
    output_root_override: str,
    remote_python_bin: str,
    timeout_s: float,
) -> dict[str, Any] | None:
    watch_mode_norm = _normalize_watch_mode(watch_mode)
    config_path_py = json.dumps(str(config_path), ensure_ascii=False)
    remote_workdir_py = json.dumps(str(remote_workdir), ensure_ascii=False)
    output_root_override_py = json.dumps(str(output_root_override), ensure_ascii=False)
    watch_mode_py = json.dumps(watch_mode_norm, ensure_ascii=False)
    remote_py = f"""
cd {_q(remote_workdir)} && {_q(remote_python_bin)} - <<'PY'
import json
import os

watch_mode = {watch_mode_py}
config_path = {config_path_py}
workdir = {remote_workdir_py}
output_root_override = {output_root_override_py}

out = {{
    "ok": False,
    "workdir": os.path.abspath(workdir),
    "config_path": os.path.abspath(os.path.join(workdir, config_path)) if not os.path.isabs(config_path) else os.path.abspath(config_path),
}}


def _safe_read_json(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass
    return None


def _to_int(v):
    try:
        return int(v)
    except Exception:
        return None


cfg = {{}}
cfg_path_abs = out["config_path"]
try:
    import yaml  # type: ignore
except Exception:
    yaml = None

if yaml is not None and os.path.isfile(cfg_path_abs):
    try:
        with open(cfg_path_abs, "r", encoding="utf-8") as f:
            loaded = yaml.safe_load(f) or {{}}
        if isinstance(loaded, dict):
            cfg = loaded
    except Exception:
        cfg = {{}}

output_root = str(output_root_override or "").strip()
if not output_root:
    output_root = "runs/free_loss_discovery" if watch_mode == "free_loss" else "runs/pref_loss_coevo"
if isinstance(cfg, dict) and not str(output_root_override or "").strip():
    output_root = str(cfg.get("output_root", output_root) or output_root)
if not os.path.isabs(output_root):
    output_root = os.path.abspath(os.path.join(workdir, output_root))
else:
    output_root = os.path.abspath(output_root)
out["output_root"] = output_root

if not os.path.isdir(output_root):
    out["reason"] = "output_root_missing"
    print(json.dumps(out, ensure_ascii=False))
    raise SystemExit(0)

candidates = []
for name in os.listdir(output_root):
    p = os.path.join(output_root, name)
    if not os.path.isdir(p):
        continue
    if os.path.isfile(os.path.join(p, "checkpoint.json")):
        candidates.append(p)

if not candidates:
    out["reason"] = "no_resumable_runs"
    print(json.dumps(out, ensure_ascii=False))
    raise SystemExit(0)

latest = sorted(candidates)[-1]
ckpt = _safe_read_json(os.path.join(latest, "checkpoint.json")) or {{}}
runtime = _safe_read_json(os.path.join(latest, "runtime_status.json")) or {{}}
summary = _safe_read_json(os.path.join(latest, "summary.json")) or {{}}

next_generation = _to_int(ckpt.get("next_generation"))
target_generations = None
if isinstance(cfg, dict):
    if watch_mode == "free_loss":
        target_generations = _to_int(cfg.get("generations"))
    else:
        budgets = cfg.get("budgets")
        if isinstance(budgets, dict):
            target_generations = _to_int(budgets.get("generations"))

status = str(runtime.get("status", "")).strip().lower()
reason = str(runtime.get("reason", "")).strip().lower()
completed = bool(status == "finished" and reason == "completed")
if (not completed) and (next_generation is not None) and (target_generations is not None):
    if int(next_generation) >= int(target_generations):
        completed = True

out.update({{
    "ok": True,
    "latest_run_dir": os.path.abspath(latest),
    "checkpoint_next_generation": next_generation,
    "target_generations": target_generations,
    "runtime_status": status,
    "runtime_reason": reason,
    "summary_last_generation": _to_int(summary.get("last_generation")) if isinstance(summary, dict) else None,
    "completed": bool(completed),
}})
print(json.dumps(out, ensure_ascii=False))
PY
"""
    rs = client.run(remote_py, timeout_s=timeout_s, use_bash_lc=True)
    if rs.returncode != 0:
        err = str(rs.stderr or "").strip().replace("\n", " ")
        out = str(rs.stdout or "").strip().replace("\n", " ")
        if len(err) > 200:
            err = err[:200] + "..."
        if len(out) > 200:
            out = out[:200] + "..."
        return {
            "ok": False,
            "reason": "probe_exec_failed",
            "ssh_returncode": int(rs.returncode),
            "stderr": err,
            "stdout": out,
        }
    payload = _safe_json_loads(rs.stdout)
    if isinstance(payload, dict):
        return payload
    return {
        "ok": False,
        "reason": "probe_json_parse_failed",
        "stdout": str(rs.stdout or "").strip(),
        "stderr": str(rs.stderr or "").strip(),
    }


def _remote_process_running(
    client: RemoteSSH,
    *,
    process_needle: str,
    timeout_s: float,
) -> tuple[list[str] | None, str | None]:
    rs = client.run(f"pgrep -af {_q(process_needle)} || true", timeout_s=timeout_s)
    if rs.returncode != 0:
        err = str(rs.stderr or "").strip().replace("\n", " ")
        if len(err) > 200:
            err = err[:200] + "..."
        return None, (err or f"ssh_rc={rs.returncode}")
    lines: list[str] = []
    for ln in str(rs.stdout or "").splitlines():
        s = ln.strip()
        if not s:
            continue
        if process_needle in s and "pgrep -af" in s:
            continue
        lines.append(s)
    return lines, None


def _launch_resume(
    client: RemoteSSH,
    *,
    watch_mode: str,
    remote_workdir: str,
    config_path: str,
    remote_log_dir: str,
    python_bin: str,
    log_tz: str,
    log_level: str,
    timeout_s: float,
) -> tuple[int | None, str | None]:
    defaults = _watch_mode_defaults(watch_mode)
    ts = time.strftime("%Y%m%d-%H%M%S", time.localtime())
    remote_log_dir_norm = str(remote_log_dir).replace("\\", "/").rstrip("/")
    if not remote_log_dir_norm:
        remote_log_dir_norm = "logs"
    log_file = f"{remote_log_dir_norm}/auto_resume_{defaults['resume_label']}_{ts}.out"
    resume_cmd = _build_resume_command(
        watch_mode=watch_mode,
        python_bin=python_bin,
        config_path=config_path,
    )
    cmd = " && ".join(
        [
            f"cd {_q(remote_workdir)}",
            f"mkdir -p {_q(remote_log_dir)}",
            f"export PYTHONPATH={_q(remote_workdir)}:{_q(os.path.join(remote_workdir, 'PTP'))}:${{PYTHONPATH:-}}",
            f"export LOG_TZ={_q(log_tz)}",
            f"export LOG_LEVEL={_q(log_level)}",
            ("nohup " + resume_cmd + f" > {_q(log_file)} 2>&1 < /dev/null & echo $!"),
        ]
    )
    rs = client.run(cmd, timeout_s=timeout_s, use_bash_lc=True)
    if rs.returncode != 0:
        return None, None
    raw = str(rs.stdout or "").strip().splitlines()
    pid: int | None = None
    if raw:
        last = raw[-1].strip()
        if last.isdigit():
            pid = int(last)
    return pid, log_file


def _attempt_resume(
    client: RemoteSSH,
    *,
    watch_mode: str,
    reason: str,
    remote_workdir: str,
    config_path: str,
    output_root_override: str,
    remote_log_dir: str,
    python_bin: str,
    log_tz: str,
    log_level: str,
    cmd_timeout_s: float,
    boot_grace_s: float,
    dry_run: bool,
) -> bool:
    defaults = _watch_mode_defaults(watch_mode)
    _log(f"resume_check reason={reason}")
    if boot_grace_s > 0:
        _log(f"waiting_boot_grace_s={boot_grace_s:.1f}")
        time.sleep(boot_grace_s)

    # If the target process is already running, do not probe/launch anything else.
    # This avoids misreporting an older completed run when the current run has not
    # yet emitted a resumable checkpoint.json.
    running, running_err = _remote_process_running(
        client,
        process_needle=defaults["process_needle"],
        timeout_s=cmd_timeout_s,
    )
    if running is None:
        _log(f"proc_query_degraded: reason={running_err}; defer_resume_check_to_online_retry")
        return False
    elif running:
        _log(f"resume_check_noop: {defaults['resume_label']}_process_already_running")
        for ln in running[:3]:
            _log(f"running_proc: {ln}")
        return False

    latest = _probe_latest_run(
        client,
        watch_mode=watch_mode,
        remote_workdir=remote_workdir,
        config_path=config_path,
        output_root_override=output_root_override,
        remote_python_bin=python_bin,
        timeout_s=cmd_timeout_s,
    )
    if not isinstance(latest, dict):
        _log("resume_check_degraded: latest_run_probe_returned_non_dict")
        return False

    if not bool(latest.get("ok")):
        latest_reason = str(latest.get("reason", "probe_failed") or "probe_failed")
        if latest_reason in {"output_root_missing", "no_resumable_runs"}:
            _log(f"resume_check_noop: latest_run_unavailable reason={latest_reason}")
            return False
        _log(
            "resume_check_degraded: "
            + f"latest_run_probe_failed reason={latest_reason} "
            + f"stderr={str(latest.get('stderr', '') or '').strip()} "
            + f"stdout={str(latest.get('stdout', '') or '').strip()}"
        )
        return False

    latest_run_dir = str(latest.get("latest_run_dir", "") or "")
    next_generation = latest.get("checkpoint_next_generation")
    target_generations = latest.get("target_generations")
    completed = bool(latest.get("completed"))
    _log(
        "latest_run_probe: "
        + f"run_dir={latest_run_dir} "
        + f"next_generation={next_generation} "
        + f"target_generations={target_generations} "
        + f"completed={completed}"
    )
    if completed:
        _log("resume_check_noop: latest_run_already_completed")
        return False

    if dry_run:
        _log("dry_run_resume: " + _build_resume_command(
            watch_mode=watch_mode,
            python_bin=python_bin,
            config_path=config_path,
        ))
        return True

    pid, log_file = _launch_resume(
        client,
        watch_mode=watch_mode,
        remote_workdir=remote_workdir,
        config_path=config_path,
        remote_log_dir=remote_log_dir,
        python_bin=python_bin,
        log_tz=log_tz,
        log_level=log_level,
        timeout_s=cmd_timeout_s,
    )
    if pid is None:
        _log("resume_launch_failed")
        return False

    _log(f"resume_launched pid={pid} remote_log={log_file}")
    return True


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Continuously watch a remote server via SSH and auto-resume the latest "
            "incomplete remote run after reboot/reconnect."
        )
    )
    p.add_argument("--host", type=str, required=True, help="SSH target, e.g. user@server")
    p.add_argument(
        "--watch-mode",
        type=str,
        choices=list(WATCH_MODE_CHOICES),
        default=WATCH_MODE_PREF_LOSS,
        help="Which remote workflow to monitor and auto-resume.",
    )
    p.add_argument("--remote-workdir", type=str, default="/data1/gushengda/eam-rl4co")
    p.add_argument(
        "--config",
        type=str,
        default="PTP/configs/experiment/pref_loss_coevo/loss_only_simple.yaml",
        help="Remote config path (absolute or relative to --remote-workdir).",
    )
    p.add_argument("--poll-s", type=float, default=15.0, help="Polling interval in seconds.")
    p.add_argument("--boot-grace-s", type=float, default=45.0, help="Wait time before attempting resume after reconnect/reboot.")
    p.add_argument(
        "--online-retry-s",
        type=float,
        default=120.0,
        help="While online, periodically retry resume checks every N seconds (0 to disable).",
    )
    p.add_argument("--command-timeout-s", type=float, default=20.0)
    p.add_argument("--connect-timeout-s", type=float, default=8.0)
    p.add_argument("--ssh-bin", type=str, default="ssh")
    p.add_argument(
        "--ssh-arg",
        action="append",
        default=[],
        help="Extra SSH arg. Repeat as needed, e.g. --ssh-arg=-i --ssh-arg=/path/key",
    )
    p.add_argument("--python-bin", type=str, default="python")
    p.add_argument("--remote-log-dir", type=str, default="logs")
    p.add_argument(
        "--output-root-override",
        type=str,
        default="",
        help="Optional remote run root override. If set, watcher probes this path instead of reading output_root from YAML.",
    )
    p.add_argument("--log-tz", type=str, default="Asia/Shanghai")
    p.add_argument("--log-level", type=str, default="INFO")
    p.add_argument("--resume-on-start", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--once", action="store_true", help="Run one connectivity cycle then exit.")
    return p


def main() -> int:
    args = _build_parser().parse_args()

    client = RemoteSSH(
        args.host,
        ssh_bin=args.ssh_bin,
        ssh_args=list(args.ssh_arg or []),
        connect_timeout_s=float(args.connect_timeout_s),
    )

    poll_s = max(2.0, float(args.poll_s))
    cmd_timeout_s = max(2.0, float(args.command_timeout_s))
    boot_grace_s = max(0.0, float(args.boot_grace_s))
    online_retry_s = max(0.0, float(args.online_retry_s))

    _log(
        "watcher_start "
        + f"host={args.host} poll_s={poll_s:.1f} boot_grace_s={boot_grace_s:.1f} "
        + f"watch_mode={args.watch_mode} resume_on_start={bool(args.resume_on_start)} dry_run={bool(args.dry_run)} "
        + f"output_root_override={str(args.output_root_override or '').strip() or '<yaml>'}"
    )
    _log(
        "watcher_env "
        + f"pid={os.getpid()} script={os.path.abspath(__file__)} cwd={os.getcwd()} "
        + f"python={sys.executable} ssh_bin={args.ssh_bin} "
        + f"connect_timeout_s={float(args.connect_timeout_s):.1f} command_timeout_s={cmd_timeout_s:.1f}"
    )
    if args.ssh_arg:
        _log("watcher_ssh_args " + " ".join(str(x) for x in list(args.ssh_arg or [])))

    was_online = False
    last_boot_id: str | None = None
    startup_checked = False
    offline_last_log_epoch_s = 0.0
    offline_log_interval_s = max(30.0, poll_s * 4.0)
    next_online_retry_epoch_s = 0.0

    try:
        while True:
            boot_id, offline_reason = _read_remote_boot_id(client, timeout_s=cmd_timeout_s)

            if boot_id is None:
                now = time.time()
                if was_online:
                    _log(f"remote_offline reason={offline_reason}")
                    offline_last_log_epoch_s = now
                elif (now - offline_last_log_epoch_s) >= offline_log_interval_s:
                    _log(f"remote_unreachable reason={offline_reason}")
                    offline_last_log_epoch_s = now
                was_online = False
                next_online_retry_epoch_s = 0.0
            else:
                if not was_online:
                    _log(f"remote_online boot_id={boot_id}")
                    if (not startup_checked) and bool(args.resume_on_start):
                        _attempt_resume(
                            client,
                            watch_mode=args.watch_mode,
                            reason="startup",
                            remote_workdir=args.remote_workdir,
                            config_path=args.config,
                            output_root_override=args.output_root_override,
                            remote_log_dir=args.remote_log_dir,
                            python_bin=args.python_bin,
                            log_tz=args.log_tz,
                            log_level=args.log_level,
                            cmd_timeout_s=cmd_timeout_s,
                            boot_grace_s=boot_grace_s,
                            dry_run=bool(args.dry_run),
                        )
                        startup_checked = True
                    elif last_boot_id and boot_id != last_boot_id:
                        _attempt_resume(
                            client,
                            watch_mode=args.watch_mode,
                            reason="reconnect_boot_id_changed",
                            remote_workdir=args.remote_workdir,
                            config_path=args.config,
                            output_root_override=args.output_root_override,
                            remote_log_dir=args.remote_log_dir,
                            python_bin=args.python_bin,
                            log_tz=args.log_tz,
                            log_level=args.log_level,
                            cmd_timeout_s=cmd_timeout_s,
                            boot_grace_s=boot_grace_s,
                            dry_run=bool(args.dry_run),
                        )
                    if online_retry_s > 0.0:
                        next_online_retry_epoch_s = time.time() + online_retry_s
                else:
                    if last_boot_id and boot_id != last_boot_id:
                        _log(f"boot_id_changed old={last_boot_id} new={boot_id}")
                        _attempt_resume(
                            client,
                            watch_mode=args.watch_mode,
                            reason="boot_id_changed",
                            remote_workdir=args.remote_workdir,
                            config_path=args.config,
                            output_root_override=args.output_root_override,
                            remote_log_dir=args.remote_log_dir,
                            python_bin=args.python_bin,
                            log_tz=args.log_tz,
                            log_level=args.log_level,
                            cmd_timeout_s=cmd_timeout_s,
                            boot_grace_s=boot_grace_s,
                            dry_run=bool(args.dry_run),
                        )
                        if online_retry_s > 0.0:
                            next_online_retry_epoch_s = time.time() + online_retry_s
                    elif online_retry_s > 0.0:
                        now = time.time()
                        if now >= next_online_retry_epoch_s:
                            _attempt_resume(
                                client,
                                watch_mode=args.watch_mode,
                                reason="periodic_online_retry",
                                remote_workdir=args.remote_workdir,
                                config_path=args.config,
                                output_root_override=args.output_root_override,
                                remote_log_dir=args.remote_log_dir,
                                python_bin=args.python_bin,
                                log_tz=args.log_tz,
                                log_level=args.log_level,
                                cmd_timeout_s=cmd_timeout_s,
                                boot_grace_s=0.0,
                                dry_run=bool(args.dry_run),
                            )
                            next_online_retry_epoch_s = now + online_retry_s

                was_online = True
                last_boot_id = boot_id
                offline_last_log_epoch_s = 0.0

            if args.once:
                break
            time.sleep(poll_s)
    except KeyboardInterrupt:
        _log("watcher_stopped_by_keyboard_interrupt")
        return 130

    _log("watcher_exit")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
