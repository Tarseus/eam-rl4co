#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Sequence


DEFAULT_HOSTS = ("g48", "g49", "g50", "g51", "g52", "g54", "g55")


@dataclass
class CommandResult:
    returncode: int
    stdout: str
    stderr: str


@dataclass
class ProcessInfo:
    pid: int
    ppid: int
    elapsed_s: int
    cpu: str
    mem: str
    state: str
    exe: str
    cwd: str
    cmd: str


@dataclass
class HostQueryResult:
    host: str
    ok: bool
    user: str
    processes: list[ProcessInfo]
    error: str
    stderr: str


def _is_noise_process(proc: ProcessInfo) -> bool:
    cmd = str(proc.cmd or "").strip()
    if not cmd:
        return True
    if cmd in {"bash", "-bash", "sh", "-sh", "zsh", "-zsh", "fish", "-fish"}:
        return True
    if cmd == "(sd-pam)":
        return True
    if cmd.startswith("/lib/systemd/systemd --user"):
        return True
    if cmd.startswith("systemd --user"):
        return True
    if cmd == "screen" or cmd.startswith("screen "):
        return True
    if cmd.startswith("sshd: ") and "@notty" in cmd:
        return True
    if cmd == "tmux" or cmd.startswith("tmux "):
        return True
    if cmd.startswith("dbus-daemon --session"):
        return True
    if cmd.startswith("dbus-broker-launch"):
        return True
    if cmd.startswith("dbus-broker "):
        return True
    if "/.vscode-server/" in cmd:
        return True
    if "command-shell --cli-data-dir" in cmd:
        return True
    return False


def _matches_target_command(proc: ProcessInfo) -> bool:
    cmd = str(proc.cmd or "").strip().lower()
    exe_name = PurePosixPath(str(proc.exe or "").strip()).name.lower()
    if not cmd and not exe_name:
        return False
    if "nohup" in cmd:
        return True
    if exe_name.startswith("python"):
        return True
    if cmd.startswith("python ") or cmd.startswith("python3 "):
        return True
    if " python " in f" {cmd} " or " python3 " in f" {cmd} ":
        return True
    return False


def _run(argv: Sequence[str], *, timeout_s: float) -> CommandResult:
    try:
        cp = subprocess.run(
            list(argv),
            text=True,
            capture_output=True,
            timeout=max(1.0, float(timeout_s)),
            check=False,
        )
        return CommandResult(
            returncode=int(cp.returncode),
            stdout=str(cp.stdout or ""),
            stderr=str(cp.stderr or ""),
        )
    except subprocess.TimeoutExpired as exc:
        return CommandResult(
            returncode=124,
            stdout=str(exc.stdout or ""),
            stderr=f"timeout after {timeout_s:.1f}s",
        )


def _ssh_bash(
    host: str,
    script: str,
    *,
    ssh_bin: str,
    ssh_args: Sequence[str],
    timeout_s: float,
) -> CommandResult:
    remote_cmd = f"bash -lc {shlex.quote(script)}"
    return _run(
        [
            ssh_bin,
            *list(ssh_args),
            host,
            remote_cmd,
        ],
        timeout_s=timeout_s,
    )


def _remote_probe(
    host: str,
    *,
    ssh_bin: str,
    ssh_args: Sequence[str],
    timeout_s: float,
    target_user: str,
) -> HostQueryResult:
    remote_script = f"""
set -euo pipefail
if command -v python3 >/dev/null 2>&1; then
  REMOTE_PYTHON=python3
elif command -v python >/dev/null 2>&1; then
  REMOTE_PYTHON=python
else
  echo '{{"ok": false, "error": "python_not_found"}}'
  exit 0
fi
TARGET_USER={shlex.quote(target_user)}
export TARGET_USER
"$REMOTE_PYTHON" - <<'PY'
import json
import os
import pwd
import subprocess


def _readlink(path: str) -> str:
    try:
        return os.readlink(path)
    except Exception:
        return ""


target_user = os.environ.get("TARGET_USER", "").strip()
if not target_user:
    target_user = subprocess.check_output(["id", "-un"], universal_newlines=True).strip()

try:
    target_uid = int(pwd.getpwnam(target_user).pw_uid)
except KeyError:
    print(json.dumps({{"ok": False, "error": "user_not_found:" + target_user}}, ensure_ascii=False))
    raise SystemExit(0)

self_pids = {{os.getpid(), os.getppid()}}
cp = subprocess.run(
    ["ps", "-eo", "pid=,ppid=,uid=,etimes=,%cpu=,%mem=,state=,args=", "--sort=-etimes"],
    universal_newlines=True,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    check=False,
)
if cp.returncode != 0:
    print(json.dumps({{"ok": False, "error": "ps_failed:" + str(cp.returncode)}}))
    raise SystemExit(0)

rows = []
for line in cp.stdout.splitlines():
    raw = line.strip()
    if not raw:
        continue
    parts = raw.split(None, 7)
    if len(parts) < 8:
        continue
    pid_s, ppid_s, uid_s, etimes_s, cpu_s, mem_s, state_s, cmd = parts
    try:
        pid = int(pid_s)
        ppid = int(ppid_s)
        uid = int(uid_s)
        elapsed_s = int(float(etimes_s))
    except ValueError:
        continue
    if uid != target_uid:
        continue
    if pid in self_pids or ppid in self_pids:
        continue
    rows.append(
        {{
            "pid": pid,
            "ppid": ppid,
            "elapsed_s": elapsed_s,
            "cpu": cpu_s,
            "mem": mem_s,
            "state": state_s,
            "exe": _readlink(f"/proc/{{pid}}/exe"),
            "cwd": _readlink(f"/proc/{{pid}}/cwd"),
            "cmd": cmd,
        }}
    )

print(json.dumps({{"ok": True, "user": target_user, "processes": rows}}, ensure_ascii=False))
PY
"""
    rs = _ssh_bash(
        host,
        remote_script,
        ssh_bin=ssh_bin,
        ssh_args=ssh_args,
        timeout_s=timeout_s,
    )
    if rs.returncode != 0:
        return HostQueryResult(
            host=host,
            ok=False,
            user=target_user,
            processes=[],
            error=f"ssh_failed:{rs.returncode}",
            stderr=rs.stderr.strip(),
        )

    payload: dict[str, object] | None = None
    for candidate in reversed([ln.strip() for ln in rs.stdout.splitlines() if ln.strip()]):
        try:
            loaded = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(loaded, dict):
            payload = loaded
            break

    if payload is None:
        return HostQueryResult(
            host=host,
            ok=False,
            user=target_user,
            processes=[],
            error="invalid_json",
            stderr=rs.stderr.strip() or rs.stdout.strip(),
        )

    if not bool(payload.get("ok")):
        return HostQueryResult(
            host=host,
            ok=False,
            user=str(payload.get("user") or target_user),
            processes=[],
            error=str(payload.get("error") or "remote_error"),
            stderr=rs.stderr.strip(),
        )

    processes: list[ProcessInfo] = []
    for item in list(payload.get("processes") or []):
        if not isinstance(item, dict):
            continue
        try:
            processes.append(
                ProcessInfo(
                    pid=int(item.get("pid")),
                    ppid=int(item.get("ppid")),
                    elapsed_s=int(item.get("elapsed_s")),
                    cpu=str(item.get("cpu") or ""),
                    mem=str(item.get("mem") or ""),
                    state=str(item.get("state") or ""),
                    exe=str(item.get("exe") or ""),
                    cwd=str(item.get("cwd") or ""),
                    cmd=str(item.get("cmd") or ""),
                )
            )
        except (TypeError, ValueError):
            continue

    return HostQueryResult(
        host=host,
        ok=True,
        user=str(payload.get("user") or target_user),
        processes=processes,
        error="",
        stderr=rs.stderr.strip(),
    )


def _format_elapsed(elapsed_s: int) -> str:
    elapsed_s = max(0, int(elapsed_s))
    days, rem = divmod(elapsed_s, 24 * 3600)
    hours, rem = divmod(rem, 3600)
    minutes, seconds = divmod(rem, 60)
    if days > 0:
        return f"{days}d{hours:02d}h{minutes:02d}m"
    if hours > 0:
        return f"{hours}h{minutes:02d}m{seconds:02d}s"
    return f"{minutes}m{seconds:02d}s"


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Check which processes the current SSH user is running on multiple remote hosts "
            "and print the full launch command for python/nohup processes."
        )
    )
    p.add_argument("--hosts", nargs="+", default=list(DEFAULT_HOSTS))
    p.add_argument("--user", default="", help="Remote username. Defaults to the SSH login user on each host.")
    p.add_argument("--ssh-bin", default="ssh")
    p.add_argument(
        "--ssh-arg",
        action="append",
        default=[],
        help="Extra SSH argument. Repeat as needed.",
    )
    p.add_argument("--timeout-s", type=float, default=20.0)
    p.add_argument(
        "--max-workers",
        type=int,
        default=min(len(DEFAULT_HOSTS), 8),
        help="How many hosts to query in parallel.",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Limit displayed processes per host. 0 means show all.",
    )
    p.add_argument(
        "--json",
        action="store_true",
        help="Print machine-readable JSON instead of the default text view.",
    )
    p.add_argument(
        "--all",
        action="store_true",
        help="Include session/helper processes such as sshd and systemd --user.",
    )
    return p


def _render_text(results: Sequence[HostQueryResult], *, limit: int) -> str:
    lines: list[str] = []
    for result in results:
        lines.append(f"== {result.host} ==")
        if not result.ok:
            lines.append(f"status: error")
            lines.append(f"error: {result.error}")
            if result.stderr:
                lines.append(f"stderr: {result.stderr}")
            lines.append("")
            continue

        shown = result.processes
        if limit > 0:
            shown = shown[:limit]
        lines.append(f"user: {result.user}")
        lines.append(f"process_count: {len(result.processes)}")
        if not shown:
            lines.append("no user processes found")
            lines.append("")
            continue

        for proc in shown:
            lines.append(
                f"[pid={proc.pid} ppid={proc.ppid} etime={_format_elapsed(proc.elapsed_s)} "
                + f"cpu={proc.cpu}% mem={proc.mem}% state={proc.state}]"
            )
            lines.append(f"exe: {proc.exe or '<unknown>'}")
            lines.append(f"cwd: {proc.cwd or '<unknown>'}")
            lines.append(f"cmd: {proc.cmd or '<empty>'}")
            lines.append("")

        if limit > 0 and len(result.processes) > len(shown):
            lines.append(f"... truncated {len(result.processes) - len(shown)} more process(es)")
            lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    hosts = [str(host).strip() for host in list(args.hosts or []) if str(host).strip()]
    if not hosts:
        raise RuntimeError("No hosts were provided")

    ssh_args = list(args.ssh_arg or [])
    if not any(str(arg).startswith("-oBatchMode=") for arg in ssh_args):
        ssh_args.insert(0, "-oBatchMode=yes")
    if not any(str(arg).startswith("-oConnectionAttempts=") for arg in ssh_args):
        ssh_args.insert(1, "-oConnectionAttempts=1")

    results_by_host: dict[str, HostQueryResult] = {}
    max_workers = max(1, min(int(args.max_workers), len(hosts)))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_map = {
            executor.submit(
                _remote_probe,
                host,
                ssh_bin=str(args.ssh_bin),
                ssh_args=ssh_args,
                timeout_s=float(args.timeout_s),
                target_user=str(args.user or "").strip(),
            ): host
            for host in hosts
        }
        for future in as_completed(future_map):
            host = future_map[future]
            try:
                results_by_host[host] = future.result()
            except Exception as exc:  # noqa: BLE001
                results_by_host[host] = HostQueryResult(
                    host=host,
                    ok=False,
                    user=str(args.user or "").strip(),
                    processes=[],
                    error=f"unexpected_error:{type(exc).__name__}:{exc}",
                    stderr="",
                )

    ordered_results = [results_by_host[host] for host in hosts]
    if not args.all:
        ordered_results = [
            HostQueryResult(
                host=result.host,
                ok=result.ok,
                user=result.user,
                processes=[
                    proc
                    for proc in result.processes
                    if (not _is_noise_process(proc)) and _matches_target_command(proc)
                ],
                error=result.error,
                stderr=result.stderr,
            )
            if result.ok
            else result
            for result in ordered_results
        ]
    if args.json:
        payload = [
            {
                "host": result.host,
                "ok": result.ok,
                "user": result.user,
                "error": result.error,
                "stderr": result.stderr,
                "processes": [
                    {
                        "pid": proc.pid,
                        "ppid": proc.ppid,
                        "elapsed_s": proc.elapsed_s,
                        "cpu": proc.cpu,
                        "mem": proc.mem,
                        "state": proc.state,
                        "exe": proc.exe,
                        "cwd": proc.cwd,
                        "cmd": proc.cmd,
                    }
                    for proc in (
                        result.processes[: int(args.limit)]
                        if int(args.limit) > 0
                        else result.processes
                    )
                ],
            }
            for result in ordered_results
        ]
        sys.stdout.write(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    else:
        sys.stdout.write(_render_text(ordered_results, limit=int(args.limit)))

    return 1 if any(not result.ok for result in ordered_results) else 0


if __name__ == "__main__":
    raise SystemExit(main())
