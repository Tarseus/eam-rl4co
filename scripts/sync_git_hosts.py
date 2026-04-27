#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from dataclasses import dataclass
from typing import Sequence


DEFAULT_HOSTS = ("g49", "g51", "g53", "g52")
DEFAULT_REMOTE_REPO = "/data1/gushengda/eam-rl4co"


@dataclass
class CommandResult:
    returncode: int
    stdout: str
    stderr: str


def _run(
    argv: Sequence[str],
    *,
    cwd: str | None = None,
    timeout_s: float = 120.0,
) -> CommandResult:
    try:
        cp = subprocess.run(
            list(argv),
            cwd=cwd,
            text=True,
            capture_output=True,
            timeout=max(1.0, float(timeout_s)),
            check=False,
        )
        return CommandResult(int(cp.returncode), str(cp.stdout or ""), str(cp.stderr or ""))
    except subprocess.TimeoutExpired as exc:
        return CommandResult(
            124,
            str(exc.stdout or ""),
            f"timeout after {timeout_s:.1f}s",
        )


def _git_current_branch(repo_path: str) -> str:
    rs = _run(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=repo_path)
    if rs.returncode != 0:
        raise RuntimeError(f"Failed to detect current branch for {repo_path}: {rs.stderr.strip()}")
    return rs.stdout.strip()


def _git_is_dirty(repo_path: str) -> bool:
    rs = _run(["git", "status", "--short"], cwd=repo_path)
    if rs.returncode != 0:
        raise RuntimeError(f"Failed to inspect git status for {repo_path}: {rs.stderr.strip()}")
    return bool(rs.stdout.strip())


def _ssh_bash(host: str, script: str, *, ssh_bin: str, timeout_s: float) -> CommandResult:
    remote_cmd = f"bash -lc {shlex.quote(script)}"
    return _run(
        [
            ssh_bin,
            "-oBatchMode=yes",
            "-oConnectionAttempts=1",
            host,
            remote_cmd,
        ],
        timeout_s=timeout_s,
    )


def _quote_join(parts: Sequence[str]) -> str:
    return " ".join(shlex.quote(str(p)) for p in parts)


def _remote_status(
    host: str,
    *,
    ssh_bin: str,
    repo_path: str,
    timeout_s: float,
) -> CommandResult:
    script = f"""
set -euo pipefail
cd {shlex.quote(repo_path)}
echo "host={host}"
echo "repo=$(pwd)"
echo "branch=$(git rev-parse --abbrev-ref HEAD)"
echo "remote=$(git remote get-url origin)"
echo "dirty_count=$(git status --short | wc -l | tr -d ' ')"
echo "--- status ---"
git status --short | sed -n '1,80p'
"""
    return _ssh_bash(host, script, ssh_bin=ssh_bin, timeout_s=timeout_s)


def _remote_sync(
    host: str,
    *,
    ssh_bin: str,
    repo_path: str,
    branch: str,
    message: str,
    timeout_s: float,
    checkout_branch: bool,
    all_changes: bool,
    paths: Sequence[str],
) -> CommandResult:
    stage_cmd = ""
    if all_changes:
        stage_cmd = "git add -A"
    elif paths:
        stage_cmd = "git add -A -- " + _quote_join(paths)

    checkout_flag = "1" if checkout_branch else "0"
    script_lines = [
        "set -euo pipefail",
        f"cd {shlex.quote(repo_path)}",
        'current_branch="$(git rev-parse --abbrev-ref HEAD)"',
        f'target_branch={shlex.quote(branch)}',
        f'commit_message={shlex.quote(message)}',
        f"checkout_branch={checkout_flag}",
        'if [ "$current_branch" != "$target_branch" ]; then',
        '  if [ "$checkout_branch" = "1" ]; then',
        '    git checkout "$target_branch"',
        '    current_branch="$target_branch"',
        "  else",
        '    echo "ERROR: branch mismatch on host '
        + host
        + ': current=$current_branch target=$target_branch" >&2',
        "    exit 21",
        "  fi",
        "fi",
        "git fetch origin \"$target_branch\"",
        'dirty_before="$(git status --short)"',
        'if [ -n "$dirty_before" ]; then',
    ]
    if stage_cmd:
        script_lines.extend(
            [
                f"  {stage_cmd}",
                "  if ! git diff --cached --quiet; then",
                '    git commit -m "$commit_message"',
                "  fi",
            ]
        )
    else:
        script_lines.extend(
            [
                '  echo "ERROR: host '
                + host
                + ' has local changes but neither --all-changes nor --path was provided" >&2',
                "  exit 22",
            ]
        )
    script_lines.extend(
        [
            "fi",
            'dirty_after="$(git status --short)"',
            'if [ -n "$dirty_after" ]; then',
            '  echo "ERROR: host '
            + host
            + ' still dirty after staging/commit; refusing pull/push" >&2',
            '  printf "%s\\n" "$dirty_after" >&2',
            "  exit 23",
            "fi",
            'git pull --rebase origin "$target_branch"',
            'git push origin HEAD:"$target_branch"',
            'echo "host=' + host + '"',
            'echo "branch=$(git rev-parse --abbrev-ref HEAD)"',
            'echo "head=$(git rev-parse HEAD)"',
        ]
    )
    return _ssh_bash(host, "\n".join(script_lines), ssh_bin=ssh_bin, timeout_s=timeout_s)


def _local_fetch_or_rebase(
    *,
    repo_path: str,
    branch: str,
    mode: str,
) -> CommandResult:
    if mode == "none":
        return CommandResult(0, "local sync skipped\n", "")
    if mode == "fetch":
        return _run(["git", "fetch", "origin", branch], cwd=repo_path)
    if mode == "rebase":
        return _run(["git", "pull", "--rebase", "origin", branch], cwd=repo_path)
    raise ValueError(f"Unsupported local sync mode: {mode}")


def _resolve_local_mode(repo_path: str, requested: str) -> str:
    if requested != "auto":
        return requested
    return "fetch" if _git_is_dirty(repo_path) else "rebase"


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Inspect or sync git worktrees across remote hosts, then update the local repo. "
            "Remote SSH connectivity relies on your local ~/.ssh/config."
        )
    )
    p.add_argument("mode", choices=("status", "sync"))
    p.add_argument("--hosts", nargs="+", default=list(DEFAULT_HOSTS))
    p.add_argument("--ssh-bin", default="ssh")
    p.add_argument("--remote-repo", default=DEFAULT_REMOTE_REPO)
    p.add_argument("--local-repo", default=os.getcwd())
    p.add_argument("--branch", default="")
    p.add_argument("--timeout-s", type=float, default=180.0)
    p.add_argument("--checkout-branch", action="store_true")
    p.add_argument("--message", default="")
    p.add_argument("--all-changes", action="store_true")
    p.add_argument("--path", action="append", default=[])
    p.add_argument(
        "--local-sync",
        choices=("auto", "fetch", "rebase", "none"),
        default="auto",
        help="After remote pushes: fetch only, pull --rebase, skip, or auto (fetch when local dirty, else rebase).",
    )
    return p


def _print_block(title: str, stdout: str, stderr: str) -> None:
    print(f"== {title} ==")
    if stdout.strip():
        print(stdout.rstrip())
    if stderr.strip():
        print("[stderr]")
        print(stderr.rstrip())


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    local_repo = os.path.abspath(str(args.local_repo))
    branch = str(args.branch or "").strip() or _git_current_branch(local_repo)

    print(f"local_repo={local_repo}")
    print(f"branch={branch}")
    print(f"hosts={','.join(args.hosts)}")

    any_fail = False
    for host in args.hosts:
        if args.mode == "status":
            rs = _remote_status(
                host,
                ssh_bin=args.ssh_bin,
                repo_path=args.remote_repo,
                timeout_s=args.timeout_s,
            )
        else:
            if not str(args.message or "").strip():
                raise RuntimeError("--message is required in sync mode")
            rs = _remote_sync(
                host,
                ssh_bin=args.ssh_bin,
                repo_path=args.remote_repo,
                branch=branch,
                message=str(args.message),
                timeout_s=args.timeout_s,
                checkout_branch=bool(args.checkout_branch),
                all_changes=bool(args.all_changes),
                paths=list(args.path or []),
            )
        _print_block(host, rs.stdout, rs.stderr)
        if rs.returncode != 0:
            any_fail = True
            if args.mode == "sync":
                print(f"sync aborted after host failure: {host}", file=sys.stderr)
                return rs.returncode

    if args.mode == "sync":
        local_mode = _resolve_local_mode(local_repo, str(args.local_sync))
        rs_local = _local_fetch_or_rebase(repo_path=local_repo, branch=branch, mode=local_mode)
        _print_block(f"local ({local_mode})", rs_local.stdout, rs_local.stderr)
        if rs_local.returncode != 0:
            return rs_local.returncode

    return 1 if any_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
