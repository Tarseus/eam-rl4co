---
name: g51-remote-run
description: Use this skill whenever the user asks to connect to g51, run remote commands on g51, start or monitor remote training jobs, inspect g51 logs, check remote GPU/process status, rsync/scp results from g51, or get timely feedback from commands running under /data1/gushengda/eam-rl4co. Prefer this skill for any g51/remote-cluster workflow even if the user only says remote, server, that machine, run it there, check logs, or sync results in the context of this repo.
---

# g51 Remote Run

This skill standardizes remote work on `g51` for the `pre-finder` / `eam-rl4co` workflow. It is meant to make remote command execution observable: submit the command, capture the remote PID/log path, poll status, tail logs, and summarize what happened.

## Defaults

- Local repo: `E:\CAS\perfer\pre-finder`
- Remote host: `g51`
- Remote repo: `/data1/gushengda/eam-rl4co`
- Remote Python: `/data1/gushengda/anaconda3/envs/rlco1/bin/python`
- Remote log root: `/data1/gushengda/eam-rl4co/logs/codex_remote`
- SSH should rely on the user's local SSH config/key. Do not put passwords, private keys, tokens, or host secrets into commands or files.

## Safety

- Treat `g51` as a shared remote machine. Before launching long GPU jobs, check current processes and GPU occupancy.
- Do not run destructive remote commands such as `rm -rf`, `git reset --hard`, mass `kill`, or overwrite checkpoints unless the user explicitly asks for that exact action.
- Prefer `-oBatchMode=yes`, `-oConnectionAttempts=1`, and a short connect timeout so failed connections return quickly.
- If local sandbox/network restrictions block SSH, rerun the SSH command with escalated permissions and a concise justification. Ask for a scoped prefix rule such as `["ssh"]` only when repeated remote checks are needed.
- Keep local and remote shell quoting simple. If the command is complex, put it in a remote `bash -lc '...'` script via the helper rather than building nested quoting by hand.

## Recommended Workflow

1. Confirm the concrete remote action from the user's prompt:
   - inspect status/logs
   - run a short foreground command
   - start a long background training/eval job
   - sync results back to local
2. Check g51 health before long jobs:
   ```powershell
   powershell -ExecutionPolicy Bypass -File .codex\skills\g51-remote-run\scripts\g51_remote.ps1 -Action check
   ```
3. For a quick command, run it in the remote repo and show stdout/stderr:
   ```powershell
   powershell -ExecutionPolicy Bypass -File .codex\skills\g51-remote-run\scripts\g51_remote.ps1 -Action exec -Command "pwd; git status --short; nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv"
   ```
4. For a long job, submit it in the background:
   ```powershell
   powershell -ExecutionPolicy Bypass -File .codex\skills\g51-remote-run\scripts\g51_remote.ps1 -Action start -JobName "jssp10x10_rl_smoke" -Command "export PYTHONPATH=/data1/gushengda/eam-rl4co:${PYTHONPATH:-}; CUDA_VISIBLE_DEVICES=0 python run.py experiment=scheduling/mgl-jssp-rl-batch-10x10"
   ```
5. Immediately report:
   - remote PID
   - remote log path
   - exact command
   - first log tail if available
6. Poll while the command is relevant to the user's request:
   ```powershell
   powershell -ExecutionPolicy Bypass -File .codex\skills\g51-remote-run\scripts\g51_remote.ps1 -Action poll -RemotePid <PID> -LogPath "<REMOTE_LOG_PATH>" -PollCount 6 -PollSeconds 10
   ```
7. Tail an existing log:
   ```powershell
   powershell -ExecutionPolicy Bypass -File .codex\skills\g51-remote-run\scripts\g51_remote.ps1 -Action tail -LogPath "<REMOTE_LOG_PATH>" -Lines 160
   ```
8. Summarize the important feedback for the user. Include concrete error messages, timestamps, PIDs, and whether the process is still running.

## Existing Repo Helpers

Use these when they fit better than raw SSH:

- `scripts/check_remote_user_processes.py`: check user processes across remote hosts, including `g51`.
  ```powershell
  python scripts\check_remote_user_processes.py --hosts g51 --ssh-arg=-oBatchMode=yes --ssh-arg=-oConnectTimeout=5
  ```
- `scripts/sync_git_hosts.py`: inspect or sync git worktrees across hosts.
  ```powershell
  python scripts\sync_git_hosts.py --hosts g51 --mode status
  ```
- `scripts/watch_remote_reboot_resume.ps1`: existing remote resume watcher for configured pref-loss jobs. Read its config before using.

## Output Format

When reporting remote feedback, use this compact structure:

```text
g51 status:
- command: <short command or job name>
- pid: <pid or n/a>
- log: <remote log path or n/a>
- state: running | exited <code> | failed-to-start | connection-failed
- key output: <2-8 high-signal lines>
- next step: <what to do next, if any>
```

For failures, lead with the root cause if the log makes it clear: missing file/config, import error, CUDA OOM, device-side assert, solver timeout, bad alloc, killed process, or SSH/connectivity failure.

## Notes For Training Jobs

- Keep dataloader batch dimension and rollout dimension separate when launching JSSP batching experiments.
- Prefer smoke runs before full runs for RL, PO, and BOPO.
- For full remote jobs, always write logs under `logs/codex_remote` or the script's existing log directory so later tail/sync commands can find them.
- When a job fails, inspect both the process state and the last 200 log lines before drawing conclusions.
