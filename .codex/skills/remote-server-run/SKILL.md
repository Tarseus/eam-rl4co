---
name: remote-server-run
description: Use when connecting to remote GPU servers such as g48-g55, checking GPU/process status, starting long training/evaluation jobs, tailing logs, polling remote PIDs, or syncing results for the pre-finder/eam-rl4co repo. This replaces host-specific remote skills by accepting a host parameter.
---

# Remote Server Run

Use `scripts/remote_server.ps1` for observable remote work on GPU servers.

Defaults:
- Local repo: `E:\CAS\perfer\pre-finder`
- Remote repo: `/data1/gushengda/eam-rl4co`
- Remote Python: `/data1/gushengda/anaconda3/envs/rlco1/bin/python`
- Remote log root: `/data1/gushengda/eam-rl4co/logs/codex_remote`
- Hosts usually available through SSH config: `g48`, `g49`, `g50`, `g51`, `g52`, `g53`, `g54`, `g55`

## Workflow

1. Check host status before long GPU jobs:
   ```powershell
   powershell -ExecutionPolicy Bypass -File .codex\skills\remote-server-run\scripts\remote_server.ps1 -Action check -HostName g51
   ```
2. Run a quick command:
   ```powershell
   powershell -ExecutionPolicy Bypass -File .codex\skills\remote-server-run\scripts\remote_server.ps1 -Action exec -HostName g51 -Command "pwd; nvidia-smi"
   ```
3. Start a long job in the remote repo:
   ```powershell
   powershell -ExecutionPolicy Bypass -File .codex\skills\remote-server-run\scripts\remote_server.ps1 -Action start -HostName g51 -JobName "job_name" -Command "export PYTHONPATH=/data1/gushengda/eam-rl4co:${PYTHONPATH:-}; python run.py ..."
   ```
4. Poll a job:
   ```powershell
   powershell -ExecutionPolicy Bypass -File .codex\skills\remote-server-run\scripts\remote_server.ps1 -Action poll -HostName g51 -RemotePid <PID> -LogPath "<REMOTE_LOG_PATH>" -PollCount 6 -PollSeconds 10
   ```
5. Tail a log:
   ```powershell
   powershell -ExecutionPolicy Bypass -File .codex\skills\remote-server-run\scripts\remote_server.ps1 -Action tail -HostName g51 -LogPath "<REMOTE_LOG_PATH>" -Lines 160
   ```

## Safety

- Treat all GPU hosts as shared machines. Check user processes and `nvidia-smi` before launching.
- Keep long-running logs under `logs/codex_remote` or existing project `logs/`.
- Avoid destructive remote commands unless explicitly requested.
- Prefer one host-specific command per action and report host, PID, log path, and high-signal log tail.
