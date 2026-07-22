param(
    [ValidateSet("check", "exec", "start", "poll", "tail")]
    [string]$Action = "check",
    [string]$HostName = "g51",
    [string]$RemoteWorkdir = "/data1/gushengda/eam-rl4co",
    [string]$RemoteLogDir = "logs/codex_remote",
    [string]$RemotePython = "/data1/gushengda/anaconda3/envs/rlco1/bin/python",
    [string]$Command = "",
    [string]$JobName = "codex_remote_job",
    [int]$RemotePid = 0,
    [string]$LogPath = "",
    [int]$Lines = 120,
    [int]$PollCount = 3,
    [int]$PollSeconds = 10,
    [int]$ConnectTimeoutSeconds = 8
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
$script:RemoteExitCode = 0

function Quote-BashArg {
    param([Parameter(Mandatory=$true)][string]$Text)
    return "'" + ($Text -replace "'", "'\''") + "'"
}

function Invoke-G51Bash {
    param([Parameter(Mandatory=$true)][string]$Script)
    $remote = "bash -lc " + (Quote-BashArg -Text $Script)
    $sshArgs = @(
        "-o", "BatchMode=yes",
        "-o", ("ConnectTimeout={0}" -f $ConnectTimeoutSeconds),
        "-o", "ConnectionAttempts=1",
        "-o", "ServerAliveInterval=10",
        "-o", "ServerAliveCountMax=3",
        $HostName,
        $remote
    )
    & ssh @sshArgs
    if ($null -ne $LASTEXITCODE) {
        $script:RemoteExitCode = [int]$LASTEXITCODE
    } else {
        $script:RemoteExitCode = 0
    }
}

function New-SafeJobName {
    param([string]$Name)
    $safe = ($Name -replace "[^A-Za-z0-9_.-]", "_").Trim("_")
    if ([string]::IsNullOrWhiteSpace($safe)) {
        return "codex_remote_job"
    }
    return $safe
}

if ($Action -eq "check") {
    $script = @"
set -euo pipefail
cd $(Quote-BashArg -Text $RemoteWorkdir)
echo "host=`$(hostname)"
echo "user=`$(id -un)"
echo "repo=`$(pwd)"
echo "git_branch=`$(git rev-parse --abbrev-ref HEAD 2>/dev/null || true)"
echo "git_status_begin"
git status --short 2>/dev/null || true
echo "git_status_end"
echo "python=`$($RemotePython -V 2>&1 || true)"
if command -v nvidia-smi >/dev/null 2>&1; then
  echo "gpu_status_begin"
  nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader
  echo "gpu_status_end"
fi
echo "process_status_begin"
ps -u "`$(id -un)" -o pid,ppid,etime,stat,pcpu,pmem,cmd --sort=-pcpu | head -n 25
echo "process_status_end"
"@
    Invoke-G51Bash -Script $script
    exit $script:RemoteExitCode
}

if ($Action -eq "exec") {
    if ([string]::IsNullOrWhiteSpace($Command)) {
        throw "-Command is required for -Action exec"
    }
    $script = @"
set -euo pipefail
cd $(Quote-BashArg -Text $RemoteWorkdir)
$Command
"@
    Invoke-G51Bash -Script $script
    exit $script:RemoteExitCode
}

if ($Action -eq "start") {
    if ([string]::IsNullOrWhiteSpace($Command)) {
        throw "-Command is required for -Action start"
    }
    $safeName = New-SafeJobName -Name $JobName
    $stamp = Get-Date -Format "yyyyMMdd-HHmmss"
    $remoteLog = "$RemoteWorkdir/$RemoteLogDir/${safeName}_${stamp}.log"
    $commandBase64 = [Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes($Command))
    $script = @"
set -euo pipefail
cd $(Quote-BashArg -Text $RemoteWorkdir)
mkdir -p $(Quote-BashArg -Text $RemoteLogDir)
echo "remote_log=$remoteLog"
printf '%s' $(Quote-BashArg -Text $commandBase64) | base64 --decode | sed 's/^/command=/' > $(Quote-BashArg -Text $remoteLog)
printf '\n' >> $(Quote-BashArg -Text $remoteLog)
echo "started_at=`$(date -Is)" >> $(Quote-BashArg -Text $remoteLog)
nohup bash -lc $(Quote-BashArg -Text $Command) >> $(Quote-BashArg -Text $remoteLog) 2>&1 < /dev/null &
pid=`$!
echo "pid=`$pid"
echo "log=$remoteLog"
sleep 2
if ps -p "`$pid" >/dev/null 2>&1; then
  echo "state=running"
else
  echo "state=exited-early"
fi
echo "tail_begin"
tail -n 80 $(Quote-BashArg -Text $remoteLog) || true
echo "tail_end"
"@
    Invoke-G51Bash -Script $script
    exit $script:RemoteExitCode
}

if ($Action -eq "tail") {
    if ([string]::IsNullOrWhiteSpace($LogPath)) {
        throw "-LogPath is required for -Action tail"
    }
    $script = @"
set -euo pipefail
if [ ! -f $(Quote-BashArg -Text $LogPath) ]; then
  echo "missing_log=$LogPath"
  exit 2
fi
tail -n $Lines $(Quote-BashArg -Text $LogPath)
"@
    Invoke-G51Bash -Script $script
    exit $script:RemoteExitCode
}

if ($Action -eq "poll") {
    if ($RemotePid -le 0) {
        throw "-RemotePid is required for -Action poll"
    }
    if ([string]::IsNullOrWhiteSpace($LogPath)) {
        throw "-LogPath is required for -Action poll"
    }
    for ($i = 1; $i -le $PollCount; $i++) {
        Write-Output ("poll={0}/{1}" -f $i, $PollCount)
        $script = @"
set -euo pipefail
if ps -p $RemotePid >/dev/null 2>&1; then
  echo "state=running"
  ps -p $RemotePid -o pid,ppid,etime,stat,pcpu,pmem,cmd
else
  echo "state=not-running"
fi
if [ -f $(Quote-BashArg -Text $LogPath) ]; then
  echo "tail_begin"
  tail -n $Lines $(Quote-BashArg -Text $LogPath) || true
  echo "tail_end"
else
  echo "missing_log=$LogPath"
fi
"@
        Invoke-G51Bash -Script $script
        $code = $script:RemoteExitCode
        if ($code -ne 0) {
            exit $code
        }
        if ($i -lt $PollCount) {
            Start-Sleep -Seconds $PollSeconds
        }
    }
    exit 0
}
