param(
    [ValidateSet("start", "stop", "status", "restart", "test")]
    [string]$Action = "start",

    [string]$ConfigPath = (Join-Path $PSScriptRoot "watch_remote_reboot_resume.config.ps1"),

    [switch]$Foreground,

    [int]$TestAttempts = 8
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Get-WatcherProcesses {
    $needle = "watch_remote_reboot_resume.py"
    Get-CimInstance Win32_Process |
        Where-Object { $_.CommandLine -and $_.CommandLine -like "*$needle*" } |
        Sort-Object ProcessId
}

function Load-WatcherConfig {
    param([string]$Path)

    if (-not (Test-Path -LiteralPath $Path)) {
        throw "Config file not found: $Path"
    }

    . $Path

    $var = Get-Variable -Name WatchRemoteRebootResumeConfig -ErrorAction SilentlyContinue
    if ($null -eq $var) {
        throw "Config file must define `$WatchRemoteRebootResumeConfig hashtable: $Path"
    }

    $cfg = $var.Value
    if ($cfg -isnot [hashtable]) {
        throw "`$WatchRemoteRebootResumeConfig must be a hashtable: $Path"
    }

    return $cfg
}

function Build-Args {
    param([hashtable]$Cfg, [string]$ScriptPath)

    $argList = @(
        $ScriptPath,
        "--host", [string]$Cfg.Host,
        "--remote-workdir", [string]$Cfg.RemoteWorkdir,
        "--config", [string]$Cfg.RemoteConfigPath,
        "--poll-s", [string]$Cfg.PollSeconds,
        "--boot-grace-s", [string]$Cfg.BootGraceSeconds,
        "--online-retry-s", [string]$Cfg.OnlineRetrySeconds,
        "--command-timeout-s", [string]$Cfg.CommandTimeoutSeconds,
        "--connect-timeout-s", [string]$Cfg.ConnectTimeoutSeconds,
        "--ssh-bin", [string]$Cfg.SshExe,
        "--python-bin", [string]$Cfg.RemotePythonBin,
        "--remote-log-dir", [string]$Cfg.RemoteLogDir,
        "--log-tz", [string]$Cfg.LogTz,
        "--log-level", [string]$Cfg.LogLevel
    )

    foreach ($item in @($Cfg.SshArgs)) {
        $argList += ("--ssh-arg=" + [string]$item)
    }

    if (-not [bool]$Cfg.ResumeOnStart) {
        $argList += "--no-resume-on-start"
    }

    if ([bool]$Cfg.DryRun) {
        $argList += "--dry-run"
    }

    return $argList
}

function Ensure-LogParent {
    param([string]$FilePath)

    $parent = Split-Path -Path $FilePath -Parent
    if ($parent -and -not (Test-Path -LiteralPath $parent)) {
        New-Item -ItemType Directory -Path $parent -Force | Out-Null
    }
}

function Start-Watcher {
    param([hashtable]$Cfg, [switch]$ForceForeground)

    $scriptPath = Join-Path $PSScriptRoot "watch_remote_reboot_resume.py"
    if (-not (Test-Path -LiteralPath $scriptPath)) {
        throw "Python watcher script not found: $scriptPath"
    }

    $running = Get-WatcherProcesses
    if ($running) {
        Write-Host "Watcher already running:"
        $running | Select-Object ProcessId, Name, CommandLine | Format-Table -AutoSize
        return
    }

    $argList = Build-Args -Cfg $Cfg -ScriptPath $scriptPath

    $localPython = [string]$Cfg.PythonExe
    if ([string]::IsNullOrWhiteSpace($localPython)) {
        throw "PythonExe is empty in config"
    }

    $workdir = [string]$Cfg.WorkingDirectory
    if ([string]::IsNullOrWhiteSpace($workdir)) {
        $workdir = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
    }

    $runInBackground = [bool]$Cfg.RunInBackground
    if ($ForceForeground.IsPresent) {
        $runInBackground = $false
    }

    if (-not $runInBackground) {
        Write-Host "Starting watcher in foreground..."
        & $localPython @argList
        return
    }

    $stdout = [string]$Cfg.StdoutLog
    $stderr = [string]$Cfg.StderrLog
    if ([string]::IsNullOrWhiteSpace($stdout) -or [string]::IsNullOrWhiteSpace($stderr)) {
        throw "StdoutLog/StderrLog must be configured for background mode"
    }

    Ensure-LogParent -FilePath $stdout
    Ensure-LogParent -FilePath $stderr

    $proc = Start-Process -FilePath $localPython `
        -ArgumentList $argList `
        -WorkingDirectory $workdir `
        -WindowStyle Hidden `
        -PassThru `
        -RedirectStandardOutput $stdout `
        -RedirectStandardError $stderr

    Write-Host "Watcher started. PID=$($proc.Id)"
    Write-Host "STDOUT: $stdout"
    Write-Host "STDERR: $stderr"
}

function Stop-Watcher {
    $running = Get-WatcherProcesses
    if (-not $running) {
        Write-Host "Watcher is not running."
        return
    }

    foreach ($p in $running) {
        Stop-Process -Id $p.ProcessId -Force
        Write-Host "Stopped PID=$($p.ProcessId)"
    }
}

function Show-Status {
    $running = Get-WatcherProcesses
    if (-not $running) {
        Write-Host "Watcher status: stopped"
        return
    }

    Write-Host "Watcher status: running"
    $running | Select-Object ProcessId, Name, CommandLine | Format-Table -AutoSize
}

function Quote-BashArg {
    param([string]$Text)

    return [string]$Text
}

function Invoke-ExternalWithTimeout {
    param(
        [string]$FilePath,
        [string[]]$ArgumentList,
        [double]$TimeoutSeconds = 20.0
    )

    $stdoutFile = [System.IO.Path]::GetTempFileName()
    $stderrFile = [System.IO.Path]::GetTempFileName()
    $sw = [Diagnostics.Stopwatch]::StartNew()
    try {
        $proc = Start-Process -FilePath $FilePath `
            -ArgumentList $ArgumentList `
            -PassThru `
            -NoNewWindow `
            -RedirectStandardOutput $stdoutFile `
            -RedirectStandardError $stderrFile

        $finished = $proc.WaitForExit([int]([Math]::Max(1000.0, $TimeoutSeconds * 1000.0)))
        if (-not $finished) {
            Stop-Process -Id $proc.Id -Force -ErrorAction SilentlyContinue
            $sw.Stop()
            return @{
                timeout = $true
                returncode = 124
                elapsed_s = [Math]::Round($sw.Elapsed.TotalSeconds, 2)
                stdout = ""
                stderr = "local_timeout"
            }
        }

        $sw.Stop()
        $stdout = ""
        $stderr = ""
        if (Test-Path -LiteralPath $stdoutFile) {
            $stdout = [string](Get-Content -LiteralPath $stdoutFile -Raw -ErrorAction SilentlyContinue)
        }
        if (Test-Path -LiteralPath $stderrFile) {
            $stderr = [string](Get-Content -LiteralPath $stderrFile -Raw -ErrorAction SilentlyContinue)
        }
        if ($null -eq $stdout) { $stdout = "" }
        if ($null -eq $stderr) { $stderr = "" }

        return @{
            timeout = $false
            returncode = [int]$proc.ExitCode
            elapsed_s = [Math]::Round($sw.Elapsed.TotalSeconds, 2)
            stdout = $stdout.TrimEnd()
            stderr = $stderr.TrimEnd()
        }
    } finally {
        Remove-Item -LiteralPath $stdoutFile -Force -ErrorAction SilentlyContinue
        Remove-Item -LiteralPath $stderrFile -Force -ErrorAction SilentlyContinue
    }
}

function Test-WatcherSsh {
    param([hashtable]$Cfg, [int]$Attempts = 8)

    $attemptsSafe = [Math]::Max(1, [int]$Attempts)
    $sshExe = [string]$Cfg.SshExe
    if ([string]::IsNullOrWhiteSpace($sshExe)) {
        $sshExe = "ssh"
    }

    $connectTimeout = [int]$Cfg.ConnectTimeoutSeconds
    if ($connectTimeout -lt 1) {
        $connectTimeout = 8
    }

    $cmdTimeout = [double]$Cfg.CommandTimeoutSeconds
    if ($cmdTimeout -lt 2.0) {
        $cmdTimeout = 20.0
    }

    $sshCommon = @()
    foreach ($a in @($Cfg.SshArgs)) {
        $sshCommon += [string]$a
    }
    $sshCommon += "-o"
    $sshCommon += ("ConnectTimeout=" + $connectTimeout)
    $sshCommon += [string]$Cfg.Host

    $remoteCmds = @(
        @{ name = "boot_id"; cmd = "cat /proc/sys/kernel/random/boot_id" },
        @{ name = "pgrep"; cmd = "pgrep -af run_pref_loss_coevo.py" },
        @{ name = "remote_py"; cmd = ("cd " + [string]$Cfg.RemoteWorkdir + " && " + [string]$Cfg.RemotePythonBin + " -V") }
    )

    Write-Host ("Testing SSH path with attempts={0}, command_timeout_s={1}, connect_timeout_s={2}" -f $attemptsSafe, $cmdTimeout, $connectTimeout)
    Write-Host ("Host={0}" -f [string]$Cfg.Host)
    Write-Host ""

    $stats = @{}
    foreach ($rc in $remoteCmds) {
        $stats[$rc.name] = @{ ok = 0; fail = 0; timeout = 0 }
    }

    for ($i = 1; $i -le $attemptsSafe; $i++) {
        Write-Host ("[Attempt {0}/{1}]" -f $i, $attemptsSafe)
        foreach ($rc in $remoteCmds) {
            $remoteShell = (Quote-BashArg -Text ([string]$rc.cmd))
            $args = @($sshCommon + $remoteShell)
            $res = Invoke-ExternalWithTimeout -FilePath $sshExe -ArgumentList $args -TimeoutSeconds $cmdTimeout

            $stdout = [string]$res.stdout
            $stderr = [string]$res.stderr
            if ($stdout.Length -gt 100) { $stdout = $stdout.Substring(0, 100) + "..." }
            if ($stderr.Length -gt 120) { $stderr = $stderr.Substring(0, 120) + "..." }

            if ([bool]$res.timeout) {
                $stats[$rc.name].timeout++
                Write-Host ("  {0,-10} TIMEOUT {1,6}s stderr={2}" -f $rc.name, $res.elapsed_s, $stderr)
                continue
            }

            if ([int]$res.returncode -eq 0) {
                $stats[$rc.name].ok++
                Write-Host ("  {0,-10} OK      {1,6}s out={2}" -f $rc.name, $res.elapsed_s, $stdout)
            } else {
                $stats[$rc.name].fail++
                Write-Host ("  {0,-10} FAIL rc={1} {2,6}s stderr={3}" -f $rc.name, $res.returncode, $res.elapsed_s, $stderr)
            }
        }
        Write-Host ""
        Start-Sleep -Milliseconds 500
    }

    Write-Host "Summary:"
    foreach ($rc in $remoteCmds) {
        $s = $stats[$rc.name]
        Write-Host ("  {0,-10} ok={1} fail={2} timeout={3}" -f $rc.name, $s.ok, $s.fail, $s.timeout)
    }
}

$cfg = Load-WatcherConfig -Path $ConfigPath

switch ($Action) {
    "start" {
        Start-Watcher -Cfg $cfg -ForceForeground:$Foreground
    }
    "stop" {
        Stop-Watcher
    }
    "status" {
        Show-Status
    }
    "restart" {
        Stop-Watcher
        Start-Sleep -Seconds 1
        Start-Watcher -Cfg $cfg -ForceForeground:$Foreground
    }
    "test" {
        Test-WatcherSsh -Cfg $cfg -Attempts $TestAttempts
    }
}
