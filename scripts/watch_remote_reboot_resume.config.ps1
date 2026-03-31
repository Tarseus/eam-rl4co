$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path

$WatchRemoteRebootResumeConfig = @{
    Host                  = "gushengda@g53"
    WatchMode             = "pref_loss"
    RemoteWorkdir         = "/data1/gushengda/eam-rl4co"
    RemoteConfigPath      = "PTP/configs/experiment/pref_loss_coevo/loss_only_ffsp100_discovery.yaml"
    RemoteOutputRoot      = ""

    PollSeconds           = 15
    BootGraceSeconds      = 45
    OnlineRetrySeconds    = 120
    CommandTimeoutSeconds = 90
    ConnectTimeoutSeconds = 8

    PythonExe             = "python"
    SshExe                = "ssh"
    SshArgs               = @(
        "-i",
        (Join-Path $env:USERPROFILE ".ssh\id_ed25519"),
        "-oBatchMode=yes",
        "-oConnectionAttempts=1",
        "-oServerAliveInterval=10",
        "-oServerAliveCountMax=3"
    )

    RemotePythonBin       = "/data1/gushengda/anaconda3/envs/rlco1/bin/python3.11"
    RemoteLogDir          = "logs"
    LogTz                 = "Asia/Shanghai"
    LogLevel              = "INFO"

    ResumeOnStart         = $true
    DryRun                = $false
    RunInBackground       = $true

    WorkingDirectory      = $RepoRoot
    StdoutLog             = (Join-Path $RepoRoot "logs\watch_remote_reboot_resume.out")
    StderrLog             = (Join-Path $RepoRoot "logs\watch_remote_reboot_resume.err")
}
