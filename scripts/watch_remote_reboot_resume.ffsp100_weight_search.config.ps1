$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path

$WatchRemoteRebootResumeConfig = @{
    Host                  = "g49"
    WatchMode             = "pref_loss"
    RemoteWorkdir         = "/data1/gushengda/eam-rl4co"
    RemoteConfigPath      = "PTP/configs/experiment/pref_loss_coevo/ffsp100_builder_weight_search_from_archive.yaml"
    RemoteOutputRoot      = "runs/pref_builder_weight_search_ffsp100"

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
    StdoutLog             = (Join-Path $RepoRoot "logs\watch_remote_reboot_resume_ffsp100_weight_search.out")
    StderrLog             = (Join-Path $RepoRoot "logs\watch_remote_reboot_resume_ffsp100_weight_search.err")
}
