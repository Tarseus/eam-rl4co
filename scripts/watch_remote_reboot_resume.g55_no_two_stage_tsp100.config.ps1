$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path

$WatchRemoteRebootResumeConfig = @{
    Host                  = "g55"
    WatchMode             = "pref_loss"
    RemoteWorkdir         = "/data1/gushengda/eam-rl4co"
    RemoteConfigPath      = "logs/no_two_stage_tsp100_g55_4gpu_resume_20260709-080646.yaml"
    RemoteOutputRoot      = "runs/pref_loss_tsp100_no_two_stage_joint_pair"

    PollSeconds           = 30
    BootGraceSeconds      = 90
    OnlineRetrySeconds    = 180
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

    RemotePythonBin       = "/data1/gushengda/anaconda3/envs/rlco1/bin/python"
    RemoteCudaVisibleDevices = "0,1,2,3"
    RemoteEnvFile         = "/data1/gushengda/.secrets/naapi_openai.env"
    RemoteProcessNeedle   = "no_two_stage_tsp100_g55_4gpu_resume_20260709-080646.yaml"
    RemoteLogDir          = "logs"
    LogTz                 = "Asia/Shanghai"
    LogLevel              = "INFO"

    ResumeOnStart         = $true
    DryRun                = $false
    RunInBackground       = $true

    WorkingDirectory      = $RepoRoot
    StdoutLog             = (Join-Path $RepoRoot "logs\watch_remote_reboot_resume_g55_no_two_stage_tsp100.out")
    StderrLog             = (Join-Path $RepoRoot "logs\watch_remote_reboot_resume_g55_no_two_stage_tsp100.err")
}
