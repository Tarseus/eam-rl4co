param(
    [ValidateSet("start", "stop", "status", "restart", "test")]
    [string]$Action = "start",

    [switch]$Foreground,

    [int]$TestAttempts = 8
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$runner = Join-Path $PSScriptRoot "watch_remote_reboot_resume.ps1"
$config = Join-Path $PSScriptRoot "watch_remote_reboot_resume.ffsp100_weight_search.config.ps1"

if (-not (Test-Path -LiteralPath $runner)) {
    throw "Watcher runner not found: $runner"
}

& $runner -Action $Action -ConfigPath $config -Foreground:$Foreground -TestAttempts $TestAttempts
