param(
    [string]$OutputDir = "",
    [string]$MiniZincSourceUrl = "https://github.com/MiniZinc/libminizinc/archive/refs/tags/2.9.5.tar.gz",
    [string]$ChuffedSourceUrl = "https://github.com/chuffed/chuffed/archive/refs/heads/develop.tar.gz",
    [switch]$Force
)

$ErrorActionPreference = "Stop"

function Write-Log {
    param([string]$Message)
    Write-Host "[download_minizinc_scheduling_sources] $Message"
}

function Get-RepoRoot {
    $scriptDir = $PSScriptRoot
    return (Resolve-Path (Join-Path $scriptDir "..")).Path
}

function Download-File {
    param(
        [string]$Url,
        [string]$Destination,
        [bool]$Overwrite
    )

    if ((Test-Path -LiteralPath $Destination) -and -not $Overwrite) {
        Write-Log "Reusing existing file: $Destination"
        return
    }

    $parent = Split-Path -Parent $Destination
    if (-not (Test-Path -LiteralPath $parent)) {
        New-Item -ItemType Directory -Path $parent -Force | Out-Null
    }

    Write-Log "Downloading $Url"
    Invoke-WebRequest -Uri $Url -OutFile $Destination
}

$repoRoot = Get-RepoRoot
if ([string]::IsNullOrWhiteSpace($OutputDir)) {
    $OutputDir = Join-Path $repoRoot "tools\src"
}

New-Item -ItemType Directory -Path $OutputDir -Force | Out-Null
$OutputDir = (Resolve-Path -LiteralPath $OutputDir).Path
$minizincArchivePath = Join-Path $OutputDir "libminizinc-src.tar.gz"
$chuffedArchivePath = Join-Path $OutputDir "chuffed-src.tar.gz"

Download-File -Url $MiniZincSourceUrl -Destination $minizincArchivePath -Overwrite:$Force.IsPresent
Download-File -Url $ChuffedSourceUrl -Destination $chuffedArchivePath -Overwrite:$Force.IsPresent

Write-Log "Done."
Write-Log "MiniZinc source archive: $minizincArchivePath"
Write-Log "Chuffed source archive: $chuffedArchivePath"
Write-Log "Upload these two files to the Linux server, then run:"
Write-Log "  bash scripts/install_minizinc_scheduling_solvers_from_source.sh"
