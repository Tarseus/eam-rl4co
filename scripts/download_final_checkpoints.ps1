param(
    [ValidateSet("best", "last", "both")]
    [string]$CheckpointMode = "best",

    [string]$OutputRoot = "downloads/final_checkpoints",

    [bool]$IncludeMetadata = $true,

    [switch]$DryRun,

    [switch]$ForceRedownload
)

$ErrorActionPreference = "Stop"

$RepoRoot = Split-Path -Parent $PSScriptRoot
$ResolvedOutputRoot = if ([System.IO.Path]::IsPathRooted($OutputRoot)) {
    $OutputRoot
} else {
    Join-Path $RepoRoot $OutputRoot
}

$RemoteRepoRoot = "/data1/gushengda/eam-rl4co"
$RemoteRunsRoot = "$RemoteRepoRoot/logs/train/runs"
$AccessibleHosts = @("g48", "g49", "g51", "g52")
$RoutingHosts = @("g51", "g48", "g49", "g52")
$SchedulingHosts = @("g48", "g49", "g51", "g52")

New-Item -ItemType Directory -Force -Path $ResolvedOutputRoot | Out-Null

function New-Experiment {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Problem,

        [Parameter(Mandatory = $true)]
        [string]$Method,

        [Parameter(Mandatory = $true)]
        [ValidateSet("direct_file", "run_path_pattern", "hparams_grep")]
        [string]$Kind,

        [string[]]$Hosts = $AccessibleHosts,

        [string]$RemotePath,

        [string]$RunPathPattern,

        [int]$RunFindMaxDepth = 1,

        [string]$HparamsContains,

        [string]$Metric,

        [ValidateSet("min", "max")]
        [string]$MetricMode,

        [string]$Notes = ""
    )

    [pscustomobject]@{
        Problem         = $Problem
        Method          = $Method
        Kind            = $Kind
        Hosts           = @($Hosts)
        RemotePath      = $RemotePath
        RunPathPattern  = $RunPathPattern
        RunFindMaxDepth = $RunFindMaxDepth
        HparamsContains = $HparamsContains
        Metric          = $Metric
        MetricMode      = $MetricMode
        Notes           = $Notes
    }
}

function Invoke-RemoteCommand {
    param(
        [Parameter(Mandatory = $true)]
        [string]$RemoteHost,

        [Parameter(Mandatory = $true)]
        [string]$Command
    )

    $output = & ssh -o BatchMode=yes -o ConnectTimeout=5 $RemoteHost $Command 2>$null
    if ($LASTEXITCODE -ne 0) {
        return $null
    }
    return $output
}

function Find-LatestRemoteRunByPathPattern {
    param(
        [Parameter(Mandatory = $true)]
        [string]$RemoteHost,

        [Parameter(Mandatory = $true)]
        [string]$Pattern,

        [Parameter(Mandatory = $true)]
        [int]$MaxDepth
    )

    $remoteCommand = "cd '$RemoteRunsRoot' && find . -maxdepth $MaxDepth -mindepth 1 -type d -path '$Pattern' -printf '%T@ %p`n' | sort -n | tail -n 1"
    $result = Invoke-RemoteCommand -RemoteHost $RemoteHost -Command $remoteCommand
    if (-not $result) {
        return $null
    }

    $line = ($result | Select-Object -Last 1).Trim()
    if ([string]::IsNullOrWhiteSpace($line)) {
        return $null
    }

    $parts = $line -split "\s+", 2
    if ($parts.Count -lt 2) {
        return $null
    }

    $relative = $parts[1].Trim()
    if ($relative.StartsWith("./")) {
        $relative = $relative.Substring(2)
    }

    [pscustomobject]@{
        Host        = $RemoteHost
        RelativeRun = $relative
        RunDir      = "$RemoteRunsRoot/$relative"
    }
}

function Find-LatestRemoteRunByHparamsContent {
    param(
        [Parameter(Mandatory = $true)]
        [string]$RemoteHost,

        [Parameter(Mandatory = $true)]
        [string]$Needle
    )

    $escapedNeedle = $Needle.Replace("'", "'\''")
    $remoteCommand = @"
cd '$RemoteRunsRoot' &&
find . -maxdepth 5 -name hparams.yaml -print |
while read file; do
  if grep -F -q -- '$escapedNeedle' "`$file"; then
    run_dir="`$(dirname "`$(dirname "`$(dirname "`$file")")")"
    printf '%s %s\n' "`$(stat -c %Y "`$run_dir")" "`$run_dir"
  fi
done | sort -n | tail -n 1
"@
    $result = Invoke-RemoteCommand -RemoteHost $RemoteHost -Command $remoteCommand
    if (-not $result) {
        return $null
    }

    $line = ($result | Select-Object -Last 1).Trim()
    if ([string]::IsNullOrWhiteSpace($line)) {
        return $null
    }

    $parts = $line -split "\s+", 2
    if ($parts.Count -lt 2) {
        return $null
    }

    $relative = $parts[1].Trim()
    if ($relative.StartsWith("./")) {
        $relative = $relative.Substring(2)
    }

    [pscustomobject]@{
        Host        = $RemoteHost
        RelativeRun = $relative
        RunDir      = "$RemoteRunsRoot/$relative"
    }
}

function Get-LatestRemoteArtifactPath {
    param(
        [Parameter(Mandatory = $true)]
        [string]$RemoteHost,

        [Parameter(Mandatory = $true)]
        [string]$RunDir,

        [Parameter(Mandatory = $true)]
        [string]$FileName
    )

    $remoteCommand = "find '$RunDir' -type f -name '$FileName' | sort | tail -n 1"
    $result = Invoke-RemoteCommand -RemoteHost $RemoteHost -Command $remoteCommand
    if (-not $result) {
        return $null
    }

    $path = ($result | Select-Object -Last 1).Trim()
    if ([string]::IsNullOrWhiteSpace($path)) {
        return $null
    }
    return $path
}

function Get-RemoteArtifactPaths {
    param(
        [Parameter(Mandatory = $true)]
        [string]$RemoteHost,

        [Parameter(Mandatory = $true)]
        [string]$RunDir,

        [Parameter(Mandatory = $true)]
        [string]$FileName
    )

    $remoteCommand = "find '$RunDir' -type f -name '$FileName' | sort"
    $result = Invoke-RemoteCommand -RemoteHost $RemoteHost -Command $remoteCommand
    if (-not $result) {
        return @()
    }

    return @(
        $result |
            Where-Object { -not [string]::IsNullOrWhiteSpace($_) } |
            ForEach-Object { $_.Trim() }
    )
}

function Get-CompanionRemoteArtifactPath {
    param(
        [Parameter(Mandatory = $true)]
        [string]$RemoteHost,

        [Parameter(Mandatory = $true)]
        [string]$ReferencePath,

        [Parameter(Mandatory = $true)]
        [string]$FileName
    )

    $referenceDir = [System.IO.Path]::GetDirectoryName($ReferencePath).Replace('\', '/')
    $candidatePath = "$referenceDir/$FileName"
    $remoteCommand = "test -f '$candidatePath' && printf '%s\n' '$candidatePath'"
    $result = Invoke-RemoteCommand -RemoteHost $RemoteHost -Command $remoteCommand
    if (-not $result) {
        return $null
    }

    $path = ($result | Select-Object -Last 1).Trim()
    if ([string]::IsNullOrWhiteSpace($path)) {
        return $null
    }

    return $path
}

function Get-PreferredMetricsArtifactPath {
    param(
        [Parameter(Mandatory = $true)]
        [string]$RemoteHost,

        [Parameter(Mandatory = $true)]
        [string]$RunDir,

        [Parameter(Mandatory = $true)]
        [string]$Metric
    )

    $metricPaths = Get-RemoteArtifactPaths -RemoteHost $RemoteHost -RunDir $RunDir -FileName "metrics.csv"
    if ($metricPaths.Count -eq 0) {
        return $null
    }

    $bestPath = $null
    $bestCount = -1
    $bestMaxEpoch = -1

    foreach ($path in $metricPaths) {
        $csvText = Invoke-RemoteCommand -RemoteHost $RemoteHost -Command "cat '$path'"
        if (-not $csvText) {
            continue
        }

        $rowCount = 0
        $maxEpoch = -1
        $rows = $csvText | ConvertFrom-Csv
        foreach ($row in $rows) {
            $epochRaw = $row.epoch
            $metricProp = $row.PSObject.Properties[$Metric]
            if ($null -eq $metricProp) {
                continue
            }

            $metricRaw = $metricProp.Value
            if ([string]::IsNullOrWhiteSpace($epochRaw) -or [string]::IsNullOrWhiteSpace($metricRaw)) {
                continue
            }

            $epoch = 0
            if (-not [int]::TryParse($epochRaw, [ref]$epoch)) {
                continue
            }

            $metricValue = 0.0
            if (-not [double]::TryParse($metricRaw, [System.Globalization.NumberStyles]::Float, [System.Globalization.CultureInfo]::InvariantCulture, [ref]$metricValue)) {
                continue
            }

            $rowCount += 1
            if ($epoch -gt $maxEpoch) {
                $maxEpoch = $epoch
            }
        }

        if (($rowCount -gt $bestCount) -or (($rowCount -eq $bestCount) -and ($maxEpoch -gt $bestMaxEpoch))) {
            $bestPath = $path
            $bestCount = $rowCount
            $bestMaxEpoch = $maxEpoch
        }
    }

    if (-not [string]::IsNullOrWhiteSpace($bestPath)) {
        return $bestPath
    }

    return $metricPaths[-1]
}

function Get-MergedMetricsRows {
    param(
        [Parameter(Mandatory = $true)]
        [pscustomobject]$ResolvedExperiment
    )

    if ([string]::IsNullOrWhiteSpace($ResolvedExperiment.ResolvedRunDir)) {
        return @()
    }

    $metricPaths = Get-RemoteArtifactPaths -RemoteHost $ResolvedExperiment.Host -RunDir $ResolvedExperiment.ResolvedRunDir -FileName "metrics.csv"
    if ($metricPaths.Count -eq 0) {
        return @()
    }

    $rowsByEpoch = @{}

    foreach ($path in $metricPaths) {
        $csvText = Invoke-RemoteCommand -RemoteHost $ResolvedExperiment.Host -Command "cat '$path'"
        if (-not $csvText) {
            continue
        }

        $rows = $csvText | ConvertFrom-Csv
        foreach ($row in $rows) {
            $epochRaw = $row.epoch
            $metricProp = $row.PSObject.Properties[$ResolvedExperiment.Metric]
            if ($null -eq $metricProp) {
                continue
            }

            $metricRaw = $metricProp.Value
            if ([string]::IsNullOrWhiteSpace($epochRaw) -or [string]::IsNullOrWhiteSpace($metricRaw)) {
                continue
            }

            $epoch = 0
            if (-not [int]::TryParse($epochRaw, [ref]$epoch)) {
                continue
            }

            $metricValue = 0.0
            if (-not [double]::TryParse($metricRaw, [System.Globalization.NumberStyles]::Float, [System.Globalization.CultureInfo]::InvariantCulture, [ref]$metricValue)) {
                continue
            }

            $rowsByEpoch[$epoch] = [pscustomobject]@{
                Epoch       = $epoch
                MetricValue = $metricValue
                SourcePath  = $path
            }
        }
    }

    return @($rowsByEpoch.Values | Sort-Object Epoch)
}

function Get-AvailableEpochCheckpointRecords {
    param(
        [Parameter(Mandatory = $true)]
        [pscustomobject]$ResolvedExperiment
    )

    if ([string]::IsNullOrWhiteSpace($ResolvedExperiment.ResolvedRunDir)) {
        return @()
    }

    $remoteCommand = "find '$($ResolvedExperiment.ResolvedRunDir)' -type f -name 'epoch_*.ckpt' | sort"
    $paths = Invoke-RemoteCommand -RemoteHost $ResolvedExperiment.Host -Command $remoteCommand
    if (-not $paths) {
        return @()
    }

    $records = @()
    foreach ($path in $paths) {
        if ([string]::IsNullOrWhiteSpace($path)) {
            continue
        }

        $trimmedPath = $path.Trim()
        $fileName = [System.IO.Path]::GetFileName($trimmedPath)
        if ($fileName -notmatch '^epoch_(\d+)\.ckpt$') {
            continue
        }

        $records += [pscustomobject]@{
            FileName   = $fileName
            Epoch      = [int]$Matches[1]
            RemotePath = $trimmedPath
        }
    }

    return @($records)
}

function Test-RunHasAnyCheckpoint {
    param(
        [Parameter(Mandatory = $true)]
        [string]$RemoteHost,

        [Parameter(Mandatory = $true)]
        [string]$RunDir
    )

    $remoteCommand = "find '$RunDir' -type f \( -name 'epoch_*.ckpt' -o -name 'last.ckpt' \) | head -n 1"
    $result = Invoke-RemoteCommand -RemoteHost $RemoteHost -Command $remoteCommand
    if (-not $result) {
        return $false
    }

    $line = ($result | Select-Object -First 1).Trim()
    return -not [string]::IsNullOrWhiteSpace($line)
}

function Resolve-Experiment {
    param(
        [Parameter(Mandatory = $true)]
        [pscustomobject]$Experiment
    )

    $fallbackResolved = $null

    foreach ($remoteHost in $Experiment.Hosts) {
        switch ($Experiment.Kind) {
            "direct_file" {
                $remotePath = "$RemoteRepoRoot/$($Experiment.RemotePath)"
                $check = Invoke-RemoteCommand -RemoteHost $remoteHost -Command "test -f '$remotePath' && printf '%s\n' '$remotePath'"
                if ($check) {
                    return [pscustomobject]@{
                        Problem         = $Experiment.Problem
                        Method          = $Experiment.Method
                        Kind            = $Experiment.Kind
                        Host            = $remoteHost
                        ResolvedRunDir  = $null
                        ResolvedRemote  = $remotePath
                        MetricsPath     = $null
                        HparamsPath     = $null
                        Metric          = $Experiment.Metric
                        MetricMode      = $Experiment.MetricMode
                        Notes           = $Experiment.Notes
                    }
                }
            }
            "run_path_pattern" {
                $resolvedRun = Find-LatestRemoteRunByPathPattern -RemoteHost $remoteHost -Pattern $Experiment.RunPathPattern -MaxDepth $Experiment.RunFindMaxDepth
                if ($resolvedRun) {
                    $candidate = [pscustomobject]@{
                        Problem         = $Experiment.Problem
                        Method          = $Experiment.Method
                        Kind            = $Experiment.Kind
                        Host            = $remoteHost
                        ResolvedRunDir  = $resolvedRun.RunDir
                        ResolvedRemote  = $null
                        MetricsPath     = Get-PreferredMetricsArtifactPath -RemoteHost $remoteHost -RunDir $resolvedRun.RunDir -Metric $Experiment.Metric
                        HparamsPath     = $null
                        Metric          = $Experiment.Metric
                        MetricMode      = $Experiment.MetricMode
                        Notes           = $Experiment.Notes
                    }
                    if (-not [string]::IsNullOrWhiteSpace($candidate.MetricsPath)) {
                        $candidate.HparamsPath = Get-CompanionRemoteArtifactPath -RemoteHost $remoteHost -ReferencePath $candidate.MetricsPath -FileName "hparams.yaml"
                    }
                    if ([string]::IsNullOrWhiteSpace($candidate.HparamsPath)) {
                        $candidate.HparamsPath = Get-LatestRemoteArtifactPath -RemoteHost $remoteHost -RunDir $resolvedRun.RunDir -FileName "hparams.yaml"
                    }
                    if (Test-RunHasAnyCheckpoint -RemoteHost $remoteHost -RunDir $resolvedRun.RunDir) {
                        return $candidate
                    }
                    if ($null -eq $fallbackResolved) {
                        $fallbackResolved = $candidate
                    }
                }
            }
            "hparams_grep" {
                $resolvedRun = Find-LatestRemoteRunByHparamsContent -RemoteHost $remoteHost -Needle $Experiment.HparamsContains
                if ($resolvedRun) {
                    $candidate = [pscustomobject]@{
                        Problem         = $Experiment.Problem
                        Method          = $Experiment.Method
                        Kind            = $Experiment.Kind
                        Host            = $remoteHost
                        ResolvedRunDir  = $resolvedRun.RunDir
                        ResolvedRemote  = $null
                        MetricsPath     = Get-PreferredMetricsArtifactPath -RemoteHost $remoteHost -RunDir $resolvedRun.RunDir -Metric $Experiment.Metric
                        HparamsPath     = $null
                        Metric          = $Experiment.Metric
                        MetricMode      = $Experiment.MetricMode
                        Notes           = $Experiment.Notes
                    }
                    if (-not [string]::IsNullOrWhiteSpace($candidate.MetricsPath)) {
                        $candidate.HparamsPath = Get-CompanionRemoteArtifactPath -RemoteHost $remoteHost -ReferencePath $candidate.MetricsPath -FileName "hparams.yaml"
                    }
                    if ([string]::IsNullOrWhiteSpace($candidate.HparamsPath)) {
                        $candidate.HparamsPath = Get-LatestRemoteArtifactPath -RemoteHost $remoteHost -RunDir $resolvedRun.RunDir -FileName "hparams.yaml"
                    }
                    if (Test-RunHasAnyCheckpoint -RemoteHost $remoteHost -RunDir $resolvedRun.RunDir) {
                        return $candidate
                    }
                    if ($null -eq $fallbackResolved) {
                        $fallbackResolved = $candidate
                    }
                }
            }
        }
    }

    return $fallbackResolved
}

function Get-BestCheckpointFileName {
    param(
        [Parameter(Mandatory = $true)]
        [pscustomobject]$ResolvedExperiment
    )

    $metricRows = Get-MergedMetricsRows -ResolvedExperiment $ResolvedExperiment
    if ($metricRows.Count -eq 0) {
        $fallback = Get-SoleEpochCheckpointFileName -ResolvedExperiment $ResolvedExperiment
        if ($fallback) {
            return $fallback
        }
        throw "Cannot resolve best checkpoint without metrics.csv for $($ResolvedExperiment.Problem)/$($ResolvedExperiment.Method)"
    }

    $bestEpoch = $null
    $bestValue = $null

    foreach ($row in $metricRows) {
        if ($null -eq $bestEpoch) {
            $bestEpoch = $row.Epoch
            $bestValue = $row.MetricValue
            continue
        }

        $isBetter = if ($ResolvedExperiment.MetricMode -eq "min") {
            $row.MetricValue -lt $bestValue
        } else {
            $row.MetricValue -gt $bestValue
        }

        if ($isBetter) {
            $bestEpoch = $row.Epoch
            $bestValue = $row.MetricValue
        }
    }

    if ($null -eq $bestEpoch) {
        $fallback = Get-SoleEpochCheckpointFileName -ResolvedExperiment $ResolvedExperiment
        if ($fallback) {
            return $fallback
        }
        throw "Unable to infer best epoch from metrics.csv for $($ResolvedExperiment.Problem)/$($ResolvedExperiment.Method)"
    }

    $bestFileName = ("epoch_{0:D3}.ckpt" -f [int]$bestEpoch)
    $availableEpochRecords = Get-AvailableEpochCheckpointRecords -ResolvedExperiment $ResolvedExperiment
    if ($availableEpochRecords.Count -eq 0) {
        return $bestFileName
    }

    $exactMatch = $availableEpochRecords | Where-Object { $_.Epoch -eq $bestEpoch } | Select-Object -First 1
    if ($exactMatch) {
        return $exactMatch.FileName
    }

    $metricRowsByEpoch = @{}
    foreach ($row in $metricRows) {
        $metricRowsByEpoch[$row.Epoch] = $row
    }

    $availableWithMetrics = @(
        $availableEpochRecords |
            Where-Object { $metricRowsByEpoch.ContainsKey($_.Epoch) }
    )

    if ($availableWithMetrics.Count -gt 0) {
        $chosen = if ($ResolvedExperiment.MetricMode -eq "min") {
            $availableWithMetrics | Sort-Object { $metricRowsByEpoch[$_.Epoch].MetricValue }, Epoch | Select-Object -First 1
        } else {
            $availableWithMetrics | Sort-Object @{ Expression = { $metricRowsByEpoch[$_.Epoch].MetricValue }; Descending = $true }, Epoch | Select-Object -First 1
        }
    } else {
        $chosen = $availableEpochRecords | Sort-Object @{ Expression = { [Math]::Abs($_.Epoch - $bestEpoch) } }, Epoch | Select-Object -First 1
    }

    if ($chosen) {
        Write-Host ("Best epoch {0:D3} was not checkpointed for {1}/{2}; using available checkpoint {3}." -f $bestEpoch, $ResolvedExperiment.Problem, $ResolvedExperiment.Method, $chosen.FileName)
        return $chosen.FileName
    }

    return $bestFileName
}

function Get-SoleEpochCheckpointFileName {
    param(
        [Parameter(Mandatory = $true)]
        [pscustomobject]$ResolvedExperiment
    )

    $epochRecords = Get-AvailableEpochCheckpointRecords -ResolvedExperiment $ResolvedExperiment
    if ($epochRecords.Count -eq 1) {
        return $epochRecords[0].FileName
    }

    return $null
}

function Resolve-CheckpointRemotePath {
    param(
        [Parameter(Mandatory = $true)]
        [pscustomobject]$ResolvedExperiment,

        [Parameter(Mandatory = $true)]
        [string]$FileName
    )

    if ($ResolvedExperiment.Kind -eq "direct_file") {
        return $ResolvedExperiment.ResolvedRemote
    }

    if ([string]::IsNullOrWhiteSpace($ResolvedExperiment.ResolvedRunDir)) {
        return $null
    }

    $preferredPath = "$($ResolvedExperiment.ResolvedRunDir)/checkpoints/$FileName"
    $remoteCommand = @"
if test -f '$preferredPath'; then
  printf '%s\n' '$preferredPath'
else
  find '$($ResolvedExperiment.ResolvedRunDir)' -type f -name '$FileName' | sort | tail -n 1
fi
"@
    $result = Invoke-RemoteCommand -RemoteHost $ResolvedExperiment.Host -Command $remoteCommand
    if (-not $result) {
        return $null
    }

    $resolvedPath = ($result | Select-Object -Last 1).Trim()
    if ([string]::IsNullOrWhiteSpace($resolvedPath)) {
        return $null
    }

    return $resolvedPath
}

function Get-CheckpointFileNames {
    param(
        [Parameter(Mandatory = $true)]
        [pscustomobject]$ResolvedExperiment,

        [Parameter(Mandatory = $true)]
        [string]$Mode
    )

    if ($ResolvedExperiment.Kind -eq "direct_file") {
        return @([System.IO.Path]::GetFileName($ResolvedExperiment.ResolvedRemote))
    }

    $files = @()
    if ($Mode -in @("best", "both")) {
        $files += Get-BestCheckpointFileName -ResolvedExperiment $ResolvedExperiment
    }
    if ($Mode -in @("last", "both")) {
        $files += "last.ckpt"
    }
    return $files | Select-Object -Unique
}

function Invoke-ScpDownload {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Remote,

        [Parameter(Mandatory = $true)]
        [string]$LocalPath
    )

    $localDir = Split-Path -Parent $LocalPath
    New-Item -ItemType Directory -Force -Path $localDir | Out-Null

    if ((-not $ForceRedownload) -and (Test-Path -LiteralPath $LocalPath)) {
        Write-Host "Skip existing: $LocalPath"
        return
    }

    Write-Host ("scp {0} {1}" -f $Remote, $LocalPath)
    if (-not $DryRun) {
        & scp $Remote $LocalPath
        if ($LASTEXITCODE -ne 0) {
            throw "scp failed: $Remote"
        }
    }
}

$experiments = @(
    New-Experiment -Problem "tsp100"    -Method "po"         -Kind "direct_file"      -Hosts $RoutingHosts    -RemotePath "baseline/tsp100_epoch_135.ckpt"                    -Notes "Routing PO baseline checkpoint"
    New-Experiment -Problem "tsp100"    -Method "sll"        -Kind "run_path_pattern" -Hosts $RoutingHosts    -RunPathPattern "./tsp100_cvrp100_baselines_*/tsp100_sll"      -RunFindMaxDepth 2 -Metric "val/reward" -MetricMode "max"
    New-Experiment -Problem "tsp100"    -Method "bopo"       -Kind "run_path_pattern" -Hosts $RoutingHosts    -RunPathPattern "./tsp100_cvrp100_baselines_*/tsp100_bopo"     -RunFindMaxDepth 2 -Metric "val/reward" -MetricMode "max"
    New-Experiment -Problem "tsp100"    -Method "loss_only"  -Kind "run_path_pattern" -Hosts $RoutingHosts    -RunPathPattern "./my_loss_TSP100"                             -RunFindMaxDepth 1 -Metric "val/reward" -MetricMode "max"
    New-Experiment -Problem "tsp100"    -Method "weighting"  -Kind "run_path_pattern" -Hosts $RoutingHosts    -RunPathPattern "./loss_weighting_tsp100"                      -RunFindMaxDepth 1 -Metric "val/reward" -MetricMode "max"

    New-Experiment -Problem "cvrp100"   -Method "po"         -Kind "direct_file"      -Hosts $RoutingHosts    -RemotePath "baseline/cvrp100_epoch_100.ckpt"                  -Notes "Routing PO baseline checkpoint"
    New-Experiment -Problem "cvrp100"   -Method "sll"        -Kind "run_path_pattern" -Hosts $RoutingHosts    -RunPathPattern "./tsp100_cvrp100_baselines_*/cvrp100_sll"    -RunFindMaxDepth 2 -Metric "val/reward" -MetricMode "max"
    New-Experiment -Problem "cvrp100"   -Method "bopo"       -Kind "run_path_pattern" -Hosts $RoutingHosts    -RunPathPattern "./tsp100_cvrp100_baselines_*/cvrp100_bopo"   -RunFindMaxDepth 2 -Metric "val/reward" -MetricMode "max"
    New-Experiment -Problem "cvrp100"   -Method "loss_only"  -Kind "hparams_grep"     -Hosts $RoutingHosts    -HparamsContains "pref_pair_json_path: runs/pref_loss_cvrp100_from_tsp100_elite/" -Metric "val/reward" -MetricMode "max"
    New-Experiment -Problem "cvrp100"   -Method "weighting"  -Kind "hparams_grep"     -Hosts $RoutingHosts    -HparamsContains "pref_pair_json_path: runs/pref_builder_weight_search_cvrp100/"  -Metric "val/reward" -MetricMode "max"

    New-Experiment -Problem "ffsp50"    -Method "rl"         -Kind "run_path_pattern" -Hosts $SchedulingHosts -RunPathPattern "./ffsp_matnet_rl_50_*"                       -RunFindMaxDepth 1 -Metric "val/reward" -MetricMode "max"
    New-Experiment -Problem "ffsp50"    -Method "po"         -Kind "run_path_pattern" -Hosts $SchedulingHosts -RunPathPattern "./ffsp_matnet_po_50_*"                       -RunFindMaxDepth 1 -Metric "val/reward" -MetricMode "max"
    New-Experiment -Problem "ffsp50"    -Method "bopo"       -Kind "run_path_pattern" -Hosts $SchedulingHosts -RunPathPattern "./ffsp_matnet_bopo_50_*"                     -RunFindMaxDepth 1 -Metric "val/reward" -MetricMode "max"
    New-Experiment -Problem "ffsp50"    -Method "loss_only"  -Kind "run_path_pattern" -Hosts $SchedulingHosts -RunPathPattern "./ffsp50_loss_only_pref_*"                   -RunFindMaxDepth 1 -Metric "val/reward" -MetricMode "max"
    New-Experiment -Problem "ffsp50"    -Method "weighting"  -Kind "run_path_pattern" -Hosts $SchedulingHosts -RunPathPattern "./ffsp50_weighting_pref_*"                   -RunFindMaxDepth 1 -Metric "val/reward" -MetricMode "max"

    New-Experiment -Problem "jssp10x10" -Method "rl"         -Kind "run_path_pattern" -Hosts $SchedulingHosts -RunPathPattern "./mgl-jssp-rl_paper_10x10_*"                 -RunFindMaxDepth 1 -Metric "val/gap"    -MetricMode "min"
    New-Experiment -Problem "jssp10x10" -Method "po"         -Kind "run_path_pattern" -Hosts $SchedulingHosts -RunPathPattern "./mgl-jssp-po_paper_10x10_*"                 -RunFindMaxDepth 1 -Metric "val/gap"    -MetricMode "min"
    New-Experiment -Problem "jssp10x10" -Method "bopo"       -Kind "run_path_pattern" -Hosts $SchedulingHosts -RunPathPattern "./mgl-jssp-bopo_paper_10x10_*"               -RunFindMaxDepth 1 -Metric "val/gap"    -MetricMode "min"
    New-Experiment -Problem "jssp10x10" -Method "sll"        -Kind "run_path_pattern" -Hosts $SchedulingHosts -RunPathPattern "./mgl-jssp-sll_10x10_*"                      -RunFindMaxDepth 1 -Metric "val/gap"    -MetricMode "min"
    New-Experiment -Problem "jssp10x10" -Method "loss_only"  -Kind "run_path_pattern" -Hosts $SchedulingHosts -RunPathPattern "./mgl-jssp-bopo-lossonly_10x10_*"            -RunFindMaxDepth 1 -Metric "val/gap"    -MetricMode "min"
    New-Experiment -Problem "jssp10x10" -Method "weighting"  -Kind "run_path_pattern" -Hosts $SchedulingHosts -RunPathPattern "./mgl-jssp-bopo-pref_10x10_*"                -RunFindMaxDepth 1 -Metric "val/gap"    -MetricMode "min"

    New-Experiment -Problem "jssp15x15" -Method "rl"         -Kind "run_path_pattern" -Hosts $SchedulingHosts -RunPathPattern "./mgl-jssp-rl_bucketed-multishape_15x15_*"   -RunFindMaxDepth 1 -Metric "val/gap"    -MetricMode "min" -Notes "Output keeps only jssp15x15 label"
    New-Experiment -Problem "jssp15x15" -Method "po"         -Kind "run_path_pattern" -Hosts $SchedulingHosts -RunPathPattern "./mgl-jssp-po_bucketed-multishape_15x15_*"   -RunFindMaxDepth 1 -Metric "val/gap"    -MetricMode "min" -Notes "Output keeps only jssp15x15 label"
    New-Experiment -Problem "jssp15x15" -Method "bopo"       -Kind "run_path_pattern" -Hosts $SchedulingHosts -RunPathPattern "./mgl-jssp-bopo_bucketed-multishape_15x15_*" -RunFindMaxDepth 1 -Metric "val/gap"    -MetricMode "min" -Notes "Output keeps only jssp15x15 label"
    New-Experiment -Problem "jssp15x15" -Method "sll"        -Kind "run_path_pattern" -Hosts $SchedulingHosts -RunPathPattern "./mgl-jssp-sll_15x15_*"                      -RunFindMaxDepth 1 -Metric "val/gap"    -MetricMode "min"
    New-Experiment -Problem "jssp15x15" -Method "loss_only"  -Kind "run_path_pattern" -Hosts $SchedulingHosts -RunPathPattern "./mgl-jssp-bopo-lossonly_15x15_*"            -RunFindMaxDepth 1 -Metric "val/gap"    -MetricMode "min"
    New-Experiment -Problem "jssp15x15" -Method "weighting"  -Kind "run_path_pattern" -Hosts $SchedulingHosts -RunPathPattern "./mgl-jssp-bopo-best_15x15_*"                -RunFindMaxDepth 1 -Metric "val/gap"    -MetricMode "min"
)

$resolvedExperiments = @()

foreach ($exp in $experiments) {
    Write-Host ("Resolving {0}/{1} ..." -f $exp.Problem, $exp.Method)
    $resolved = Resolve-Experiment -Experiment $exp
    if ($null -eq $resolved) {
        Write-Warning ("Skipping unresolved experiment: {0}/{1}" -f $exp.Problem, $exp.Method)
        continue
    }
    $resolvedExperiments += $resolved
}

$manifestPath = Join-Path $ResolvedOutputRoot "manifest.csv"
$resolvedExperiments | Select-Object Problem, Method, Kind, Host, ResolvedRunDir, ResolvedRemote, MetricsPath, HparamsPath, Metric, MetricMode, Notes |
    Export-Csv -NoTypeInformation -Encoding UTF8 -Path $manifestPath
Write-Host "Manifest written to $manifestPath"
Write-Host "JSSP output labels are restricted to jssp10x10 and jssp15x15."

foreach ($exp in $resolvedExperiments) {
    $destDir = Join-Path $ResolvedOutputRoot (Join-Path $exp.Problem $exp.Method)
    $checkpointFiles = Get-CheckpointFileNames -ResolvedExperiment $exp -Mode $CheckpointMode

    foreach ($fileName in $checkpointFiles) {
        $localPath = Join-Path $destDir $fileName
        if ((-not $ForceRedownload) -and (Test-Path -LiteralPath $localPath)) {
            Write-Host "Skip existing: $localPath"
            continue
        }

        $remotePath = Resolve-CheckpointRemotePath -ResolvedExperiment $exp -FileName $fileName
        if ([string]::IsNullOrWhiteSpace($remotePath)) {
            $existingLocalCheckpoints = @()
            if ((-not $ForceRedownload) -and (Test-Path -LiteralPath $destDir)) {
                $existingLocalCheckpoints = @(
                    Get-ChildItem -LiteralPath $destDir -File -Filter "*.ckpt" -ErrorAction SilentlyContinue |
                        Select-Object -ExpandProperty Name
                )
            }

            if ($existingLocalCheckpoints.Count -gt 0) {
                Write-Host ("Keep existing local checkpoint(s) for {0}/{1}: {2}" -f $exp.Problem, $exp.Method, ($existingLocalCheckpoints -join ", "))
                continue
            }

            Write-Warning ("Missing remote checkpoint for {0}/{1}: {2}" -f $exp.Problem, $exp.Method, $fileName)
            continue
        }

        $remote = "{0}:{1}" -f $exp.Host, $remotePath
        $localPath = Join-Path $destDir ([System.IO.Path]::GetFileName($remotePath))
        Invoke-ScpDownload -Remote $remote -LocalPath $localPath
    }

    if ($IncludeMetadata -and $exp.Kind -ne "direct_file") {
        if (-not [string]::IsNullOrWhiteSpace($exp.MetricsPath)) {
            $metricsLocalPath = Join-Path $destDir "metrics.csv"
            Invoke-ScpDownload -Remote ("{0}:{1}" -f $exp.Host, $exp.MetricsPath) -LocalPath $metricsLocalPath
        }
        if (-not [string]::IsNullOrWhiteSpace($exp.HparamsPath)) {
            $hparamsLocalPath = Join-Path $destDir "hparams.yaml"
            Invoke-ScpDownload -Remote ("{0}:{1}" -f $exp.Host, $exp.HparamsPath) -LocalPath $hparamsLocalPath
        }
    }
}

Write-Host "Finished."
