[CmdletBinding()]
param(
    [Parameter(Mandatory)]
    [ValidatePattern('^[A-Za-z0-9][A-Za-z0-9._-]{0,95}$')]
    [string]$RunId,

    [Parameter(Mandatory)]
    [ValidateSet('preflight', 'loaded', 'steady', 'result', 'postflight', 'failure')]
    [string]$Stage,

    [ValidateRange(1, 2147483647)]
    [int]$OwnedPid,

    [string]$OutputRoot,

    [switch]$NoWrite
)

$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)

if ([string]::IsNullOrWhiteSpace($OutputRoot)) {
    $OutputRoot = Join-Path $projectRoot 'results\PUBLIC-BENCH-001\evidence'
}

function Invoke-TextCommand {
    param(
        [Parameter(Mandatory)]
        [string]$FilePath,
        [string[]]$Arguments = @()
    )

    try {
        $lines = @(& $FilePath @Arguments 2>&1 | ForEach-Object { $_.ToString() })
        return [ordered]@{
            exit_code = $LASTEXITCODE
            lines = $lines
        }
    }
    catch {
        return [ordered]@{
            exit_code = -1
            lines = @($_.Exception.Message)
        }
    }
}

function Get-GitValue {
    param([string[]]$Arguments)

    $result = Invoke-TextCommand -FilePath 'git' -Arguments (@('-C', $projectRoot) + $Arguments)
    if ($result.exit_code -ne 0 -or $result.lines.Count -eq 0) {
        return $null
    }
    return ($result.lines -join "`n").Trim()
}

$capturedUtc = [DateTime]::UtcNow.ToString('o')
$fileUtc = [DateTime]::UtcNow.ToString('yyyyMMddTHHmmss.fffZ')
$gitCommit = Get-GitValue -Arguments @('rev-parse', 'HEAD')
$gitStatus = Get-GitValue -Arguments @('status', '--porcelain=v1', '--untracked-files=no')

$gpuQuery = @(
    '--query-gpu=timestamp,name,driver_version,pstate,temperature.gpu,power.draw,utilization.gpu,utilization.memory,memory.total,memory.used,memory.free',
    '--format=csv,noheader,nounits'
)
$gpu = Invoke-TextCommand -FilePath 'nvidia-smi' -Arguments $gpuQuery

$os = Get-CimInstance -ClassName Win32_OperatingSystem
$pagefiles = @(
    Get-CimInstance -ClassName Win32_PageFileUsage -ErrorAction SilentlyContinue |
        ForEach-Object {
            [ordered]@{
                allocated_mib = [int64]$_.AllocatedBaseSize
                current_mib = [int64]$_.CurrentUsage
                peak_mib = [int64]$_.PeakUsage
            }
        }
)

$drive = $null
$qualifier = Split-Path -Qualifier $projectRoot
if ($qualifier) {
    $driveName = $qualifier.TrimEnd(':', '\')
    $driveInfo = Get-PSDrive -Name $driveName -PSProvider FileSystem -ErrorAction SilentlyContinue
    if ($driveInfo) {
        $drive = [ordered]@{
            name = $driveInfo.Name
            used_bytes = [int64]$driveInfo.Used
            free_bytes = [int64]$driveInfo.Free
        }
    }
}

$owned = $null
if ($OwnedPid -gt 0) {
    $proc = Get-Process -Id $OwnedPid -ErrorAction Stop
    $exeHash = $null
    if ($proc.Path -and (Test-Path -LiteralPath $proc.Path -PathType Leaf)) {
        $exeHash = (Get-FileHash -LiteralPath $proc.Path -Algorithm SHA256).Hash.ToLowerInvariant()
    }
    $owned = [ordered]@{
        pid = $proc.Id
        name = $proc.ProcessName
        executable_sha256 = $exeHash
        working_set_bytes = [int64]$proc.WorkingSet64
        private_memory_bytes = [int64]$proc.PrivateMemorySize64
        cpu_seconds = if ($null -eq $proc.CPU) { $null } else { [double]$proc.CPU }
        handles = $proc.HandleCount
    }
}

$record = [ordered]@{
    schema = 'q38-benchmark-diagnostic-v1'
    run_id = $RunId
    stage = $Stage
    captured_utc = $capturedUtc
    git = [ordered]@{
        commit = if ($gitCommit) { $gitCommit } else { 'UNBORN_OR_UNAVAILABLE' }
        tracked_clean = [string]::IsNullOrWhiteSpace($gitStatus)
        tracked_status_entry_count = if ([string]::IsNullOrWhiteSpace($gitStatus)) { 0 } else { @($gitStatus -split "`n").Count }
    }
    gpu = [ordered]@{
        query = $gpuQuery
        exit_code = $gpu.exit_code
        columns = @('timestamp', 'name', 'driver_version', 'pstate', 'temperature_c', 'power_w', 'gpu_util_pct', 'memory_util_pct', 'memory_total_mib', 'memory_used_mib', 'memory_free_mib')
        rows = $gpu.lines
    }
    system = [ordered]@{
        total_ram_bytes = [int64]$os.TotalVisibleMemorySize * 1KB
        free_ram_bytes = [int64]$os.FreePhysicalMemory * 1KB
        total_virtual_bytes = [int64]$os.TotalVirtualMemorySize * 1KB
        free_virtual_bytes = [int64]$os.FreeVirtualMemory * 1KB
        pagefiles = $pagefiles
        project_drive = $drive
    }
    owned_process = $owned
}

$jsonPath = $null
if (-not $NoWrite) {
    $diagnosticDir = Join-Path (Join-Path $OutputRoot $RunId) 'diagnostics'
    New-Item -ItemType Directory -Path $diagnosticDir -Force | Out-Null
    $jsonPath = Join-Path $diagnosticDir "$Stage-$fileUtc.json"
    if (Test-Path -LiteralPath $jsonPath) {
        throw "Refusing to overwrite diagnostic record: $jsonPath"
    }
    $json = $record | ConvertTo-Json -Depth 8
    [System.IO.File]::WriteAllText($jsonPath, $json + [Environment]::NewLine, [System.Text.UTF8Encoding]::new($false))
}

try {
    $host.UI.RawUI.WindowTitle = "BENCH $RunId $Stage"
}
catch {
    # Non-interactive validation hosts may not expose a window title.
}
Write-Host 'QWEN3.8 BENCHMARK EVIDENCE' -ForegroundColor Cyan
Write-Host "run_id:       $RunId"
Write-Host "stage:        $Stage"
Write-Host "captured_utc: $capturedUtc"
Write-Host "git_commit:   $($record.git.commit)"
Write-Host "tracked_clean:$($record.git.tracked_clean)"
Write-Host ''
Write-Host 'GPU: timestamp, name, driver, pstate, temp C, power W, gpu %, mem %, total MiB, used MiB, free MiB' -ForegroundColor Cyan
$gpu.lines | ForEach-Object { Write-Host $_ }
Write-Host ''
Write-Host ("RAM: total {0:N2} GiB, free {1:N2} GiB" -f ($record.system.total_ram_bytes / 1GB), ($record.system.free_ram_bytes / 1GB))
Write-Host ("Virtual: total {0:N2} GiB, free {1:N2} GiB" -f ($record.system.total_virtual_bytes / 1GB), ($record.system.free_virtual_bytes / 1GB))
if ($owned) {
    Write-Host ("Owned process: pid {0}, {1}, RSS {2:N2} GiB, exe sha256 {3}" -f $owned.pid, $owned.name, ($owned.working_set_bytes / 1GB), $owned.executable_sha256)
}
if ($jsonPath) {
    Write-Host "diagnostic_json: $jsonPath"
}
