[CmdletBinding()]
param(
    [Parameter(Mandatory)]
    [ValidateNotNullOrEmpty()]
    [string]$WindowTitle,

    [Parameter(Mandatory)]
    [ValidateRange(1, 2147483647)]
    [int]$ExpectedProcessId,

    [Parameter(Mandatory)]
    [ValidateNotNullOrEmpty()]
    [string]$OutputPath,

    [switch]$AllowSharedWindowsTerminalHost
)

$ErrorActionPreference = 'Stop'

Add-Type -AssemblyName System.Drawing
Add-Type -TypeDefinition @'
using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using System.Text;

public static class BenchmarkWindowCapture {
    public delegate bool EnumWindowsProc(IntPtr hWnd, IntPtr lParam);

    [StructLayout(LayoutKind.Sequential)]
    public struct RECT {
        public int Left;
        public int Top;
        public int Right;
        public int Bottom;
    }

    [DllImport("user32.dll")]
    public static extern bool EnumWindows(EnumWindowsProc callback, IntPtr lParam);

    [DllImport("user32.dll", CharSet = CharSet.Unicode)]
    public static extern int GetWindowText(IntPtr hWnd, StringBuilder text, int maxCount);

    [DllImport("user32.dll")]
    public static extern int GetWindowTextLength(IntPtr hWnd);

    [DllImport("user32.dll")]
    public static extern bool IsWindowVisible(IntPtr hWnd);

    [DllImport("user32.dll")]
    public static extern bool IsIconic(IntPtr hWnd);

    [DllImport("user32.dll")]
    public static extern bool GetWindowRect(IntPtr hWnd, out RECT rect);

    [DllImport("user32.dll")]
    public static extern uint GetWindowThreadProcessId(IntPtr hWnd, out uint processId);

    [DllImport("user32.dll")]
    public static extern bool PrintWindow(IntPtr hWnd, IntPtr hdc, uint flags);

    public static IntPtr[] FindExactTitle(string expected) {
        var matches = new List<IntPtr>();
        EnumWindows(delegate (IntPtr hWnd, IntPtr ignored) {
            int length = GetWindowTextLength(hWnd);
            if (length < 1) return true;
            var text = new StringBuilder(length + 1);
            GetWindowText(hWnd, text, text.Capacity);
            if (String.Equals(text.ToString(), expected, StringComparison.Ordinal)) {
                matches.Add(hWnd);
            }
            return true;
        }, IntPtr.Zero);
        return matches.ToArray();
    }
}
'@

function Get-ProcessLineage {
    param([Parameter(Mandatory)][int]$ProcessId)

    $lineage = New-Object System.Collections.Generic.List[int]
    $seen = @{}
    $current = $ProcessId
    for ($depth = 0; $depth -lt 12 -and $current -gt 0; $depth++) {
        if ($seen.ContainsKey($current)) {
            break
        }
        $seen[$current] = $true
        $lineage.Add($current)
        $process = Get-CimInstance -ClassName Win32_Process -Filter "ProcessId=$current" -ErrorAction SilentlyContinue
        if ($null -eq $process -or $process.ParentProcessId -le 0) {
            break
        }
        $current = [int]$process.ParentProcessId
    }
    return @($lineage)
}

$output = [IO.Path]::GetFullPath($OutputPath)
if ([IO.Path]::GetExtension($output) -ine '.png') {
    throw 'OutputPath must end in .png'
}
if (Test-Path -LiteralPath $output) {
    throw "Refusing to overwrite existing screenshot: $output"
}

$expected = Get-CimInstance -ClassName Win32_Process -Filter "ProcessId=$ExpectedProcessId" -ErrorAction SilentlyContinue
if ($null -eq $expected) {
    throw "Expected process $ExpectedProcessId is not running"
}
if ([IO.Path]::GetFileName($expected.ExecutablePath) -ine 'powershell.exe') {
    throw "Expected process $ExpectedProcessId is not powershell.exe"
}

$matches = @([BenchmarkWindowCapture]::FindExactTitle($WindowTitle))
if ($matches.Count -ne 1) {
    throw "Expected exactly one visible window titled '$WindowTitle'; found $($matches.Count)"
}
$handle = [IntPtr]$matches[0]
if (-not [BenchmarkWindowCapture]::IsWindowVisible($handle)) {
    throw 'Target benchmark window is not visible'
}
if ([BenchmarkWindowCapture]::IsIconic($handle)) {
    throw 'Target benchmark window is minimized'
}

$windowPid = [uint32]0
[void][BenchmarkWindowCapture]::GetWindowThreadProcessId($handle, [ref]$windowPid)
$windowLineage = @(Get-ProcessLineage -ProcessId ([int]$windowPid))
$expectedLineage = @(Get-ProcessLineage -ProcessId $ExpectedProcessId)
$ancestryVerified = $windowPid -eq $ExpectedProcessId -or
    $windowLineage -contains $ExpectedProcessId -or
    $expectedLineage -contains ([int]$windowPid)
$windowProcess = Get-CimInstance -ClassName Win32_Process -Filter "ProcessId=$windowPid" -ErrorAction SilentlyContinue
$sharedTerminalBinding = (
    -not $ancestryVerified -and
    $AllowSharedWindowsTerminalHost.IsPresent -and
    $null -ne $windowProcess -and
    [IO.Path]::GetFileName($windowProcess.ExecutablePath) -ieq 'WindowsTerminal.exe' -and
    $expected.CommandLine -like "*$WindowTitle*"
)
if (-not $ancestryVerified -and -not $sharedTerminalBinding) {
    $hostName = if ($null -ne $windowProcess) { [IO.Path]::GetFileName($windowProcess.ExecutablePath) } else { 'unknown' }
    throw "Window process $windowPid ($hostName) is not related to expected process $ExpectedProcessId"
}
$rect = New-Object BenchmarkWindowCapture+RECT
if (-not [BenchmarkWindowCapture]::GetWindowRect($handle, [ref]$rect)) {
    throw 'GetWindowRect failed'
}
$width = $rect.Right - $rect.Left
$height = $rect.Bottom - $rect.Top
if ($width -lt 320 -or $height -lt 200) {
    throw "Target benchmark window is unexpectedly small: ${width}x${height}"
}

$directory = Split-Path -Parent $output
if ([string]::IsNullOrWhiteSpace($directory)) {
    throw 'OutputPath must include a parent directory'
}
[IO.Directory]::CreateDirectory($directory) | Out-Null
$temporary = Join-Path $directory ('.capture-' + [Guid]::NewGuid().ToString('N') + '.png')

$bitmap = New-Object System.Drawing.Bitmap($width, $height, [Drawing.Imaging.PixelFormat]::Format32bppArgb)
$graphics = [Drawing.Graphics]::FromImage($bitmap)
$hdc = $graphics.GetHdc()
try {
    if (-not [BenchmarkWindowCapture]::PrintWindow($handle, $hdc, 2)) {
        throw 'PrintWindow failed; full-desktop fallback is intentionally disabled'
    }
}
finally {
    $graphics.ReleaseHdc($hdc)
    $graphics.Dispose()
}

try {
    $bitmap.Save($temporary, [Drawing.Imaging.ImageFormat]::Png)
}
finally {
    $bitmap.Dispose()
}

$bytes = [IO.File]::ReadAllBytes($temporary)
if ($bytes.Length -lt 1024 -or
    $bytes[0] -ne 0x89 -or $bytes[1] -ne 0x50 -or $bytes[2] -ne 0x4E -or $bytes[3] -ne 0x47) {
    Remove-Item -LiteralPath $temporary -Force -ErrorAction SilentlyContinue
    throw 'Captured file is not a valid non-empty PNG'
}

[IO.File]::Move($temporary, $output)
$metadata = [ordered]@{
    schema_version = 'q3ple-headed-window-capture-v1'
    captured_utc = [DateTime]::UtcNow.ToString('o')
    title = $WindowTitle
    expected_process_id = $ExpectedProcessId
    window_process_id = [int]$windowPid
    process_binding = if ($ancestryVerified) { 'process_ancestry' } else { 'windows_terminal_exact_title_and_command' }
    ancestry_verified = $ancestryVerified
    shared_windows_terminal_host = $sharedTerminalBinding
    hwnd = $handle.ToInt64()
    method = 'PrintWindow.PW_RENDERFULLCONTENT'
    full_desktop_fallback = $false
    width = $width
    height = $height
    png = [IO.Path]::GetFileName($output)
    png_bytes = (Get-Item -LiteralPath $output).Length
    png_sha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath $output).Hash
}
$metadataPath = [IO.Path]::ChangeExtension($output, '.json')
$metadata | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath $metadataPath -Encoding UTF8
$metadata | ConvertTo-Json -Depth 4
