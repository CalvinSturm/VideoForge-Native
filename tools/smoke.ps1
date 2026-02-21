<#
.SYNOPSIS
    VideoForge end-to-end smoke test harness (Windows / PowerShell 5+).

.DESCRIPTION
    Validates that all prerequisites are present and the Python pipeline can
    complete a job.  Optionally tests the native engine path.

    Exits 0 on success, 1 on any failure.

.PARAMETER InputFile
    Path to a short test video (MP4 or MKV).  Required for a full pipeline run.

.PARAMETER Model
    Model name to request (e.g. RCAN_x4).  Must match a model in your weights/
    directory.  If omitted, only the IPC handshake is validated.

.PARAMETER Precision
    Inference precision: fp32 | fp16 | deterministic.  Default: fp32.

.PARAMETER Timeout
    Zenoh handshake timeout in seconds.  Default: 60.

.EXAMPLE
    # Quick prerequisite + IPC check (no video required):
    .\tools\smoke.ps1

.EXAMPLE
    # Full pipeline smoke with a test clip and RCAN_x4:
    .\tools\smoke.ps1 -InputFile C:\test\sample_720p.mp4 -Model RCAN_x4

.NOTES
    Run from the repository root.
    Requires: Rust toolchain, FFmpeg in PATH, Python venv with torch/zenoh.
#>
param(
    [string]$InputFile      = "",
    [string]$Model          = "",
    [string]$Precision      = "fp32",
    [int]   $Timeout        = 60,
    [bool]  $ShmRoundtrip   = $true
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

# ─── Helpers ──────────────────────────────────────────────────────────────────

function Pass([string]$label) {
    Write-Host "[PASS] $label" -ForegroundColor Green
}

function Fail([string]$label, [string]$detail = "") {
    Write-Host "[FAIL] $label$(if ($detail) {": $detail"})" -ForegroundColor Red
    $script:anyFailed = $true
}

function Skip([string]$label, [string]$reason) {
    Write-Host "[SKIP] $label — $reason" -ForegroundColor Yellow
}

function Section([string]$title) {
    Write-Host ""
    Write-Host "── $title $(("-" * [Math]::Max(0, 60 - $title.Length)))" -ForegroundColor Cyan
}

$script:anyFailed = $false

# ─── 0. Banner ────────────────────────────────────────────────────────────────

Write-Host ""
Write-Host "=== VideoForge Smoke Test ===" -ForegroundColor White
Write-Host "Date   : $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
Write-Host "Repo   : $PSScriptRoot\.."

# ─── 1. Prerequisites ─────────────────────────────────────────────────────────

Section "Prerequisites"

# FFmpeg
try {
    $null = & ffmpeg -version 2>&1
    if ($LASTEXITCODE -eq 0) { Pass "FFmpeg in PATH" }
    else { Fail "FFmpeg in PATH" "exit code $LASTEXITCODE" }
} catch {
    Fail "FFmpeg in PATH" "command not found — install FFmpeg and add to PATH"
}

# FFprobe
try {
    $null = & ffprobe -version 2>&1
    if ($LASTEXITCODE -eq 0) { Pass "FFprobe in PATH" }
    else { Fail "FFprobe in PATH" "exit code $LASTEXITCODE" }
} catch {
    Fail "FFprobe in PATH" "command not found"
}

# Rust / Cargo
try {
    $rustc = & rustc --version 2>&1
    Pass "Rust toolchain ($rustc)"
} catch {
    Fail "Rust toolchain" "rustc not found — install from https://rustup.rs"
}

# Cargo manifest
$manifest = Join-Path $PSScriptRoot ".." "src-tauri" "Cargo.toml"
if (Test-Path $manifest) {
    Pass "src-tauri/Cargo.toml found"
} else {
    Fail "src-tauri/Cargo.toml" "not found at $manifest"
}

# Input file (optional)
if ($InputFile) {
    if (Test-Path $InputFile) {
        Pass "Input file: $InputFile"
    } else {
        Fail "Input file" "not found: $InputFile"
    }
}

# Model (optional) — check weights directory
if ($Model) {
    $repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
    $weightsDir = Join-Path $repoRoot "weights"
    $weightsFound = $false
    if (Test-Path $weightsDir) {
        $modelFiles = Get-ChildItem -Path $weightsDir -Recurse -File `
            -Filter "*$($Model.Replace('_','*'))*" 2>$null
        if ($modelFiles.Count -gt 0) {
            Pass "Model weights found: $($modelFiles[0].FullName)"
            $weightsFound = $true
        }
    }
    if (-not $weightsFound) {
        Fail "Model weights ($Model)" "no matching file under $weightsDir — place weights there before running"
    }
}

# ─── 2. Build smoke binary ────────────────────────────────────────────────────

Section "Build smoke binary"

$buildArgs = @(
    "build",
    "--manifest-path", (Join-Path $PSScriptRoot ".." "src-tauri" "Cargo.toml"),
    "--bin", "smoke",
    "--quiet"
)

Write-Host "  cargo $($buildArgs -join ' ')"
try {
    & cargo @buildArgs
    if ($LASTEXITCODE -ne 0) {
        Fail "smoke binary build" "cargo exited with code $LASTEXITCODE"
    } else {
        Pass "smoke binary build"
    }
} catch {
    Fail "smoke binary build" $_.Exception.Message
}

# ─── 3. Run Rust smoke binary ─────────────────────────────────────────────────

Section "Python IPC Handshake (via smoke binary)"

if ($script:anyFailed) {
    Skip "smoke binary run" "prerequisite check failed above"
} else {
    $smokeExe = Join-Path $PSScriptRoot ".." "src-tauri" "target" "debug" "smoke.exe"
    if (-not (Test-Path $smokeExe)) {
        $smokeExe = Join-Path $PSScriptRoot ".." "src-tauri" "target" "release" "smoke.exe"
    }

    $smokeArgs = @("--precision", $Precision, "--timeout", $Timeout)
    if ($Model) { $smokeArgs += @("--model", $Model) }

    Write-Host "  $smokeExe $($smokeArgs -join ' ')"
    try {
        & $smokeExe @smokeArgs
        if ($LASTEXITCODE -eq 0) {
            Pass "Python IPC smoke (all checks)"
        } else {
            Fail "Python IPC smoke" "smoke binary exited with code $LASTEXITCODE"
        }
    } catch {
        Fail "Python IPC smoke" $_.Exception.Message
    }
}

# ─── 4. SHM Roundtrip ────────────────────────────────────────────────────────

Section "SHM Roundtrip (no FFmpeg, no UI)"

if ($script:anyFailed) {
    Skip "SHM roundtrip" "earlier check failed"
} elseif ($ShmRoundtrip) {
    $smokeArgs = @("--shm-roundtrip", "--precision", $Precision, "--timeout", $Timeout)
    Write-Host "  $smokeExe $($smokeArgs -join ' ')"
    try {
        & $smokeExe @smokeArgs
        if ($LASTEXITCODE -eq 0) { Pass "SHM roundtrip" }
        else { Fail "SHM roundtrip" "smoke binary exited $LASTEXITCODE" }
    } catch {
        Fail "SHM roundtrip" $_.Exception.Message
    }
} else {
    Skip "SHM roundtrip" "skipped by -ShmRoundtrip:`$false"
}

# ─── 5. Full pipeline (if input file provided) ────────────────────────────────

Section "Full pipeline (Python)"

if (-not $InputFile) {
    Skip "Full pipeline" "no --InputFile provided; skipping video encode test"
} elseif ($script:anyFailed) {
    Skip "Full pipeline" "earlier check failed"
} else {
    Write-Host "  NOTE: Full pipeline requires the app to be running (npm run dev)."
    Write-Host "  To run manually, launch the app and invoke from DevTools:"
    Write-Host "    await window.__TAURI__.core.invoke('upscale_request', {"
    Write-Host "      inputPath: '$InputFile',"
    Write-Host "      outputPath: '',"
    Write-Host "      model: '$Model',"
    Write-Host "      editConfig: { trim_start:0, trim_end:0, crop:null, rotation:0, flip_h:false, flip_v:false, fps:0, color:{brightness:0,contrast:0,saturation:0,gamma:1} },"
    Write-Host "      scale: 4"
    Write-Host "    })"
    Skip "Full pipeline" "manual step — see SMOKE_TESTS.md for full walkthrough"
}

# ─── 6. Native engine ─────────────────────────────────────────────────────────

Section "Native engine"

Write-Host "  Status: BLOCKED — ort ^2.0 not published as stable on crates.io."
Write-Host "  See docs/NATIVE_ENGINE_MVP.md for resolution steps."
Skip "Native engine" "feature flag off by default (ort dependency unresolved)"

# ─── 7. Summary ───────────────────────────────────────────────────────────────

Write-Host ""
Write-Host "─────────────────────────────────────────────────────────────────" -ForegroundColor Cyan

if (-not $script:anyFailed) {
    Write-Host "Result: ALL CHECKS PASSED" -ForegroundColor Green
    exit 0
} else {
    Write-Host "Result: ONE OR MORE CHECKS FAILED" -ForegroundColor Red
    Write-Host "  → See red [FAIL] lines above for details." -ForegroundColor Red
    exit 1
}
