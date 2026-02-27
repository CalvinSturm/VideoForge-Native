# check_deps.ps1 — Verify required runtime dependencies are available.
# Called by run_release_signoff_windows.ps1 during CI release signoff.

$ErrorActionPreference = "Stop"
$failed = @()

function Test-Cmd([string]$Name) {
  if (Get-Command $Name -ErrorAction SilentlyContinue) {
    Write-Host "[PASS] $Name found: $((Get-Command $Name).Source)"
  } else {
    Write-Host "[FAIL] $Name not found in PATH"
    $script:failed += $Name
  }
}

Write-Host "=== Dependency Check ==="

# Core toolchain
Test-Cmd "cargo"
Test-Cmd "node"
Test-Cmd "npm"

# FFmpeg (bundled or PATH)
$ffmpegBundled = Join-Path (Get-Location) "third_party\ffmpeg7\bin\ffmpeg.exe"
if (Test-Path $ffmpegBundled) {
  Write-Host "[PASS] ffmpeg found (bundled): $ffmpegBundled"
} else {
  Test-Cmd "ffmpeg"
}

# Python packages (best-effort — only check if python is available)
$python = Get-Command python -ErrorAction SilentlyContinue
if ($python) {
  Write-Host "[INFO] Python: $($python.Source)"
  $requiredPkgs = @("torch", "zenoh")
  foreach ($pkg in $requiredPkgs) {
    $check = & python -c "import $pkg" 2>&1
    if ($LASTEXITCODE -eq 0) {
      Write-Host "[PASS] Python package '$pkg' importable"
    } else {
      Write-Host "[WARN] Python package '$pkg' not importable (non-blocking)"
    }
  }
} else {
  Write-Host "[WARN] Python not found — skipping package checks (non-blocking)"
}

if ($failed.Count -gt 0) {
  throw "Missing required dependencies: $($failed -join ', ')"
}

Write-Host ""
Write-Host "[PASS] All required dependencies found."
