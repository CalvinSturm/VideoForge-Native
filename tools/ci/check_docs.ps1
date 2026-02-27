# check_docs.ps1 — Verify required documentation files exist and are non-empty.
# Called by run_release_signoff_windows.ps1 during CI release signoff.

$ErrorActionPreference = "Stop"
$failed = @()

$requiredDocs = @(
  "README.md",
  "ARCHITECTURE_GUIDE.md",
  "docs\SMOKE_TESTS.md",
  "docs\NATIVE_ENGINE_MVP.md",
  "docs\RELEASE_SIGNOFF.md",
  "docs\WINDOWS_RAVE_RUNTIME.md",
  "docs\CI_GPU_STABILIZATION_CHECKLIST.md",
  "docs\VIDEOFORGE_2_WEEK_EXECUTION_PLAN.md"
)

Write-Host "=== Documentation Check ==="

foreach ($doc in $requiredDocs) {
  $path = Join-Path (Get-Location) $doc
  if (Test-Path $path) {
    $size = (Get-Item $path).Length
    if ($size -gt 0) {
      Write-Host "[PASS] $doc ($size bytes)"
    } else {
      Write-Host "[FAIL] $doc exists but is empty"
      $failed += $doc
    }
  } else {
    Write-Host "[FAIL] $doc not found"
    $failed += $doc
  }
}

if ($failed.Count -gt 0) {
  throw "Missing or empty documentation files: $($failed -join ', ')"
}

Write-Host ""
Write-Host "[PASS] All required documentation files present."
