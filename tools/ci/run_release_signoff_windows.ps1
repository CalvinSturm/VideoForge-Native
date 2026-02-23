param(
  [string]$RepoRoot = ".",
  [switch]$SkipClippy,
  [switch]$SkipCargoTest
)

$ErrorActionPreference = "Stop"

function Invoke-Step([string]$Name, [scriptblock]$Action) {
  Write-Host ""
  Write-Host "=== $Name ==="
  try {
    & $Action | ForEach-Object { Write-Host $_ }
    Write-Host "[PASS] $Name"
    return [ordered]@{ name = $Name; pass = $true; detail = "ok" }
  } catch {
    Write-Host "[FAIL] ${Name}: $($_.Exception.Message)"
    return [ordered]@{ name = $Name; pass = $false; detail = $_.Exception.Message }
  }
}

Push-Location $RepoRoot
try {
  $results = @()

  $results += Invoke-Step "cargo fmt --check (src-tauri)" {
    Push-Location "src-tauri"
    try {
      cargo fmt --all -- --check
      if ($LASTEXITCODE -ne 0) { throw "cargo fmt failed with exit code $LASTEXITCODE" }
    } finally {
      Pop-Location
    }
  }

  if (-not $SkipClippy) {
    $results += Invoke-Step "cargo clippy --workspace --all-targets -D warnings (src-tauri)" {
      Push-Location "src-tauri"
      try {
        cargo clippy --workspace --all-targets -- -D warnings
        if ($LASTEXITCODE -ne 0) { throw "cargo clippy failed with exit code $LASTEXITCODE" }
      } finally {
        Pop-Location
      }
    }
  }

  if (-not $SkipCargoTest) {
    $results += Invoke-Step "cargo test (src-tauri)" {
      Push-Location "src-tauri"
      try {
        cargo test
        if ($LASTEXITCODE -ne 0) { throw "cargo test failed with exit code $LASTEXITCODE" }
      } finally {
        Pop-Location
      }
    }
  }

  $checkDeps = Join-Path (Get-Location) "scripts/check_deps.sh"
  $checkDocs = Join-Path (Get-Location) "scripts/check_docs.sh"

  if (Test-Path $checkDeps) {
    $results += Invoke-Step "./scripts/check_deps.sh" {
      bash ./scripts/check_deps.sh
      if ($LASTEXITCODE -ne 0) { throw "check_deps.sh failed with exit code $LASTEXITCODE" }
    }
  } else {
    $results += [ordered]@{
      name = "./scripts/check_deps.sh"
      pass = $true
      detail = "skipped (file not present in this repo)"
    }
  }

  if (Test-Path $checkDocs) {
    $results += Invoke-Step "./scripts/check_docs.sh" {
      bash ./scripts/check_docs.sh
      if ($LASTEXITCODE -ne 0) { throw "check_docs.sh failed with exit code $LASTEXITCODE" }
    }
  } else {
    $results += [ordered]@{
      name = "./scripts/check_docs.sh"
      pass = $true
      detail = "skipped (file not present in this repo)"
    }
  }

  $summary = [ordered]@{
    generated_at_utc = (Get-Date).ToUniversalTime().ToString("o")
    all_pass = ($results | Where-Object { -not $_.pass }).Count -eq 0
    results = $results
  }

  Write-Host ""
  Write-Host "=== Release Signoff Summary ==="
  $summary | ConvertTo-Json -Depth 5 | Write-Host

  if (-not $summary.all_pass) {
    throw "Release signoff checks failed."
  }
} finally {
  Pop-Location
}
