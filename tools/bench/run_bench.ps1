param(
  [Parameter(Mandatory = $true)]
  [Alias("Input")]
  [string]$InputPath,

  [Parameter(Mandatory = $true)]
  [Alias("Output")]
  [string]$OutputPath,

  [Parameter(Mandatory = $true)]
  [string]$Model,

  [Parameter(Mandatory = $true)]
  [uint32]$Scale,

  [Parameter(Mandatory = $true)]
  [ValidateSet("fp16", "fp32")]
  [string]$Precision,

  [switch]$Deterministic,
  [switch]$DryRun,
  [string]$EditConfigJson
)

$ErrorActionPreference = "Stop"

$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\\..")).Path
$SrcTauri = Join-Path $RepoRoot "src-tauri"
$OutDir = Join-Path $PSScriptRoot "out"
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null

$Timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$JsonlPath = Join-Path $OutDir ("bench_{0}.jsonl" -f $Timestamp)

$CargoArgs = @(
  "run",
  "--bin", "videoforge_bench",
  "--",
  "--input", $InputPath,
  "--output", $OutputPath,
  "--model", $Model,
  "--scale", $Scale.ToString(),
  "--precision", $Precision
)

if ($Deterministic) {
  $CargoArgs += "--deterministic"
}
if ($DryRun) {
  $CargoArgs += "--dry-run"
}
if ($EditConfigJson) {
  $CargoArgs += @("--edit-config", $EditConfigJson)
}

$sw = [System.Diagnostics.Stopwatch]::StartNew()
Push-Location $SrcTauri
try {
  & cargo @CargoArgs | Tee-Object -FilePath $JsonlPath
  $ExitCode = $LASTEXITCODE
} finally {
  Pop-Location
  $sw.Stop()
}

Write-Host ""
Write-Host "Bench summary"
Write-Host ("  Exit code : {0}" -f $ExitCode)
Write-Host ("  Elapsed   : {0:N2}s" -f $sw.Elapsed.TotalSeconds)
Write-Host ("  Output    : {0}" -f $OutputPath)
Write-Host ("  JSONL log : {0}" -f $JsonlPath)

exit $ExitCode
