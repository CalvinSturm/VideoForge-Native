param(
  [Parameter(Mandatory = $true)]
  [string]$CurrentJson,

  [Parameter(Mandatory = $true)]
  [string]$BaselineJson,

  [Parameter(Mandatory = $true)]
  [string]$ReportJson,

  [switch]$Enforce
)

$ErrorActionPreference = "Stop"

function Require-Path([string]$Path, [string]$Label) {
  if (-not (Test-Path $Path)) {
    throw "$Label not found: $Path"
  }
}

function Read-Json([string]$Path) {
  return Get-Content -Raw -Path $Path | ConvertFrom-Json
}

function To-DoubleOrNull($value) {
  if ($null -eq $value) { return $null }
  if ($value -is [double] -or $value -is [float] -or $value -is [decimal]) { return [double]$value }
  if ($value -is [int] -or $value -is [long]) { return [double]$value }
  return $null
}

Require-Path -Path $CurrentJson -Label "Current benchmark JSON"
Require-Path -Path $BaselineJson -Label "Baseline JSON"

$current = Read-Json -Path $CurrentJson
$baseline = Read-Json -Path $BaselineJson

if ($current.command -ne "benchmark") {
  throw "Current JSON is not benchmark output (command=$($current.command))."
}

$baseMetrics = $baseline.baseline
$thresholds = $baseline.thresholds
if ($null -eq $baseMetrics -or $null -eq $thresholds) {
  throw "Baseline JSON must include 'baseline' and 'thresholds' objects."
}

$currentFps = To-DoubleOrNull $current.fps
$baseFps = To-DoubleOrNull $baseMetrics.fps
$currentInfer = To-DoubleOrNull $current.stages.infer
$baseInfer = To-DoubleOrNull $baseMetrics.stages.infer
$currentDecode = To-DoubleOrNull $current.stages.decode
$baseDecode = To-DoubleOrNull $baseMetrics.stages.decode
$currentEncode = To-DoubleOrNull $current.stages.encode
$baseEncode = To-DoubleOrNull $baseMetrics.stages.encode

if ($null -eq $currentFps -or $null -eq $baseFps -or $baseFps -le 0) {
  throw "Invalid fps values in current/baseline JSON."
}
if ($null -eq $currentInfer -or $null -eq $baseInfer -or $baseInfer -le 0) {
  throw "Invalid infer stage values in current/baseline JSON."
}

$fpsDropPct = [double]$thresholds.fps_drop_pct
$inferIncreasePct = [double]$thresholds.infer_increase_pct
$decodeIncreasePct = [double]$thresholds.decode_increase_pct
$encodeIncreasePct = [double]$thresholds.encode_increase_pct

$fpsMin = $baseFps * (1.0 - ($fpsDropPct / 100.0))
$inferMax = $baseInfer * (1.0 + ($inferIncreasePct / 100.0))
$decodeMax = if ($null -ne $baseDecode) { $baseDecode * (1.0 + ($decodeIncreasePct / 100.0)) } else { $null }
$encodeMax = if ($null -ne $baseEncode) { $baseEncode * (1.0 + ($encodeIncreasePct / 100.0)) } else { $null }

$checks = @()
$failures = @()

$fpsPass = $currentFps -ge $fpsMin
$checks += [ordered]@{
  metric = "fps"
  current = $currentFps
  baseline = $baseFps
  limit = $fpsMin
  comparator = ">="
  pass = $fpsPass
}
if (-not $fpsPass) {
  $failures += "fps regression: current=$currentFps < min=$([Math]::Round($fpsMin,3))"
}

$inferPass = $currentInfer -le $inferMax
$checks += [ordered]@{
  metric = "stages.infer"
  current = $currentInfer
  baseline = $baseInfer
  limit = $inferMax
  comparator = "<="
  pass = $inferPass
}
if (-not $inferPass) {
  $failures += "infer regression: current=$currentInfer > max=$([Math]::Round($inferMax,3))"
}

if ($null -ne $decodeMax -and $null -ne $currentDecode) {
  $decodePass = $currentDecode -le $decodeMax
  $checks += [ordered]@{
    metric = "stages.decode"
    current = $currentDecode
    baseline = $baseDecode
    limit = $decodeMax
    comparator = "<="
    pass = $decodePass
  }
  if (-not $decodePass) {
    $failures += "decode regression: current=$currentDecode > max=$([Math]::Round($decodeMax,3))"
  }
}

if ($null -ne $encodeMax -and $null -ne $currentEncode) {
  $encodePass = $currentEncode -le $encodeMax
  $checks += [ordered]@{
    metric = "stages.encode"
    current = $currentEncode
    baseline = $baseEncode
    limit = $encodeMax
    comparator = "<="
    pass = $encodePass
  }
  if (-not $encodePass) {
    $failures += "encode regression: current=$currentEncode > max=$([Math]::Round($encodeMax,3))"
  }
}

$report = [ordered]@{
  generated_at_utc = (Get-Date).ToUniversalTime().ToString("o")
  enforce = [bool]$Enforce
  baseline_file = (Resolve-Path $BaselineJson).Path
  current_file = (Resolve-Path $CurrentJson).Path
  pass = ($failures.Count -eq 0)
  checks = $checks
  failures = $failures
}

$reportDir = Split-Path -Parent $ReportJson
if ($reportDir -and -not (Test-Path $reportDir)) {
  New-Item -ItemType Directory -Force -Path $reportDir | Out-Null
}

$report | ConvertTo-Json -Depth 8 | Set-Content -Path $ReportJson -Encoding utf8
Write-Host "Wrote benchmark threshold report: $ReportJson"

if ($Enforce -and $failures.Count -gt 0) {
  throw "Benchmark thresholds failed: $($failures -join '; ')"
}

if (-not $Enforce -and $failures.Count -gt 0) {
  Write-Warning "Benchmark thresholds would fail (enforcement disabled): $($failures -join '; ')"
}
