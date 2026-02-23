param(
  [Parameter(Mandatory = $true)]
  [string]$InputGlob,

  [Parameter(Mandatory = $true)]
  [string]$OutputJson,

  [string]$ExistingBaselineJson,

  [string]$Target = "windows-gpu"
)

$ErrorActionPreference = "Stop"

function Read-Json([string]$Path) {
  return Get-Content -Raw -Path $Path | ConvertFrom-Json
}

function To-DoubleOrNull($value) {
  if ($null -eq $value) { return $null }
  if ($value -is [double] -or $value -is [float] -or $value -is [decimal]) { return [double]$value }
  if ($value -is [int] -or $value -is [long]) { return [double]$value }
  return $null
}

function Get-Sorted([double[]]$values) {
  if ($null -eq $values -or $values.Count -eq 0) { return @() }
  return @($values | Sort-Object)
}

function Get-Median([double[]]$values) {
  $sorted = Get-Sorted -values $values
  if ($sorted.Count -eq 0) { throw "Cannot compute median from empty set." }
  $mid = [int]($sorted.Count / 2)
  if (($sorted.Count % 2) -eq 1) { return [double]$sorted[$mid] }
  return [double](($sorted[$mid - 1] + $sorted[$mid]) / 2.0)
}

function Get-Percentile([double[]]$values, [double]$pct) {
  $sorted = Get-Sorted -values $values
  if ($sorted.Count -eq 0) { throw "Cannot compute percentile from empty set." }
  if ($pct -le 0.0) { return [double]$sorted[0] }
  if ($pct -ge 100.0) { return [double]$sorted[$sorted.Count - 1] }

  $rank = ($pct / 100.0) * ($sorted.Count - 1)
  $lo = [Math]::Floor($rank)
  $hi = [Math]::Ceiling($rank)
  if ($lo -eq $hi) { return [double]$sorted[$lo] }
  $weight = $rank - $lo
  return [double]($sorted[$lo] + (($sorted[$hi] - $sorted[$lo]) * $weight))
}

$inputFiles = @(Get-ChildItem -Path $InputGlob -File -ErrorAction SilentlyContinue)
if ($inputFiles.Count -eq 0) {
  throw "No benchmark JSON files matched InputGlob: $InputGlob"
}

$fps = @()
$decode = @()
$infer = @()
$encode = @()
$acceptedFiles = @()

foreach ($file in $inputFiles) {
  $json = Read-Json -Path $file.FullName
  if ($json.command -ne "benchmark") {
    Write-Warning "Skipping non-benchmark JSON: $($file.FullName)"
    continue
  }

  $f = To-DoubleOrNull $json.fps
  $d = To-DoubleOrNull $json.stages.decode
  $i = To-DoubleOrNull $json.stages.infer
  $e = To-DoubleOrNull $json.stages.encode

  if ($null -eq $f -or $null -eq $i) {
    Write-Warning "Skipping benchmark missing numeric fps/infer stage: $($file.FullName)"
    continue
  }

  $fps += $f
  if ($null -ne $d) { $decode += $d }
  $infer += $i
  if ($null -ne $e) { $encode += $e }
  $acceptedFiles += $file.FullName
}

if ($acceptedFiles.Count -eq 0) {
  throw "No valid benchmark JSON files found in InputGlob: $InputGlob"
}

$thresholds = [ordered]@{
  fps_drop_pct = 70.0
  infer_increase_pct = 150.0
  decode_increase_pct = 200.0
  encode_increase_pct = 200.0
}

if ($ExistingBaselineJson -and (Test-Path $ExistingBaselineJson)) {
  $existing = Read-Json -Path $ExistingBaselineJson
  if ($null -ne $existing.thresholds) {
    $thresholds = [ordered]@{
      fps_drop_pct = [double]$existing.thresholds.fps_drop_pct
      infer_increase_pct = [double]$existing.thresholds.infer_increase_pct
      decode_increase_pct = [double]$existing.thresholds.decode_increase_pct
      encode_increase_pct = [double]$existing.thresholds.encode_increase_pct
    }
  }
}

$baseline = [ordered]@{
  schema_version = 1
  target = $Target
  note = "Auto-generated baseline from benchmark artifacts. Keep thresholds aligned to runner stability."
  source_files = $acceptedFiles
  sample_count = $acceptedFiles.Count
  baseline = [ordered]@{
    fps = [Math]::Round((Get-Median -values $fps), 3)
    stages = [ordered]@{
      decode = if ($decode.Count -gt 0) { [Math]::Round((Get-Median -values $decode), 3) } else { $null }
      infer = [Math]::Round((Get-Median -values $infer), 3)
      encode = if ($encode.Count -gt 0) { [Math]::Round((Get-Median -values $encode), 3) } else { $null }
    }
  }
  thresholds = $thresholds
}

$outDir = Split-Path -Parent $OutputJson
if ($outDir -and -not (Test-Path $outDir)) {
  New-Item -ItemType Directory -Force -Path $outDir | Out-Null
}

$baseline | ConvertTo-Json -Depth 8 | Set-Content -Path $OutputJson -Encoding utf8

$summary = [ordered]@{
  sample_count = $acceptedFiles.Count
  fps = [ordered]@{
    p10 = [Math]::Round((Get-Percentile -values $fps -pct 10.0), 3)
    p50 = [Math]::Round((Get-Median -values $fps), 3)
    p90 = [Math]::Round((Get-Percentile -values $fps -pct 90.0), 3)
  }
  infer = [ordered]@{
    p10 = [Math]::Round((Get-Percentile -values $infer -pct 10.0), 3)
    p50 = [Math]::Round((Get-Median -values $infer), 3)
    p90 = [Math]::Round((Get-Percentile -values $infer -pct 90.0), 3)
  }
}

if ($decode.Count -gt 0) {
  $summary.decode = [ordered]@{
    p10 = [Math]::Round((Get-Percentile -values $decode -pct 10.0), 3)
    p50 = [Math]::Round((Get-Median -values $decode), 3)
    p90 = [Math]::Round((Get-Percentile -values $decode -pct 90.0), 3)
  }
}

if ($encode.Count -gt 0) {
  $summary.encode = [ordered]@{
    p10 = [Math]::Round((Get-Percentile -values $encode -pct 10.0), 3)
    p50 = [Math]::Round((Get-Median -values $encode), 3)
    p90 = [Math]::Round((Get-Percentile -values $encode -pct 90.0), 3)
  }
}

Write-Host "Wrote baseline: $OutputJson"
Write-Host ""
Write-Host "Calibration summary:"
$summary | ConvertTo-Json -Depth 5 | Write-Host
