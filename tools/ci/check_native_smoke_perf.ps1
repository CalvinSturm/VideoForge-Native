param(
  [Parameter(Mandatory = $true)]
  [string]$Input,

  [Parameter(Mandatory = $true)]
  [string]$Onnx,

  [int]$Scale = 2,
  [ValidateSet("fp32", "fp16", "deterministic")]
  [string]$Precision = "fp32",
  [int]$Runs = 3,
  [double]$MinMedianFps = 40.0,
  [string]$SmokeExe = ".\src-tauri\target\debug\smoke.exe",
  [string]$OutJson = ".\artifacts\native_perf_report.json"
)

$ErrorActionPreference = "Stop"

function Require-Path([string]$Path, [string]$Label) {
  if (-not (Test-Path $Path)) {
    throw "$Label not found: $Path"
  }
}

if ($Runs -lt 1) {
  throw "Runs must be >= 1."
}

Require-Path -Path $Input -Label "Input video"
Require-Path -Path $Onnx -Label "ONNX model"
Require-Path -Path $SmokeExe -Label "smoke.exe"

$ffprobeOk = $false
try {
  & ffprobe -version *> $null
  if ($LASTEXITCODE -eq 0) { $ffprobeOk = $true }
} catch {}
if (-not $ffprobeOk) {
  throw "ffprobe not found in PATH."
}

$frameCountRaw = & ffprobe -v error -count_frames -select_streams v:0 -show_entries stream=nb_read_frames -of default=noprint_wrappers=1:nokey=1 "$Input"
$frameCount = 0
if (-not [int]::TryParse(($frameCountRaw | Select-Object -First 1).Trim(), [ref]$frameCount) -or $frameCount -le 0) {
  throw "Could not determine frame count for input: $Input"
}

$outDir = Split-Path -Parent $OutJson
if ($outDir -and -not (Test-Path $outDir)) {
  New-Item -ItemType Directory -Force -Path $outDir | Out-Null
}

$runRows = @()
for ($i = 1; $i -le $Runs; $i++) {
  Write-Host "Run $i/$Runs ..."
  $sw = [System.Diagnostics.Stopwatch]::StartNew()
  & $SmokeExe --e2e-native --native-direct --input "$Input" --e2e-onnx "$Onnx" --e2e-scale $Scale --precision $Precision --keep-temp
  $exitCode = $LASTEXITCODE
  $sw.Stop()

  $secs = [Math]::Max($sw.Elapsed.TotalSeconds, 0.001)
  $fps = $frameCount / $secs
  $ok = ($exitCode -eq 0)

  $runRows += [ordered]@{
    run = $i
    exit_code = $exitCode
    elapsed_sec = [Math]::Round($secs, 3)
    fps = [Math]::Round($fps, 3)
    pass = $ok
  }

  if (-not $ok) {
    throw "Native smoke run $i failed (exit code $exitCode)."
  }
}

$fpsValues = @($runRows | ForEach-Object { [double]$_.fps } | Sort-Object)
$mid = [int][Math]::Floor($fpsValues.Count / 2)
$medianFps = if (($fpsValues.Count % 2) -eq 1) {
  $fpsValues[$mid]
} else {
  ($fpsValues[$mid - 1] + $fpsValues[$mid]) / 2.0
}

$pass = ($medianFps -ge $MinMedianFps)

$report = [ordered]@{
  generated_at_utc = (Get-Date).ToUniversalTime().ToString("o")
  input = (Resolve-Path $Input).Path
  onnx = (Resolve-Path $Onnx).Path
  scale = $Scale
  precision = $Precision
  frames = $frameCount
  runs = $Runs
  min_median_fps = $MinMedianFps
  median_fps = [Math]::Round($medianFps, 3)
  pass = $pass
  results = $runRows
}

$report | ConvertTo-Json -Depth 8 | Set-Content -Path $OutJson -Encoding utf8
Write-Host "Perf report written: $OutJson"
Write-Host ("Median FPS: {0} (threshold: {1})" -f ([Math]::Round($medianFps, 3)), $MinMedianFps)

if (-not $pass) {
  throw ("Native perf regression: median FPS {0} is below threshold {1}" -f ([Math]::Round($medianFps, 3)), $MinMedianFps)
}
