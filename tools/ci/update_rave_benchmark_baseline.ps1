param(
  [Parameter(Mandatory = $true)]
  [string]$InputGlob,

  [Parameter(Mandatory = $true)]
  [string]$OutputJson,

  [string]$ExistingBaselineJson,

  [string]$Target = "windows-gpu"
)

$ErrorActionPreference = "Stop"
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$impl = Join-Path $scriptDir "calibrate_rave_gpu_baseline.ps1"

if (-not (Test-Path $impl)) {
  throw "Missing implementation script: $impl"
}

& $impl `
  -InputGlob $InputGlob `
  -OutputJson $OutputJson `
  -ExistingBaselineJson $ExistingBaselineJson `
  -Target $Target
