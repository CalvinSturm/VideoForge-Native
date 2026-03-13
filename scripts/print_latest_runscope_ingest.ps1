param(
    [string]$SearchRoot = (Join-Path $PSScriptRoot ".."),
    [switch]$IncludeManifestOnly
)

$resolvedRoot = (Resolve-Path $SearchRoot).Path

$patterns = @("videoforge_run.json")
if ($IncludeManifestOnly) {
    $patterns += "videoforge.run_manifest.v1.json"
}

$files = foreach ($pattern in $patterns) {
    Get-ChildItem -Path $resolvedRoot -Recurse -File -Filter $pattern -ErrorAction SilentlyContinue
}

$latest = $files |
    Sort-Object LastWriteTimeUtc -Descending |
    Select-Object -First 1

if (-not $latest) {
    Write-Host "No RunScope-ingestible VideoForge artifact bundle was found under:"
    Write-Host "  $resolvedRoot"
    Write-Host ""
    Write-Host "Run a VideoForge job with VIDEOFORGE_ENABLE_RUN_ARTIFACTS=1 first."
    exit 1
}

$bundleDir = $latest.Directory.FullName
$command = 'runscope ingest "{0}"' -f $bundleDir

Write-Host "Latest VideoForge artifact bundle:"
Write-Host "  $bundleDir"
Write-Host ""
Write-Host "Suggested RunScope ingest command:"
Write-Host "  $command"
