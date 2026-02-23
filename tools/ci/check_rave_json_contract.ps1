param(
  [Parameter(Mandatory = $true)]
  [string]$JsonPath,

  [Parameter(Mandatory = $true)]
  [string]$ExpectedCommand,

  [Parameter(Mandatory = $true)]
  [string]$RequiredFieldsCsv
)

$ErrorActionPreference = "Stop"

if (-not (Test-Path $JsonPath)) {
  throw "JSON file not found: $JsonPath"
}

try {
  $raw = Get-Content -Raw -Path $JsonPath
  $json = $raw | ConvertFrom-Json
} catch {
  throw "Invalid JSON at ${JsonPath}: $($_.Exception.Message)"
}

function Has-Field($obj, [string]$path) {
  $parts = $path.Split('.')
  $current = $obj

  foreach ($part in $parts) {
    if ($null -eq $current) { return $false }

    if ($current -is [System.Management.Automation.PSCustomObject]) {
      $prop = $current.PSObject.Properties[$part]
      if ($null -eq $prop) { return $false }
      $current = $prop.Value
      continue
    }

    if ($current -is [hashtable]) {
      if (-not $current.ContainsKey($part)) { return $false }
      $current = $current[$part]
      continue
    }

    return $false
  }

  return $true
}

if (-not (Has-Field $json "schema_version")) {
  throw "Contract check failed ($ExpectedCommand): missing required field 'schema_version' in $JsonPath"
}

$schemaVersion = $json.schema_version
if ($schemaVersion -isnot [int] -and $schemaVersion -isnot [long]) {
  throw "Contract check failed ($ExpectedCommand): schema_version must be an integer in $JsonPath"
}

if (-not (Has-Field $json "command")) {
  throw "Contract check failed ($ExpectedCommand): missing required field 'command' in $JsonPath"
}

if ($json.command -ne $ExpectedCommand) {
  throw "Contract check failed: expected command='$ExpectedCommand' but found '$($json.command)' in $JsonPath"
}

$missing = @()
foreach ($field in ($RequiredFieldsCsv.Split(',') | ForEach-Object { $_.Trim() } | Where-Object { $_ })) {
  if (-not (Has-Field $json $field)) {
    $missing += $field
  }
}

if ($missing.Count -gt 0) {
  throw "Contract check failed ($ExpectedCommand): missing required fields: $($missing -join ', ')"
}

Write-Host "Contract check passed: $ExpectedCommand ($JsonPath)"
