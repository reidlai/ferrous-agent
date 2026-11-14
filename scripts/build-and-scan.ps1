[CmdletBinding()]
param(
    [switch]$DryRun,
    [string]$ImageTag = "ferrous-agent:latest",
    [int]$MaxSizeMb = 55
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$AgentDir = Split-Path -Parent $PSScriptRoot
$MaxSizeBytes = $MaxSizeMb * 1MB

function Invoke-Step {
    param(
        [string]$Description,
        [scriptblock]$Action,
        [string]$DryRunCommand
    )

    Write-Host "==> $Description" -ForegroundColor Cyan
    if ($DryRun) {
        Write-Host "[DRY-RUN] $DryRunCommand" -ForegroundColor Yellow
        return
    }

    & $Action
    if ($LASTEXITCODE -ne 0) {
        throw "Step failed ($LASTEXITCODE): $Description"
    }
}

Push-Location $AgentDir
try {
    Invoke-Step `
        -Description "Lint Dockerfile with hadolint" `
        -DryRunCommand "hadolint Dockerfile" `
        -Action { hadolint Dockerfile }

    Invoke-Step `
        -Description "Build Docker image ($ImageTag)" `
        -DryRunCommand "docker build -t $ImageTag ." `
        -Action { docker build -t $ImageTag . }

    Write-Host "==> Verify image size < ${MaxSizeMb} MB" -ForegroundColor Cyan
    if ($DryRun) {
        Write-Host "[DRY-RUN] docker image inspect $ImageTag --format '{{.Size}}' (< $MaxSizeBytes bytes)" -ForegroundColor Yellow
    } else {
        $sizeRaw = docker image inspect $ImageTag --format '{{.Size}}'
        if ($LASTEXITCODE -ne 0) {
            throw "Failed to inspect image size for $ImageTag"
        }
        $sizeBytes = [int64]$sizeRaw.Trim()
        $sizeMb = [math]::Round($sizeBytes / 1MB, 2)
        Write-Host "Image size: ${sizeMb} MB ($sizeBytes bytes)"
        if ($sizeBytes -ge $MaxSizeBytes) {
            throw "Image size ${sizeMb} MB exceeds ${MaxSizeMb} MB limit"
        }
    }

    Invoke-Step `
        -Description "Run cargo-deny supply-chain checks" `
        -DryRunCommand "cargo deny check" `
        -Action { cargo deny check }

    Invoke-Step `
        -Description "Scan container image with Trivy" `
        -DryRunCommand "trivy image --severity HIGH,CRITICAL --exit-code 1 $ImageTag" `
        -Action { trivy image --severity HIGH,CRITICAL --exit-code 1 $ImageTag }

    Write-Host "All build-and-scan checks completed." -ForegroundColor Green
}
finally {
    Pop-Location
}
