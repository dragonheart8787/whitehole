# Build image and run smoke checks inside Docker
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $Root

try {
    docker info 2>&1 | Out-Null
} catch {
    Write-Error "Docker daemon not running. Start Docker Desktop (Linux engine) and retry."
}

Write-Host "Building whitesearch:latest ..." -ForegroundColor Cyan
docker build -f containers/Dockerfile -t whitesearch:latest .
if ($LASTEXITCODE -ne 0) { throw "docker build failed" }

Write-Host "`nSmoke: whitesearch --help" -ForegroundColor Cyan
docker run --rm whitesearch:latest --help
if ($LASTEXITCODE -ne 0) { throw "help failed" }

Write-Host "`nSmoke: compare (mock, toy via quick nlive)" -ForegroundColor Cyan
docker run --rm -v "${Root}:/workspace" -w /workspace whitesearch:latest `
    compare --model bounce --null null --alt bh_ringdown `
    --channel gw --data mock --nlive 20 --outdir artifacts/docker_compare

Write-Host "`nSmoke: calibrate quick" -ForegroundColor Cyan
docker run --rm -v "${Root}:/workspace" -w /workspace whitesearch:latest `
    calibrate --profile quick --outdir artifacts/docker_calibrate

if (-not (Test-Path "artifacts/docker_calibrate/index.md")) {
    throw "calibration index.md missing"
}

Write-Host "`nDocker verification PASSED" -ForegroundColor Green
