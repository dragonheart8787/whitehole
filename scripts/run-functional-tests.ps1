# WhiteSearch full functional / E2E smoke test (fast toy sampler via env var)
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot\..

$env:WHITESEARCH_FORCE_TOY = "1"
$stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$OUT = "artifacts/functional_$stamp"
New-Item -ItemType Directory -Force -Path $OUT | Out-Null

function Step($name, $cmd) {
    Write-Host "`n========== $name ==========" -ForegroundColor Cyan
    Invoke-Expression $cmd
    if ($LASTEXITCODE -ne 0) { throw "FAILED: $name (exit $LASTEXITCODE)" }
    Write-Host "OK: $name" -ForegroundColor Green
}

Write-Host "=== Phase 1: unit tests ===" -ForegroundColor Yellow
python -m pytest tests/ -q --tb=line
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "`n=== Phase 2: CLI end-to-end (outdir=$OUT) ===" -ForegroundColor Yellow

Step "fit (gw mock)" `
    "python -m whitesearch fit --model bounce --channel gw --data mock --nlive 50 --outdir `"$OUT/fit`""

Step "fit inject-model mismatch" `
    "python -m whitesearch fit --model bh_ringdown --channel gw --data mock --inject-model bounce --nlive 50 --outdir `"$OUT/fit_mismatch`""

Step "compare (core workflow)" `
    "python -m whitesearch compare --model bounce --null null --alt bh_ringdown --channel gw --data mock --nlive 50 --outdir `"$OUT/compare`""

Step "rank (gw-compatible models)" `
    "python -m whitesearch rank --models bounce,bh_ringdown,null --channel gw --data mock --nlive 50 --outdir `"$OUT/rank`""

Step "rank radio channel" `
    "python -m whitesearch rank --models pbh_tunneling,magnetar,null --channel radio --data mock --nlive 50 --outdir `"$OUT/rank_radio`""

Step "inject recovery" `
    "python -m whitesearch inject --model bounce --channel gw --n-injections 5 --nlive 30 --outdir `"$OUT/inject`""

Step "sensitivity" `
    "python -m whitesearch sensitivity --model pbh_tunneling --channel radio --n-injections 8 --outdir `"$OUT/sensitivity`""

Step "report" `
    "python -m whitesearch report --run-dir `"$OUT/compare`" --output `"$OUT/report.md`""

Write-Host "`n========== fail-closed (expect error) ==========" -ForegroundColor Cyan
$prevEA = $ErrorActionPreference
$ErrorActionPreference = "Continue"
& python -m whitesearch fit --model bounce --channel gw --data heasarc --nlive 30 --outdir "$OUT/fail_closed" 2>&1 | Out-Null
$failClosedExit = $LASTEXITCODE
$ErrorActionPreference = $prevEA
if ($failClosedExit -eq 0) { throw "FAILED: heasarc should fail without --allow-mock-fallback" }
Write-Host "OK: heasarc fail-closed (exit $failClosedExit)" -ForegroundColor Green

Step "fallback with flag" `
    "python -m whitesearch fit --model bounce --channel gw --data heasarc --allow-mock-fallback --nlive 30 --outdir `"$OUT/fallback`""

# Artifact checks
$required = @(
    "$OUT/compare/compare_summary.json",
    "$OUT/report.md"
)
foreach ($f in $required) {
    if (-not (Test-Path $f)) { throw "Missing artifact: $f" }
}

Write-Host "`n========================================" -ForegroundColor Green
Write-Host "ALL FUNCTIONAL TESTS PASSED" -ForegroundColor Green
Write-Host "Artifacts: $OUT" -ForegroundColor Green
Write-Host "========================================`n" -ForegroundColor Green
