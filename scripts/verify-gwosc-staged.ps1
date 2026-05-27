# Staged GWOSC + dynesty verification (A -> B -> C)
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot\..

Remove-Item Env:WHITESEARCH_FORCE_TOY -ErrorAction SilentlyContinue
$OUT = "artifacts/verify_gwosc_staged"
$NLIVE = 25
$OPTS = @(
    "--data", "gwosc", "--event", "GW150914", "--channel", "gw",
    "--likelihood-mode", "mf", "--nlive", $NLIVE,
    "--dynesty-bound", "live", "--dynesty-sample", "rwalk"
)

function Step($name, $cmd) {
    Write-Host "`n========== $name ==========" -ForegroundColor Cyan
    Invoke-Expression $cmd
    if ($LASTEXITCODE -ne 0) { throw "FAILED: $name" }
}

Step "A: bh_ringdown fit" "python -m whitesearch fit --model bh_ringdown --outdir `"$OUT/A_bh`" @OPTS"
Step "B: bounce fit" "python -m whitesearch fit --model bounce --outdir `"$OUT/B_bounce`" @OPTS"
Step "C: compare" "python -m whitesearch compare --model bounce --null null --alt bh_ringdown --outdir `"$OUT/C_compare`" @OPTS"

Write-Host "`nStaged GWOSC verification PASSED -> $OUT" -ForegroundColor Green
