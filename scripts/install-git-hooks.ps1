# 安裝 post-commit 自動 push hook（只需執行一次）
$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $Root

if (-not (Test-Path ".git")) {
    Write-Error "請先在專案根目錄執行 git init"
}

$hooksDir = ".git\hooks"
$src = "scripts\hooks\post-commit"
$dst = Join-Path $hooksDir "post-commit"

Copy-Item $src $dst -Force
Write-Host "已安裝 post-commit hook -> $dst"
Write-Host "之後每次 git commit 會自動 git push 到 origin"
