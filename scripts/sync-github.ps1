# 同步專案到 GitHub: https://github.com/dragonheart8787/whitehole
# 用法:
#   .\scripts\sync-github.ps1
#   .\scripts\sync-github.ps1 -Message "描述此次更新"

param(
    [string]$Message = "",
    [string]$Remote = "origin",
    [string]$Branch = "main"
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $Root

if (-not (Test-Path ".git")) {
    Write-Error "尚未初始化 git。請先執行: git init"
}

$status = git status --porcelain
if (-not $status) {
    Write-Host "沒有變更，略過 commit。"
} else {
    if (-not $Message) {
        $Message = "Update: $(Get-Date -Format 'yyyy-MM-dd HH:mm')"
    }
    git add -A
    git commit -m $Message
    Write-Host "已提交: $Message"
}

Write-Host "推送到 $Remote/$Branch ..."
git push -u $Remote $Branch
Write-Host "完成: https://github.com/dragonheart8787/whitehole"
