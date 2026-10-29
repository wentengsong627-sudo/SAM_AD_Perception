[CmdletBinding()]
param([int]$Port=8850,[string]$Bind='127.0.0.1')
$ErrorActionPreference='Stop'
Set-Location -LiteralPath $PSScriptRoot
$env:PYTHONUTF8='1'
$env:PYTHONDONTWRITEBYTECODE='1'
& (Join-Path $PSScriptRoot '.venv/Scripts/python.exe') -B -m sam_ad.server --backend A --port $Port --bind $Bind
exit $LASTEXITCODE
