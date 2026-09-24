# ColPali RAG - start the web app (owns the Qdrant local-mode index).
# First run loads the ColPali + BGE-M3 models (downloads once into data/hf-cache).
$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

$port = if ($env:API_PORT) { $env:API_PORT } else { 8000 }
$inUse = Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue
if ($inUse) {
    Write-Host "Port $port already in use - an instance may already be running." -ForegroundColor Yellow
    Write-Host '  -> stop it with:  .\powershell\stop-app.ps1' -ForegroundColor Yellow
    exit 1
}

Write-Host "Starting ColPali RAG on http://localhost:$port  (Ctrl+C to stop)"
uv run python -m colpali_rag.api