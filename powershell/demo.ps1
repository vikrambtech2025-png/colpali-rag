# ColPali RAG - full demo: setup + ingest demo corpus + start app.
$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
& (Join-Path $root 'powershell\setup.ps1')

Write-Host 'Starting app in a new window...'
Start-Process powershell -ArgumentList "-NoExit", "-Command", "& '$root\powershell\start-app.ps1'"
Start-Sleep -Seconds 4

Write-Host 'Ingesting demo corpus...'
& (Join-Path $root 'powershell\ingest.ps1') -Path (Join-Path $root 'data\corpus')

Write-Host ''
Write-Host 'Demo ready. Open http://localhost:8000 and ask e.g.:'
Write-Host '  "What does the revenue chart show for each product line?"'