# ColPali RAG - one-time setup: install deps + generate demo corpus.
$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

Write-Host '[1/3] Installing dependencies (first run downloads CUDA torch, may take a while)...'
uv sync

Write-Host '[2/3] Generating demo corpus PDFs (text + figures)...'
uv run python -m scripts.make_demo_corpus

Write-Host '[3/3] Done.'
Write-Host ''
Write-Host 'Next:  .\powershell\start-app.ps1   (then open http://localhost:8000)'
Write-Host '       .\powershell\ingest.ps1 -Path ".\data\corpus\*.pdf"'