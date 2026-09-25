# Prepare the college-showcase demo: stop app -> generate demo PDFs -> ingest -> start app.
# Uses the offline ingest CLI, which is safe only while the server is down (single-writer rule).
$root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $root

Write-Output '[1/4] stopping app (single-writer rule)...'
& powershell -NoProfile -ExecutionPolicy Bypass -File '.\powershell\stop-app.ps1' 2>$null

Write-Output '[2/4] generating demo PDFs (matplotlib + pymupdf)...'
& '.\.venv\Scripts\python.exe' '.\demo\make_demo_pdfs.py'
if ($LASTEXITCODE -ne 0) { Write-Error 'demo PDF generation failed'; exit 1 }

Write-Output '[3/4] ingesting demo corpus (offline CLI)...'
& '.\.venv\Scripts\python.exe' -m colpali_rag.ingest '.\demo\pdfs' --no-mlflow
if ($LASTEXITCODE -ne 0) { Write-Error 'demo ingest failed'; exit 1 }

Write-Output '[4/4] starting app (warmup_on_start=true keeps the first query instant)...'
& powershell -NoProfile -ExecutionPolicy Bypass -File '.\powershell\start-app.ps1'