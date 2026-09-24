# ColPali RAG - ingest PDFs through the running API (curl.exe multipart POST).
param(
    [Parameter(Mandatory = $true)]
    [string[]]$Path,
    [int]$Port = 8000
)
$base = "http://127.0.0.1:$Port/v1/ingest"
foreach ($item in $Path) {
    $files = @()
    if (Test-Path $item -PathType Container) { $files = Get-ChildItem $item -Filter *.pdf }
    elseif (Test-Path $item) { $files = Get-Item $item }
    else { Write-Host "Not found: $item" -ForegroundColor Red; continue }
    foreach ($f in $files) {
        Write-Host "Uploading $($f.FullName)"
        $json = & curl.exe -s -X POST -F "file=@$($f.FullName)" $base
        $job = $json | ConvertFrom-Json
        Write-Host "  job $($job.job_id): $($job.status)"
        $jobId = $job.job_id
        do {
            Start-Sleep -Seconds 2
            $st = (& curl.exe -s "$base/$jobId") | ConvertFrom-Json
            Write-Host "  status: $($st.status)"
        } while ($st.status -eq 'queued' -or $st.status -eq 'running')
    }
}
Write-Host 'Done.'