# ColPali RAG - stop the running app (kills the python process listening on the port).
$port = if ($env:API_PORT) { $env:API_PORT } else { 8000 }
$conn = Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue
if (-not $conn) { Write-Host "No process on port $port."; exit 0 }
$pidToKill = $conn[0].OwningProcess
Write-Host "Stopping process $pidToKill on port $port"
Stop-Process -Id $pidToKill -Force
Start-Sleep -Seconds 1
Write-Host 'Stopped.'