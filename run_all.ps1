# Start backend API (FastAPI) and frontend static server
# Usage: PowerShell -ExecutionPolicy Bypass -File .\run_all.ps1

$ErrorActionPreference = 'Stop'

$repoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $repoRoot

# Start backend (uvicorn) in a new window
$backendPath = Join-Path $repoRoot 'fnmvp/fake-news-moderator-mvp/backend'
$backendCmd = 'uvicorn app:app --host 127.0.0.1 --port 8000 --reload'
Start-Process -FilePath 'powershell' -ArgumentList "-NoProfile","-NoExit","-Command","Set-Location `"$backendPath`"; $backendCmd"

# Start frontend (Python http.server) in a new window
$frontendPath = Join-Path $repoRoot 'fnmvp/fake-news-moderator-mvp/frontend'
$frontendCmd = 'python -m http.server 5500 --directory .'
Start-Process -FilePath 'powershell' -ArgumentList "-NoProfile","-NoExit","-Command","Set-Location `"$frontendPath`"; $frontendCmd"

Write-Host "Servers starting..."
Write-Host "Backend: http://127.0.0.1:8000 (health: /healthz)"
Write-Host "Frontend: http://localhost:5500/index.html"

