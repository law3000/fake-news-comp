@echo off
setlocal
REM Start backend API (FastAPI) and frontend static server in separate CMD windows
REM Usage: double-click run_all.bat or run from CMD

set REPO_ROOT=%~dp0
pushd "%REPO_ROOT%"

REM Start backend (FastAPI) in a new window
start "Backend (FastAPI)" cmd /k "cd /d fnmvp\fake-news-moderator-mvp\backend && python -m uvicorn app:app --host 127.0.0.1 --port 8000 --reload"

REM Start frontend (Python http.server) in a new window
start "Frontend (Static)" cmd /k "cd /d fnmvp\fake-news-moderator-mvp\frontend && python -m http.server 5500 --directory ."

echo Servers starting...
echo Backend:  http://127.0.0.1:8000   (health: /healthz)
echo Frontend: http://localhost:5500/index.html

popd
endlocal

