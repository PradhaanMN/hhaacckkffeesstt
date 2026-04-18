@echo off
REM ================================================================================
REM Solar Panel Detection System - Start Server
REM ================================================================================

echo.
echo ================================================================================
echo   SOLAR PANEL DETECTION SYSTEM
echo   Starting Web Server...
echo ================================================================================
echo.

REM ── Locate repo root (works whether you double-click or run from anywhere) ──────
set "SCRIPT_DIR=%~dp0"
cd /d "%SCRIPT_DIR%"

REM ── Activate virtual environment if present, else use system Python ───────────
if exist ".venv\Scripts\activate.bat" (
    echo [INFO] Activating virtual environment...
    call .venv\Scripts\activate.bat
) else (
    echo [INFO] No .venv found, using system Python.
    echo        Run setup.bat to create a virtual environment.
)
echo.

REM ── Verify backend exists ─────────────────────────────────────────────────────
if not exist "pipeline_code\backend\main.py" (
    echo [ERROR] pipeline_code\backend\main.py not found!
    echo         Make sure you are running this from the project root folder.
    pause
    exit /b 1
)

REM ── Check browser availability ────────────────────────────────────────────────
echo [INFO] Checking browser for satellite imagery...
set BROWSER_FOUND=0
where chrome.exe   >nul 2>&1 && set BROWSER_FOUND=1 && echo [OK] Chrome available
if %BROWSER_FOUND%==0 where msedge.exe >nul 2>&1 && set BROWSER_FOUND=1 && echo [OK] Microsoft Edge available
if %BROWSER_FOUND%==0 where firefox.exe >nul 2>&1 && set BROWSER_FOUND=1 && echo [OK] Firefox available
if %BROWSER_FOUND%==0 if exist "C:\Program Files\BraveSoftware\Brave-Browser\Application\brave.exe" set BROWSER_FOUND=1 && echo [OK] Brave available
if %BROWSER_FOUND%==0 where opera.exe >nul 2>&1 && set BROWSER_FOUND=1 && echo [OK] Opera available
if %BROWSER_FOUND%==0 (
    echo [WARNING] No browser detected - satellite imagery fetching may fail.
)
echo.

REM ── Start FastAPI server from pipeline_code directory ─────────────────────────
echo [INFO] Starting FastAPI server...
echo.
echo   Dashboard : http://localhost:8000
echo   API docs  : http://localhost:8000/docs
echo.
echo   Press Ctrl+C to stop the server.
echo ================================================================================
echo.

cd /d "%SCRIPT_DIR%pipeline_code"
python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload

pause
