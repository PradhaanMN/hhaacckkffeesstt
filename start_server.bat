@echo off
REM ================================================================================
REM Solar Panel Detection System - Start Server
REM EcoInnovators Ideathon 2026
REM ================================================================================

echo.
echo ================================================================================
echo   SOLAR PANEL DETECTION SYSTEM
echo   Starting Web Server...
echo ================================================================================
echo.

REM Check if virtual environment exists
if not exist .venv (
    echo [ERROR] Virtual environment not found!
    echo.
    echo Please run setup.bat first to install dependencies.
    echo.
    pause
    exit /b 1
)

REM Activate virtual environment
echo [1/2] Activating virtual environment...
call .venv\Scripts\activate.bat
echo.

REM Check if in pipeline_code directory structure
if exist pipeline_code\backend\main.py (
    set BACKEND_PATH=pipeline_code.backend.main
) else if exist backend\main.py (
    set BACKEND_PATH=backend.main
) else (
    echo [ERROR] Backend not found!
    echo Expected: backend\main.py or pipeline_code\backend\main.py
    pause
    exit /b 1
)

REM Check browser availability
echo [2/3] Checking browser for satellite imagery...
set BROWSER_FOUND=0

where chrome.exe >nul 2>&1 && set BROWSER_FOUND=1 && echo [OK] Chrome available
if %BROWSER_FOUND%==0 where msedge.exe >nul 2>&1 && set BROWSER_FOUND=1 && echo [OK] Microsoft Edge available
if %BROWSER_FOUND%==0 where firefox.exe >nul 2>&1 && set BROWSER_FOUND=1 && echo [OK] Firefox available
if %BROWSER_FOUND%==0 if exist "C:\Program Files\BraveSoftware\Brave-Browser\Application\brave.exe" set BROWSER_FOUND=1 && echo [OK] Brave available
if %BROWSER_FOUND%==0 where opera.exe >nul 2>&1 && set BROWSER_FOUND=1 && echo [OK] Opera available

if %BROWSER_FOUND%==0 (
    echo [WARNING] No browser detected!
    echo The system needs Chrome, Edge, Firefox, Brave, or Opera to fetch satellite imagery.
    echo Server will start, but imagery fetching may fail.
    echo.
)
echo.

REM Start the server
echo [3/3] Starting FastAPI server...
echo.
echo Server will be available at: http://localhost:8000
echo.
echo Press Ctrl+C to stop the server
echo.
echo ================================================================================
echo.

python -m uvicorn %BACKEND_PATH%:app --host 0.0.0.0 --port 8000

pause
