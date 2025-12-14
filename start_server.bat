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

REM Start the server
echo [2/2] Starting FastAPI server...
echo.
echo Server will be available at: http://localhost:8000
echo.
echo Press Ctrl+C to stop the server
echo.
echo ================================================================================
echo.

python -m uvicorn %BACKEND_PATH%:app --host 0.0.0.0 --port 8000

pause
