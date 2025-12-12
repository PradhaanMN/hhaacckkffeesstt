@echo off
REM ================================================================================
REM Solar Panel Detection System - Quick Start Server
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

REM Start the server
echo [2/2] Starting FastAPI server...
echo.
echo Server will be available at: http://localhost:8000
echo.
echo Press Ctrl+C to stop the server
echo.
echo ================================================================================
echo.

python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000

pause
