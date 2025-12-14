@echo off
REM ================================================================================
REM Solar Panel Detection System - Quick Setup
REM EcoInnovators Ideathon 2026
REM ================================================================================

echo.
echo ================================================================================
echo   SOLAR PANEL DETECTION SYSTEM - SETUP
echo   EcoInnovators Ideathon 2026
echo ================================================================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python is not installed or not in PATH!
    echo.
    echo Please install Python 3.10 or higher from: https://www.python.org/downloads/
    echo Make sure to check "Add Python to PATH" during installation.
    echo.
    pause
    exit /b 1
)

echo [1/4] Checking Python version...
python --version
echo.

REM Check Python version (must be 3.10 or higher)
python -c "import sys; exit(0 if sys.version_info >= (3, 10) else 1)" >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python 3.10 or higher required!
    echo Current version is too old. Please upgrade Python.
    echo.
    pause
    exit /b 1
)

echo [SUCCESS] Python version compatible!
echo.

REM Create virtual environment
echo [2/4] Creating virtual environment...
if exist .venv (
    echo Virtual environment already exists. Skipping creation.
) else (
    python -m venv .venv
    if errorlevel 1 (
        echo [ERROR] Failed to create virtual environment!
        pause
        exit /b 1
    )
    echo [SUCCESS] Virtual environment created!
)
echo.

REM Activate virtual environment
echo [3/4] Activating virtual environment...
call .venv\Scripts\activate.bat
if errorlevel 1 (
    echo [ERROR] Failed to activate virtual environment!
    pause
    exit /b 1
)
echo [SUCCESS] Virtual environment activated!
echo.

REM Install requirements
echo [4/4] Installing dependencies (this may take 3-5 minutes)...
python -m pip install --upgrade pip --quiet
pip install -r requirements.txt

echo.
echo ================================================================================
echo   SETUP COMPLETE!
echo ================================================================================
echo.
echo Virtual environment created at: .venv\
echo All dependencies installed successfully!
echo.
echo NEXT STEPS:
echo.
echo 1. To start the web server, run:
echo    start_server.bat
echo.
echo 2. Or manually:
echo    .venv\Scripts\activate
echo    python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000
echo.
echo 3. Then open your browser at:
echo    http://localhost:8000
echo.
echo ================================================================================
pause
