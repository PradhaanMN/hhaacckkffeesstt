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
pip install -r environment_details\requirements.txt

echo.
echo [5/5] Checking browser availability...
echo.

REM Check for Chrome
where chrome.exe >nul 2>&1
if %errorlevel%==0 (
    echo [OK] Chrome detected
) else (
    echo [INFO] Chrome not found in PATH
)

REM Check for Edge
where msedge.exe >nul 2>&1
if %errorlevel%==0 (
    echo [OK] Microsoft Edge detected
) else (
    echo [INFO] Edge not found in PATH
)

REM Check for Firefox
where firefox.exe >nul 2>&1
if %errorlevel%==0 (
    echo [OK] Firefox detected
) else (
    echo [INFO] Firefox not found in PATH
)

REM Check for Opera
where opera.exe >nul 2>&1
if %errorlevel%==0 (
    echo [OK] Opera detected
) else (
    echo [INFO] Opera not found in PATH
)

REM Check for Brave (common locations)
if exist "C:\Program Files\BraveSoftware\Brave-Browser\Application\brave.exe" (
    echo [OK] Brave detected
) else if exist "C:\Program Files (x86)\BraveSoftware\Brave-Browser\Application\brave.exe" (
    echo [OK] Brave detected
) else (
    echo [INFO] Brave not found
)

REM Check for Vivaldi (common locations)
if exist "C:\Program Files\Vivaldi\Application\vivaldi.exe" (
    echo [OK] Vivaldi detected
) else if exist "C:\Program Files (x86)\Vivaldi\Application\vivaldi.exe" (
    echo [OK] Vivaldi detected
) else (
    echo [INFO] Vivaldi not found
)

REM Check for Chromium (common locations)
if exist "C:\Program Files\Chromium\Application\chrome.exe" (
    echo [OK] Chromium detected
) else if exist "C:\Program Files (x86)\Chromium\Application\chrome.exe" (
    echo [OK] Chromium detected
) else (
    echo [INFO] Chromium not found
)

echo.
echo NOTE: The system will automatically use any available browser.
echo      (Chrome, Chromium, Edge, Firefox, Brave, Vivaldi, or Opera)
echo.

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
echo 2. Then open your browser at:
echo    http://localhost:8000
echo.
echo ================================================================================
pause
