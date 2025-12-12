@echo off
REM ================================================================================
REM Solar Panel Detection System - Automated Setup Script
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
    echo Please install Python 3.10 or 3.11 from: https://www.python.org/downloads/
    echo Make sure to check "Add Python to PATH" during installation.
    echo.
    pause
    exit /b 1
)

echo [1/5] Checking Python version...
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
echo [2/5] Creating virtual environment...
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
echo [3/5] Activating virtual environment...
call .venv\Scripts\activate.bat
if errorlevel 1 (
    echo [ERROR] Failed to activate virtual environment!
    pause
    exit /b 1
)
echo [SUCCESS] Virtual environment activated!
echo.

REM Upgrade pip
echo [4/5] Upgrading pip to latest version...
python -m pip install --upgrade pip
echo.

REM Install all requirements
echo [5/5] Installing all dependencies...
echo This may take 5-10 minutes depending on your internet connection...
echo.

echo Installing core requirements...
pip install torch>=2.0.0 torchvision>=0.15.0 ultralytics>=8.0.0

echo.
echo Installing computer vision libraries...
pip install opencv-python>=4.8.0 Pillow>=10.0.0

echo.
echo Installing automated imagery retrieval...
pip install selenium>=4.0.0

echo.
echo Installing data processing libraries...
pip install numpy>=1.24.0 pandas>=2.0.0

echo.
echo Installing visualization tools...
pip install matplotlib>=3.7.0

echo.
echo Installing web framework and API...
pip install fastapi>=0.104.0 uvicorn[standard]>=0.24.0 pydantic>=2.4.0 python-multipart>=0.0.6

echo.
echo Installing additional utilities...
pip install pycocotools>=2.0.6 tqdm>=4.65.0 PyYAML>=6.0 scipy>=1.11.0

echo.
echo Installing optional packages...
pip install albumentations>=1.3.0 openpyxl>=3.0.0

echo.
echo Installing Jupyter support (optional)...
pip install jupyter>=1.0.0 ipykernel>=6.25.0

echo.
echo Installing requests library...
pip install requests>=2.28.0

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
echo 1. To activate the environment manually in future:
echo    .venv\Scripts\activate
echo.
echo 2. To start the web server:
echo    python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000
echo.
echo 3. Then open your browser:
echo    http://localhost:8000
echo.
echo 4. For batch processing:
echo    python pipeline/main.py inputs/samples.xlsx
echo.
echo ================================================================================
echo   Ready to use! Press any key to exit...
echo ================================================================================
pause >nul
