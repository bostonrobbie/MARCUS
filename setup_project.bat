@echo off
echo ==========================================
echo Quant_Lab Portable Setup
echo ==========================================

REM Check Python
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo Error: Python is not installed or not in PATH.
    pause
    exit /b
)

REM Create Virtual Environment
if not exist ".venv" (
    echo Creating virtual environment...
    python -m venv .venv
) else (
    echo Virtual environment already exists.
)

REM Activate and Install
echo Installing dependencies...
call .venv\Scripts\activate
pip install --upgrade pip
if exist "requirements.txt" (
    pip install -r requirements.txt
) else (
    echo Warning: requirements.txt not found. Skipping dependency install.
)

REM Extract Data
echo Checking for compressed data...
for %%f in (data\*.zip) do (
    echo Extracting %%f...
    tar -xf "%%f" -C data
)

REM Create Directories
if not exist "data" mkdir data
if not exist "logs" mkdir logs
if not exist "outputs" mkdir outputs
if not exist "cache" mkdir cache

echo ==========================================
echo Setup Complete!
echo To start: .venv\Scripts\activate
echo ==========================================
pause
