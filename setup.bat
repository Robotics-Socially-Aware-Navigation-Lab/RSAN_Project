@echo off
echo ==============================================
echo RSAN_Project Environment Setup (Windows)
echo ==============================================

:: Step 1: Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo Python not found! Please install Python 3.9+ and add it to PATH.
    pause
    exit /b
)
for /f "tokens=2 delims= " %%v in ('python --version') do set PY_VER=%%v
echo Python version: %PY_VER%

:: Step 2: Create virtual environment
python -m venv env
call env\Scripts\activate

:: Step 3: Upgrade pip
python -m pip install --upgrade pip

:: Step 4: Install dependencies
if exist requirements.txt (
    echo Installing dependencies...
    pip install -r requirements.txt
) else (
    echo No requirements.txt found, skipping...
)

:: Step 5: Install dev tools
pip install black isort flake8

:: Step 6: Run verifications
echo Running format/lint checks...
black --check src/
isort --check-only src/
flake8 src/

echo ==============================================
echo RSAN_Project environment setup complete!
echo To activate later: call env\Scripts\activate
echo ==============================================
pause