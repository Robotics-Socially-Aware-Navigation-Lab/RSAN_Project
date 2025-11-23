@echo off
echo ==============================================
echo RSAN_Project Smart Environment Setup (Windows)
echo ==============================================

:: Step 1: Check if Conda is installed
where conda >nul 2>&1
if %errorlevel%==0 (
    echo Conda detected - using Conda environment setup.
    set USE_CONDA=true
) else (
    echo Conda not found. Falling back to Python venv setup.
    set USE_CONDA=false
)

:: Step 2: Create or activate environment
if "%USE_CONDA%"=="true" (
    set ENV_NAME=rsan_env
    echo Checking if Conda environment "%ENV_NAME%" exists...
    conda env list | findstr /I "%ENV_NAME%" >nul
    if %errorlevel%==0 (
        echo Environment already exists. Skipping creation.
    ) else (
        echo Creating Conda environment from environment.yml...
        conda env create -f environment.yml
    )
    echo Activating Conda environment...
    call conda activate %ENV_NAME%
) else (
    echo Creating Python virtual environment...
    python -m venv env
    call env\Scripts\activate
    echo Upgrading pip...
    python -m pip install --upgrade pip

    if exist requirements.txt (
        echo Installing dependencies from requirements.txt...
        pip install -r requirements.txt
    ) else (
        echo Installing base dependencies (pip fallback)...
        pip install ultralytics fiftyone python-dotenv requests pyyaml rich plotly ftfy imageio rtree scikit-learn scikit-image opencv-python pillow pymongo mongoengine motor
    )
)

:: Step 3: Verify key libraries
echo Verifying key libraries...
python - <<EOF
import torch, cv2, fiftyone, numpy
print("\n RSAN environment verification successful!")
print(f"Torch: {torch.__version__}")
print(f"OpenCV: {cv2.__version__}")
print(f"FiftyOne: {fiftyone.__version__}")
print(f"Numpy: {numpy.__version__}")
EOF

echo ==============================================
echo RSAN_Project environment setup complete!
if "%USE_CONDA%"=="true" (
    echo To activate later: call conda activate rsan_env
) else (
    echo To activate later: call env\Scripts\activate
)
echo ==============================================
pause
