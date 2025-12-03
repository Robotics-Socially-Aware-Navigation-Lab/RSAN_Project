@echo off
SETLOCAL ENABLEEXTENSIONS ENABLEDELAYEDEXPANSION

echo ========================================================
echo      RSAN_Project - Smart Environment Setup (Windows)
echo ========================================================

:: ------------------------------------------------------------
:: Check for Conda
:: ------------------------------------------------------------
where conda >nul 2>&1
if %errorlevel%==0 (
    echo Conda detected.
    set USE_CONDA=true
) else (
    echo Conda NOT found. Using Python venv fallback.
    set USE_CONDA=false
)

set ENV_NAME=rsan_env

:: ------------------------------------------------------------
:: Conda Path Hook (IMPORTANT)
:: ------------------------------------------------------------
if "%USE_CONDA%"=="true" (
    call "%USERPROFILE%\anaconda3\Scripts\activate.bat" >nul 2>&1
)

:: ------------------------------------------------------------
:: Conda Setup
:: ------------------------------------------------------------
if "%USE_CONDA%"=="true" (
    echo Checking if "%ENV_NAME%" exists...
    conda env list | findstr /I "%ENV_NAME%" >nul
    if %errorlevel%==0 (
        echo Environment already exists.
    ) else (
        echo Creating Conda environment from environment.yml...
        conda env create -f environment.yml
        if %errorlevel% NEQ 0 (
            echo Conda creation failed. Falling back to pip+venv...
            set USE_CONDA=false
        )
    )
)

:: ------------------------------------------------------------
:: Activate Conda or Create Venv
:: ------------------------------------------------------------
if "%USE_CONDA%"=="true" (
    echo Activating Conda environment...
    call conda activate %ENV_NAME%
) else (
    echo Creating Python virtual environment "env"...
    python -m venv env
    call env\Scripts\activate
    python -m pip install --upgrade pip

    if exist requirements.txt (
        echo Installing Python dependencies from requirements.txt...
        pip install -r requirements.txt
    ) else (
        echo Installing minimal dependencies...
        pip install ultralytics fiftyone python-dotenv requests pyyaml rich plotly ftfy imageio rtree scikit-learn scikit-image opencv-python pillow pymongo mongoengine motor
    )
)

:: ------------------------------------------------------------
:: Verification Step
:: ------------------------------------------------------------

echo.
echo Verifying the environment...
echo.

python -c "import torch, cv2, fiftyone, numpy; \
print('Torch:', torch.__version__); \
print('CUDA:', torch.cuda.is_available()); \
print('OpenCV:', cv2.__version__); \
print('FiftyOne:', fiftyone.__version__); \
print('Numpy:', numpy.__version__);"

echo.
echo ========================================================
echo RSAN_Project environment setup COMPLETE!
if "%USE_CONDA%"=="true" (
    echo To activate later:   conda activate rsan_env
) else (
    echo To activate later:   env\Scripts\activate
)
echo ========================================================
pause