#!/usr/bin/env bash
# ================================================================
# RSAN_Project - Smart Environment Setup Script
# Author: Robotics Socially Aware Navigation Lab (RSAN Lab)
# Purpose: Automatically sets up and verifies the development
#          environment for any team member (Mac, Linux, Windows)
# ================================================================

# Detect Operating System
echo "Detecting your operating system..."
OS="unknown"
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    OS="Linux"
elif [[ "$OSTYPE" == "darwin"* ]]; then
    OS="Mac"
elif [[ "$OSTYPE" == "cygwin" || "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
    OS="Windows"
fi
echo "Detected OS: $OS"


# Check for Conda
if command -v conda &>/dev/null; then
    echo "Conda detected — using Conda environment setup."
    USE_CONDA=true
else
    echo "Conda not found. Falling back to Python venv setup."
    USE_CONDA=false
fi

# Create Environment
if [ "$USE_CONDA" = true ]; then
    # Use Conda
    ENV_NAME="rsan_env"
    echo "Checking if Conda environment '$ENV_NAME' already exists..."
    if conda info --envs | grep -q "$ENV_NAME"; then
        echo "Environment '$ENV_NAME' already exists. Skipping creation."
    else
        echo "Creating Conda environment from environment.yml..."
        conda env create -f environment.yml || {
            echo "Conda environment creation failed!"
            exit 1
        }
    fi

    echo "Activating Conda environment..."
    eval "$(conda shell.bash hook)"
    conda activate "$ENV_NAME"

else
    # Use Python venv
    echo "Creating virtual environment using Python venv..."
    if command -v python3 &>/dev/null; then
        PYTHON_CMD="python3"
    elif command -v python &>/dev/null; then
        PYTHON_CMD="python"
    else
        echo "Python not found! Please install Python 3.9+ and re-run."
        exit 1
    fi

    $PYTHON_CMD -m venv env
    source env/bin/activate
    echo "Virtual environment activated."

    echo "Installing dependencies from requirements.txt..."
    pip install --upgrade pip
    pip install -r requirements.txt
fi

# Verify Installation
echo ""
echo "Verifying key libraries..."
python - <<'EOF'
import torch, cv2, fiftyone, numpy
print("\n RSAN environment verification successful!")
print(f"Torch: {torch.__version__}")
print(f"OpenCV: {cv2.__version__}")
print(f"FiftyOne: {fiftyone.__version__}")
print(f"Numpy: {numpy.__version__}")
EOF

# Auto-sync environment files (best practice)
if [ "$USE_CONDA" = true ]; then
    echo ""
    echo " Syncing environment files for RSAN_Project..."
    conda env export > environment.yml
    pip freeze > requirements.txt
    # Remove system-specific prefix for portability
    sed -i '' '/^prefix:/d' environment.yml 2>/dev/null || true
    echo " environment.yml and requirements.txt updated successfully."
fi


# Final Message
echo ""
echo " RSAN_Project environment setup complete!"
echo "---------------------------------------------------"
if [ "$USE_CONDA" = true ]; then
    echo "To activate later:  conda activate rsan_env"
else
    echo "To activate later:  source env/bin/activate"
fi
echo "---------------------------------------------------"
echo "Next steps:"
echo "1️⃣ Run notebooks in 'colab/' or Python files in 'src/'."
echo "2️⃣ Use GitHub Actions for automated formatting and linting."
echo "3️⃣ Have fun building socially aware robots "
echo "---------------------------------------------------"