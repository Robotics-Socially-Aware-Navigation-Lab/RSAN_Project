# #!/usr/bin/env bash
# # ================================================================
# # RSAN_Project - Smart Environment Setup Script (AUTO-RECREATE)
# # Author: Robotics Socially Aware Navigation Lab (RSAN Lab)
# # Purpose: Guaranteed consistent environment for ALL teammates:
# #          • Mac (Intel / M1 / M2 / M3)
# #          • Linux
# #          • Windows (WSL)
# # ================================================================

# echo "-----------------------------------------------------"
# echo "   RSAN PROJECT — SMART ENVIRONMENT SETUP"
# echo "-----------------------------------------------------"

# # Detect OS
# OS="unknown"
# if [[ "$OSTYPE" == "linux-gnu"* ]]; then
#     OS="Linux"
# elif [[ "$OSTYPE" == "darwin"* ]]; then
#     OS="Mac"
# elif [[ "$OSTYPE" == *"msys"* || "$OSTYPE" == *"cygwin"* ]]; then
#     OS="Windows"
# fi
# echo "Detected OS: $OS"

# # ------------------------------------------------------
# # Detect Conda
# # ------------------------------------------------------
# if command -v conda &>/dev/null; then
#     echo "Conda detected — using Conda for environment setup."
#     USE_CONDA=true
# else
#     echo "Conda NOT found — using Python venv fallback."
#     USE_CONDA=false
# fi

# # ------------------------------------------------------
# # Create (or RECREATE) Environment
# # ------------------------------------------------------
# ENV_NAME="rsan_env"

# if [ "$USE_CONDA" = true ]; then
#     echo ""
#     echo "Checking for existing environment '$ENV_NAME'..."

#     if conda info --envs | grep -q "$ENV_NAME"; then
#         echo "Environment exists — removing for a clean rebuild..."
#         conda env remove -n "$ENV_NAME" -y
#     else
#         echo "No previous environment found."
#     fi

#     echo "Creating fresh Conda environment from environment.yml..."
#     conda env create -f environment.yml || {
#         echo "❌ FAILED to create conda environment!"
#         exit 1
#     }

#     echo "Activating environment..."
#     eval "$(conda shell.bash hook)"
#     conda activate "$ENV_NAME"

# else
#     # ------------------------------
#     # Python venv fallback
#     # ------------------------------
#     echo ""
#     echo "Using Python venv fallback..."

#     echo "Removing old virtual env..."
#     rm -rf env

#     echo "Creating new virtual environment..."
#     python3 -m venv env
#     source env/bin/activate

#     echo "Installing dependencies (requirements.txt)..."
#     pip install --upgrade pip
#     pip install -r requirements.txt
# fi

# # ------------------------------------------------------
# # VERIFY INSTALLATION
# # ------------------------------------------------------
# echo ""
# echo "Verifying core libraries..."

# python - <<'EOF'
# import torch, cv2, fiftyone, numpy
# print("\n RSAN environment verification successful!")
# print(f"Torch: {torch.__version__}")
# print(f"OpenCV: {cv2.__version__}")
# print(f"FiftyOne: {fiftyone.__version__}")
# print(f"Numpy: {numpy.__version__}")
# EOF

# echo ""
# echo "-----------------------------------------------------"
# echo "  RSAN_Project environment setup COMPLETE 🎉"
# echo "-----------------------------------------------------"
# if [ "$USE_CONDA" = true ]; then
#     echo "To activate later:   conda activate rsan_env"
# else
#     echo "To activate later:   source env/bin/activate"
# fi
# echo "-----------------------------------------------------"
# echo "Next steps:"
# echo " 1️⃣ Run: python -m src.tools.run_unified_pipeline image.jpg"
# echo " 2️⃣ Or   python -m src.tools.run_unified_pipeline webcam"
# echo " 3️⃣ Make sure OPENAI_API_KEY is set if you want LLM reasoning."
# echo "-----------------------------------------------------"


#!/usr/bin/env bash
# ================================================================
# RSAN_Project - Smart Environment Setup Script (FIXED)
# Works on:
#   • Mac (Intel / M1 / M2 / M3)
#   • Linux
#   • Windows (WSL)
# ================================================================

echo "-----------------------------------------------------"
echo "   RSAN PROJECT — SMART ENVIRONMENT SETUP"
echo "-----------------------------------------------------"

# Detect OS
OS="unknown"
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    OS="Linux"
elif [[ "$OSTYPE" == "darwin"* ]]; then
    OS="Mac"
elif [[ "$OSTYPE" == *"msys"* || "$OSTYPE" == *"cygwin"* ]]; then
    OS="Windows"
fi
echo "Detected OS: $OS"

# ------------------------------------------------------
# Detect Conda
# ------------------------------------------------------
if command -v conda &>/dev/null; then
    echo "Conda detected — using Conda for environment setup."
    USE_CONDA=true
else
    echo "Conda NOT found — using Python venv fallback."
    USE_CONDA=false
fi

ENV_NAME="rsan_env"

# ======================================================
# FIX #1 — Force stable channels to avoid Mac M1 errors
# ======================================================
if [ "$USE_CONDA" = true ]; then
    echo ""
    echo "Applying safe Conda channel configuration..."
    conda config --set channel_priority strict
    conda config --remove channels conda-forge 2>/dev/null
    conda config --add channels pytorch
    conda config --add channels defaults
fi


# ------------------------------------------------------
# Create (or RECREATE) Environment
# ------------------------------------------------------
if [ "$USE_CONDA" = true ]; then
    echo ""
    echo "Checking for existing environment '$ENV_NAME'..."

    if conda info --envs | grep -q "$ENV_NAME"; then
        echo "Environment exists — removing for a clean rebuild..."
        conda env remove -n "$ENV_NAME" -y
    else
        echo "No previous environment found."
    fi

    echo ""
    echo "Creating fresh Conda environment from environment.yml..."
    conda env create -f environment.yml || {
        echo "⚠️ Conda install failed — retrying using pip fallback..."
        
        # Create a tiny base conda env and pip install inside it
        conda create -y -n "$ENV_NAME" python=3.10
        eval "$(conda shell.bash hook)"
        conda activate "$ENV_NAME"
        
        pip install --upgrade pip
        pip install -r requirements.txt
    }

    echo ""
    echo "Activating environment..."
    eval "$(conda shell.bash hook)"
    conda activate "$ENV_NAME"

else
    # ------------------------------
    # Python venv fallback
    # ------------------------------
    echo ""
    echo "Using Python venv fallback..."

    echo "Removing old virtual env..."
    rm -rf env

    echo "Creating new virtual environment..."
    python3 -m venv env
    source env/bin/activate

    echo "Installing dependencies (requirements.txt)..."
    pip install --upgrade pip
    pip install -r requirements.txt
fi

# ------------------------------------------------------
# VERIFY INSTALLATION
# ------------------------------------------------------
echo ""
echo "Verifying core libraries..."

python - <<'EOF'
import torch, cv2, fiftyone, numpy
print("\n RSAN environment verification successful!")
print(f"Torch: {torch.__version__}")
print(f"MPS Available: {torch.backends.mps.is_available()}")
print(f"OpenCV: {cv2.__version__}")
print(f"FiftyOne: {fiftyone.__version__}")
print(f"Numpy: {numpy.__version__}")
EOF

echo ""
echo "-----------------------------------------------------"
echo "  RSAN_Project environment setup COMPLETE 🎉"
echo "-----------------------------------------------------"

if [ "$USE_CONDA" = true ]; then
    echo "To activate later:   conda activate rsan_env"
else
    echo "To activate later:   source env/bin/activate"
fi

echo "-----------------------------------------------------"
echo "Next steps:"
echo " 1️⃣ python -m src.tools.run_unified_pipeline image.jpg"
echo " 2️⃣ python -m src.tools.run_unified_pipeline webcam"
echo " 3️⃣ Make sure OPENAI_API_KEY is set for LLM reasoning."
echo "-----------------------------------------------------"