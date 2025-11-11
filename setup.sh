#!/usr/bin/env bash
# RSAN_Project - Environment Setup Script
# Author: Robotics Socially Aware Navigation Lab (RSAN Lab)
# Purpose: One-command setup for all team members (Mac, Windows, Linux)

# Step 1: Detect OS
echo "Detecting your operating system..."
OS="unknown"
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    OS="Linux"
elif [[ "$OSTYPE" == "darwin"* ]]; then
    OS="Mac"
elif [[ "$OSTYPE" == "cygwin" || "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
    OS="Windows"
fi
echo "Detected: $OS"

# Step 2: Check Python installation
echo "Checking Python installation..."
if command -v python3 &>/dev/null; then
    PYTHON_CMD="python3"
elif command -v python &>/dev/null; then
    PYTHON_CMD="python"
else
    echo "Python not found! Please install Python 3.9+ and re-run this script."
    exit 1
fi
echo "Python version: $($PYTHON_CMD --version)"

# Step 3: Create virtual environment
echo "Creating virtual environment..."
if [[ "$OS" == "Windows" ]]; then
    $PYTHON_CMD -m venv env
    source env/Scripts/activate
else
    $PYTHON_CMD -m venv env
    source env/bin/activate
fi
echo "Virtual environment activated."

# Step 4: Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Step 5: Install dependencies
if [ -f "requirements.txt" ]; then
    echo "Installing project dependencies..."
    pip install -r requirements.txt
else
    echo "requirements.txt not found! Skipping dependency installation."
fi

# Step 6: Verify important tools
echo "Installing development tools (black, isort, flake8)..."
pip install black isort flake8

# Step 7: Run quick verification
echo "Running formatter and linter checks..."
black --check src/ || echo "Code not formatted (Black)"
isort --check-only src/ || echo "Imports not sorted (isort)"
flake8 src/ || echo "Lint warnings found (flake8)"

# Step 8: Done
echo ""
echo "RSAN_Project environment setup complete!"
echo "---------------------------------------------------"
echo "To activate the environment in future sessions:"
if [[ "$OS" == "Windows" ]]; then
    echo "Run: source env/Scripts/activate"
else
    echo "Run: source env/bin/activate"
fi
echo "---------------------------------------------------"
echo "Next steps:"
echo "1️⃣ Run the project notebooks in 'colab/' or scripts in 'src/'"
echo "2️⃣ Use GitHub Actions for automated linting and formatting"
echo "3️⃣ Have fun building socially aware robots "