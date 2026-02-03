#!/bin/bash
# Setup script for creating conda environment for SemFlow-MPPI

set -e  # Exit on error

ENV_NAME="semflow-mppi"
ENV_FILE="environment.yml"

echo "=========================================="
echo "SemFlow-MPPI Conda Environment Setup"
echo "=========================================="

# Check if conda is installed
if ! command -v conda &> /dev/null; then
    echo "Error: conda is not installed. Please install Anaconda or Miniconda first."
    exit 1
fi

# Check if environment already exists
if conda env list | grep -q "^${ENV_NAME} "; then
    echo "Warning: Environment '${ENV_NAME}' already exists."
    read -p "Do you want to remove it and create a new one? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Removing existing environment..."
        conda env remove -n "${ENV_NAME}" -y
    else
        echo "Keeping existing environment. Activate it with: conda activate ${ENV_NAME}"
        exit 0
    fi
fi

# Create environment from yml file
echo "Creating conda environment from ${ENV_FILE}..."
conda env create -f "${ENV_FILE}"

# Activate environment
echo "Activating environment..."
eval "$(conda shell.bash hook)"
conda activate "${ENV_NAME}"

# Install PyTorch with CUDA support (optional)
echo ""
echo "=========================================="
echo "PyTorch Installation"
echo "=========================================="
echo "Do you want to install PyTorch with CUDA support?"
echo "1) Yes, install PyTorch with CUDA 11.8"
echo "2) Yes, install PyTorch with CUDA 12.1"
echo "3) No, use CPU-only version (already in environment.yml)"
read -p "Enter choice (1/2/3): " -n 1 -r
echo

if [[ $REPLY =~ ^[1]$ ]]; then
    echo "Installing PyTorch with CUDA 11.8..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
elif [[ $REPLY =~ ^[2]$ ]]; then
    echo "Installing PyTorch with CUDA 12.1..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
else
    echo "Using CPU-only PyTorch from environment.yml"
fi

# Verify installation
echo ""
echo "=========================================="
echo "Verifying Installation"
echo "=========================================="
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import numpy; print(f'NumPy version: {numpy.__version__}')"
python -c "import scipy; print(f'SciPy version: {scipy.__version__}')"

echo ""
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo "To activate the environment, run:"
echo "  conda activate ${ENV_NAME}"
echo ""
echo "To deactivate, run:"
echo "  conda deactivate"

