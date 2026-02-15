#!/bin/bash
# Cadabrio Environment Setup
# Creates the conda environment with all dependencies

echo "============================================"
echo " Cadabrio Environment Setup"
echo "============================================"
echo

# Check for conda
if ! command -v conda &> /dev/null; then
    echo "ERROR: Conda not found. Please install Miniconda or Anaconda."
    exit 1
fi

echo "Creating conda environment from environment.yml..."
conda env create -f environment.yml

if [ $? -ne 0 ]; then
    echo
    echo "Environment creation failed. Try updating:"
    echo "  conda env update -f environment.yml --prune"
    exit 1
fi

echo
echo "============================================"
echo " Setup complete!"
echo " Activate with: conda activate cadabrio"
echo " Run with:      python -m cadabrio"
echo "============================================"
