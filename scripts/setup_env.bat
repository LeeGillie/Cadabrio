@echo off
REM Cadabrio Environment Setup - Windows
REM Creates the conda environment with all dependencies

echo ============================================
echo  Cadabrio Environment Setup
echo ============================================
echo.

REM Check for conda
where conda >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Conda not found. Please install Miniconda or Anaconda.
    exit /b 1
)

echo Creating conda environment from environment.yml...
conda env create -f environment.yml

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo Environment creation failed. Try updating:
    echo   conda env update -f environment.yml --prune
    exit /b 1
)

echo.
echo ============================================
echo  Setup complete!
echo  Activate with: conda activate cadabrio
echo  Run with:      python -m cadabrio
echo ============================================
