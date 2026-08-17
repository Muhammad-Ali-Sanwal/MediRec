@echo off
:: run_app.bat — One-click Runner for Windows
:: Double click this file to auto-setup & run the application

title MediRec - Medicine Recommendation System

echo =====================================================
echo  💊 Medicine Recommendation System (Windows Setup & Run)
echo =====================================================

cd /d "%~dp0"

:: 1. Check Virtual Environment
if not exist "venv\" (
    echo.
    echo [1/4] Virtual environment not found. Creating venv...
    python -m venv venv
    if errorlevel 1 (
        echo ERROR: Python is not installed or not in system PATH.
        echo Please install Python 3.8+ from https://www.python.org/
        pause
        exit /b 1
    )
) else (
    echo.
    echo [1/4] Virtual environment (venv) found.
)

:: Activate virtual environment
call venv\Scripts\activate.bat

:: 2. Check Dependencies
python -c "import streamlit, pandas, sklearn, plotly" >nul 2>&1
if errorlevel 1 (
    echo.
    echo [2/4] Dependencies missing. Installing from requirements.txt...
    python -m pip install --upgrade pip --quiet
    pip install -r requirements.txt --quiet
    echo ✓ Dependencies installed successfully.
) else (
    echo.
    echo [2/4] All required Python dependencies are already installed.
)

:: 3. Check Dataset & Metadata
if not exist "data\patient_records.csv" (
    echo.
    echo [3/4] Dataset missing. Generating patient dataset...
    cd data
    python generate_dataset.py
    cd ..
) else (
    echo.
    echo [3/4] Dataset found.
)

:: 4. Check Models
if not exist "models\disease_model.pkl" (
    echo.
    echo [4/4] Trained models missing. Training ML/DL models now...
    python models/train_models.py
) else (
    echo.
    echo [4/4] Trained models found.
)

echo.
echo =====================================================
echo 🚀 Launching MediRec Streamlit Application...
echo =====================================================

python -m streamlit run src/streamlit_app.py

pause
