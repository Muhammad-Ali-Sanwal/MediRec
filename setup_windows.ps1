<#
setup_windows.ps1 - One-click setup for Windows (PowerShell)

Run this script from an elevated or normal PowerShell prompt.
If your execution policy blocks scripts, run the command below from an Administrator PowerShell once:

    powershell -ExecutionPolicy Bypass -File .\setup_windows.ps1

Or from a normal PowerShell prompt to bypass for this run:

    powershell -ExecutionPolicy Bypass -Command "& { .\setup_windows.ps1 }"

This script will:
 - create a `venv` virtual environment (if missing)
 - activate the venv
 - upgrade pip and install `requirements.txt`
 - generate the synthetic dataset (`data/generate_dataset.py`)
 - run model training (`models/train_models.py`)

Keep `setup.sh` untouched for Unix systems.
#>

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

Write-Host "============================================="
Write-Host "  Medicine Recommendation System – Setup (Windows)"
Write-Host "  CS619 | Spring 2026"
Write-Host "============================================="

Write-Host "`n[1/4] Checking Python availability..."
if (-not (Get-Command python -ErrorAction SilentlyContinue)) {
    Write-Error "Python not found in PATH. Install Python 3.8+ and ensure 'python' is on PATH."
    exit 1
}

# 1. Create & activate venv
if (-not $env:VIRTUAL_ENV) {
    if (-not (Test-Path -Path .\venv)) {
        Write-Host "[0/4] Creating virtual environment 'venv'..."
        python -m venv venv
    }
    Write-Host "Activating virtual environment..."
    # For PowerShell activation
    .\venv\Scripts\Activate.ps1
}

Write-Host "Upgrading pip and installing requirements..."
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt

# 2. Generate dataset
Write-Host "`n[2/4] Generating dataset..."
Push-Location data
python generate_dataset.py
Pop-Location

# 3. Train models
Write-Host "`n[3/4] Training ML models (this may take a few minutes)..."
python models/train_models.py

Write-Host "`n[4/4] Setup complete!`n"
Write-Host "▶  To launch the Web UI:"
Write-Host "   streamlit run src/streamlit_app.py" -ForegroundColor Cyan
Write-Host "▶  To launch the REST API:"
Write-Host "   python src/app.py" -ForegroundColor Cyan
Write-Host "▶  To run hyperparameter tuning:"
Write-Host "   python utils/hyperparameter_tuning.py" -ForegroundColor Cyan
Write-Host "▶  To open the Jupyter notebook:"
Write-Host "   jupyter notebook notebooks/EDA_and_Model_Analysis.ipynb" -ForegroundColor Cyan
Write-Host ""
Write-Host "✅ All models saved to ./models/" -ForegroundColor Green
Write-Host "   Run: python src/app.py  to start the Flask API" -ForegroundColor Cyan
Write-Host "   Run: streamlit run src/streamlit_app.py  for the Web UI" -ForegroundColor Cyan

# Start services: run Flask API in background and Streamlit in foreground
Write-Host "`n[+] Starting services: Flask API (background) and Streamlit UI (foreground)..." -ForegroundColor Yellow
if (-not (Test-Path -Path .\logs)) { New-Item -ItemType Directory -Path .\logs | Out-Null }
Write-Host "Starting Flask API in background (logs\flask.log)..." -ForegroundColor Cyan
$flask = Start-Process -FilePath python -ArgumentList 'src/app.py' -RedirectStandardOutput '.\logs\flask.log' -RedirectStandardError '.\logs\flask.log' -PassThru
Start-Sleep -Seconds 1
Write-Host "Launching Streamlit UI (will run in foreground). Use Ctrl+C to stop." -ForegroundColor Yellow

try {
    python -m streamlit run src/streamlit_app.py
}
finally {
    if ($flask -and -not $flask.HasExited) {
        Write-Host "Stopping Flask process (ID: $($flask.Id))..." -ForegroundColor Yellow
        Stop-Process -Id $flask.Id -Force -ErrorAction SilentlyContinue
    }
}
