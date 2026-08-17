<#
start_windows.ps1 - Start Flask API (background) and Streamlit UI (foreground) on Windows
Usage: powershell -ExecutionPolicy Bypass -File .\start_windows.ps1
#>

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

Write-Host "============================================="
Write-Host "  MediRec — Start Services (Windows)"
Write-Host "============================================="

# Ensure Python is available
if (-not (Get-Command python -ErrorAction SilentlyContinue)) {
    Write-Error "Python not found in PATH. Install Python 3.8+ and ensure 'python' is on PATH."
    exit 1
}

# Create/activate venv
if (Test-Path .\venv) {
    Write-Host "Activating existing virtualenv..."
    .\venv\Scripts\Activate.ps1
} else {
    Write-Host "Virtualenv not found — creating 'venv' and installing requirements (may take a while)..."
    python -m venv venv
    .\venv\Scripts\Activate.ps1
    python -m pip install --upgrade pip setuptools wheel
    python -m pip install -r requirements.txt
}

if (-not (Test-Path -Path .\logs)) { New-Item -ItemType Directory -Path .\logs | Out-Null }

Write-Host "Starting Flask API in background (logs\flask.log)..."
$flask = Start-Process -FilePath python -ArgumentList 'src/app.py' -RedirectStandardOutput '.\logs\flask.log' -RedirectStandardError '.\logs\flask.log' -PassThru
Start-Sleep -Seconds 1

Write-Host "Launching Streamlit UI (foreground). Press Ctrl+C to stop both services."
try {
    python -m streamlit run src/streamlit_app.py
}
finally {
    if ($flask -and -not $flask.HasExited) {
        Write-Host "Stopping Flask process (ID: $($flask.Id))..."
        Stop-Process -Id $flask.Id -Force -ErrorAction SilentlyContinue
    }
}

Write-Host "All services stopped. Logs: .\logs\flask.log"
