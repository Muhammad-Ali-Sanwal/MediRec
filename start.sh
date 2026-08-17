#!/usr/bin/env bash
# start.sh — Start Flask API (background) and Streamlit UI (foreground)
# Usage: bash start.sh

set -e

echo "============================================="
echo "  MediRec — Start Services"
echo "============================================="

# Activate virtualenv if present; otherwise create one and install requirements
if [ -d "venv" ]; then
    echo "Activating existing virtualenv..."
    # shellcheck disable=SC1091
    source venv/bin/activate
else
    echo "Virtualenv not found — creating 'venv' and installing requirements (may take a while)..."
    python3 -m venv venv
    # shellcheck disable=SC1091
    source venv/bin/activate
    python -m pip install --upgrade pip setuptools wheel
    python -m pip install -r requirements.txt
fi

mkdir -p logs

echo "Starting Flask API in background (logs/flask.log)..."
nohup python src/app.py > logs/flask.log 2>&1 &
FLASK_PID=$!
echo "Flask PID: ${FLASK_PID}"

# Ensure clean shutdown of Flask when this script receives Ctrl+C or terminates
trap 'echo "Stopping services..."; kill "${FLASK_PID}" 2>/dev/null || true; exit 0' SIGINT SIGTERM

sleep 1
echo "Launching Streamlit UI (foreground). Press Ctrl+C to stop both services."
python -m streamlit run src/streamlit_app.py

# Streamlit exited — stop Flask as well
echo "Streamlit exited, stopping Flask (${FLASK_PID})"
kill "${FLASK_PID}" 2>/dev/null || true
trap - SIGINT SIGTERM

echo "All services stopped. Logs: logs/flask.log"
