#!/usr/bin/env bash
# setup.sh – One-click setup for Medicine Recommendation System
# Usage: bash setup.sh

set -e

echo "============================================="
echo "  Medicine Recommendation System – Setup"
echo "  CS619 | Spring 2026"
echo "============================================="

# 1. Install dependencies
echo -e "\n[1/4] Installing Python dependencies..."

if [ -z "$VIRTUAL_ENV" ]; then
	if [ ! -d "venv" ]; then
		echo "[0/4] Creating virtual environment 'venv'..."
		python3 -m venv venv
	fi
	echo "Activating virtual environment..."
	source venv/bin/activate
fi

echo "Upgrading pip and installing requirements..."
python -m pip install --upgrade pip setuptools wheel --quiet
python -m pip install -r requirements.txt --quiet

# 2. Generate dataset
echo -e "\n[2/4] Generating dataset..."
cd data && python generate_dataset.py && cd ..

# 3. Train models
echo -e "\n[3/4] Training ML models (this may take a few minutes)..."
python models/train_models.py

echo -e "\n[4/4] Setup complete!"
echo ""
echo "▶  To launch the Web UI:"
echo "   streamlit run src/streamlit_app.py"
echo ""
echo "▶  To launch the REST API:"
echo "   python src/app.py"
echo ""
echo "▶  To run hyperparameter tuning:"
echo "   python utils/hyperparameter_tuning.py"
echo ""
echo "▶  To open the Jupyter notebook:"
echo "   jupyter notebook notebooks/EDA_and_Model_Analysis.ipynb"
echo ""
echo "✅ All models saved to ./models/"
echo "   Run: python src/app.py  to start the Flask API"
echo "   Run: streamlit run src/streamlit_app.py  for the Web UI"

# Start services: run Flask API in background and Streamlit in foreground
echo "\n[+] Starting services: Flask API (background) and Streamlit UI (foreground)"
mkdir -p logs
echo "Starting Flask API in background (logs/flask.log)..."
nohup python src/app.py > logs/flask.log 2>&1 &
FLASK_PID=$!
echo "Flask PID: ${FLASK_PID}"

# Ensure both processes are killed when this script receives SIGINT/SIGTERM
trap 'echo "Stopping services..."; kill "${FLASK_PID}" 2>/dev/null || true; exit 0' SIGINT SIGTERM

sleep 1
echo "Launching Streamlit UI (will run in foreground). Use Ctrl+C to stop." 
python -m streamlit run src/streamlit_app.py

# When Streamlit exits normally, stop Flask as well
echo "Streamlit exited, stopping Flask (${FLASK_PID})"
kill "${FLASK_PID}" 2>/dev/null || true
trap - SIGINT SIGTERM
