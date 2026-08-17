#!/usr/bin/env bash
# run_app.sh — One-click Runner for Linux & macOS
# Usage: ./run_app.sh  or  bash run_app.sh

set -e

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

echo "====================================================="
echo " 💊 Medicine Recommendation System (Linux / macOS)"
echo "====================================================="

# 1. Virtual Environment Check & Setup
if [ ! -d "venv" ]; then
    echo -e "\n[1/4] Virtual environment not found. Creating venv..."
    python3 -m venv venv
else
    echo -e "\n[1/4] Virtual environment (venv) found."
fi

# Activate virtual environment
source venv/bin/activate

# 2. Dependencies Check & Installation
if ! python -c "import streamlit, pandas, sklearn, plotly" 2>/dev/null; then
    echo -e "\n[2/4] Dependencies missing. Installing from requirements.txt..."
    pip install --upgrade pip --quiet
    pip install -r requirements.txt --quiet
    echo "✓ Dependencies installed successfully."
else
    echo -e "\n[2/4] All required Python dependencies are already installed."
fi

# 3. Dataset & Models Check
if [ ! -f "data/patient_records.csv" ] || [ ! -f "data/metadata.json" ]; then
    echo -e "\n[3/4] Dataset missing. Generating patient dataset..."
    cd data && python generate_dataset.py && cd ..
else
    echo -e "\n[3/4] Dataset found."
fi

if [ ! -f "models/disease_model.pkl" ] || [ ! -f "models/medicine_model.pkl" ]; then
    echo -e "\n[4/4] Trained models missing. Training ML/DL models now..."
    python models/train_models.py
else
    echo -e "\n[4/4] Trained models found."
fi

echo -e "\n====================================================="
echo "🚀 Launching MediRec Streamlit Application..."
echo "====================================================="

python -m streamlit run src/streamlit_app.py
