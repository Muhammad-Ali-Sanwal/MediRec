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
pip install -r requirements.txt --quiet

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
