# 💊 MediRec — Medicine Recommendation System
### CS619 Final Year Project | Spring 2026
**Supervisor:** Dr. Mushtaq Hussain | mushtaq.hussain@vu.edu.pk  
**Domain:** Machine Learning · Deep Learning · Healthcare AI

---

## 📌 Table of Contents

1. [Project Overview](#1-project-overview)
2. [Key Features](#2-key-features)
3. [System Architecture](#3-system-architecture)
4. [Technology Stack](#4-technology-stack)
5. [Project Structure](#5-project-structure)
6. [ML Models Used](#6-ml-models-used)
7. [Diseases & Medicines Covered](#7-diseases--medicines-covered)
8. [How to Run the Project (Step by Step)](#8-how-to-run-the-project-step-by-step)
9. [Testing the Application](#9-testing-the-application)
10. [API Reference](#10-api-reference)
11. [Troubleshooting](#11-troubleshooting)
12. [Disclaimer](#12-disclaimer)

---

## 1. Project Overview

The **Medicine Recommendation System** is an AI-powered virtual assistant designed to enhance patient care by providing accurate and personalized medicine suggestions. The system accepts patient symptoms, age, gender, severity level, and medical history as inputs, then uses a combination of trained Machine Learning models and clinical rule-based logic to recommend the most relevant medicines.

The system is built for three types of users:

- **Patients** — who want to understand what medicines relate to their symptoms
- **Healthcare Providers** — who want AI-assisted decision support
- **Pharmacists** — who want to cross-reference symptom-based suggestions

> ⚠️ **Important:** This system is for educational and research purposes only. Always consult a licensed healthcare professional before taking any medication.

---

## 2. Key Features

| Feature | Description |
|---|---|
| 🔍 Symptom-Based Diagnosis | Predicts disease from a selection of symptoms |
| 💊 Medicine Recommendation | Ranks medicines by ML confidence + clinical rules |
| 👤 Patient Personalization | Age, gender, severity, and history are factored in |
| 📊 Analytics Dashboard | Visual comparison of all trained ML models |
| 🗄️ Medicine Database | Searchable database with dosage, side effects, contraindications |
| 🌐 Web Interface | Clean 4-page Streamlit UI |
| 🔌 REST API | Flask API for integration with other systems |
| 🧠 Multiple ML Models | 6+ models trained and compared automatically |
| ⚙️ Hyperparameter Tuning | Optuna-based optimization for best model |
| 📓 Jupyter Notebook | Full EDA + model analysis notebook included |

---

## 3. System Architecture

```
┌─────────────────────────────────────────────────┐
│              Patient Input Layer                │
│   Symptoms · Age · Gender · Severity · History  │
└────────────────────┬────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────┐
│           Feature Engineering                   │
│   One-hot symptoms · Encoding · Scaling         │
└────────────────────┬────────────────────────────┘
                     │
          ┌──────────┴──────────┐
          ▼                     ▼
┌─────────────────┐   ┌─────────────────────────┐
│  ML Disease     │   │  ML Medicine            │
│  Predictor      │   │  Predictor              │
│  (Best Model)   │   │  (Best Model)           │
└────────┬────────┘   └───────────┬─────────────┘
         │                        │
         └──────────┬─────────────┘
                    ▼
┌─────────────────────────────────────────────────┐
│         Rule-Based Clinical Engine              │
│   Merges ML output with disease-medicine rules  │
└────────────────────┬────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────┐
│         Ranked Recommendations                  │
│   Medicine · Confidence · Source · Drug Info    │
└────────┬──────────────────────────┬─────────────┘
         │                          │
         ▼                          ▼
┌─────────────────┐       ┌─────────────────────┐
│  Streamlit UI   │       │   Flask REST API     │
│  localhost:8501 │       │   localhost:5000     │
└─────────────────┘       └─────────────────────┘
```

---

## 4. Technology Stack

| Layer | Tools |
|---|---|
| **Language** | Python 3.10+ |
| **ML Models** | Scikit-learn, XGBoost, LightGBM |
| **Deep Learning** | TensorFlow, Keras |
| **Hyperparameter Tuning** | Optuna |
| **Explainability** | SHAP |
| **Data Processing** | Pandas, NumPy, SciPy |
| **Visualization** | Plotly, Matplotlib, Seaborn |
| **Web UI** | Streamlit |
| **REST API** | Flask, Flask-CORS |
| **NLP** | NLTK |
| **IDE** | VS Code / Google Colab |
| **Notebook** | Jupyter |

---

## 5. Project Structure

```
medicine_recommendation_system/
│
├── data/
│   ├── generate_dataset.py       # Generates 3000 synthetic patient records
│   ├── patient_records.csv       # Auto-generated training dataset
│   └── metadata.json             # Disease, symptom, medicine metadata
│
├── models/
│   ├── train_models.py           # Full ML training pipeline
│   ├── disease_model.pkl         # Trained disease prediction model
│   ├── medicine_model.pkl        # Trained medicine prediction model
│   ├── label_encoder_disease.pkl
│   ├── label_encoder_medicine.pkl
│   ├── scaler.pkl                # Feature scaler
│   ├── feature_columns.pkl       # Feature column order
│   └── training_results.json     # Accuracy comparison of all models
│
├── src/
│   ├── __init__.py
│   ├── recommender.py            # Core recommendation engine
│   ├── app.py                    # Flask REST API (5 endpoints)
│   └── streamlit_app.py          # Streamlit Web UI (4 pages)
│
├── utils/
│   ├── preprocessing.py          # Text normalization utilities
│   ├── evaluate.py               # Confusion matrix + SHAP analysis
│   └── hyperparameter_tuning.py  # Optuna tuning for best model
│
├── notebooks/
│   └── EDA_and_Model_Analysis.ipynb   # Full analysis notebook
│
├── requirements.txt              # All Python dependencies
├── setup.sh                      # One-click setup script
└── README.md                     # Project readme
```

---

## 6. ML Models Used

| # | Model | Type | Notes |
|---|---|---|---|
| 1 | Logistic Regression | Linear | Baseline model |
| 2 | Random Forest | Ensemble | High accuracy, robust |
| 3 | Gradient Boosting | Ensemble | Often best performer |
| 4 | Support Vector Machine | Kernel-based | Good on small features |
| 5 | K-Nearest Neighbors | Instance-based | Simple, interpretable |
| 6 | Naive Bayes | Probabilistic | Fast, good baseline |
| 7 | XGBoost | Boosting | Industry standard |
| 8 | LightGBM | Boosting | Fast, memory efficient |
| 9 | Neural Network | Deep Learning | TensorFlow/Keras |

The system **automatically selects and saves the best model** based on test accuracy and F1 score.

---

## 7. Diseases & Medicines Covered

**15 Diseases:**

| Disease | Severity |
|---|---|
| Common Cold | Mild |
| Flu (Influenza) | Moderate |
| Allergic Rhinitis | Mild |
| Insomnia | Mild |
| Migraine | Moderate |
| Gastritis | Moderate |
| Urinary Tract Infection | Moderate |
| Anxiety Disorder | Moderate |
| Hypertension | Serious |
| Type 2 Diabetes | Serious |
| Asthma | Serious |
| Depression | Serious |
| Arthritis | Serious |
| Pneumonia | Serious |
| Hypothyroidism | Serious |

**Sample Medicines:** Paracetamol, Metformin, Amlodipine, Salbutamol, Sumatriptan, Omeprazole, Amoxicillin, Levothyroxine, Cetirizine, Ibuprofen, and 40+ more.

---

## 8. How to Run the Project (Step by Step)

### Prerequisites
- Ubuntu 20.04 or later
- Internet connection (for pip installs)
- At least 2GB free disk space

---

### Step 1 — Open Terminal

Press `Ctrl + Alt + T`

---

### Step 2 — Navigate to Project Folder

```bash
cd ~/Desktop/medicine_recommendation_system./medicine_recommendation_system
```

Confirm files are present:

```bash
ls
```

Expected output:
```
data/  models/  notebooks/  requirements.txt  README.md  setup.sh  src/  utils/
```

---

### Step 3 — Install Python & Required Tools

```bash
sudo apt update
sudo apt install python3 python3-pip python3-venv -y
```

Check Python version (must be 3.8 or higher):

```bash
python3 --version
```

---

### Step 4 — Create a Virtual Environment

A virtual environment keeps your project's packages isolated from the rest of your system.

```bash
python3 -m venv venv
```

Activate it:

```bash
source venv/bin/activate
```

Your terminal prompt will now show `(venv)` — this confirms it is active.

> Every time you open a new terminal to work on this project, run `source venv/bin/activate` first.

---

### Step 5 — Update requirements.txt (Fix Version Conflicts)

Replace the strict version requirements with flexible ones that work with your Python:

```bash
cat > requirements.txt << 'EOF'
streamlit
pandas
numpy
scikit-learn
xgboost
lightgbm
matplotlib
seaborn
plotly
joblib
scipy
nltk
flask
flask-cors
imbalanced-learn
optuna
pillow
requests
python-dotenv
EOF
```

---

### Step 6 — Install All Dependencies

```bash
pip install -r requirements.txt
```

> This will take **3–5 minutes**. You will see packages downloading one by one. This is normal.

Verify key packages installed correctly:

```bash
python -c "import streamlit, sklearn, pandas, flask; print('All packages OK')"
```

Expected output:
```
All packages OK
```

---

### Step 7 — Generate the Dataset

```bash
cd data
python generate_dataset.py
cd ..
```

Expected output:
```
Generating patient dataset...
✓ Generated 3000 patient records → patient_records.csv
✓ Saved metadata → metadata.json
```

This creates two files: `patient_records.csv` (3000 rows of patient data) and `metadata.json` (disease/medicine information).

---

### Step 8 — Train the ML Models

```bash
python models/train_models.py
```

Expected output (takes 1–3 minutes):
```
Loaded 3000 records with 82 columns

Training for: Disease Prediction
  Logistic Regression     CV=1.000  Test=0.997
  Random Forest           CV=0.998  Test=0.997
  Gradient Boosting       CV=0.998  Test=1.000
  ...
  ✓ Best model: Gradient Boosting (accuracy=1.000)

Training for: Medicine Prediction
  ...
  ✓ Best model: Logistic Regression (accuracy=1.000)

✅ All models saved to ./models/
```

All model files (`.pkl`) are now saved in the `models/` folder.

---

### Step 9 — Launch the Web Application

```bash
python -m streamlit run src/streamlit_app.py
```

Expected output:
```
  You can now view your Streamlit app in your browser.
  Local URL: http://localhost:8501
  Network URL: http://192.168.x.x:8501
```

Your browser will open automatically at **http://localhost:8501**

To stop the app: press `Ctrl + C` in the terminal.

---

### Step 10 — (Optional) Launch the Flask REST API

Open a **second terminal** window, then:

```bash
cd ~/Desktop/medicine_recommendation_system./medicine_recommendation_system
source venv/bin/activate
python src/app.py
```

Expected output:
```
🚀 Starting Medicine Recommendation API on http://localhost:5000
 * Running on http://0.0.0.0:5000
```

---

### Step 11 — (Optional) Run Jupyter Notebook for Analysis

```bash
pip install jupyter
jupyter notebook notebooks/EDA_and_Model_Analysis.ipynb
```

This opens the notebook in your browser with full EDA charts, model comparison graphs, and SHAP analysis.

---

### Step 12 — (Optional) Run Hyperparameter Tuning

```bash
pip install optuna
python utils/hyperparameter_tuning.py
```

This uses Optuna to find the best parameters for Random Forest and XGBoost. Takes 5–10 minutes. Saves the tuned model as `models/disease_model_tuned.pkl`.

---

### Every Time You Return to the Project

You only need to run Steps 7 and 8 **once**. After that, each time:

```bash
cd ~/Desktop/medicine_recommendation_system./medicine_recommendation_system
source venv/bin/activate
python -m streamlit run src/streamlit_app.py
```

---

## 9. Testing the Application

### Test Case 1 — Flu
- Age: `28` | Gender: `Male` | Severity: `Moderate`
- Symptoms: `high fever`, `body ache`, `fatigue`, `headache`, `chills`
- ✅ Expected: **Flu (Influenza)** — Oseltamivir, Paracetamol, Ibuprofen

### Test Case 2 — Diabetes
- Age: `45` | Gender: `Female` | Severity: `Serious`
- Symptoms: `frequent urination`, `excessive thirst`, `fatigue`, `blurred vision`
- ✅ Expected: **Type 2 Diabetes** — Metformin, Glibenclamide

### Test Case 3 — Migraine
- Age: `32` | Gender: `Female` | Severity: `Moderate`
- Symptoms: `severe headache`, `nausea`, `light sensitivity`, `vomiting`
- ✅ Expected: **Migraine** — Sumatriptan, Rizatriptan

### Test Case 4 — Asthma
- Age: `19` | Gender: `Male` | Severity: `Serious`
- Symptoms: `shortness of breath`, `wheezing`, `chest tightness`, `cough`
- ✅ Expected: **Asthma** — Salbutamol, Formoterol

---

## 10. API Reference

Base URL: `http://localhost:5000`

### GET /
Health check.
```bash
curl http://localhost:5000/
```

### POST /api/recommend
Get medicine recommendations.
```bash
curl -X POST http://localhost:5000/api/recommend \
  -H "Content-Type: application/json" \
  -d '{
    "symptoms": ["headache", "fever", "body ache"],
    "age": 25,
    "gender": "Male",
    "severity": 2
  }'
```

### GET /api/symptoms
Get list of all supported symptoms.
```bash
curl http://localhost:5000/api/symptoms
```

### GET /api/diseases
Get list of all diseases.
```bash
curl http://localhost:5000/api/diseases
```

### GET /api/medicine/{name}
Get detailed info about a medicine.
```bash
curl http://localhost:5000/api/medicine/Metformin
```

---

## 11. Troubleshooting

| Error | Cause | Fix |
|---|---|---|
| `command not found: python` | Ubuntu uses `python3` | Always use `python3` or activate venv |
| `command not found: streamlit` | Streamlit not in PATH | Use `python -m streamlit run ...` |
| `ModuleNotFoundError` | Package not installed | Run `pip install -r requirements.txt` inside venv |
| `No matching distribution` | Version conflict | Replace requirements.txt with flexible versions (Step 5) |
| `externally-managed-environment` | venv not active | Run `source venv/bin/activate` |
| Port 8501 already in use | Another app running | Use `--server.port 8502` flag |
| Port 5000 already in use | Another Flask app | Use `port=5001` in `app.py` |
| Models not found error | Training not done | Run `python models/train_models.py` first |
| Dataset not found error | Data not generated | Run `python data/generate_dataset.py` first |

---

## 12. Disclaimer

> This Medicine Recommendation System is developed purely for **educational and research purposes** as part of the CS619 Final Year Project at Virtual University of Pakistan.
>
> The recommendations generated by this system are **AI-based suggestions only** and should **never** replace the advice, diagnosis, or treatment provided by a licensed and qualified healthcare professional.
>
> Always consult your doctor or pharmacist before taking any medication.

---

*CS619 — Spring 2026 | Virtual University of Pakistan*  
*Supervisor: Dr. Mushtaq Hussain | mushtaq.hussain@vu.edu.pk*
