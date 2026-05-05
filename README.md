# 💊 Medicine Recommendation System

**CS619 – Final Year Project | Spring 2026**
**Supervisor:** Dr. Mushtaq Hussain | mushtaq.hussain@vu.edu.pk
**Student:** Muhammad Ali Sanwal | bc240440384mas@vu.edu.pk

---

## Overview

An AI-powered medicine recommendation system that provides personalized medicine
suggestions based on patient symptoms, age, gender, and medical history. The system
uses multiple machine learning and deep learning models with a rule-based augmentation
layer for high accuracy.

---

## Project Structure

```
medicine_recommendation_system/
│
├── data/
│   ├── generate_dataset.py      # Generates synthetic patient dataset
│   ├── patient_records.csv      # (auto-generated) Training data
│   └── metadata.json            # (auto-generated) Disease/medicine metadata
│
├── models/
│   ├── train_models.py          # ML training pipeline (6+ models + DL)
│   ├── disease_model.pkl        # (auto-generated)
│   ├── medicine_model.pkl       # (auto-generated)
│   ├── label_encoder_disease.pkl
│   ├── label_encoder_medicine.pkl
│   ├── scaler.pkl
│   └── training_results.json   # Accuracy comparison
│
├── src/
│   ├── recommender.py           # Core recommendation engine
│   ├── app.py                   # Flask REST API
│   └── streamlit_app.py         # Streamlit Web UI (4 pages)
│
├── utils/
│   └── hyperparameter_tuning.py # Optuna-based tuning
│
├── notebooks/
│   └── EDA_and_Model_Analysis.ipynb
│
├── requirements.txt
├── setup.sh                     # One-click setup
└── README.md
```

---

## Quick Start

### Option 1 – One-click setup (Linux/Mac)
```bash
bash setup.sh
```

### Option 2 – Manual steps
```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Generate dataset
cd data && python generate_dataset.py && cd ..

# 3. Train models
python models/train_models.py

# 4. Launch Streamlit UI
streamlit run src/streamlit_app.py
```

---

## ML Models Included

| Model | Type |
|---|---|
| Logistic Regression | Baseline |
| Random Forest | Ensemble |
| Gradient Boosting | Ensemble |
| Support Vector Machine | Kernel |
| K-Nearest Neighbors | Instance |
| Naive Bayes | Probabilistic |
| XGBoost | Boosting |
| LightGBM | Boosting |
| Neural Network (TF/Keras) | Deep Learning |

The best-performing model is automatically selected and saved.

---

## Diseases Covered (15)

Common Cold, Flu, Hypertension, Type 2 Diabetes, Migraine, Allergic Rhinitis,
Asthma, Gastritis, Urinary Tract Infection, Anxiety Disorder, Depression,
Arthritis, Pneumonia, Insomnia, Hypothyroidism

---

## Flask API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET  | `/` | Health check |
| POST | `/api/recommend` | Get recommendations |
| GET  | `/api/symptoms` | List all symptoms |
| GET  | `/api/medicine/<name>` | Medicine details |
| GET  | `/api/diseases` | List all diseases |

### Example API Call
```bash
curl -X POST http://localhost:5000/api/recommend \
  -H "Content-Type: application/json" \
  -d '{
    "symptoms": ["headache", "fever", "body ache"],
    "age": 28,
    "gender": "Female",
    "severity": 2
  }'
```

---

## Streamlit UI Pages

1. **Home & Recommend** – Patient form + real-time recommendations
2. **Analytics Dashboard** – Model comparison charts, dataset stats
3. **Medicine Database** – Searchable medicine reference
4. **About** – Project info and architecture

---

## Disclaimer

> This system is for **educational purposes only**. Always consult a licensed
> healthcare professional before taking any medication.
