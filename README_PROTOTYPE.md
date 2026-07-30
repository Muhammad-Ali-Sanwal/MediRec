# Medicine Recommendation System — Prototype Phase
## CS619 | Spring 2026 | Virtual University of Pakistan

**Student Name:** Muhammad Ali Sanwal
**Student ID:** BC240440384
**Student Email:** bc240440384mas@vu.edu.pk
**Group ID:** themushtaq48
**Supervisor:** Dr. Mushtaq Hussain | mushtaq.hussain@vu.edu.pk
**Domain:** Machine Learning, Deep Learning, Web Application

---

## What This Prototype Demonstrates

This prototype covers all functional requirements defined in the Prototype Phase assignment. Below is a direct mapping between each requirement and where it is implemented.

---

## Requirement 1 — Data Collection and Pre-processing Module

**Implemented in:** `data/generate_dataset.py`

The script generates a synthetic but realistic dataset of **3000 patient records** covering **15 diseases** and **50+ medicines**. It performs the following preprocessing steps:

- Handles symptom encoding using **one-hot encoding** (68 symptom columns)
- Applies **label encoding** to disease and medicine target variables
- Applies **StandardScaler normalization** to numerical features (age, severity)
- Splits dataset into **80% training / 20% testing** for model evaluation
- Saves prepared dataset as `patient_records.csv`
- Saves disease, symptom, and medicine metadata as `metadata.json`

**To run:**
```bash
cd data
python generate_dataset.py
```

**Output files:**
- `data/patient_records.csv` — 3000 rows, 71 columns
- `data/metadata.json` — disease details, symptom list, medicine info

---

## Requirement 2 — Web-Based User Interface

**Implemented in:** `src/streamlit_app.py`

The Streamlit web application has four pages:

| Page | What it Demonstrates |
|---|---|
| Page 1 — Home & Recommend | Symptom input form, predicted diseases with confidence scores, ranked medicine recommendations with expandable drug info cards |
| Page 2 — Analytics Dashboard | Dataset preview, model accuracy comparison charts, severity distribution pie chart, records per disease histogram |
| Page 3 — Medicine Database | Searchable medicine reference with dosage, side effects, contraindications, and OTC/Rx category |
| Page 4 — About | Project overview, architecture, technology stack |

**To run:**
```bash
python -m streamlit run src/streamlit_app.py
```
Opens at: **http://localhost:8501**

---

## How to Set Up and Run the Full Prototype

### Step 1 — Install dependencies
```bash
pip install -r requirements.txt
```

### Step 2 — Generate dataset
```bash
cd data
python generate_dataset.py
cd ..
```

### Step 3 — Train ML models
```bash
python models/train_models.py
```

### Step 4 — Launch the web application
```bash
python -m streamlit run src/streamlit_app.py
```

---

## ML Models Trained in Prototype

| Model | Test Accuracy |
|---|---|
| Gradient Boosting | 1.000 |
| Random Forest | 0.998 |
| XGBoost | 0.997 |
| Logistic Regression | 0.997 |
| LightGBM | 0.996 |
| SVM | 0.980 |
| KNN | 0.950 |
| Naive Bayes | 0.890 |

Best model is automatically selected and saved as `models/disease_model.pkl`.

---

## Diseases Covered (15)

Common Cold, Flu (Influenza), Hypertension, Type 2 Diabetes, Migraine,
Allergic Rhinitis, Asthma, Gastritis, Urinary Tract Infection,
Anxiety Disorder, Depression, Arthritis, Pneumonia, Insomnia, Hypothyroidism

---

## Project Structure

```
medicine_recommendation_system/
├── data/
│   ├── generate_dataset.py         ← Data collection & synthetic dataset generation
│   ├── patient_records.csv         ← Generated patient training dataset
│   └── metadata.json               ← Disease, symptom, and medicine metadata
├── models/
│   ├── train_models.py             ← ML/DL model training pipeline
│   ├── disease_model.pkl           ← Best trained disease prediction model
│   ├── medicine_model.pkl          ← Best trained medicine prediction model
│   ├── label_encoder_disease.pkl   ← Label encoder for disease classes
│   ├── label_encoder_medicine.pkl  ← Label encoder for medicine classes
│   ├── feature_columns.pkl         ← Feature column mapping
│   ├── scaler.pkl                  ← Feature scaling transformer
│   └── training_results.json      ← Accuracy & F1-score comparison of all models
├── src/
│   ├── __init__.py
│   ├── recommender.py              ← Core recommendation engine & safety logic
│   ├── app.py                      ← Flask REST API
│   └── streamlit_app.py            ← Interactive Web-based UI
├── utils/
│   ├── hyperparameter_tuning.py    ← Optuna-based model optimization
│   ├── evaluate.py                 ← Confusion matrix & SHAP analysis utilities
│   └── preprocessing.py           ← Text normalization & symptom mapping
├── notebooks/
│   └── EDA_and_Model_Analysis.ipynb
├── requirements.txt                ← Dependency list
├── setup.sh                        ← One-click setup script
├── MediRec_Viva_and_Demo_Guide.md  ← Presentation & Viva defense guide
├── MediRec_Project_Guide.md        ← Project documentation guide
├── README.md                       ← General README
└── README_PROTOTYPE.md             ← Requirements & prototype documentation
```

---

## Tools Used

| Tool | Purpose |
|---|---|
| Python | Core programming language |
| Google Colab / VS Code | Development environment |
| Streamlit | Web-based user interface |
| Pandas, NumPy | Data collection and preprocessing |
| Scikit-learn | ML model training and evaluation |
| XGBoost, LightGBM | Boosting models |
| TensorFlow / Keras | Deep learning neural network |
| Matplotlib, Seaborn, Plotly | Visualization |
| Optuna | Hyperparameter tuning |
| Jupyter Notebook | EDA and analysis notebook |

---

## Disclaimer

This Medicine Recommendation System is developed purely for educational and
research purposes as part of CS619 Final Year Project at Virtual University
of Pakistan. The AI-generated recommendations must not replace the advice of
a licensed healthcare professional. Always consult your doctor before taking
any medication.

---

*CS619 — Spring 2026 | Virtual University of Pakistan*
*Supervisor: Dr. Mushtaq Hussain | mushtaq.hussain@vu.edu.pk*
*Student: Muhammad Ali Sanwal | BC240440384 | bc240440384mas@vu.edu.pk*