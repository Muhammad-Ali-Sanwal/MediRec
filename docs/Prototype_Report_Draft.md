# Prototype Phase Report — Medicine Recommendation System

## Student
- Name: M. Ali Sanwal
- VUID: BC240440384
- Email: bc240440384mas@vu.edu.pk

## Supervisor
- Name: Dr. Mushtaq Hussain
- Email: mushtaq.hussain@vu.edu.pk

## Project Overview
The Medicine Recommendation System is an AI-powered assistant that recommends medicines based on patient symptoms, medical history, and existing conditions. The prototype implements data collection and preprocessing, model training and evaluation, a medicine recommendation engine combining ML and clinical rules, and a Streamlit web UI for demonstration.

## Functional Requirements — Mapping to Implementation

1. Data Collection & Preprocessing
   - Implemented in: `data/generate_dataset.py`, `utils/preprocessing.py`
   - Details: synthetic dataset generation combining disease definitions, symptoms, and prescribed medicines. Preprocessing includes normalization, encoding (`gender_enc`), and feature vector assembly based on `feature_columns.pkl`.

2. Model Training & Testing
   - Implemented in: `models/train_models.py`
   - Details: trains disease and medicine predictors (classical ML and optional DL). Training outputs saved to `models/` as `.pkl` artifacts and `models/training_results.json` with metrics.

3. Model Fine-Tuning & Optimization
   - Implemented in: `utils/hyperparameter_tuning.py` (uses Optuna where applicable).

4. Recommendation Engine
   - Implemented in: `src/recommender.py`
   - Details: combines predicted medicines from ML models and rule-based medicines from `data/metadata.json`. Ranks and boosts items appearing in both sources. Exposes `recommend()` API used by Streamlit frontend and REST API.

5. Web-Based UI
   - Implemented in: `src/streamlit_app.py`
   - Details: Streamlit app with pages for Home & Recommend, Analytics Dashboard, Medicine Database, and About. Includes pagination, search, and result explanation.

## Methodology
- Data: synthetic/generated patient records in `data/patient_records.csv` and medicine metadata in `data/metadata.json`.
- Features: binary symptom indicators, `age`, `gender_enc`, `severity` and engineered features saved in `feature_columns.pkl`.
- Models: baseline classifiers (Random Forest / XGBoost / LightGBM) with hyperparameter tuning. Models saved as `disease_model.pkl` and `medicine_model.pkl`.
- Evaluation: classification report, confusion matrix, ROC/PR where applicable, and SHAP feature importance. Evaluation utilities in `utils/evaluate.py`.

## Results (to be populated after running setup and evaluation)
- Overall accuracy, precision, recall for disease predictor: **(populate)**
- Confusion matrix: `models/confusion_matrix.png`
- SHAP feature importance: `models/shap_importance.png`
- Sample recommendations and confidence scores: include 3–5 representative cases.

## Limitations
- Synthetic dataset may not capture full clinical complexity. Real-world deployment requires clinical validation and regulatory compliance.
- Model predictions are only as good as the data and rules; false positives/negatives can occur.

## Future Work
- Integrate real patient EHR data with privacy safeguards.
- Add dosage personalization, drug–drug interaction checks, and allergy constraints.
- Deploy as a HIPAA-compliant service with clinician-in-the-loop controls.

## Reproducibility & How to Run Locally
1. Linux/macOS
```bash
bash setup.sh
```
2. Windows PowerShell
```powershell
powershell -ExecutionPolicy Bypass -File .\setup_windows.ps1
```

After setup, start app (setup scripts will start services automatically). Evaluation scripts: `python utils/evaluate.py`.

## Appendix (to be filled)
- Data schema description
- Model architectures and hyperparameters
- Training logs and selected hyperparameter trials

---
*Draft prepared automatically. Replace placeholders with exact metrics and screenshots after running the evaluation scripts.*
