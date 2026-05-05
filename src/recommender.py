"""
Medicine Recommendation Engine
Core prediction & ranking logic used by both Flask API and Streamlit UI.
"""
import os
import json
import numpy as np
import joblib
import pandas as pd
from typing import List, Dict, Any, Optional


class MedicineRecommender:
    """
    Loads trained models and produces personalized medicine recommendations.
    """

    def __init__(self, model_dir: str = "models",
                 metadata_path: str = "data/metadata.json"):
        self.model_dir = model_dir
        self._load_metadata(metadata_path)
        self._load_models()

    # ── Loading ───────────────────────────────────────────────────────────────
    def _load_metadata(self, path: str):
        with open(path) as f:
            meta = json.load(f)
        self.all_symptoms    = meta["all_symptoms"]
        self.all_medicines   = meta["all_medicines"]
        self.disease_details = meta["disease_details"]
        self.medicine_info   = meta.get("medicine_info", {})

    def _load_models(self):
        def _load(name):
            p = os.path.join(self.model_dir, name)
            return joblib.load(p) if os.path.exists(p) else None

        self.disease_model  = _load("disease_model.pkl")
        self.medicine_model = _load("medicine_model.pkl")
        self.le_disease     = _load("label_encoder_disease.pkl")
        self.le_medicine    = _load("label_encoder_medicine.pkl")
        self.scaler         = _load("scaler.pkl")
        self.feature_cols   = _load("feature_columns.pkl")

        self.models_loaded = all([
            self.disease_model, self.medicine_model,
            self.le_disease, self.le_medicine,
            self.scaler, self.feature_cols
        ])

    # ── Feature Engineering ───────────────────────────────────────────────────
    def _build_features(self, symptoms: List[str], age: int,
                        gender: str, severity: int) -> np.ndarray:
        row = {}
        for sym in self.all_symptoms:
            row[f"sym_{sym.replace(' ', '_')}"] = 1 if sym in symptoms else 0
        row["age"] = age
        row["severity"] = severity
        row["gender_enc"] = 1 if gender.lower() == "male" else 0

        # Align to training feature order
        values = [row.get(c, 0) for c in self.feature_cols]
        X = np.array(values, dtype=float).reshape(1, -1)
        return self.scaler.transform(X)

    # ── Prediction ────────────────────────────────────────────────────────────
    def predict_disease(self, X: np.ndarray) -> Dict[str, Any]:
        probs = self.disease_model.predict_proba(X)[0]
        top_idx = np.argsort(probs)[::-1][:3]
        return [
            {
                "disease": self.le_disease.inverse_transform([i])[0],
                "confidence": round(float(probs[i]) * 100, 2)
            }
            for i in top_idx if probs[i] > 0.01
        ]

    def predict_medicine(self, X: np.ndarray) -> Dict[str, Any]:
        probs = self.medicine_model.predict_proba(X)[0]
        top_idx = np.argsort(probs)[::-1][:5]
        return [
            {
                "medicine": self.le_medicine.inverse_transform([i])[0],
                "confidence": round(float(probs[i]) * 100, 2)
            }
            for i in top_idx if probs[i] > 0.01
        ]

    # ── Rule-Based Augmentation ───────────────────────────────────────────────
    def _rule_based_medicines(self, disease: str) -> List[str]:
        details = self.disease_details.get(disease, {})
        return details.get("medicines", [])

    def _rank_medicines(self, ml_meds: List[Dict],
                        rule_meds: List[str],
                        disease: str) -> List[Dict]:
        """Merge ML predictions with rule-based recommendations."""
        ranked = []
        seen = set()

        # ML predictions first
        for item in ml_meds:
            name = item["medicine"]
            if name not in seen:
                item["source"] = "ML Model"
                item["info"] = self.medicine_info.get(name, {})
                ranked.append(item)
                seen.add(name)

        # Rule-based fill
        for med in rule_meds:
            if med not in seen:
                ranked.append({
                    "medicine": med,
                    "confidence": 0.0,
                    "source": "Clinical Rules",
                    "info": self.medicine_info.get(med, {})
                })
                seen.add(med)

        # Boost if medicine appears in both
        ml_names = {m["medicine"] for m in ml_meds}
        rule_set = set(rule_meds)
        for item in ranked:
            if item["medicine"] in ml_names and item["medicine"] in rule_set:
                item["confidence"] = min(item["confidence"] * 1.2, 100.0)
                item["source"] = "ML + Clinical"

        return ranked[:6]

    # ── Public API ────────────────────────────────────────────────────────────
    def recommend(self, symptoms: List[str], age: int = 30,
                  gender: str = "Male", severity: int = 2,
                  medical_history: Optional[str] = "") -> Dict[str, Any]:
        """
        Main recommendation function.
        Returns diseases, medicines, and metadata.
        """
        if not self.models_loaded:
            return self._rule_only(symptoms)

        X = self._build_features(symptoms, age, gender, severity)
        diseases = self.predict_disease(X)
        ml_meds  = self.predict_medicine(X)

        primary_disease = diseases[0]["disease"] if diseases else ""
        rule_meds = self._rule_based_medicines(primary_disease)
        medicines = self._rank_medicines(ml_meds, rule_meds, primary_disease)

        severity_labels = {1: "Mild", 2: "Moderate", 3: "Serious"}
        disease_severity = self.disease_details.get(
            primary_disease, {}).get("severity", "unknown")

        return {
            "status": "success",
            "predicted_diseases": diseases,
            "recommended_medicines": medicines,
            "severity_level": severity_labels.get(severity, "Moderate"),
            "disease_severity": disease_severity,
            "symptom_count": len(symptoms),
            "model_used": "ML + Rules",
            "disclaimer": (
                "This is an AI-generated recommendation for educational purposes only. "
                "Always consult a licensed healthcare professional before taking any medication."
            )
        }

    def _rule_only(self, symptoms: List[str]) -> Dict[str, Any]:
        """Fallback when models are not trained yet."""
        matched = {}
        for disease, info in self.disease_details.items():
            overlap = len(set(symptoms) & set(info["symptoms"]))
            if overlap:
                matched[disease] = overlap

        if not matched:
            return {"status": "no_match", "message": "No matching diseases found."}

        top_disease = max(matched, key=matched.get)
        meds = self.disease_details[top_disease]["medicines"][:5]

        return {
            "status": "success",
            "predicted_diseases": [{"disease": top_disease, "confidence": 70.0}],
            "recommended_medicines": [
                {"medicine": m, "confidence": 0.0,
                 "source": "Clinical Rules",
                 "info": self.medicine_info.get(m, {})}
                for m in meds
            ],
            "model_used": "Rules Only",
            "disclaimer": "AI recommendation — consult a doctor."
        }

    def get_all_symptoms(self) -> List[str]:
        return self.all_symptoms

    def get_medicine_info(self, name: str) -> Dict:
        return self.medicine_info.get(name, {})
