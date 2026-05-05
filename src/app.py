"""
Flask REST API for the Medicine Recommendation System
Endpoints:
  GET  /                  → health check
  POST /api/recommend     → main recommendation
  GET  /api/symptoms      → list all symptoms
  GET  /api/medicine/<n>  → medicine details
  GET  /api/diseases      → list all diseases
"""
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from flask import Flask, request, jsonify
from flask_cors import CORS
from src.recommender import MedicineRecommender

app = Flask(__name__)
CORS(app)

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
recommender = MedicineRecommender(
    model_dir=os.path.join(BASE_DIR, "models"),
    metadata_path=os.path.join(BASE_DIR, "data", "metadata.json")
)


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/", methods=["GET"])
def health():
    return jsonify({
        "status": "running",
        "service": "Medicine Recommendation API",
        "version": "1.0.0",
        "models_loaded": recommender.models_loaded
    })


@app.route("/api/recommend", methods=["POST"])
def recommend():
    """
    Body (JSON):
    {
        "symptoms": ["headache", "fever", "cough"],
        "age": 28,
        "gender": "Female",
        "severity": 2,
        "medical_history": "hypertension"
    }
    """
    data = request.get_json(silent=True) or {}

    symptoms = data.get("symptoms", [])
    if not symptoms or not isinstance(symptoms, list):
        return jsonify({"error": "Provide a list of symptoms."}), 400

    age      = int(data.get("age", 30))
    gender   = data.get("gender", "Male")
    severity = int(data.get("severity", 2))
    history  = data.get("medical_history", "")

    result = recommender.recommend(symptoms, age, gender, severity, history)
    return jsonify(result)


@app.route("/api/symptoms", methods=["GET"])
def symptoms():
    return jsonify({"symptoms": recommender.get_all_symptoms()})


@app.route("/api/medicine/<name>", methods=["GET"])
def medicine_info(name):
    info = recommender.get_medicine_info(name)
    if not info:
        return jsonify({"error": f"No info found for '{name}'"}), 404
    return jsonify({"medicine": name, "details": info})


@app.route("/api/diseases", methods=["GET"])
def diseases():
    result = []
    for name, info in recommender.disease_details.items():
        result.append({
            "disease": name,
            "severity": info.get("severity", "unknown"),
            "symptom_count": len(info.get("symptoms", []))
        })
    return jsonify({"diseases": result})


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("🚀 Starting Medicine Recommendation API on http://localhost:5000")
    app.run(debug=True, host="0.0.0.0", port=5000)
