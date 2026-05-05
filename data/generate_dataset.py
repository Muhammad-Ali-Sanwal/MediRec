"""
Dataset generator for Medicine Recommendation System.
Generates synthetic but realistic medical data for training.
"""
import pandas as pd
import numpy as np
import json
import random

random.seed(42)
np.random.seed(42)

# ─── Disease & Symptom Definitions ───────────────────────────────────────────
DISEASES = {
    "Common Cold": {
        "symptoms": ["runny nose", "sneezing", "sore throat", "cough", "mild fever",
                     "headache", "body ache", "fatigue"],
        "medicines": ["Paracetamol", "Cetirizine", "Dextromethorphan", "Loratadine",
                      "Pseudoephedrine", "Zinc Supplements"],
        "severity": "mild"
    },
    "Flu (Influenza)": {
        "symptoms": ["high fever", "body ache", "fatigue", "headache", "cough",
                     "sore throat", "chills", "vomiting"],
        "medicines": ["Oseltamivir (Tamiflu)", "Paracetamol", "Ibuprofen",
                      "Zanamivir", "Amantadine"],
        "severity": "moderate"
    },
    "Hypertension": {
        "symptoms": ["headache", "dizziness", "blurred vision", "chest pain",
                     "shortness of breath", "nosebleed"],
        "medicines": ["Amlodipine", "Lisinopril", "Losartan", "Metoprolol",
                      "Hydrochlorothiazide", "Atenolol"],
        "severity": "serious"
    },
    "Type 2 Diabetes": {
        "symptoms": ["frequent urination", "excessive thirst", "fatigue", "blurred vision",
                     "slow healing wounds", "tingling hands feet", "weight loss"],
        "medicines": ["Metformin", "Glibenclamide", "Sitagliptin", "Empagliflozin",
                      "Insulin Glargine", "Pioglitazone"],
        "severity": "serious"
    },
    "Migraine": {
        "symptoms": ["severe headache", "nausea", "vomiting", "light sensitivity",
                     "sound sensitivity", "visual aura", "dizziness"],
        "medicines": ["Sumatriptan", "Rizatriptan", "Topiramate", "Propranolol",
                      "Amitriptyline", "Ibuprofen"],
        "severity": "moderate"
    },
    "Allergic Rhinitis": {
        "symptoms": ["sneezing", "runny nose", "itchy eyes", "nasal congestion",
                     "watery eyes", "itchy throat"],
        "medicines": ["Cetirizine", "Loratadine", "Fexofenadine", "Montelukast",
                      "Fluticasone Nasal Spray", "Budesonide"],
        "severity": "mild"
    },
    "Asthma": {
        "symptoms": ["shortness of breath", "wheezing", "chest tightness", "cough",
                     "difficulty breathing", "breathlessness at night"],
        "medicines": ["Salbutamol (Albuterol)", "Formoterol", "Fluticasone Inhaler",
                      "Budesonide Inhaler", "Montelukast", "Ipratropium"],
        "severity": "serious"
    },
    "Gastritis": {
        "symptoms": ["stomach pain", "nausea", "vomiting", "bloating", "loss of appetite",
                     "indigestion", "burning sensation stomach"],
        "medicines": ["Omeprazole", "Pantoprazole", "Ranitidine", "Antacids",
                      "Metronidazole", "Clarithromycin"],
        "severity": "moderate"
    },
    "Urinary Tract Infection": {
        "symptoms": ["frequent urination", "burning urination", "cloudy urine",
                     "pelvic pain", "strong urine smell", "lower back pain"],
        "medicines": ["Ciprofloxacin", "Nitrofurantoin", "Trimethoprim", "Fosfomycin",
                      "Amoxicillin", "Co-Amoxiclav"],
        "severity": "moderate"
    },
    "Anxiety Disorder": {
        "symptoms": ["excessive worry", "restlessness", "fatigue", "difficulty concentrating",
                     "irritability", "sleep problems", "muscle tension", "palpitations"],
        "medicines": ["Sertraline", "Escitalopram", "Buspirone", "Alprazolam",
                      "Clonazepam", "Venlafaxine"],
        "severity": "moderate"
    },
    "Depression": {
        "symptoms": ["persistent sadness", "loss of interest", "fatigue", "sleep problems",
                     "appetite changes", "difficulty concentrating", "hopelessness"],
        "medicines": ["Sertraline", "Fluoxetine", "Escitalopram", "Venlafaxine",
                      "Bupropion", "Mirtazapine"],
        "severity": "serious"
    },
    "Arthritis": {
        "symptoms": ["joint pain", "joint stiffness", "swelling joints", "reduced mobility",
                     "warmth around joints", "joint tenderness"],
        "medicines": ["Ibuprofen", "Naproxen", "Diclofenac", "Methotrexate",
                      "Hydroxychloroquine", "Prednisone"],
        "severity": "serious"
    },
    "Pneumonia": {
        "symptoms": ["cough with phlegm", "fever", "chills", "shortness of breath",
                     "chest pain", "fatigue", "nausea", "vomiting"],
        "medicines": ["Amoxicillin", "Azithromycin", "Doxycycline", "Levofloxacin",
                      "Co-Amoxiclav", "Cefuroxime"],
        "severity": "serious"
    },
    "Insomnia": {
        "symptoms": ["difficulty falling asleep", "waking at night", "waking early",
                     "daytime fatigue", "irritability", "difficulty concentrating"],
        "medicines": ["Zolpidem", "Melatonin", "Doxylamine", "Temazepam",
                      "Eszopiclone", "Trazodone"],
        "severity": "mild"
    },
    "Hypothyroidism": {
        "symptoms": ["fatigue", "weight gain", "cold sensitivity", "constipation",
                     "dry skin", "hair loss", "muscle weakness", "depression"],
        "medicines": ["Levothyroxine", "Liothyronine", "Desiccated Thyroid",
                      "Selenium Supplements"],
        "severity": "serious"
    }
}

ALL_SYMPTOMS = sorted(set(
    s for d in DISEASES.values() for s in d["symptoms"]
))

ALL_MEDICINES = sorted(set(
    m for d in DISEASES.values() for m in d["medicines"]
))

# ─── Generate Patient Records ─────────────────────────────────────────────────
def generate_patient_data(n_samples=3000):
    records = []
    disease_list = list(DISEASES.keys())

    for i in range(n_samples):
        disease_name = random.choice(disease_list)
        disease_info = DISEASES[disease_name]

        # Primary symptoms (always present)
        core_symptoms = disease_info["symptoms"]
        n_core = max(2, int(len(core_symptoms) * random.uniform(0.5, 0.9)))
        selected_symptoms = random.sample(core_symptoms, n_core)

        # Random noise symptoms
        other_symptoms = [s for s in ALL_SYMPTOMS if s not in core_symptoms]
        n_noise = random.randint(0, 2)
        noise_symptoms = random.sample(other_symptoms, min(n_noise, len(other_symptoms)))
        final_symptoms = list(set(selected_symptoms + noise_symptoms))

        # Recommended medicines (top 3)
        rec_medicines = disease_info["medicines"][:3]
        primary_medicine = rec_medicines[0]

        age = random.randint(5, 85)
        gender = random.choice(["Male", "Female"])
        severity_map = {"mild": 1, "moderate": 2, "serious": 3}
        severity = severity_map[disease_info["severity"]]

        record = {
            "patient_id": f"P{i+1000:05d}",
            "age": age,
            "gender": gender,
            "disease": disease_name,
            "severity": severity,
            "primary_medicine": primary_medicine,
            "recommended_medicines": "|".join(rec_medicines),
            "symptom_count": len(final_symptoms),
        }
        # One-hot encode symptoms
        for sym in ALL_SYMPTOMS:
            record[f"sym_{sym.replace(' ', '_')}"] = 1 if sym in final_symptoms else 0

        records.append(record)

    return pd.DataFrame(records)


# ─── Medicine Info Dataset ────────────────────────────────────────────────────
MEDICINE_INFO = {
    "Paracetamol": {
        "generic_name": "Acetaminophen",
        "drug_class": "Analgesic / Antipyretic",
        "dosage": "500-1000 mg every 4-6 hours (max 4g/day)",
        "side_effects": ["nausea", "liver damage (overdose)", "rash"],
        "contraindications": ["severe liver disease", "alcohol dependence"],
        "category": "OTC"
    },
    "Ibuprofen": {
        "generic_name": "Ibuprofen",
        "drug_class": "NSAID",
        "dosage": "200-400 mg every 4-6 hours (max 1200 mg/day OTC)",
        "side_effects": ["stomach upset", "nausea", "heartburn", "dizziness"],
        "contraindications": ["peptic ulcer", "renal failure", "pregnancy (3rd trimester)"],
        "category": "OTC/Prescription"
    },
    "Amoxicillin": {
        "generic_name": "Amoxicillin",
        "drug_class": "Penicillin Antibiotic",
        "dosage": "250-500 mg three times daily for 5-10 days",
        "side_effects": ["diarrhea", "nausea", "rash", "allergic reaction"],
        "contraindications": ["penicillin allergy"],
        "category": "Prescription"
    },
    "Metformin": {
        "generic_name": "Metformin HCl",
        "drug_class": "Biguanide (Antidiabetic)",
        "dosage": "500-850 mg twice daily with meals",
        "side_effects": ["nausea", "diarrhea", "stomach pain", "lactic acidosis (rare)"],
        "contraindications": ["renal impairment", "hepatic failure", "alcohol abuse"],
        "category": "Prescription"
    },
    "Amlodipine": {
        "generic_name": "Amlodipine Besylate",
        "drug_class": "Calcium Channel Blocker",
        "dosage": "5-10 mg once daily",
        "side_effects": ["ankle swelling", "flushing", "headache", "dizziness"],
        "contraindications": ["cardiogenic shock", "severe aortic stenosis"],
        "category": "Prescription"
    },
    "Cetirizine": {
        "generic_name": "Cetirizine HCl",
        "drug_class": "Antihistamine",
        "dosage": "10 mg once daily",
        "side_effects": ["drowsiness", "dry mouth", "fatigue"],
        "contraindications": ["severe renal impairment"],
        "category": "OTC"
    },
    "Omeprazole": {
        "generic_name": "Omeprazole",
        "drug_class": "Proton Pump Inhibitor",
        "dosage": "20-40 mg once daily before meal",
        "side_effects": ["headache", "diarrhea", "nausea", "abdominal pain"],
        "contraindications": ["hypersensitivity to PPIs"],
        "category": "OTC/Prescription"
    },
    "Salbutamol (Albuterol)": {
        "generic_name": "Salbutamol Sulfate",
        "drug_class": "Beta-2 Agonist (Bronchodilator)",
        "dosage": "100-200 mcg inhaled every 4-6 hours as needed",
        "side_effects": ["tremor", "palpitations", "headache", "nervousness"],
        "contraindications": ["hypersensitivity", "tachyarrhythmia"],
        "category": "Prescription"
    },
    "Sumatriptan": {
        "generic_name": "Sumatriptan Succinate",
        "drug_class": "Triptan (5-HT1 Agonist)",
        "dosage": "50-100 mg at onset of migraine; repeat after 2 hours if needed",
        "side_effects": ["tingling", "flushing", "chest tightness", "dizziness"],
        "contraindications": ["ischemic heart disease", "uncontrolled hypertension"],
        "category": "Prescription"
    },
    "Levothyroxine": {
        "generic_name": "Levothyroxine Sodium",
        "drug_class": "Thyroid Hormone",
        "dosage": "25-200 mcg once daily (individualized)",
        "side_effects": ["palpitations", "weight loss", "insomnia", "tremor"],
        "contraindications": ["untreated adrenal insufficiency", "thyrotoxicosis"],
        "category": "Prescription"
    }
}


if __name__ == "__main__":
    print("Generating patient dataset...")
    df = generate_patient_data(3000)
    df.to_csv("patient_records.csv", index=False)
    print(f"✓ Generated {len(df)} patient records → patient_records.csv")

    # Save metadata
    meta = {
        "diseases": list(DISEASES.keys()),
        "all_symptoms": ALL_SYMPTOMS,
        "all_medicines": ALL_MEDICINES,
        "disease_details": DISEASES,
        "medicine_info": MEDICINE_INFO
    }
    with open("metadata.json", "w") as f:
        json.dump(meta, f, indent=2)
    print("✓ Saved metadata → metadata.json")
