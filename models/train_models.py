"""
Model Training & Evaluation Pipeline
Trains multiple ML/DL models and saves the best one.
"""
import os
import json
import joblib
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (classification_report, confusion_matrix,
                             accuracy_score, f1_score)
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    from lightgbm import LGBMClassifier
    HAS_LGB = True
except ImportError:
    HAS_LGB = False


# ─── Load Data ────────────────────────────────────────────────────────────────
def load_data(csv_path: str):
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} records with {df.shape[1]} columns")
    return df


# ─── Preprocessing ────────────────────────────────────────────────────────────
def preprocess(df: pd.DataFrame):
    sym_cols = [c for c in df.columns if c.startswith("sym_")]
    feature_cols = sym_cols + ["age", "severity"]

    # Encode gender
    df["gender_enc"] = (df["gender"] == "Male").astype(int)
    feature_cols.append("gender_enc")

    X = df[feature_cols].values
    y_disease = df["disease"].values
    y_medicine = df["primary_medicine"].values

    le_disease = LabelEncoder().fit(y_disease)
    le_medicine = LabelEncoder().fit(y_medicine)

    y_disease_enc = le_disease.transform(y_disease)
    y_medicine_enc = le_medicine.transform(y_medicine)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return (X_scaled, y_disease_enc, y_medicine_enc,
            le_disease, le_medicine, scaler, feature_cols)


# ─── Model Zoo ────────────────────────────────────────────────────────────────
def get_models():
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=150, random_state=42),
        "SVM": SVC(kernel="rbf", probability=True, random_state=42),
        "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=7),
        "Naive Bayes": GaussianNB(),
    }
    if HAS_XGB:
        models["XGBoost"] = XGBClassifier(n_estimators=200, random_state=42,
                                           use_label_encoder=False,
                                           eval_metric="mlogloss")
    if HAS_LGB:
        models["LightGBM"] = LGBMClassifier(n_estimators=200, random_state=42,
                                             verbose=-1)
    return models


# ─── Training Loop ────────────────────────────────────────────────────────────
def train_and_evaluate(X, y, label: str):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    models = get_models()
    results = {}
    best_score = -1
    best_model = None
    best_name = ""

    print(f"\n{'='*60}")
    print(f"  Training for: {label}")
    print(f"{'='*60}")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for name, model in models.items():
        try:
            cv_scores = cross_val_score(model, X_train, y_train,
                                        cv=cv, scoring="accuracy", n_jobs=-1)
            model.fit(X_train, y_train)
            test_acc = accuracy_score(y_test, model.predict(X_test))
            f1 = f1_score(y_test, model.predict(X_test), average="weighted")

            results[name] = {
                "cv_mean": round(float(cv_scores.mean()), 4),
                "cv_std": round(float(cv_scores.std()), 4),
                "test_accuracy": round(test_acc, 4),
                "f1_score": round(f1, 4)
            }
            print(f"  {name:<25} CV={cv_scores.mean():.3f}±{cv_scores.std():.3f}  "
                  f"Test={test_acc:.3f}  F1={f1:.3f}")

            if test_acc > best_score:
                best_score = test_acc
                best_model = model
                best_name = name
        except Exception as e:
            print(f"  {name:<25} FAILED: {e}")

    print(f"\n  ✓ Best model: {best_name} (accuracy={best_score:.3f})")
    return best_model, best_name, results, X_test, y_test


# ─── Deep Learning Model (TF/Keras) ──────────────────────────────────────────
def build_deep_model(input_dim: int, num_classes: int):
    try:
        import tensorflow as tf
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
        from tensorflow.keras.callbacks import EarlyStopping
        from tensorflow.keras.utils import to_categorical

        model = Sequential([
            Dense(256, activation="relu", input_shape=(input_dim,)),
            BatchNormalization(),
            Dropout(0.3),
            Dense(128, activation="relu"),
            BatchNormalization(),
            Dropout(0.2),
            Dense(64, activation="relu"),
            Dropout(0.1),
            Dense(num_classes, activation="softmax")
        ])
        model.compile(optimizer="adam",
                      loss="sparse_categorical_crossentropy",
                      metrics=["accuracy"])
        return model, True
    except ImportError:
        return None, False


# ─── Main ─────────────────────────────────────────────────────────────────────
def main(data_path: str = "data/patient_records.csv",
         output_dir: str = "models"):
    os.makedirs(output_dir, exist_ok=True)

    df = load_data(data_path)
    (X, y_disease, y_medicine,
     le_disease, le_medicine,
     scaler, feature_cols) = preprocess(df)

    # ── Disease Prediction ──
    disease_model, disease_best, disease_results, Xd_test, yd_test = \
        train_and_evaluate(X, y_disease, "Disease Prediction")

    # ── Medicine Prediction ──
    med_model, med_best, med_results, Xm_test, ym_test = \
        train_and_evaluate(X, y_medicine, "Medicine Prediction")

    # ── Deep Learning (optional) ──
    dl_acc = None
    nn_model, nn_ok = build_deep_model(X.shape[1], len(np.unique(y_disease)))
    if nn_ok:
        try:
            from tensorflow.keras.callbacks import EarlyStopping
            X_tr, X_te, y_tr, y_te = train_test_split(
                X, y_disease, test_size=0.2, random_state=42, stratify=y_disease
            )
            es = EarlyStopping(patience=5, restore_best_weights=True)
            nn_model.fit(X_tr, y_tr, epochs=50, batch_size=64,
                         validation_split=0.1, callbacks=[es], verbose=0)
            dl_acc = float(nn_model.evaluate(X_te, y_te, verbose=0)[1])
            nn_model.save(os.path.join(output_dir, "deep_disease_model.h5"))
            print(f"\n  ✓ Deep Learning model accuracy: {dl_acc:.3f}")
        except Exception as e:
            print(f"\n  Deep Learning skipped: {e}")

    # ── Save Artifacts ──
    joblib.dump(disease_model, os.path.join(output_dir, "disease_model.pkl"))
    joblib.dump(med_model,     os.path.join(output_dir, "medicine_model.pkl"))
    joblib.dump(le_disease,    os.path.join(output_dir, "label_encoder_disease.pkl"))
    joblib.dump(le_medicine,   os.path.join(output_dir, "label_encoder_medicine.pkl"))
    joblib.dump(scaler,        os.path.join(output_dir, "scaler.pkl"))
    joblib.dump(feature_cols,  os.path.join(output_dir, "feature_columns.pkl"))

    # ── Save Results ──
    summary = {
        "disease_best_model": disease_best,
        "medicine_best_model": med_best,
        "disease_results": disease_results,
        "medicine_results": med_results,
        "deep_learning_accuracy": dl_acc,
        "feature_count": len(feature_cols)
    }
    with open(os.path.join(output_dir, "training_results.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("\n✅ All models saved to ./models/")
    print("   Run: python src/app.py  to start the Flask API")
    print("   Run: streamlit run src/streamlit_app.py  for the Web UI")
    return summary


if __name__ == "__main__":
    main()
