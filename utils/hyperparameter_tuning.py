"""
Hyperparameter Tuning with Optuna
Optimizes the best-performing model (RandomForest / XGBoost) for disease prediction.
Run: python utils/hyperparameter_tuning.py
"""
import os
import sys
import json
import joblib
import warnings
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False


BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def load_data():
    df = pd.read_csv(os.path.join(BASE, "data", "patient_records.csv"))
    sym_cols = [c for c in df.columns if c.startswith("sym_")]
    feature_cols = sym_cols + ["age", "severity"]
    df["gender_enc"] = (df["gender"] == "Male").astype(int)
    feature_cols.append("gender_enc")

    X = df[feature_cols].values
    y = LabelEncoder().fit_transform(df["disease"].values)
    X = StandardScaler().fit_transform(X)
    return X, y


def tune_random_forest(X, y, n_trials=30):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    def objective(trial):
        params = {
            "n_estimators":   trial.suggest_int("n_estimators", 100, 500),
            "max_depth":      trial.suggest_int("max_depth", 3, 20),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
            "min_samples_leaf":  trial.suggest_int("min_samples_leaf", 1, 10),
            "max_features":   trial.suggest_categorical("max_features", ["sqrt", "log2"]),
        }
        model = RandomForestClassifier(**params, random_state=42, n_jobs=-1)
        scores = cross_val_score(model, X, y, cv=cv, scoring="accuracy")
        return scores.mean()

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    return study.best_params, study.best_value


def tune_xgboost(X, y, n_trials=30):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    def objective(trial):
        params = {
            "n_estimators":   trial.suggest_int("n_estimators", 100, 400),
            "max_depth":      trial.suggest_int("max_depth", 3, 10),
            "learning_rate":  trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample":      trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_alpha":      trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
            "reg_lambda":     trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
        }
        model = XGBClassifier(**params, random_state=42, use_label_encoder=False,
                               eval_metric="mlogloss", n_jobs=-1)
        scores = cross_val_score(model, X, y, cv=cv, scoring="accuracy")
        return scores.mean()

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    return study.best_params, study.best_value


def main():
    if not HAS_OPTUNA:
        print("Optuna not installed. Run: pip install optuna")
        return

    print("Loading data…")
    X, y = load_data()
    results = {}

    print("\n🔧 Tuning Random Forest…")
    rf_params, rf_score = tune_random_forest(X, y, n_trials=50)
    results["random_forest"] = {"best_params": rf_params, "best_cv_accuracy": rf_score}
    print(f"  Best RF accuracy: {rf_score:.4f}")
    print(f"  Best params: {rf_params}")

    if HAS_XGB:
        print("\n🔧 Tuning XGBoost…")
        xgb_params, xgb_score = tune_xgboost(X, y, n_trials=50)
        results["xgboost"] = {"best_params": xgb_params, "best_cv_accuracy": xgb_score}
        print(f"  Best XGB accuracy: {xgb_score:.4f}")
        print(f"  Best params: {xgb_params}")

    # Retrain best model on full data
    best_name = max(results, key=lambda k: results[k]["best_cv_accuracy"])
    best_params = results[best_name]["best_params"]
    print(f"\n✅ Best overall: {best_name} — retraining on full dataset…")

    if best_name == "random_forest":
        best_model = RandomForestClassifier(**best_params, random_state=42, n_jobs=-1)
    else:
        best_model = XGBClassifier(**best_params, random_state=42,
                                    use_label_encoder=False, eval_metric="mlogloss")
    best_model.fit(X, y)
    joblib.dump(best_model, os.path.join(BASE, "models", "disease_model_tuned.pkl"))

    out_path = os.path.join(BASE, "models", "tuning_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"✓ Tuning results saved → {out_path}")
    print("✓ Tuned model saved  → models/disease_model_tuned.pkl")


if __name__ == "__main__":
    main()
