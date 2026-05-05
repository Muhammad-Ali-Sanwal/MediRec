"""
Evaluation utilities – confusion matrix, classification report, SHAP explainability.
Run standalone: python utils/evaluate.py
"""
import os, sys, json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import joblib
import pandas as pd
from sklearn.metrics import (classification_report, confusion_matrix,
                             ConfusionMatrixDisplay, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler


BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def load_artifacts():
    def _l(name):
        return joblib.load(os.path.join(BASE, "models", name))
    return {
        "disease_model":  _l("disease_model.pkl"),
        "le_disease":     _l("label_encoder_disease.pkl"),
        "scaler":         _l("scaler.pkl"),
        "feature_cols":   _l("feature_columns.pkl"),
    }


def load_data(a):
    df = pd.read_csv(os.path.join(BASE, "data", "patient_records.csv"))
    df["gender_enc"] = (df["gender"] == "Male").astype(int)
    X = df[a["feature_cols"]].values
    y = a["le_disease"].transform(df["disease"].values)
    X = a["scaler"].transform(X)
    return X, y, a["le_disease"].classes_


def plot_confusion_matrix(model, X_test, y_test, labels, save_path=None):
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)

    fig, ax = plt.subplots(figsize=(14, 12))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(ax=ax, xticks_rotation=45, colorbar=True, cmap="Blues")
    ax.set_title("Confusion Matrix – Disease Prediction", fontsize=14)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved → {save_path}")
    plt.show()


def print_classification_report(model, X_test, y_test, labels):
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred, target_names=labels))


def shap_feature_importance(model, X_test, feature_cols, save_path=None):
    try:
        import shap
        explainer = shap.TreeExplainer(model)
        shap_vals = explainer.shap_values(X_test[:200])
        if isinstance(shap_vals, list):
            shap_vals = np.abs(np.array(shap_vals)).mean(axis=0)
        mean_shap = np.abs(shap_vals).mean(axis=0)
        top_idx = np.argsort(mean_shap)[::-1][:20]

        fig, ax = plt.subplots(figsize=(10, 7))
        ax.barh([feature_cols[i] for i in top_idx[::-1]],
                mean_shap[top_idx[::-1]], color="steelblue")
        ax.set_title("Top 20 Features by SHAP Importance")
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150)
        plt.show()
    except ImportError:
        print("SHAP not installed. Run: pip install shap")
    except Exception as e:
        print(f"SHAP failed: {e}")


def main():
    print("Loading artifacts…")
    a = load_artifacts()
    X, y, labels = load_data(a)
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("\nClassification Report:")
    print_classification_report(a["disease_model"], X_test, y_test, labels)

    plot_confusion_matrix(
        a["disease_model"], X_test, y_test, labels,
        save_path=os.path.join(BASE, "models", "confusion_matrix.png")
    )

    shap_feature_importance(
        a["disease_model"], X_test, a["feature_cols"],
        save_path=os.path.join(BASE, "models", "shap_importance.png")
    )


if __name__ == "__main__":
    main()
