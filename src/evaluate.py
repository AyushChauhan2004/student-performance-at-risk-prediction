"""
evaluate.py
===========
Final model evaluation on the held-out test set.

This script:
  1. Loads the saved model artifact.
  2. Loads the test split.
  3. Computes and prints all required metrics.
  4. Saves visualisations to reports/figures/.
  5. Appends test metrics to model_metadata.json.

IMPORTANT: The test set is used ONLY here, after model selection is final.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")          # non-interactive backend for scripts
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    ConfusionMatrixDisplay,
)
from sklearn.inspection import permutation_importance

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from preprocessing import (
    load_raw_data, validate_data, chronological_split,
    get_X_y, FEATURE_COLUMNS
)

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "student_risk_model.pkl"
METADATA_PATH = BASE_DIR / "models" / "model_metadata.json"
FIGURES_DIR = BASE_DIR / "reports" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}. Run train.py first.")
    model = joblib.load(MODEL_PATH)
    print(f"[evaluate] Model loaded from {MODEL_PATH}")
    return model


def get_test_data():
    df = load_raw_data()
    df = validate_data(df)
    _, _, test_df = chronological_split(df)
    X_test, y_test = get_X_y(test_df)
    return X_test, y_test


def compute_metrics(pipeline, X_test, y_test) -> dict:
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
        "recall": round(recall_score(y_test, y_pred, zero_division=0), 4),
        "f1_score": round(f1_score(y_test, y_pred, zero_division=0), 4),
        "roc_auc": round(roc_auc_score(y_test, y_proba), 4),
    }

    print("\n" + "=" * 50)
    print("TEST SET EVALUATION RESULTS")
    print("=" * 50)
    for k, v in metrics.items():
        print(f"  {k:12s}: {v}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["Not At Risk", "At Risk"], zero_division=0))

    # ── Confusion matrix interpretation ───────────────────────────────────────
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    print("Confusion Matrix Interpretation:")
    print(f"  True Negatives  (TN): {tn}  — correctly identified as Not At Risk")
    print(f"  False Positives (FP): {fp}  — flagged At Risk but actually Not At Risk")
    print(f"  False Negatives (FN): {fn}  — missed At Risk students (highest cost)")
    print(f"  True Positives  (TP): {tp}  — correctly identified as At Risk")
    print(f"\n  FP consequence: Teacher checks in with a student who didn't need it — low harm.")
    print(f"  FN consequence: At-risk student is missed and receives no support — HIGH harm.")
    print("  → This is why we prioritise RECALL over raw accuracy.")

    metrics["confusion_matrix"] = cm.tolist()
    return metrics, y_pred, y_proba


def plot_confusion_matrix(pipeline, X_test, y_test) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay.from_estimator(
        pipeline, X_test, y_test,
        display_labels=["Not At Risk", "At Risk"],
        colorbar=False, ax=ax, cmap="Blues"
    )
    ax.set_title("Confusion Matrix — Test Set", fontsize=13, fontweight="bold")
    plt.tight_layout()
    out = FIGURES_DIR / "confusion_matrix.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"[evaluate] Saved → {out}")


def plot_permutation_importance(pipeline, X_test, y_test) -> None:
    result = permutation_importance(
        pipeline, X_test, y_test,
        n_repeats=20,
        random_state=42,
        scoring="f1",
    )
    perm_imp = pd.DataFrame({
        "feature": FEATURE_COLUMNS,
        "importance_mean": result.importances_mean,
        "importance_std": result.importances_std,
    }).sort_values("importance_mean", ascending=True)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(
        perm_imp["feature"],
        perm_imp["importance_mean"],
        xerr=perm_imp["importance_std"],
        color="steelblue", alpha=0.8, capsize=4
    )
    ax.set_xlabel("Mean Decrease in F1 Score (permutation importance)")
    ax.set_title("Permutation Feature Importance — Test Set", fontweight="bold")
    ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
    plt.tight_layout()
    out = FIGURES_DIR / "permutation_importance.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"[evaluate] Saved → {out}")

    # Also save to JSON for Streamlit
    perm_dict = perm_imp.set_index("feature")["importance_mean"].to_dict()
    perm_path = BASE_DIR / "models" / "permutation_importance.json"
    with open(perm_path, "w") as f:
        json.dump(perm_dict, f, indent=2)
    print(f"[evaluate] Permutation importance saved → {perm_path}")


def plot_model_comparison(metadata: dict) -> None:
    val_results = metadata.get("model_results_on_validation", {})
    if not val_results:
        print("[evaluate] No validation results found in metadata — skipping comparison plot.")
        return

    names = list(val_results.keys())
    f1s = [val_results[n]["val_f1"] for n in names]
    recalls = [val_results[n]["val_recall"] for n in names]
    precisions = [val_results[n]["val_precision"] for n in names]

    x = np.arange(len(names))
    width = 0.25
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - width, f1s, width, label="F1-Score", color="steelblue")
    ax.bar(x,         recalls, width, label="Recall",   color="coral")
    ax.bar(x + width, precisions, width, label="Precision", color="seagreen")
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=11)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Score")
    ax.set_title("Model Comparison — Validation Set", fontweight="bold")
    ax.legend()
    ax.yaxis.grid(True, alpha=0.3)
    plt.tight_layout()
    out = FIGURES_DIR / "model_comparison.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"[evaluate] Saved → {out}")


def main():
    print("=" * 60)
    print("STUDENT PERFORMANCE & AT-RISK PREDICTION — EVALUATION")
    print("=" * 60)

    pipeline = load_model()
    X_test, y_test = get_test_data()

    metrics, y_pred, y_proba = compute_metrics(pipeline, X_test, y_test)
    plot_confusion_matrix(pipeline, X_test, y_test)
    plot_permutation_importance(pipeline, X_test, y_test)

    # Load and update metadata
    with open(METADATA_PATH, "r") as f:
        metadata = json.load(f)

    metadata["test_metrics"] = metrics
    with open(METADATA_PATH, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"[evaluate] Test metrics saved → {METADATA_PATH}")

    plot_model_comparison(metadata)
    print("\n[evaluate] Evaluation complete.")


if __name__ == "__main__":
    main()

