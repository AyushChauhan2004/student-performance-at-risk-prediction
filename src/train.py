"""
train.py
========
Model training script for the Student Performance & At-Risk Prediction project.

Models trained
--------------
1. Logistic Regression  — interpretable baseline
2. Random Forest        — ensemble, non-linear
3. XGBoost              — gradient-boosted trees

Each model is wrapped in a full sklearn Pipeline that includes the
StandardScaler, ensuring the scaler is fitted only on training data and
consistently applied at inference time.

The best model (by F1-score on the validation set) is saved as:
  models/student_risk_model.pkl
  models/model_metadata.json

Usage
-----
  python src/train.py
"""

from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, classification_report
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

# Local imports
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from preprocessing import (
    load_raw_data, validate_data, chronological_split,
    get_X_y, FEATURE_COLUMNS, TARGET_COLUMN, build_preprocessor
)

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"
MODEL_PATH = MODELS_DIR / "student_risk_model.pkl"
METADATA_PATH = MODELS_DIR / "model_metadata.json"

RANDOM_SEED = 42


def build_model_pipelines() -> dict[str, Pipeline]:
    """
    Build a dict of name → sklearn Pipeline, each containing a
    StandardScaler and a classifier.

    Why a full Pipeline?
    --------------------
    Wrapping the scaler inside the pipeline means the final model artifact
    includes the fitted scaler.  At inference time, raw features are fed
    directly into pipeline.predict() — no manual scaling step needed and
    no risk of applying a differently-fitted scaler.
    """
    models = {
        "LogisticRegression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                class_weight="balanced",   # handles mild class imbalance
                max_iter=1000,
                random_state=RANDOM_SEED,
                C=0.5,                     # moderate regularisation
            )),
        ]),

        "RandomForest": Pipeline([
            ("scaler", StandardScaler()),  # RF is scale-invariant but kept for consistency
            ("clf", RandomForestClassifier(
                n_estimators=200,
                max_depth=8,               # controls depth to limit overfitting
                min_samples_leaf=10,       # avoids tiny leaves = overfit
                class_weight="balanced",
                random_state=RANDOM_SEED,
                n_jobs=-1,
            )),
        ]),

        "XGBoost": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", xgb.XGBClassifier(
                n_estimators=200,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                scale_pos_weight=1,        # adjusted below after split
                use_label_encoder=False,
                eval_metric="logloss",
                random_state=RANDOM_SEED,
                verbosity=0,
            )),
        ]),
    }
    return models


def train_and_select(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
) -> tuple[str, Pipeline, dict]:
    """
    Train all candidate models on X_train / y_train.
    Select the best model using F1-score on the validation set.

    Returns
    -------
    best_name : str
    best_pipeline : fitted Pipeline
    results : dict  (all model metrics on validation set)
    """
    # Adjust XGBoost scale_pos_weight to handle class imbalance
    neg_count = (y_train == 0).sum()
    pos_count = (y_train == 1).sum()
    spw = neg_count / pos_count if pos_count > 0 else 1.0
    print(f"[train] Class balance — Negative: {neg_count}, Positive: {pos_count}, scale_pos_weight: {spw:.2f}")

    models = build_model_pipelines()
    # Update XGBoost's scale_pos_weight
    models["XGBoost"].named_steps["clf"].set_params(scale_pos_weight=spw)

    results = {}
    best_name = None
    best_f1 = -1.0
    best_pipeline = None

    for name, pipeline in models.items():
        t0 = time.time()
        pipeline.fit(X_train, y_train)
        elapsed = time.time() - t0

        val_preds = pipeline.predict(X_val)
        val_proba = pipeline.predict_proba(X_val)[:, 1]
        f1 = f1_score(y_val, val_preds, zero_division=0)

        report = classification_report(y_val, val_preds, output_dict=True, zero_division=0)
        results[name] = {
            "val_f1": round(f1, 4),
            "val_precision": round(report["1"]["precision"], 4),
            "val_recall": round(report["1"]["recall"], 4),
            "val_accuracy": round(report["accuracy"], 4),
            "train_time_s": round(elapsed, 2),
        }

        print(
            f"[train] {name:25s} | "
            f"Val F1={f1:.4f} | "
            f"P={results[name]['val_precision']:.4f} | "
            f"R={results[name]['val_recall']:.4f} | "
            f"Acc={results[name]['val_accuracy']:.4f} | "
            f"Time={elapsed:.2f}s"
        )

        if f1 > best_f1:
            best_f1 = f1
            best_name = name
            best_pipeline = pipeline

    print(f"\n[train] ✅ Best model on validation: {best_name} (F1={best_f1:.4f})")
    return best_name, best_pipeline, results


def save_artifacts(
    best_name: str,
    best_pipeline: Pipeline,
    results: dict,
    n_train: int,
    n_val: int,
    n_test: int,
) -> None:
    """Persist the trained pipeline and metadata JSON."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(best_pipeline, MODEL_PATH)
    print(f"[train] Model saved → {MODEL_PATH}")

    metadata = {
        "selected_model": best_name,
        "feature_columns": FEATURE_COLUMNS,
        "target_column": TARGET_COLUMN,
        "at_risk_threshold": 40,
        "dataset_type": "Synthetic (reproducible, seed=42)",
        "n_train_observations": n_train,
        "n_val_observations": n_val,
        "n_test_observations": n_test,
        "training_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "random_seed": RANDOM_SEED,
        "model_results_on_validation": results,
        "responsible_ai_note": (
            "This model is an advisory decision-support tool. "
            "All predictions must be reviewed by a qualified educator. "
            "The model must NOT be used to automatically punish students, "
            "assign grades, make admission decisions, or replace human judgment."
        ),
    }
    with open(METADATA_PATH, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"[train] Metadata saved → {METADATA_PATH}")


def main() -> None:
    print("=" * 60)
    print("STUDENT PERFORMANCE & AT-RISK PREDICTION — TRAINING")
    print("=" * 60)

    # Load and prepare data
    df = load_raw_data()
    df = validate_data(df)
    train_df, val_df, test_df = chronological_split(df)

    X_train, y_train = get_X_y(train_df)
    X_val,   y_val   = get_X_y(val_df)
    X_test,  y_test  = get_X_y(test_df)

    # Train and select
    best_name, best_pipeline, results = train_and_select(
        X_train, y_train, X_val, y_val
    )

    # Save
    save_artifacts(
        best_name, best_pipeline, results,
        n_train=len(X_train), n_val=len(X_val), n_test=len(X_test),
    )

    print("\n[train] Training complete. Run evaluate.py for test-set metrics.")


if __name__ == "__main__":
    main()

