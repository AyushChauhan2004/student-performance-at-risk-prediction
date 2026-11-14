"""
predict.py
==========
Inference script for the Student Performance & At-Risk Prediction project.

Loads the saved model pipeline and runs a prediction for a single student
observation supplied as a dictionary of feature values.

Usage (command line)
---------------------
  python src/predict.py

Usage (programmatic)
---------------------
  from predict import predict_single
  result = predict_single({
      "assessment_period": 3,
      "attendance_percentage": 55.0,
      "assignment_completion_percentage": 60.0,
      "previous_assessment_score": 38.0,
      "earlier_assessment_score": 42.0,
      "missed_assignments": 4,
      "average_previous_score": 40.0,
      "performance_trend": -4.0,
  })
  print(result)
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from preprocessing import FEATURE_COLUMNS
from explain import explain_prediction

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "student_risk_model.pkl"


def load_pipeline():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model artifact not found at {MODEL_PATH}. "
            "Run `python src/train.py` first."
        )
    return joblib.load(MODEL_PATH)


def validate_input(input_dict: dict) -> None:
    """
    Validate that input values are within expected ranges.
    Raises ValueError with a descriptive message on invalid input.
    """
    pct_fields = ["attendance_percentage", "assignment_completion_percentage"]
    for field in pct_fields:
        val = input_dict.get(field)
        if val is not None and not (0 <= val <= 100):
            raise ValueError(f"{field} must be between 0 and 100 (got {val}).")

    score_fields = ["previous_assessment_score", "earlier_assessment_score"]
    for field in score_fields:
        val = input_dict.get(field)
        if val is not None and not (0 <= val <= 100):
            raise ValueError(f"{field} must be between 0 and 100 (got {val}).")

    missed = input_dict.get("missed_assignments")
    if missed is not None and missed < 0:
        raise ValueError(f"missed_assignments cannot be negative (got {missed}).")

    period = input_dict.get("assessment_period")
    if period is not None and period < 1:
        raise ValueError(f"assessment_period must be >= 1 (got {period}).")


def predict_single(input_dict: dict) -> dict:
    """
    Run inference for a single student.

    Parameters
    ----------
    input_dict : dict
        Must contain all keys in FEATURE_COLUMNS with numeric values.

    Returns
    -------
    dict with keys: at_risk, probability, risk_band, explanation, disclaimer
    """
    validate_input(input_dict)

    missing = set(FEATURE_COLUMNS) - set(input_dict.keys())
    if missing:
        raise ValueError(f"Missing required features: {missing}")

    pipeline = load_pipeline()

    X = pd.DataFrame([{col: input_dict[col] for col in FEATURE_COLUMNS}])
    prediction = int(pipeline.predict(X)[0])
    probability = float(pipeline.predict_proba(X)[0][1])

    explanation = explain_prediction(input_dict, probability, pipeline)

    return {
        "at_risk": prediction,
        "probability": round(probability, 4),
        **explanation,
    }


if __name__ == "__main__":
    sample = {
        "assessment_period": 3,
        "attendance_percentage": 55.0,
        "assignment_completion_percentage": 60.0,
        "previous_assessment_score": 38.0,
        "earlier_assessment_score": 42.0,
        "missed_assignments": 4,
        "average_previous_score": 40.0,
        "performance_trend": -4.0,
    }

    result = predict_single(sample)
    print("\n=== PREDICTION RESULT ===")
    for k, v in result.items():
        if k == "top_signals":
            print(f"  top_signals:")
            for s in v:
                print(f"    - {s}")
        else:
            print(f"  {k}: {v}")

