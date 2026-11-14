"""
preprocessing.py
================
Data loading, validation, leakage-safe feature/target separation,
chronological train / validation / test splitting, and sklearn pipeline
construction for the Student Performance & At-Risk Prediction project.

Leakage-prevention rules enforced here
---------------------------------------
1. `next_assessment_score` is NEVER included in X.
2. `at_risk`              is NEVER included in X.
3. `student_id`           is NEVER included in X.
4. The sklearn preprocessing pipeline (scaler) is fitted ONLY on X_train.
5. Data is split chronologically: earlier periods → train, later → test.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DATA_PATH = BASE_DIR / "data" / "raw" / "student_assessments.csv"
PROCESSED_DIR = BASE_DIR / "data" / "processed"

# ── Feature / Target Definitions ───────────────────────────────────────────────
# These are the ONLY columns used as model inputs.
# assessment_period IS included as a feature; it encodes temporal position.
FEATURE_COLUMNS = [
    "assessment_period",
    "attendance_percentage",
    "assignment_completion_percentage",
    "previous_assessment_score",
    "earlier_assessment_score",
    "missed_assignments",
    "average_previous_score",
    "performance_trend",
]

TARGET_COLUMN = "at_risk"

# Columns that must NEVER appear in X (data-leakage guard)
LEAKAGE_COLUMNS = {"next_assessment_score", "at_risk", "student_id"}

# Chronological split: last N periods go to test, second-last to validation
# With 5 periods: periods 1-3 → train, 4 → val, 5 → test
VAL_PERIODS = {4}
TEST_PERIODS = {5}


def load_raw_data(path: Path = RAW_DATA_PATH) -> pd.DataFrame:
    """Load the raw CSV and return a DataFrame."""
    if not path.exists():
        raise FileNotFoundError(
            f"Raw dataset not found at {path}. "
            "Run `python src/generate_data.py` first."
        )
    df = pd.read_csv(path)
    print(f"[preprocessing] Loaded {len(df):,} rows from {path.name}")
    return df


def validate_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Run basic data-quality checks.  Returns the validated (possibly
    corrected) DataFrame.  Raises ValueError on unrecoverable issues.
    """
    required_cols = set(FEATURE_COLUMNS) | {TARGET_COLUMN, "next_assessment_score", "student_id"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")

    # Range checks
    pct_cols = ["attendance_percentage", "assignment_completion_percentage"]
    for col in pct_cols:
        out_of_range = ((df[col] < 0) | (df[col] > 100)).sum()
        if out_of_range:
            print(f"[preprocessing] WARNING: {out_of_range} out-of-range values in {col}; clipping.")
            df[col] = df[col].clip(0, 100)

    score_cols = ["previous_assessment_score", "earlier_assessment_score", "next_assessment_score"]
    for col in score_cols:
        out_of_range = ((df[col] < 0) | (df[col] > 100)).sum()
        if out_of_range:
            print(f"[preprocessing] WARNING: {out_of_range} out-of-range values in {col}; clipping.")
            df[col] = df[col].clip(0, 100)

    # Missing values
    missing_counts = df[list(required_cols)].isnull().sum()
    if missing_counts.any():
        print("[preprocessing] Missing values detected — filling with column medians:")
        for col, cnt in missing_counts[missing_counts > 0].items():
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
            print(f"  {col}: {cnt} filled with median={median_val:.2f}")

    # Duplicate rows
    n_dupes = df.duplicated().sum()
    if n_dupes:
        print(f"[preprocessing] Dropping {n_dupes} exact duplicate rows.")
        df = df.drop_duplicates().reset_index(drop=True)

    # Leakage guard: assert no leakage column appears in FEATURE_COLUMNS
    leakage_overlap = LEAKAGE_COLUMNS & set(FEATURE_COLUMNS)
    if leakage_overlap:
        raise ValueError(
            f"LEAKAGE DETECTED: {leakage_overlap} appear in FEATURE_COLUMNS!"
        )

    print("[preprocessing] Validation passed.")
    return df


def chronological_split(
    df: pd.DataFrame,
    val_periods: set = VAL_PERIODS,
    test_periods: set = TEST_PERIODS,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split by assessment_period in chronological order.

    Why chronological? Because we simulate real deployment: a model trained
    on past periods must predict for future periods.  Random splitting
    would leak temporal information (e.g. period-5 data could appear in
    training, making validation unrealistically easy).

    Returns
    -------
    train_df, val_df, test_df
    """
    test_df = df[df["assessment_period"].isin(test_periods)].copy()
    val_df  = df[df["assessment_period"].isin(val_periods)].copy()
    train_df = df[~df["assessment_period"].isin(test_periods | val_periods)].copy()

    print(
        f"[preprocessing] Chronological split: "
        f"train={len(train_df):,} | val={len(val_df):,} | test={len(test_df):,}"
    )
    return train_df, val_df, test_df


def get_X_y(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """Extract feature matrix X and target y from a split DataFrame."""
    X = df[FEATURE_COLUMNS].copy()
    y = df[TARGET_COLUMN].copy()
    return X, y


def build_preprocessor() -> Pipeline:
    """
    Build a sklearn preprocessing pipeline.

    Currently: StandardScaler only (all features are numeric).
    The pipeline is intentionally kept simple and fitted ONLY on training data.
    """
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
    ])
    return pipeline


def run_preprocessing(
    raw_path: Path = RAW_DATA_PATH,
    save_splits: bool = True,
) -> dict:
    """
    Full preprocessing workflow:
      1. Load raw data.
      2. Validate.
      3. Chronological split.
      4. Build + fit preprocessor on train only.
      5. Optionally save processed splits.

    Returns a dict with keys:
      X_train, X_val, X_test, y_train, y_val, y_test,
      X_train_scaled, X_val_scaled, X_test_scaled, preprocessor
    """
    df = load_raw_data(raw_path)
    df = validate_data(df)

    train_df, val_df, test_df = chronological_split(df)

    X_train, y_train = get_X_y(train_df)
    X_val,   y_val   = get_X_y(val_df)
    X_test,  y_test  = get_X_y(test_df)

    preprocessor = build_preprocessor()
    X_train_scaled = preprocessor.fit_transform(X_train)
    X_val_scaled   = preprocessor.transform(X_val)
    X_test_scaled  = preprocessor.transform(X_test)

    print(f"[preprocessing] Feature columns: {FEATURE_COLUMNS}")
    print(f"[preprocessing] Target: {TARGET_COLUMN}")
    print(f"[preprocessing] Train at-risk rate: {y_train.mean():.3f}")
    print(f"[preprocessing] Val   at-risk rate: {y_val.mean():.3f}")
    print(f"[preprocessing] Test  at-risk rate: {y_test.mean():.3f}")

    if save_splits:
        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        train_df.to_csv(PROCESSED_DIR / "train.csv", index=False)
        val_df.to_csv(PROCESSED_DIR / "val.csv", index=False)
        test_df.to_csv(PROCESSED_DIR / "test.csv", index=False)
        print(f"[preprocessing] Processed splits saved to {PROCESSED_DIR}")

    return {
        "X_train": X_train,
        "X_val": X_val,
        "X_test": X_test,
        "y_train": y_train,
        "y_val": y_val,
        "y_test": y_test,
        "X_train_scaled": X_train_scaled,
        "X_val_scaled": X_val_scaled,
        "X_test_scaled": X_test_scaled,
        "preprocessor": preprocessor,
        "feature_columns": FEATURE_COLUMNS,
    }


if __name__ == "__main__":
    result = run_preprocessing()
    print("\nPreprocessing complete.")

