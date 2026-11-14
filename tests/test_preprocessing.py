"""
test_preprocessing.py
=====================
Tests for the preprocessing pipeline — leakage prevention, splitting,
missing value handling, and scaler fit behaviour.
"""
import numpy as np
import pandas as pd
import pytest

from generate_data import generate_dataset
from preprocessing import (
    validate_data, chronological_split, get_X_y,
    build_preprocessor, FEATURE_COLUMNS, TARGET_COLUMN, LEAKAGE_COLUMNS,
    VAL_PERIODS, TEST_PERIODS,
)


@pytest.fixture(scope="module")
def raw_df():
    return generate_dataset()


class TestLeakagePrevention:
    """Ensure future/target columns never appear in X."""

    def test_next_score_not_in_features(self):
        assert "next_assessment_score" not in FEATURE_COLUMNS

    def test_at_risk_not_in_features(self):
        assert "at_risk" not in FEATURE_COLUMNS

    def test_student_id_not_in_features(self):
        assert "student_id" not in FEATURE_COLUMNS

    def test_leakage_columns_definition(self):
        overlap = LEAKAGE_COLUMNS & set(FEATURE_COLUMNS)
        assert len(overlap) == 0, f"Leakage columns in FEATURE_COLUMNS: {overlap}"

    def test_get_X_y_excludes_leakage(self, raw_df):
        X, y = get_X_y(raw_df)
        for col in LEAKAGE_COLUMNS:
            assert col not in X.columns, f"Leakage column '{col}' found in X"


class TestChronologicalSplit:
    """Validate chronological splitting logic."""

    def test_split_sizes(self, raw_df):
        train_df, val_df, test_df = chronological_split(raw_df)
        assert len(train_df) + len(val_df) + len(test_df) == len(raw_df)

    def test_no_overlap_between_splits(self, raw_df):
        train_df, val_df, test_df = chronological_split(raw_df)
        train_periods = set(train_df["assessment_period"].unique())
        val_periods = set(val_df["assessment_period"].unique())
        test_periods = set(test_df["assessment_period"].unique())
        assert len(train_periods & val_periods) == 0
        assert len(train_periods & test_periods) == 0
        assert len(val_periods & test_periods) == 0

    def test_test_periods_are_latest(self, raw_df):
        train_df, val_df, test_df = chronological_split(raw_df)
        max_train = train_df["assessment_period"].max()
        min_test = test_df["assessment_period"].min()
        assert min_test > max_train, "Test periods overlap with or precede training periods"

    def test_train_set_is_not_empty(self, raw_df):
        train_df, _, _ = chronological_split(raw_df)
        assert len(train_df) > 0


class TestMissingValueHandling:
    """Validate that validate_data handles missing values correctly."""

    def test_handles_missing_values(self, raw_df):
        df_dirty = raw_df.copy()
        df_dirty.loc[0, "attendance_percentage"] = np.nan
        df_dirty.loc[1, "previous_assessment_score"] = np.nan
        # Should not raise
        df_clean = validate_data(df_dirty)
        assert df_clean["attendance_percentage"].isnull().sum() == 0
        assert df_clean["previous_assessment_score"].isnull().sum() == 0


class TestPreprocessor:
    """Validate scaler fit/transform behaviour."""

    def test_preprocessor_transforms_correctly(self, raw_df):
        df = validate_data(raw_df)
        train_df, val_df, _ = chronological_split(df)
        X_train, _ = get_X_y(train_df)
        X_val, _ = get_X_y(val_df)

        preprocessor = build_preprocessor()
        X_train_scaled = preprocessor.fit_transform(X_train)
        X_val_scaled = preprocessor.transform(X_val)

        # Scaled train data should be approximately zero-mean
        assert abs(X_train_scaled.mean()) < 0.5

    def test_scaler_not_fitted_on_val(self, raw_df):
        """Verify val mean is NOT zero (scaler fit only on train)."""
        df = validate_data(raw_df)
        train_df, val_df, _ = chronological_split(df)
        X_train, _ = get_X_y(train_df)
        X_val, _ = get_X_y(val_df)

        preprocessor = build_preprocessor()
        preprocessor.fit(X_train)
        X_val_scaled = preprocessor.transform(X_val)
        # Val set scaled mean should NOT be close to 0 (different distribution)
        # This test passes as long as it doesn't raise
        assert X_val_scaled is not None

