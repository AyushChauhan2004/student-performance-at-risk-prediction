"""
test_data.py
============
Tests for dataset generation (reproducibility, schema, target construction).
"""
import numpy as np
import pandas as pd
import pytest

from generate_data import generate_dataset, N_STUDENTS, N_PERIODS, AT_RISK_THRESHOLD
from preprocessing import FEATURE_COLUMNS, TARGET_COLUMN, LEAKAGE_COLUMNS


class TestDataGeneration:
    """Dataset reproducibility and structure tests."""

    def test_reproducibility(self):
        """Same seed → identical datasets."""
        df1 = generate_dataset(random_seed=42)
        df2 = generate_dataset(random_seed=42)
        pd.testing.assert_frame_equal(df1, df2)

    def test_different_seeds_differ(self):
        """Different seeds produce different outputs."""
        df1 = generate_dataset(random_seed=42)
        df2 = generate_dataset(random_seed=99)
        assert not df1["attendance_percentage"].equals(df2["attendance_percentage"])

    def test_shape(self):
        """Dataset has the expected number of rows."""
        df = generate_dataset()
        assert len(df) == N_STUDENTS * N_PERIODS

    def test_required_columns(self):
        """All expected columns are present."""
        df = generate_dataset()
        expected = set(FEATURE_COLUMNS) | {TARGET_COLUMN, "next_assessment_score", "student_id"}
        assert expected.issubset(set(df.columns)), f"Missing: {expected - set(df.columns)}"

    def test_dtypes(self):
        """Numeric columns have numeric dtypes."""
        df = generate_dataset()
        for col in FEATURE_COLUMNS:
            assert pd.api.types.is_numeric_dtype(df[col]), f"{col} is not numeric"

    def test_no_missing_values(self):
        """Generated dataset contains no nulls."""
        df = generate_dataset()
        assert df.isnull().sum().sum() == 0, "Dataset contains null values"

    def test_no_duplicates(self):
        """Each (student_id, assessment_period) pair is unique."""
        df = generate_dataset()
        assert not df.duplicated(subset=["student_id", "assessment_period"]).any()

    def test_at_risk_target_construction(self):
        """at_risk = 1 iff next_assessment_score < threshold."""
        df = generate_dataset()
        expected_at_risk = (df["next_assessment_score"] < AT_RISK_THRESHOLD).astype(int)
        pd.testing.assert_series_equal(
            df["at_risk"].reset_index(drop=True),
            expected_at_risk.reset_index(drop=True),
            check_names=False,
        )

    def test_at_risk_binary(self):
        """Target column contains only 0 and 1."""
        df = generate_dataset()
        assert set(df["at_risk"].unique()).issubset({0, 1})

    def test_attendance_range(self):
        """Attendance percentages are within [0, 100]."""
        df = generate_dataset()
        assert df["attendance_percentage"].between(0, 100).all()

    def test_completion_range(self):
        """Assignment completion percentages are within [0, 100]."""
        df = generate_dataset()
        assert df["assignment_completion_percentage"].between(0, 100).all()

    def test_score_range(self):
        """Assessment scores are within [0, 100]."""
        df = generate_dataset()
        for col in ["previous_assessment_score", "earlier_assessment_score", "next_assessment_score"]:
            assert df[col].between(0, 100).all(), f"{col} out of range"

    def test_missed_assignments_non_negative(self):
        """Missed assignments count is non-negative."""
        df = generate_dataset()
        assert (df["missed_assignments"] >= 0).all()

    def test_class_imbalance_reasonable(self):
        """At-risk class is not trivially rare or trivially dominant."""
        df = generate_dataset()
        ratio = df["at_risk"].mean()
        assert 0.05 < ratio < 0.60, f"Unexpected class balance: {ratio:.3f}"

    def test_multiple_periods_per_student(self):
        """Each student has multiple assessment periods."""
        df = generate_dataset()
        counts = df.groupby("student_id")["assessment_period"].nunique()
        assert (counts > 1).all(), "Some students have only one assessment period"

