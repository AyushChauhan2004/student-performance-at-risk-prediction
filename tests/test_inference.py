"""
test_inference.py
=================
Tests for model loading, prediction validity, input validation,
and end-to-end inference.
"""
import numpy as np
import pandas as pd
import pytest

from predict import predict_single, load_pipeline, validate_input
from preprocessing import FEATURE_COLUMNS

VALID_INPUT = {
    "assessment_period": 3,
    "attendance_percentage": 70.0,
    "assignment_completion_percentage": 65.0,
    "previous_assessment_score": 50.0,
    "earlier_assessment_score": 55.0,
    "missed_assignments": 2,
    "average_previous_score": 52.5,
    "performance_trend": -5.0,
}


class TestModelArtifact:
    """Verify the saved model artifact is loadable and usable."""

    def test_model_loads_successfully(self):
        pipeline = load_pipeline()
        assert pipeline is not None

    def test_model_has_predict_method(self):
        pipeline = load_pipeline()
        assert hasattr(pipeline, "predict")

    def test_model_has_predict_proba_method(self):
        pipeline = load_pipeline()
        assert hasattr(pipeline, "predict_proba")

    def test_model_uses_correct_feature_count(self):
        pipeline = load_pipeline()
        X = pd.DataFrame([{col: 50.0 for col in FEATURE_COLUMNS}])
        proba = pipeline.predict_proba(X)
        assert proba.shape == (1, 2)


class TestInference:
    """Verify prediction output structure and validity."""

    def test_predict_returns_dict(self):
        result = predict_single(VALID_INPUT)
        assert isinstance(result, dict)

    def test_predict_contains_at_risk(self):
        result = predict_single(VALID_INPUT)
        assert "at_risk" in result
        assert result["at_risk"] in {0, 1}

    def test_predict_probability_in_range(self):
        result = predict_single(VALID_INPUT)
        assert 0.0 <= result["probability"] <= 1.0

    def test_predict_contains_risk_band(self):
        result = predict_single(VALID_INPUT)
        assert "risk_band" in result
        assert result["risk_band"] in {"Low Risk", "Medium Risk", "High Risk"}

    def test_predict_contains_disclaimer(self):
        result = predict_single(VALID_INPUT)
        assert "disclaimer" in result
        assert len(result["disclaimer"]) > 10

    def test_predict_contains_top_signals(self):
        result = predict_single(VALID_INPUT)
        assert "top_signals" in result
        assert isinstance(result["top_signals"], list)
        assert len(result["top_signals"]) > 0

    def test_end_to_end_high_risk_student(self):
        """A student with very low attendance and scores should be high risk."""
        high_risk_input = {
            "assessment_period": 4,
            "attendance_percentage": 25.0,
            "assignment_completion_percentage": 20.0,
            "previous_assessment_score": 20.0,
            "earlier_assessment_score": 22.0,
            "missed_assignments": 8,
            "average_previous_score": 21.0,
            "performance_trend": -2.0,
        }
        result = predict_single(high_risk_input)
        # We only assert the probability is higher than a moderate student's
        assert result["probability"] > 0.3  # should be meaningfully at risk

    def test_end_to_end_low_risk_student(self):
        """A student with high attendance and scores should be low risk."""
        low_risk_input = {
            "assessment_period": 2,
            "attendance_percentage": 95.0,
            "assignment_completion_percentage": 98.0,
            "previous_assessment_score": 85.0,
            "earlier_assessment_score": 82.0,
            "missed_assignments": 0,
            "average_previous_score": 83.5,
            "performance_trend": 3.0,
        }
        result = predict_single(low_risk_input)
        assert result["probability"] < 0.5  # should not be predicted as at risk


class TestInputValidation:
    """Verify that invalid inputs are rejected gracefully."""

    def test_attendance_above_100_raises(self):
        bad_input = VALID_INPUT.copy()
        bad_input["attendance_percentage"] = 110.0
        with pytest.raises(ValueError):
            validate_input(bad_input)

    def test_attendance_negative_raises(self):
        bad_input = VALID_INPUT.copy()
        bad_input["attendance_percentage"] = -5.0
        with pytest.raises(ValueError):
            validate_input(bad_input)

    def test_completion_above_100_raises(self):
        bad_input = VALID_INPUT.copy()
        bad_input["assignment_completion_percentage"] = 101.0
        with pytest.raises(ValueError):
            validate_input(bad_input)

    def test_negative_missed_raises(self):
        bad_input = VALID_INPUT.copy()
        bad_input["missed_assignments"] = -1
        with pytest.raises(ValueError):
            validate_input(bad_input)

    def test_score_above_100_raises(self):
        bad_input = VALID_INPUT.copy()
        bad_input["previous_assessment_score"] = 105.0
        with pytest.raises(ValueError):
            validate_input(bad_input)

    def test_missing_feature_raises(self):
        incomplete = {k: v for k, v in VALID_INPUT.items() if k != "attendance_percentage"}
        with pytest.raises(ValueError):
            predict_single(incomplete)

