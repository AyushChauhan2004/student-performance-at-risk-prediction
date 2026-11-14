"""
generate_data.py
================
Reproducible synthetic dataset generator for the
Student Performance & At-Risk Prediction project.

IMPORTANT — SYNTHETIC DATA NOTICE
----------------------------------
All data produced by this script is entirely artificial.
It does NOT represent any real school, student population, or
individual student.  It MUST NOT be used to make any claim
about actual student populations or real academic outcomes.

No sensitive/protected attributes (gender, caste, religion,
disability, ethnicity) are present in this dataset.

Assumptions
-----------
- Approximately 1 000 unique simulated students.
- Each student has 5 chronological assessment periods (terms).
- Attendance follows a right-skewed Beta distribution,
  clipped to [20 %, 100 %].
- Assignment completion is correlated with attendance.
- Previous scores are auto-regressive: each term's score
  depends on the prior term's score plus noise.
- Missed assignments is inversely correlated with completion %.
- Performance trend = current minus previous score (bounded).
- Target (at_risk = 1) is defined as next_assessment_score < 40.
- Dataset is ordered chronologically by (student_id, assessment_period).

Random seed: 42 (NumPy).
"""

import numpy as np
import pandas as pd
from pathlib import Path

# ── Configuration ──────────────────────────────────────────────────────────────
RANDOM_SEED = 42
N_STUDENTS = 1_000
N_PERIODS = 5          # assessment periods per student
AT_RISK_THRESHOLD = 40 # scores below this → at_risk = 1
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
OUTPUT_FILE = OUTPUT_DIR / "student_assessments.csv"


def generate_dataset(
    n_students: int = N_STUDENTS,
    n_periods: int = N_PERIODS,
    at_risk_threshold: int = AT_RISK_THRESHOLD,
    random_seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """
    Generate a reproducible synthetic student-assessment dataset.

    Parameters
    ----------
    n_students : int
        Number of unique simulated students.
    n_periods : int
        Number of sequential assessment periods per student.
    at_risk_threshold : int
        Score boundary defining at-risk (score < threshold → at_risk = 1).
    random_seed : int
        NumPy random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        One row per (student, assessment_period).  Rows are sorted by
        student_id then assessment_period.
    """
    rng = np.random.default_rng(random_seed)
    records = []

    for student_id in range(1, n_students + 1):
        # ── Per-student latent characteristics ────────────────────────────────
        # Latent "academic ability" [0, 1] governs average performance
        ability = rng.beta(a=2.5, b=2.5)          # centred around 0.5

        # Latent "engagement" [0, 1] governs attendance / completion tendency
        engagement = rng.beta(a=3.0, b=2.0)       # slightly right-skewed

        # Initialise the first "previous" score for the auto-regressive chain
        base_score = float(np.clip(ability * 100 + rng.normal(0, 8), 10, 100))
        prev_score = base_score
        earlier_score = max(base_score - rng.uniform(0, 15), 10.0)

        for period in range(1, n_periods + 1):
            # ── Attendance ────────────────────────────────────────────────────
            # Base attendance from engagement + noise
            att_base = engagement * 90 + 10            # range roughly 10-100
            attendance_pct = float(np.clip(
                att_base + rng.normal(0, 7), 20, 100
            ))

            # ── Assignment completion ─────────────────────────────────────────
            # Correlated with attendance and engagement
            comp_base = 0.6 * engagement * 100 + 0.3 * attendance_pct * 0.9
            assignment_completion_pct = float(np.clip(
                comp_base + rng.normal(0, 8), 0, 100
            ))

            # ── Missed assignments ────────────────────────────────────────────
            # Inversely related to completion percentage; range 0–10
            missed_base = (1 - assignment_completion_pct / 100) * 8
            missed_assignments = int(np.clip(
                round(missed_base + rng.normal(0, 1)), 0, 10
            ))

            # ── Next assessment score (the FUTURE outcome we predict) ─────────
            # Driven by: ability, attendance, completion, missed, prior score + noise
            score_signal = (
                0.35 * prev_score
                + 0.20 * ability * 100
                + 0.20 * attendance_pct * 0.8
                + 0.15 * assignment_completion_pct * 0.7
                - 0.10 * missed_assignments * 5
                + rng.normal(0, 9)          # irreducible noise
            )
            # Normalise to 0-100
            next_score = float(np.clip(score_signal, 0, 100))

            # ── Derived / engineered features ─────────────────────────────────
            average_previous_score = round((prev_score + earlier_score) / 2, 2)
            performance_trend = round(prev_score - earlier_score, 2)

            # ── Target label ──────────────────────────────────────────────────
            at_risk = int(next_score < at_risk_threshold)

            records.append({
                "student_id": student_id,
                "assessment_period": period,
                "attendance_percentage": round(attendance_pct, 2),
                "assignment_completion_percentage": round(assignment_completion_pct, 2),
                "previous_assessment_score": round(prev_score, 2),
                "earlier_assessment_score": round(earlier_score, 2),
                "missed_assignments": missed_assignments,
                "average_previous_score": average_previous_score,
                "performance_trend": performance_trend,
                "next_assessment_score": round(next_score, 2),
                "at_risk": at_risk,
            })

            # Advance the auto-regressive chain
            earlier_score = prev_score
            prev_score = next_score

    df = pd.DataFrame(records)
    df = df.sort_values(["student_id", "assessment_period"]).reset_index(drop=True)
    return df


def save_dataset(df: pd.DataFrame, path: Path = OUTPUT_FILE) -> None:
    """Persist the dataset to CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"[generate_data] Saved {len(df):,} rows → {path}")


if __name__ == "__main__":
    df = generate_dataset()
    save_dataset(df)
    print(df.head(10).to_string())
    print("\nShape:", df.shape)
    print("\nAt-risk distribution:")
    print(df["at_risk"].value_counts(normalize=True).round(3))

