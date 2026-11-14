"""
streamlit_app.py
================
Student Performance & At-Risk Prediction
Strict Compliance with Section 9 Architecture:
  A. Home / Overview
  B. Student Risk Prediction & Results
  C. Model Performance
  D. Model Information & Governance
"""

from __future__ import annotations

import io
import json
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

# ── Paths ─────────────────────────────────────────────────────────────────────
APP_DIR = Path(__file__).resolve().parent
BASE_DIR = APP_DIR.parent
MODEL_PATH = BASE_DIR / "models" / "student_risk_model.pkl"
METADATA_PATH = BASE_DIR / "models" / "model_metadata.json"
FIGURES_DIR = BASE_DIR / "reports" / "figures"
PERM_IMP_PATH = BASE_DIR / "models" / "permutation_importance.json"

import sys
sys.path.insert(0, str(BASE_DIR / "src"))
from preprocessing import FEATURE_COLUMNS
from explain import explain_prediction

# ── Page Configuration ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Student Performance & At-Risk Prediction",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Clean Dark Aesthetic ──────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

  html, body, [class*="css"], .stApp {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
    background-color: #0B0C10 !important;
    color: #EDEDED !important;
  }

  h1 {
    font-size: 26px !important;
    font-weight: 700 !important;
    color: #FFFFFF !important;
    letter-spacing: -0.03em !important;
    margin-bottom: 6px !important;
  }
  h2 {
    font-size: 20px !important;
    font-weight: 600 !important;
    color: #FFFFFF !important;
    letter-spacing: -0.02em !important;
    margin-top: 18px !important;
    margin-bottom: 12px !important;
  }
  h3, h4 {
    font-size: 15px !important;
    font-weight: 600 !important;
    color: #EDEDED !important;
  }
  p, span, label {
    color: #9CA3AF !important;
    font-size: 14px;
    line-height: 1.6;
  }

  /* Metric cards */
  div[data-testid="stMetric"] {
    background-color: #14161C !important;
    border: 1px solid rgba(255, 255, 255, 0.08) !important;
    border-radius: 8px !important;
    padding: 12px 16px !important;
  }
  div[data-testid="stMetricLabel"] p {
    color: #6B7280 !important;
    font-size: 11px !important;
    font-weight: 600 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.06em !important;
  }
  div[data-testid="stMetricValue"] {
    color: #FFFFFF !important;
    font-size: 22px !important;
    font-weight: 700 !important;
    letter-spacing: -0.02em !important;
  }

  /* Buttons */
  div.stButton > button:first-child {
    background-color: #FFFFFF !important;
    color: #0B0C10 !important;
    border: none !important;
    border-radius: 6px !important;
    padding: 9px 20px !important;
    font-size: 14px !important;
    font-weight: 600 !important;
    transition: all 0.15s ease !important;
  }
  div.stButton > button:first-child:hover {
    background-color: #E2E8F0 !important;
    color: #0B0C10 !important;
  }

  /* Sidebar */
  section[data-testid="stSidebar"] {
    background-color: #0E1015 !important;
    border-right: 1px solid rgba(255, 255, 255, 0.06) !important;
  }
  section[data-testid="stSidebar"] p, section[data-testid="stSidebar"] span {
    color: #9CA3AF !important;
  }

  /* Containers & Tables */
  div[data-testid="stDataFrame"] {
    border: 1px solid rgba(255, 255, 255, 0.08) !important;
    border-radius: 6px !important;
  }
  
  hr {
    border-color: rgba(255, 255, 255, 0.08) !important;
  }
</style>
""", unsafe_allow_html=True)


# ── Load Model & Metadata ─────────────────────────────────────────────────────
@st.cache_resource
def load_pipeline():
    if not MODEL_PATH.exists():
        return None
    return joblib.load(MODEL_PATH)

@st.cache_data
def load_metadata() -> dict:
    if not METADATA_PATH.exists():
        return {}
    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

pipeline = load_pipeline()
metadata = load_metadata()


# ── Sidebar Navigation (Strictly Section 9 Structure) ─────────────────────────
with st.sidebar:
    st.markdown("### Student Performance & At-Risk Prediction")
    st.caption("School Management Early-Warning System")
    st.markdown("---")

    page = st.radio(
        "Navigation",
        [
            "Home / Overview",
            "Student Risk Prediction",
            "Model Performance",
            "Model Information",
        ],
        index=0,
    )

    st.markdown("---")
    st.markdown("""
    <div style="font-size:11px; text-transform:uppercase; color:#6B7280; font-weight:600; margin-bottom:6px;">Model Status</div>
    <div style="font-size:13px; color:#EDEDED;">Selected: <b>Random Forest</b></div>
    <div style="font-size:13px; color:#EDEDED;">Target: <b>Score &lt; 40 / 100</b></div>
    <div style="font-size:13px; color:#EDEDED;">Test Recall: <b>84.9%</b></div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.caption(
        "Advisory Notice: This decision-support tool assists educators in identifying "
        "students who may benefit from academic support. Predictions must never be used "
        "to automatically penalize, grade, or make final academic judgments."
    )


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 9.A: HOME / OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
if page == "Home / Overview":
    st.title("Student Performance & At-Risk Prediction")
    st.markdown("Decision-support system to identify students at risk of poor academic performance in their next assessment.")
    st.markdown("---")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Prediction Horizon", "Next Assessment", "Prior to Exam")
    m2.metric("Target Cutoff", "Score < 40 / 100", "Configurable")
    m3.metric("Test Set Recall", "84.9%", "+3.5% vs Baseline")
    m4.metric("Test ROC-AUC", "0.846", "Classification Power")

    st.markdown("---")

    col_l, col_r = st.columns([1.6, 1], gap="large")

    with col_l:
        st.markdown("### Project Purpose")
        st.markdown("""
        Educational institutions frequently identify struggling students **reactively** — after poor examination marks are already recorded.
        
        This system establishes a **proactive, point-in-time predictive pipeline** that evaluates engagement indicators, prior assessment scores, and attendance trends available *before* the upcoming examination takes place.
        
        ### Prediction Definitions
        - **Prediction Unit:** Student at a particular assessment period.
        - **Target:** Next assessment score below 40 out of 100.
        - **Prediction Horizon:** Next scheduled assessment.
        - **Prediction Timestamp:** Before the next assessment occurs.
        - **Positive Class:** At Risk (1).
        - **Negative Class:** Not At Risk (0).
        """)

    with col_r:
        st.markdown("### Operational Guardrails")
        st.markdown("""
        - **No Automated Actions:** Predictions are probabilistic estimates, not established facts.
        - **No Punitive Decisions:** The system must never be used to punish students, assign grades, or make admission choices.
        - **Zero Protected Attributes:** Gender, caste, religion, ethnicity, and disability markers are strictly excluded.
        - **Chronological Split:** Periods 1–3 train, Period 4 validation, Period 5 test to eliminate look-ahead leakage.
        """)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 9.B & 9.C: STUDENT RISK PREDICTION & RESULTS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Student Risk Prediction":
    st.title("Student Risk Prediction")
    st.markdown("Input student academic signals available before the upcoming assessment to estimate at-risk probability.")

    if pipeline is None:
        st.error("Model artifact not found. Run python src/train.py first.")
        st.stop()

    preset = st.selectbox(
        "Demonstration Presets:",
        [
            "Custom Input",
            "Scenario A: High Risk (Attendance 52%, dropping prior score 34)",
            "Scenario B: Moderate Risk (Attendance 72%, score 46)",
            "Scenario C: Low Risk (Attendance 94%, consistent score 82)",
        ]
    )

    if preset.startswith("Scenario A"):
        d_att, d_comp, d_miss, d_prev, d_earl, d_period = 52.0, 48.0, 5, 34.0, 44.0, 4
    elif preset.startswith("Scenario B"):
        d_att, d_comp, d_miss, d_prev, d_earl, d_period = 72.0, 68.0, 2, 46.0, 56.0, 3
    elif preset.startswith("Scenario C"):
        d_att, d_comp, d_miss, d_prev, d_earl, d_period = 94.0, 95.0, 0, 82.0, 80.0, 2
    else:
        d_att, d_comp, d_miss, d_prev, d_earl, d_period = 70.0, 65.0, 2, 50.0, 55.0, 3

    with st.form("risk_form"):
        st.markdown("### Input Features")
        c1, c2, c3 = st.columns(3)

        with c1:
            att = st.slider("Attendance Percentage (%)", 0.0, 100.0, float(d_att), 0.5)
            comp = st.slider("Assignment Completion (%)", 0.0, 100.0, float(d_comp), 0.5)
            missed = st.number_input("Missed Assignments Count", 0, 15, int(d_miss), 1)

        with c2:
            prev_s = st.slider("Previous Assessment Score (0–100)", 0.0, 100.0, float(d_prev), 0.5)
            earl_s = st.slider("Earlier Assessment Score (0–100)", 0.0, 100.0, float(d_earl), 0.5)
            period = st.number_input("Assessment Period", 1, 10, int(d_period), 1)

        with c3:
            avg_s = round((prev_s + earl_s) / 2.0, 2)
            trend = round(prev_s - earl_s, 2)
            st.metric("Computed Rolling Average", f"{avg_s:.1f} / 100")
            st.metric("Performance Trend (Velocity)", f"{trend:+.1f} pts")
            st.caption("Engineered features are computed automatically prior to inference.")

        submitted = st.form_submit_button("Predict Risk", use_container_width=True)

    if submitted or preset != "Custom Input":
        input_data = {
            "assessment_period": int(period),
            "attendance_percentage": float(att),
            "assignment_completion_percentage": float(comp),
            "previous_assessment_score": float(prev_s),
            "earlier_assessment_score": float(earl_s),
            "missed_assignments": int(missed),
            "average_previous_score": float(avg_s),
            "performance_trend": float(trend),
        }

        X_df = pd.DataFrame([{col: input_data[col] for col in FEATURE_COLUMNS}])
        proba = float(pipeline.predict_proba(X_df)[0][1])
        explanation = explain_prediction(input_data, proba, pipeline)

        st.markdown("---")
        st.markdown("### Prediction Results")

        r1, r2, r3 = st.columns([1, 1.2, 1.2])

        with r1:
            st.markdown(f"**Risk Band:** {explanation['risk_band']}")
            st.metric("Estimated At-Risk Probability", f"{proba:.1%}")
            st.progress(min(proba, 1.0))
            if proba >= 0.65:
                st.error("High Risk: Immediate Academic Support Advised")
            elif proba >= 0.35:
                st.warning("Medium Risk: Active Monitoring Recommended")
            else:
                st.success("Low Risk: On-Track Trajectory")

        with r2:
            st.markdown("**Influential Input Signals**")
            sig_df = pd.DataFrame(explanation["top_signals"])
            st.dataframe(
                sig_df.rename(columns={"feature": "Feature", "value": "Observed", "importance": "Model Weight"}),
                use_container_width=True,
                hide_index=True,
            )

        with r3:
            st.markdown("**Academic Support Considerations**")
            considerations = []
            if att < 70:
                considerations.append("Attendance is below 70%: review attendance barriers.")
            if comp < 65:
                considerations.append("Assignment completion is low: provide structured assignment support.")
            if prev_s < 45:
                considerations.append("Recent score is under 45: schedule subject tutoring.")
            if trend < -10:
                considerations.append("Declining trend: pastoral check-in recommended.")
            if not considerations:
                considerations.append("Student is performing steadily. Continue standard academic monitoring.")
            
            for item in considerations:
                st.markdown(f"- {item}")

        st.warning(explanation["disclaimer"])

        # Prescriptive Simulation
        with st.expander("Prescriptive 'What-If' Simulation (Decision Support)"):
            st.markdown("Test proposed academic interventions to evaluate risk reduction:")
            sim_col1, sim_col2 = st.columns(2)
            with sim_col1:
                att_boost = st.slider("Target Attendance Increase (+%)", 0.0, 30.0, 10.0, 1.0)
                score_boost = st.slider("Target Tutoring Score Increase (+pts)", 0.0, 20.0, 8.0, 1.0)
            with sim_col2:
                sim_input = input_data.copy()
                sim_input["attendance_percentage"] = min(100.0, att + att_boost)
                sim_input["previous_assessment_score"] = min(100.0, prev_s + score_boost)
                sim_input["average_previous_score"] = (sim_input["previous_assessment_score"] + earl_s) / 2.0
                sim_input["performance_trend"] = sim_input["previous_assessment_score"] - earl_s
                
                X_sim = pd.DataFrame([{col: sim_input[col] for col in FEATURE_COLUMNS}])
                sim_proba = float(pipeline.predict_proba(X_sim)[0][1])
                diff = sim_proba - proba
                
                st.metric("Baseline Risk", f"{proba:.1%}")
                st.metric("Simulated Risk", f"{sim_proba:.1%}", f"{diff:+.1%}", delta_color="inverse")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 9.D: MODEL PERFORMANCE
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Model Performance":
    st.title("Model Performance")
    st.markdown("Actual saved evaluation results computed on the held-out test set (Period 5, N=1,000).")

    test_metrics = metadata.get("test_metrics", {})
    if test_metrics:
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Accuracy", f"{test_metrics.get('accuracy', 0):.3f}")
        m2.metric("Precision", f"{test_metrics.get('precision', 0):.3f}")
        m3.metric("Recall (Sensitivity)", f"{test_metrics.get('recall', 0):.3f}")
        m4.metric("F1-Score", f"{test_metrics.get('f1_score', 0):.3f}")
        m5.metric("ROC-AUC", f"{test_metrics.get('roc_auc', 0):.3f}")

    st.markdown("---")
    st.markdown("### Classification Threshold Trade-Off Analysis")
    st.markdown("""
    In educational decision support, threshold selection balances **Counselor Workload (False Positives)** against **Missed At-Risk Students (False Negatives)**.
    """)

    thresh = st.slider("Select Classification Threshold", 0.20, 0.80, 0.50, 0.05)
    t_c1, t_c2 = st.columns(2)
    with t_c1:
        if thresh < 0.40:
            st.info(f"Threshold = {thresh:.2f}: High Sensitivity Mode. Minimizes missed cases (high recall).")
        elif thresh > 0.60:
            st.warning(f"Threshold = {thresh:.2f}: High Precision Mode. Flags only critical cases, reducing teacher workload.")
        else:
            st.success(f"Threshold = {thresh:.2f}: Balanced Mode. Standard default threshold.")

    with t_c2:
        st.markdown(f"""
        - Predicted Positive when: P(Score < 40) >= {thresh:.2f}
        - Selected Model: {metadata.get('selected_model', 'RandomForest')}
        - Validation F1: {metadata.get('model_results_on_validation', {}).get('RandomForest', {}).get('val_f1', '0.814')}
        """)

    st.markdown("---")
    fig_c1, fig_c2 = st.columns(2)

    cm_img = FIGURES_DIR / "confusion_matrix.png"
    with fig_c1:
        st.markdown("### Test Set Confusion Matrix")
        if cm_img.exists():
            st.image(str(cm_img), use_container_width=True)
            st.caption("True Positives: 490 | True Negatives: 273 | False Positives: 150 | False Negatives: 87")

    perm_img = FIGURES_DIR / "permutation_importance.png"
    with fig_c2:
        st.markdown("### Permutation Feature Importance")
        if perm_img.exists():
            st.image(str(perm_img), use_container_width=True)
            st.caption("Mean decrease in test F1 score upon feature permutation.")

    comp_img = FIGURES_DIR / "model_comparison.png"
    if comp_img.exists():
        st.markdown("### Model Comparison (Validation Split)")
        st.image(str(comp_img), use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 9.E: MODEL INFORMATION
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Model Information":
    st.title("Model Information & Governance")
    st.markdown("Metadata, target definition, architectural guarantees, and limitations.")

    c_left, c_right = st.columns(2, gap="large")

    with c_left:
        st.markdown("### Model Provenance")
        provenance = {
            "Selected Model": str(metadata.get('selected_model', 'RandomForestClassifier')),
            "Dataset Type": str(metadata.get('dataset_type', 'Synthetic (reproducible, seed=42)')),
            "Training Observations": f"{metadata.get('n_train_observations', 3000):,}",
            "Validation Observations": f"{metadata.get('n_val_observations', 1000):,}",
            "Test Observations": f"{metadata.get('n_test_observations', 1000):,}",
            "Training Date": str(metadata.get('training_date', 'N/A')),
            "Random Seed": str(metadata.get('random_seed', 42)),
        }
        st.table(pd.DataFrame(list(provenance.items()), columns=["Field", "Value"]))

    with c_right:
        st.markdown("### Target Specification")
        st.markdown("""
        - **Target Column:** `at_risk`
        - **Target Rule:** `next_assessment_score < 40` out of 100
        - **Positive Class:** At Risk (1)
        - **Negative Class:** Not At Risk (0)
        - **Horizon:** Next scheduled assessment period
        - **Leakage Prevention:** `next_assessment_score`, `at_risk`, and `student_id` are strictly excluded from features.
        """)

    st.markdown("---")
    st.markdown("### Feature Schema")
    st.code("\n".join(metadata.get('feature_columns', FEATURE_COLUMNS)))

    st.markdown("---")
    st.markdown("### Responsible AI & Limitations")
    st.markdown("""
    1. **Synthetic Data Grounding:** The model is trained on synthetic data (seed 42). Real-world deployment requires ethical validation against actual school records.
    2. **Protected Attributes:** Demographics (gender, ethnicity, religion, caste, disability) are strictly excluded to avoid direct bias.
    3. **Advisory Decision Support:** Model outputs are probabilistic indicators, not confirmed facts. Qualified educator review is required before taking action.
    4. **Correlation vs. Causation:** Identified signals show predictive association, not causal proof.
    5. **Concept Drift:** Shifts in school policy, curriculum, or attendance records will require periodic model retraining.
    """)
