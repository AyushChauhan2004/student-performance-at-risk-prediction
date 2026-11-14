# Technical Report
## Student Performance & At-Risk Prediction
**Algions AI Labs Private Limited | AI & Machine Learning**

*Dataset: Synthetic (reproducible). Metrics: measured from actual experiments.*

---

## 1. Abstract

This report documents the design, implementation, and evaluation of a machine learning–based decision-support system for identifying students at risk of poor academic performance. The system predicts whether a student will score below 40 out of 100 in their next scheduled assessment, using features available before that assessment takes place. Three models were trained and compared — Logistic Regression, Random Forest, and XGBoost — with Random Forest selected based on F1-score on a held-out validation set. The selected model achieved 0.763 accuracy, 0.765 precision, 0.849 recall, and 0.805 F1-score on the final test set. All data is synthetic. The system is explicitly designed as an advisory tool requiring human educator review before any action is taken.

---

## 2. Introduction

Early identification of struggling students is a persistent challenge in school management. By the time poor academic performance is observed in assessment results, a student may have already fallen significantly behind. A predictive system that uses historical patterns — attendance, assignment completion, and prior scores — can help educators intervene earlier and more deliberately.

This project builds an end-to-end ML pipeline: from reproducible synthetic data generation through model training, evaluation, explainability, and a Streamlit-based inference dashboard.

---

## 3. Business Problem

Schools currently identify struggling students reactively — after their performance has already deteriorated. The goal is to shift this to a **proactive** model: predict before the next assessment which students are likely to score below a defined threshold, enabling educators to offer targeted support in advance.

**Constraints on use:**
- The system must not punish students automatically.
- The system must not make admission or grading decisions.
- Every prediction must include an explicit human-review disclaimer.

---

## 4. Objectives

1. Build a reproducible synthetic dataset reflecting realistic educational patterns.
2. Train and compare three ML approaches.
3. Evaluate using leakage-safe chronological splitting.
4. Prioritise recall to minimise missed at-risk students.
5. Provide per-prediction explanations.
6. Deploy a Streamlit inference dashboard.
7. Document responsible AI considerations throughout.

---

## 5. Dataset Description

| Attribute | Value |
|---|---|
| Type | Synthetic (reproducible, NumPy seed=42) |
| Total observations | 5,000 |
| Unique students | 1,000 |
| Assessment periods per student | 5 |
| At-risk prevalence | ~53.5% |
| Missing values | None (by construction) |
| Sensitive attributes | None included |

**Data generation assumptions:**
- Student "ability" sampled from Beta(2.5, 2.5) — centred distribution.
- Student "engagement" sampled from Beta(3.0, 2.0) — slightly right-skewed.
- Attendance is a function of engagement plus Gaussian noise, clipped to [20, 100].
- Assignment completion is correlated with both attendance and engagement.
- Missed assignments are inversely correlated with completion percentage.
- Scores are auto-regressive: each period's score depends on the prior period's score plus ability, attendance, and noise terms.
- Irreducible noise (σ=9) prevents the target from being trivially predictable.

---

## 6. Target Definition

**Target column:** `at_risk`

**Definition:** `at_risk = 1` if `next_assessment_score < 40`, else `0`.

**Why 40?** This is a project assumption representing a commonly understood threshold for academic difficulty. It is explicitly configurable and should be validated against real school policies in any real deployment.

**Point-in-time cutoff:** All features used to predict `at_risk` are available *before* the next assessment. `next_assessment_score` is never used as a model input.

---

## 7. Feature Engineering

| Feature | Engineering |
|---|---|
| `average_previous_score` | Mean of `previous_assessment_score` and `earlier_assessment_score` |
| `performance_trend` | `previous_assessment_score − earlier_assessment_score` |
| `assignment_completion_percentage` | Directly collected; represents effort/engagement |
| `missed_assignments` | Count of unsubmitted work |

**Features explicitly excluded:**
- `next_assessment_score` — future outcome (leakage)
- `at_risk` — the target label (leakage)
- `student_id` — identifier with no predictive meaning

---

## 8. EDA Summary

- **Attendance:** Right-skewed distribution. Many students cluster around 70–90%.
- **Assignment completion:** Broad distribution; correlated with attendance (r ≈ 0.6).
- **Previous scores:** Approximately normal around 55; at-risk students cluster below 40.
- **At-risk balance:** 53.5% at-risk / 46.5% not at-risk — moderate imbalance requiring `class_weight="balanced"`.
- **Performance trend:** Negatively trending students (trend < 0) have meaningfully higher at-risk rates.
- **Correlation:** `previous_assessment_score` and `average_previous_score` are the strongest individual predictors. No single feature perfectly separates the classes, ensuring the problem is non-trivial.

---

## 9. Data Preprocessing

1. **Range validation:** Percentage columns clipped to [0, 100]; scores clipped to [0, 100].
2. **Missing values:** Filled with column medians (none present in generated data, but handled defensively).
3. **Duplicate removal:** Any exact duplicate rows are dropped.
4. **Scaling:** `StandardScaler` applied inside the sklearn `Pipeline`, fitted only on training data.

---

## 10. Leakage Prevention

| Rule | Implementation |
|---|---|
| `next_assessment_score` excluded from X | Enforced in `FEATURE_COLUMNS` and tested |
| `at_risk` excluded from X | Enforced in `LEAKAGE_COLUMNS` and tested |
| `student_id` excluded from X | Enforced in `LEAKAGE_COLUMNS` and tested |
| Scaler fitted only on train | Scaler is inside the sklearn Pipeline; `fit()` called once on training data |
| Chronological splitting | Earlier periods → train; later periods → test |
| Test set untouched during selection | Test set evaluated only in `evaluate.py`, after model selection is final |

---

## 11. Model Methodology

### Chronological Split
- Periods 1–3 → Training set (3,000 observations)
- Period 4 → Validation set (1,000 observations) — used for model selection
- Period 5 → Test set (1,000 observations) — used for final evaluation only

**Why chronological?** Random splitting would allow the model to see future assessment information during training, inflating metrics and masking real generalisation to unseen time periods.

### Model 1: Logistic Regression
- Simple linear baseline.
- `class_weight="balanced"` to handle imbalance.
- Coefficients are directly interpretable.
- C=0.5 provides moderate L2 regularisation.

### Model 2: Random Forest
- Ensemble of 200 decision trees.
- `max_depth=8` and `min_samples_leaf=10` control overfitting.
- `class_weight="balanced"` for imbalance.
- Feature importances available via `feature_importances_`.

### Model 3: XGBoost
- Gradient-boosted trees, 200 estimators, learning rate 0.05.
- `scale_pos_weight` set to ratio of negative:positive in training set (~0.94).
- Subsample and colsample_bytree=0.8 for regularisation.

---

## 12. Baseline Comparison (Validation Set)

| Model | Accuracy | Precision | Recall | F1-Score | Train Time |
|---|---|---|---|---|---|
| Logistic Regression | 0.768 | 0.769 | 0.833 | 0.800 | 0.72s |
| **Random Forest** | **0.779** | **0.764** | **0.872** | **0.814** | 0.62s |
| XGBoost | 0.777 | 0.768 | 0.858 | 0.811 | 0.45s |

**Selected model:** Random Forest — highest F1 and highest recall on validation set.

---

## 13. Model Evaluation (Test Set — Final)

| Metric | Value |
|---|---|
| Accuracy | 0.763 |
| Precision | 0.766 |
| **Recall** | **0.849** |
| F1-Score | 0.805 |
| ROC-AUC | 0.846 |

The model correctly identifies **84.9% of genuinely at-risk students** (recall), which is the primary business objective.

---

## 14. Confusion Matrix Interpretation

|  | Predicted: Not At Risk | Predicted: At Risk |
|---|---|---|
| **Actual: Not At Risk** | TN = 273 | FP = 150 |
| **Actual: At Risk** | FN = 87 | TP = 490 |

- **True Positives (490):** At-risk students correctly flagged — educator can offer support.
- **True Negatives (273):** Not-at-risk students correctly cleared — no unnecessary intervention.
- **False Positives (150):** Students flagged unnecessarily. The educator checks in with a student who didn't strictly need it. Low harm — the worst outcome is an unnecessary but benign conversation.
- **False Negatives (87):** At-risk students the model missed. These students receive no proactive support. **This is the highest-cost error.** The model's high recall (0.849) means only 87 out of 577 at-risk students are missed.

**Business reasoning:** In an educational support context, the asymmetry of costs justifies accepting more False Positives to minimise False Negatives.

---

## 15. Error Analysis

The 87 false negatives likely correspond to:
- Students who have moderate attendance and completion rates but score poorly due to test anxiety, topic difficulty, or other unmeasured factors.
- Students near the 40-point threshold where the boundary is inherently uncertain.
- Students in their first assessment period where the model has less historical context.

Recommendations:
- Educators should treat the probability (not just the binary label) as continuous risk signal.
- Students with probabilities in the 35–65% range (medium risk) should receive a lighter check-in even if not formally "at risk".

---

## 16. Explainability

**Permutation Importance (model-agnostic):**
Feature importance is computed by measuring how much the model's F1-score drops when each feature's values are randomly permuted on the test set. Features with higher drops are more important.

**Key findings** (from actual permutation importance):
- `previous_assessment_score` and `average_previous_score` are consistently the strongest predictors — recent performance history is the most informative signal.
- `attendance_percentage` and `assignment_completion_percentage` contribute meaningfully.
- `performance_trend` captures whether a student is improving or declining.

**Important caveat:** Importance values reflect the model's learned correlations. They do NOT prove causation. A student's low attendance may correlate with at-risk status without being the root cause of their academic difficulty.

**Per-prediction explanations** are displayed in the Streamlit dashboard for each individual prediction, showing the top 3 contributing features and their values.

---

## 17. Application Architecture

```
User Input (Streamlit form)
        ↓
Input Validation (predict.py → validate_input)
        ↓
Feature Dictionary → pandas DataFrame
        ↓
Pipeline.predict_proba() [scaler + Random Forest]
        ↓
Probability → Risk Band → Explanation
        ↓
Streamlit Output (risk band, probability, top signals, disclaimer)
```

The Streamlit app:
- Loads the model once using `@st.cache_resource`.
- Loads metadata once using `@st.cache_data`.
- Never hardcodes metrics or predictions.
- Displays only results computed from the actual trained artifact.

---

## 18. Responsible AI

| Consideration | Implementation |
|---|---|
| Synthetic data disclosure | Labelled throughout; explicitly stated in dashboard |
| No sensitive attributes | No gender, caste, religion, disability, or ethnicity |
| No automatic actions | Every prediction requires human review |
| False positive impact | Unnecessary check-in — documented as low harm |
| False negative impact | Missed support — documented as high harm; recall prioritised |
| Distribution shift | Documented as a known limitation |
| Model uncertainty | Probability displayed; medium-risk band explicitly flagged |
| Human oversight | Mandatory disclaimer on every prediction |
| Privacy | No real student data used or required |
| Causation disclaimer | Explicitly stated in explainability section and dashboard |

---

## 19. Limitations

1. **Synthetic data:** The model has never seen real student data. Real performance may differ substantially.
2. **Configurable threshold:** The 40/100 threshold is an assumption. Schools may define academic risk differently.
3. **No unmeasured factors:** Mental health, socioeconomic status, language barriers, and home environment are not captured.
4. **No demographic parity analysis:** With synthetic data and no sensitive attributes, fairness across demographic groups cannot be evaluated.
5. **Static model:** The model is not retrained over time. Distribution shift will degrade accuracy if student patterns change.
6. **5-period horizon:** With only 5 periods per student, the model has limited historical depth per student.
7. **Single school context:** Results may not generalise across different school systems or curricula.

---

## 20. Conclusion

The Random Forest model achieves strong predictive performance (Recall=0.849, F1=0.805) on the held-out test set, correctly identifying 84.9% of at-risk students before their next assessment. The system is transparent, reproducible, and explicitly bounded to its advisory role. It provides per-prediction explanations, documented uncertainty, and a mandatory human-review disclaimer.

The pipeline is fully end-to-end: from reproducible data generation through training, evaluation, testing, and a live Streamlit dashboard. All 45 unit tests pass.

---

## 21. Future Improvements

1. **Real data integration:** Partner with schools to use properly anonymised, ethically governed real student data.
2. **Student-level time series modelling:** Use LSTM or Transformer architectures to model individual student trajectories more accurately.
3. **Fairness evaluation:** Measure and mitigate potential disparate impact across demographic groups once real data is available.
4. **Threshold calibration:** Use Platt scaling or isotonic regression to improve probability calibration.
5. **Active learning:** Allow educators to provide feedback on predictions to improve the model iteratively.
6. **Multi-subject modelling:** Model at-risk status per subject, not just overall, to give more targeted support recommendations.
7. **Confidence intervals:** Report uncertainty bands on probabilities, not just point estimates.
8. **Continuous retraining pipeline:** Implement automated periodic retraining as new assessment data arrives.

