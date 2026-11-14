# Technical Viva & Interview Preparation Guide
## Student Performance & At-Risk Prediction
**Algions AI Labs Private Limited | AI & Machine Learning**

*All answers are grounded in the actual codebase, model choices, and experimental results.*

---

## Part 1: 20 Technical Viva Questions & Answers

**Q1. Why did you define at-risk as scoring below 40?**
> A: 40 is a commonly accepted passing threshold in school academic systems. It is an explicit, configurable project assumption documented in `model_metadata.json` and `generate_data.py` (`AT_RISK_THRESHOLD = 40`), rather than an unchangeable universal truth. If a school uses 50, only that single parameter needs updating.

**Q2. Why is chronological splitting used instead of random k-fold or random train_test_split?**
> A: Educational assessment data is inherently temporal. Students progress through assessment periods 1, 2, 3, 4, 5 sequentially. Random splitting would allow period-5 data from some students to train a model predicting period-2 data for others, causing look-ahead leakage and overly optimistic metrics. Chronological splitting (periods 1–3 for train, period 4 for validation, period 5 for test) exactly mirrors real deployment where historical data predicts future outcomes.

**Q3. What constitutes data leakage in this project, and how is it prevented?**
> A: Data leakage happens when information unavailable at inference time contaminates training. We prevent it via 4 mechanisms:
> 1. `next_assessment_score` is strictly excluded from features (it is the future outcome).
> 2. `at_risk` is excluded from features (it is the target).
> 3. `student_id` is excluded (it is an arbitrary identifier).
> 4. `StandardScaler` is fitted *only* on the training split inside an sklearn `Pipeline`.

**Q4. Why did you select Random Forest over Logistic Regression and XGBoost?**
> A: On the validation set (Period 4), Random Forest achieved the highest F1-score (0.8144) and the highest recall (0.8723). For school interventions, recall is paramount because missing a struggling student is far more damaging than a false alarm. Random Forest also offers robust non-linear boundaries and stable feature importances without severe sensitivity to scaling.

**Q5. Why did you optimize for F1-score and Recall instead of Accuracy?**
> A: Accuracy is misleading when costs of error types are asymmetrical. Predicting everyone as "not at risk" in an unbalanced or shifting class setting yields artificial accuracy while failing the core school objective. Recall directly measures the percentage of genuinely at-risk students flagged, and F1 ensures precision does not collapse.

**Q6. What does a Recall of 0.849 on the test set mean in plain English?**
> A: It means that out of every 100 students who actually scored below 40 in assessment period 5, the model successfully identified approximately 85 of them in advance, giving educators the opportunity to intervene early.

**Q7. Explain the business trade-off between False Positives and False Negatives here.**
> A: 
> - **False Positive (FP):** A student is flagged as at-risk but manages to score >= 40. The teacher has a supportive check-in or provides extra practice. The consequence is low harm and potentially positive reinforcement.
> - **False Negative (FN):** An at-risk student is missed, receives no extra help, and fails. The consequence is high harm. Therefore, we deliberately accept a higher FP rate to minimize FN.

**Q8. What is Permutation Feature Importance, and why choose it over basic tree feature importance?**
> A: Default tree MDI (Mean Decrease in Impurity) can overestimate high-cardinality or continuous features and reflects training fit. Permutation importance shuffles each feature on the held-out test set and records the resulting drop in F1-score. It is model-agnostic, computed on unseen data, and reflects actual generalization reliance.

**Q9. Why bundle the scaler and classifier inside an sklearn Pipeline?**
> A: It encapsulates the feature transformations and classifier into a single executable object. During serialization (`student_risk_model.pkl`), the fitted scaling parameters (mean and variance from training data) are stored. In inference, raw user inputs can be passed directly to `pipeline.predict_proba()` without risking manual scaling errors or test set contamination.

**Q10. Why is student_id excluded as a feature?**
> A: Student ID is a nominal identifier. If included, tree models or linear models could memorize specific student IDs from the training set, causing high overfitting and complete failure when predicting for newly enrolled students.

**Q11. How do you handle class balance during training?**
> A: In Logistic Regression and Random Forest, we use `class_weight="balanced"`, which weights samples inversely proportional to class frequencies. In XGBoost, we compute `scale_pos_weight = neg_count / pos_count`. This prevents the estimators from favoring the negative class.

**Q12. What does performance_trend capture?**
> A: `performance_trend = previous_assessment_score - earlier_assessment_score`. It captures velocity and direction: whether a student is on an upward trajectory or in a downward spiral. A student with a 45 who dropped from 65 is in much greater danger than a student with a 45 who rose from 30.

**Q13. How does input validation work in predict.py and the Streamlit app?**
> A: `validate_input()` checks that percentages (`attendance_percentage`, `assignment_completion_percentage`) and assessment scores are strictly within [0, 100], `missed_assignments` >= 0, and `assessment_period` >= 1. Any violating input raises a clear `ValueError`, preventing erroneous data from corrupting predictions.

**Q14. What does the test set ROC-AUC of 0.846 indicate?**
> A: ROC-AUC evaluates the ranking capability of the model across all possible classification thresholds. A score of 0.846 means there is an 84.6% probability that a randomly chosen at-risk student is assigned a higher predicted risk probability than a randomly chosen not-at-risk student.

**Q15. How did you ensure the synthetic dataset is realistic and not trivially separable?**
> A: We incorporated latent variables (student ability and engagement), auto-regressive score evolution (each score depends on the prior score plus attendance and noise), and an irreducible Gaussian noise term (sigma = 9). This prevents any single feature from achieving 99% accuracy and produces realistic boundary overlap.

**Q16. What steps would you take to adapt this system to real school records?**
> A: 
> 1. Partner with academic administrators to anonymize PII under FERPA/GDPR guidelines.
> 2. Validate the cutoff threshold against the school's historical grade distribution.
> 3. Perform fairness audits across demographic slices to detect disparate impact.
> 4. Establish an automated continuous monitoring pipeline for concept drift.

**Q17. Why engineer average_previous_score in addition to the individual assessment scores?**
> A: Individual assessment scores are susceptible to single-day noise (illness, bad day). The rolling average smooths short-term fluctuations to capture sustained baseline capability.

**Q18. What is the unit of prediction in this system?**
> A: The prediction unit is a **student-assessment period observation**. Each row represents one student at a specific point in time immediately prior to their next scheduled assessment.

**Q19. What is the ethical requirement for prediction output?**
> A: Every prediction must present an explicit human-in-the-loop disclaimer: the system is purely advisory, probabilities are statistical estimates rather than guarantees, and predictions must never trigger automatic sanctions, grading, or disciplinary actions.

**Q20. How would you detect model degradation in production?**
> A: By tracking actual outcomes after each exam period and calculating running Precision, Recall, and F1. We would also compute Population Stability Index (PSI) or Kolmogorov-Smirnov (KS) tests on input feature distributions (e.g., sudden attendance policy shifts) to catch data drift before performance drops.

---

## Part 2: 10 Architecture Questions & Answers

**A1. Explain the directory and module layout of the project.**
> A: The structure isolates responsibilities cleanly:
> - `data/raw/` & `data/processed/`: Reproducible data storage and splits.
> - `src/`: Modular logic (`generate_data.py`, `preprocessing.py`, `train.py`, `evaluate.py`, `explain.py`, `predict.py`).
> - `models/`: Exported pipeline artifact (`.pkl`) and metadata (`.json`).
> - `app/`: Streamlit interactive dashboard.
> - `tests/`: 45 automated unit tests with `conftest.py`.
> - `reports/`: Markdown reports and saved evaluation figures.

**A2. How does the Streamlit app communicate with the trained model?**
> A: The app imports `load_model()` which uses `joblib.load(MODEL_PATH)` cached via `@st.cache_resource`. When a user submits student details, the app formats them into a DataFrame aligned with `FEATURE_COLUMNS`, passes them through `pipeline.predict_proba()`, and passes the outputs to `explain_prediction()`.

**A3. Why cache artifacts with `@st.cache_resource` and `@st.cache_data` in Streamlit?**
> A: Streamlit re-executes the script upon every UI interaction. `@st.cache_resource` keeps the serialized pipeline in memory across reruns, while `@st.cache_data` avoids repeatedly re-reading `model_metadata.json` and `permutation_importance.json` from disk, keeping inference sub-second.

**A4. How is the feature schema strictly enforced across training, testing, and inference?**
> A: `FEATURE_COLUMNS` is defined as a single source of truth in `src/preprocessing.py`. Both `train.py`, `evaluate.py`, `predict.py`, and `test_inference.py` import `FEATURE_COLUMNS` to slice DataFrames in identical order.

**A5. What is the purpose of `model_metadata.json`?**
> A: It decouples model reporting from hardcoded values. It captures selected model architecture, validation scores, held-out test metrics, training dates, sample sizes, and responsible AI statements, allowing the Streamlit UI to dynamically display true audit telemetry.

**A6. Why use `conftest.py` instead of per-file `sys.path.append`?**
> A: `conftest.py` is automatically recognized by pytest at the project root, placing `src/` onto Python's module search path before test discovery. This prevents duplicate path hacks across test modules.

**A7. How would you expose this model as a REST API?**
> A: Wrap `predict_single()` in a FastAPI application with a Pydantic `StudentInputSchema`. The endpoint `/api/v1/predict` would accept JSON inputs, validate types and ranges, run pipeline inference, and return the risk band, probability, and top signals.

**A8. How do you prevent overfitting in the Random Forest model?**
> A: We set `max_depth=8` (limiting maximum tree depth) and `min_samples_leaf=10` (requiring at least 10 observations in any leaf node). This prevents individual trees from isolating individual training data points.

**A9. If a new feature (e.g., homework_hours) is added, what must happen?**
> A:
> 1. Add `homework_hours` to `FEATURE_COLUMNS` in `src/preprocessing.py`.
> 2. Regenerate or update the raw dataset.
> 3. Re-run `src/train.py` so the `StandardScaler` and classifier learn the new feature dimension.
> 4. Re-run `src/evaluate.py` to regenerate metrics and permutation plots.
> 5. Update input widgets in `app/streamlit_app.py`.

**A10. What ensures reproducibility across different machines?**
> A: A fixed random seed (42) in NumPy data generation and Scikit-Learn/XGBoost estimators, explicit `requirements.txt` dependency specifications, and version-controlled scripts that reproduce the dataset and models deterministically from scratch.

---

## Part 3: 10 ML Fundamentals Questions & Answers

**F1. What is the difference between Bagging and Boosting?**
> A: Bagging (e.g., Random Forest) trains independent base learners in parallel on bootstrap samples and averages their outputs, primarily reducing variance. Boosting (e.g., XGBoost) trains base learners sequentially, where each new tree focuses on the residual errors of preceding trees, reducing both bias and variance.

**F2. What is Log-Loss (Binary Cross-Entropy)?**
> A: The objective function measuring discrepancy between predicted probability $p$ and actual binary label $y \in \{0, 1\}$:
> $$\mathcal{L} = - [y \log(p) + (1-y) \log(1-p)]$$
> It heavily penalizes confident incorrect predictions.

**F3. Why does Logistic Regression require feature scaling while Random Forest does not?**
> A: Logistic Regression relies on gradient descent or quasi-Newton optimization where unscaled features create elongated loss contours and unequal gradient updates. Furthermore, L2 regularization penalizes coefficients uniformly, unfairly penalizing larger-scale features. Decision trees only evaluate monotonic split thresholds along single features, making them invariant to scale.

**F4. Explain the Bias-Variance tradeoff.**
> A: Total error comprises Bias² (error from erroneous model assumptions / underfitting), Variance (error from sensitivity to training set fluctuations / overfitting), and Irreducible Noise. Increasing model complexity decreases bias but increases variance; regularizers manage this sweet spot.

**F5. How does `scale_pos_weight` work in XGBoost?**
> A: It controls the balance of positive and negative weights via $\frac{\text{negative count}}{\text{positive count}}$. It scales the gradient and hessian of the positive instances, forcing boosting iterations to treat positive mistakes with higher penalty.

**F6. What is the difference between ROC-AUC and PR-AUC?**
> A: ROC plots True Positive Rate vs False Positive Rate. PR plots Precision vs Recall. When the negative class is vast (extreme imbalance), False Positive Rate can stay small despite many false alarms; PR-AUC is more sensitive to false positives in severe class imbalance.

**F7. What is the difference between L1 and L2 regularization?**
> A: L1 (Lasso) adds the sum of absolute weights ($\lambda \sum |w_i|$), driving weights to exact zeros (feature selection). L2 (Ridge) adds the sum of squared weights ($\lambda \sum w_i^2$), shrinking weights smoothly towards zero without setting them to exact zero.

**F8. What is data drift versus concept drift?**
> A: Data drift (covariate shift) occurs when $P(X)$ changes while $P(Y|X)$ remains constant (e.g., student attendance distributions shift). Concept drift occurs when the underlying relationship $P(Y|X)$ changes (e.g., passing threshold or exam difficulty changes).

**F9. What is a confusion matrix?**
> A: A $2 \times 2$ table cross-tabulating ground-truth binary labels against predicted labels, delineating True Positives (TP), True Negatives (TN), False Positives (FP), and False Negatives (FN).

**F10. Why is evaluating on validation data necessary before touching the test set?**
> A: If hyperparameter tuning or model selection is performed on the test set, information from the test set implicitly guides model selection decisions. The validation set acts as a tuning testbed; the final test set remains completely unseen until evaluation.

---

## Part 4: 5 Debugging & Change Exercises

**D1. Scenario: Adding a school causes test recall to plummet from 0.85 to 0.58.**
> - **Cause:** Covariate shift or differing grading practices at the new school.
> - **Action:** Check distributions of `previous_assessment_score` and attendance for the new school. If grading rubrics differ, introduce school-level normalization or include a school indicator variable.

**D2. Scenario: A student with an 85% attendance rate and a 65 previous score is flagged as "Medium Risk". Is this an error?**
> - **Cause:** Check `earlier_assessment_score` and `performance_trend`. If earlier score was 90, `performance_trend` is -25 and `missed_assignments` may be elevated. The multi-feature interaction correctly detects rapid decline.

**D3. Scenario: The threshold for at-risk is changed from 40 to 45.**
> - **Action:** In `src/generate_data.py`, modify `AT_RISK_THRESHOLD = 45`. Re-run `python src/generate_data.py`, `python src/train.py`, and `python src/evaluate.py`. The pipeline is parameter-driven and will update all datasets, models, and metadata automatically.

**D4. Scenario: XGBoost fails with `ValueError: feature_names mismatch`.**
> - **Cause:** Columns in the inference DataFrame were passed in a different order or with extra columns.
> - **Action:** In `src/predict.py`, always enforce `df = df[FEATURE_COLUMNS]` before passing to `pipeline.predict()`.

**D5. Scenario: The Streamlit app displays "N/A" across all test metrics.**
> - **Cause:** `model_metadata.json` is missing `test_metrics` because `evaluate.py` has not been executed.
> - **Action:** Run `python src/evaluate.py`, which evaluates the model on the test split and writes the results to `model_metadata.json`.

---

## Part 5: 3–5 Minute Demonstration Script

- **[0:00 - 0:45] Problem & Goal:**
  > "Hello. Today I am presenting our Student Performance & At-Risk Prediction system developed for school management. The core problem is that schools typically recognize struggling students only after an exam has been failed. Our system acts as an early-warning advisory tool, predicting whether a student is at risk of scoring below 40 in their next assessment using data available before the exam."

- **[0:45 - 1:30] Pipeline & Methodology:**
  > "Our pipeline uses 5,000 synthetic observations across 5 sequential periods. To prevent data leakage, we strictly use chronological splitting: periods 1–3 for training, period 4 for validation, and period 5 for testing. We benchmarked Logistic Regression, Random Forest, and XGBoost. Random Forest won with a validation F1 of 0.814 and a Recall of 0.872. On the final test set, it caught 84.9% of all genuinely at-risk students."

- **[1:30 - 2:30] Live Streamlit Demonstration:**
  > *(Switch to browser at localhost:8501)*
  > "Here is our Streamlit dashboard. 
  > Under 'Student Risk Prediction', let's test a student who had 62% attendance, missed 4 assignments, and had a previous score of 38 declining from 48.
  > Clicking 'Predict Risk' gives an estimated at-risk probability of ~70% — categorized in the High Risk band. 
  > Crucially, the app explains *why*: highlighting the declining trend and prior scores. It suggests actionable support considerations, and displays our mandatory human-review disclaimer."

- **[2:30 - 3:00] Verification & Testing:**
  > "We have 45 automated unit tests in `pytest` covering data integrity, target construction, point-in-time leakage prevention, and inference validation. All 45 tests pass cleanly."

- **[3:00 - 3:30] Responsible AI & Conclusion:**
  > "To ensure Responsible AI, no sensitive protected characteristics like gender or ethnicity are used. The tool is explicitly bounded as an advisory support tool that requires human educator oversight. Thank you!"
