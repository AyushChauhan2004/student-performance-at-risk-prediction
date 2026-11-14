# Student Performance & At-Risk Prediction
**Algions AI Labs Private Limited | AI & Machine Learning**

> ⚠️ **Advisory Tool Only.** This system must never be used to automatically punish students, assign grades, or make high-stakes academic decisions. All predictions require human review by a qualified educator.

---

## Table of Contents
1. [Project Overview](#1-project-overview)
2. [Problem Definition](#2-problem-definition)
3. [Dataset](#3-dataset)
4. [Environment Setup](#4-environment-setup)
5. [Running the Pipeline](#5-running-the-pipeline)
6. [Running the Application](#6-running-the-application)
7. [Running Tests](#7-running-tests)
8. [Project Structure](#8-project-structure)
9. [Limitations](#9-limitations)

---

## 1. Project Overview

This project builds an end-to-end machine learning pipeline that predicts whether a student is likely to score **below 40 out of 100** in their next scheduled academic assessment.

The goal is to give educators an early-warning signal so they can offer targeted support **before** a student's performance deteriorates — not after.

**Three models are trained and compared:**
- Logistic Regression (interpretable baseline)
- Random Forest Classifier (selected model)
- XGBoost Classifier

---

## 2. Problem Definition

| Attribute | Value |
|---|---|
| Prediction unit | Student at a given assessment period |
| Target | `next_assessment_score < 40` |
| Positive class | At Risk (1) |
| Negative class | Not At Risk (0) |
| Prediction horizon | Next scheduled assessment |
| Prediction timestamp | Before the next assessment takes place |
| At-risk threshold | 40 / 100 (configurable assumption) |

**Why recall matters:** A False Negative (missed at-risk student) means no support is offered to a student who needs it. This is the most costly error. We prioritise recall over raw accuracy.

---

## 3. Dataset

- **Type:** Synthetic (reproducible, seed=42)
- **Rows:** 5,000 (1,000 students × 5 assessment periods)
- **Target balance:** ~53% at-risk / ~47% not at-risk
- **No real student data.** No sensitive/protected attributes (gender, caste, religion, disability, ethnicity).

**Features used by the model:**

| Feature | Description |
|---|---|
| `assessment_period` | Chronological assessment number |
| `attendance_percentage` | % of classes attended |
| `assignment_completion_percentage` | % of assignments submitted |
| `previous_assessment_score` | Most recent completed score |
| `earlier_assessment_score` | Score before the most recent |
| `missed_assignments` | Count of unsubmitted assignments |
| `average_previous_score` | Mean of previous two scores |
| `performance_trend` | previous_score − earlier_score |

**Excluded from model (leakage prevention):**
- `next_assessment_score` — this is the future outcome we predict
- `at_risk` — this is the target label
- `student_id` — identifier, not a predictive signal

---

## 4. Environment Setup

### Prerequisites
- Python 3.9+ (tested on 3.14.6)
- pip

### Install dependencies
```bash
pip install -r requirements.txt
```

No virtual environment is strictly required, but one is recommended for isolation:
```bash
python -m venv venv
venv\Scripts\activate      # Windows
pip install -r requirements.txt
```

---

## 5. Running the Pipeline

Run these commands from the **project root** (`student-performance-at-risk/`).

### Step 1 — Generate the dataset
```bash
python src/generate_data.py
```
Output: `data/raw/student_assessments.csv` (5,000 rows)

### Step 2 — Train models
```bash
python src/train.py
```
Output:
- `models/student_risk_model.pkl` — best model pipeline
- `models/model_metadata.json` — training metadata and validation metrics

### Step 3 — Evaluate on test set
```bash
python src/evaluate.py
```
Output:
- Metrics printed to terminal
- `reports/figures/confusion_matrix.png`
- `reports/figures/permutation_importance.png`
- `reports/figures/model_comparison.png`
- Test metrics appended to `models/model_metadata.json`

### Step 4 — Run a sample inference
```bash
python src/predict.py
```

---

## 6. Running the Application

```bash
streamlit run app/streamlit_app.py
```

The dashboard opens at `http://localhost:8501`.

**Sections:**
- 🏠 Home / Overview — purpose, limitations, ethical notice
- 🔍 Student Risk Prediction — input student data, get a prediction
- 📊 Model Performance — real metrics, confusion matrix, feature importance
- ℹ️ Model Information — model details, feature list, responsible AI statement

---

## 7. Running Tests

```bash
python -m pytest tests/ -v
```

**45 tests** covering:
- Dataset reproducibility and schema validation
- Target construction correctness
- Leakage prevention (3 explicit checks)
- Chronological splitting
- Missing value handling
- Model artifact loading
- Inference output structure
- Risk band and probability validity
- End-to-end high-risk and low-risk student scenarios
- Input validation (6 error cases)

---

## 8. Project Structure

```
student-performance-at-risk/
│
├── data/
│   ├── raw/                        ← generated CSV
│   └── processed/                  ← train/val/test splits
│
├── notebooks/
│   └── 01_eda_and_experiments.ipynb
│
├── src/
│   ├── generate_data.py            ← synthetic dataset generator
│   ├── preprocessing.py            ← validation, splitting, pipeline
│   ├── train.py                    ← model training and selection
│   ├── evaluate.py                 ← test-set evaluation and figures
│   ├── explain.py                  ← feature importance and explanation
│   └── predict.py                  ← single-observation inference
│
├── models/
│   ├── student_risk_model.pkl      ← saved pipeline artifact
│   ├── model_metadata.json         ← training + evaluation metadata
│   └── permutation_importance.json ← feature importance scores
│
├── app/
│   └── streamlit_app.py            ← Streamlit dashboard
│
├── tests/
│   ├── test_data.py                ← 15 dataset tests
│   ├── test_preprocessing.py       ← 12 preprocessing tests
│   └── test_inference.py           ← 18 inference tests
│
├── reports/
│   ├── figures/                    ← saved plots
│   └── technical_report.md         ← full technical documentation
│
├── requirements.txt
├── README.md
├── .gitignore
└── .env.example
```

---

## 9. Limitations

1. **Synthetic data** — trained on artificial data; real-world performance is unknown.
2. **Threshold** — the 40/100 at-risk threshold is a project assumption, not a universal standard.
3. **No sensitive attributes** — the model cannot account for factors like disability, socioeconomic status, or language barriers that may explain attendance or completion patterns.
4. **Distribution shift** — if student behaviour patterns change substantially, model accuracy may degrade without retraining.
5. **Causation** — the model identifies statistical correlations. It cannot explain *why* a student might struggle.
6. **Mandatory human review** — this system is advisory only. Educators must review all flagged students before taking any action.

---

*Generated with AI assistance. The student understands the implementation and can explain all design decisions independently.*

