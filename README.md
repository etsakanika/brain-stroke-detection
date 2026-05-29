<div align="center">

# 🧠 Brain Stroke Risk Prediction
### Interpretable Machine Learning for Healthcare Screening


[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![imbalanced-learn](https://img.shields.io/badge/imbalanced--learn-SMOTE-blueviolet)](https://imbalanced-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

</div>

---

## Headline Result

> **A Logistic Regression model with balanced class weights identifies 82% of stroke cases** on the held-out test set (recall = 0.82, ROC-AUC = 0.83). The trade-off: precision is low (0.13) because the dataset is heavily imbalanced (only 4.9% positive cases). In a clinical screening context, prioritising recall over precision is the correct design choice — missed strokes carry far higher cost than false alarms.

<div align="center">

| Metric | Validation Set | Test Set |
|:---|:---:|:---:|
| **Recall (Stroke = 1)** | **0.76** | **0.82** |
| Precision (Stroke = 1) | 0.13 | 0.13 |
| F1 (Stroke = 1) | 0.22 | 0.22 |
| ROC-AUC | 0.83 | 0.83 |
| Accuracy | 0.74 | 0.72 |

*Note: in imbalanced healthcare screening, accuracy is the least informative metric - a classifier that always predicts "no stroke" would reach ~95% accuracy while being clinically useless.*

</div>

---

## Table of Contents

1. [Motivation](#-motivation)
2. [Dataset](#-dataset)
3. [Methodology](#-methodology)
4. [Key Findings](#-key-findings)
5. [Quick Start](#-quick-start)
6. [Repository Structure](#-repository-structure)
7. [Limitations and Future Work](#-limitations-and-future-work)

---

## Motivation

Stroke is among the leading causes of death and long-term disability worldwide. Early risk stratification allows clinicians to intervene with lifestyle changes, monitoring, or preventive treatment before an event occurs.

This project addresses a specific question often raised in clinical-decision-support work:

> *Can a simple, interpretable model identify patients at elevated stroke risk reliably enough to support a screening workflow — and if so, what features drive its predictions?*

The emphasis here is intentional: **interpretability over raw accuracy**. A clinician needs to understand *why* a model flags a patient, not just trust an opaque score.

---

## Dataset

[**Stroke Prediction Dataset**](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset) — 5,110 patient records with 11 demographic, lifestyle, and clinical features. Target variable `stroke` is binary (0 = no event, 1 = stroke occurred).

| Property | Value |
|---|---|
| Total records | 5,110 |
| Positive class rate | **4.87%** (249 strokes) |
| Imbalance ratio | ≈ 19.5 : 1 |
| Missing values | 201 in `bmi` (handled via mean imputation in pipeline) |

**Feature groups:**
- **Demographic** — `age`, `gender`, `ever_married`, `Residence_type`
- **Clinical** — `hypertension`, `heart_disease`, `avg_glucose_level`, `bmi`
- **Lifestyle** — `work_type`, `smoking_status`

---

## Methodology

### Experimental Protocol

| Decision | Choice | Rationale |
|---|---|---|
| Split strategy | 70% train / 15% validation / 15% test, **stratified** | Preserves the 4.87% positive rate across all splits |
| Random seed | Fixed (42) | Reproducibility |
| Preprocessing | sklearn `Pipeline` + `ColumnTransformer` | No data leakage; fit only on train |
| Numerical features | Mean imputation + `StandardScaler` | Handles missing `bmi`, normalises scale |
| Categorical features | Most-frequent imputation + `OneHotEncoder` | Standard handling of categorical inputs |
| Imbalance handling | `class_weight="balanced"` and SMOTE (compared) | Two complementary approaches |
| Primary metric | **Recall on positive class** | Missed stroke = worst possible error |
| Supporting metrics | Precision, F1, ROC-AUC, confusion matrix | Full clinical picture |

### Models Compared

| Model | Why included |
|---|---|
| **Logistic Regression** + `class_weight="balanced"` | Strong, interpretable baseline; coefficients directly explainable |
| **Logistic Regression** + **SMOTE** | Tests whether synthetic oversampling improves recall over reweighting |

Logistic regression was selected over more complex algorithms (Random Forest, XGBoost) because **interpretability is the priority** — the coefficients translate directly into odds ratios that clinicians can audit. The downstream goal is decision support, not maximum predictive accuracy on a held-out test set.

---

## Key Findings

### 1. SMOTE and class weighting are interchangeable on this dataset
Both approaches converge to essentially identical performance (recall ≈ 0.76, F1 ≈ 0.22 on validation). The simpler `class_weight="balanced"` was retained as the final model — fewer moving parts, easier to deploy.

### 2. Five features dominate the model
Coefficient analysis identifies **age, hypertension, heart disease, average glucose level, and smoking status** as the strongest positive predictors of stroke. This aligns with the established clinical literature on stroke risk factors — a useful sanity check that the model has learned medically plausible patterns.

### 3. Test-set recall reaches 0.82
On the held-out test set, the final model correctly identifies **31 out of 38 actual stroke cases** (82% sensitivity). The 7 missed cases are the most important error category to study in any deployment context.

### 4. Precision is intrinsically low — and that's expected
With 4.87% prevalence, even a perfectly-calibrated model will produce many false positives at the threshold needed to catch most true cases. **This is a feature of the screening problem, not a model defect.** A clinical workflow would treat positive predictions as "send for further evaluation", not as a diagnosis.

### 5. Accuracy is a misleading metric here
A trivial baseline that always predicts "no stroke" would reach ~95% accuracy on this dataset while catching zero strokes. This project deliberately reports recall, F1, and ROC-AUC as the meaningful metrics.

---

## Quick Start

### Option A - Run in Google Colab

1. Upload `Brain_Stroke_Detection.ipynb` and `brain_stroke.csv` to Colab
2. Runtime → Run all (no GPU required)

### Option B - Run locally

```bash
# Clone
git clone https://github.com/etsakanika/Brain-Stroke-Detection.git
cd Brain-Stroke-Detection

# Set up environment
python -m venv .venv
source .venv/bin/activate          # macOS/Linux
# .venv\Scripts\activate           # Windows
pip install -r requirements.txt

# Launch
jupyter notebook Brain_Stroke_Detection.ipynb
```

The dataset (`brain_stroke.csv`) is included in the repository — no external download required.

---

## Repository Structure

```
Brain-Stroke-Detection/
│
├── 📓 Brain_Stroke_Detection.ipynb    # Main notebook — EDA, modeling, evaluation
├── 📊 brain_stroke.csv                # Dataset (5,110 records)
├── 📄 README.md                        # This file
├── 📄 requirements.txt                 # Pinned dependencies
└── 📄 .gitignore                       # Excludes venv, checkpoints
```

---

## License

Released under the [MIT License](LICENSE). Free to use, modify, and distribute with attribution.

<div align="center">
