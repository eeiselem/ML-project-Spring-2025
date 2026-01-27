# Machine Learning Project: Spring 2025

## Project Overview

This repository documents the application of statistical learning methods to three distinct real-world problems: regression, classification, and clustering. The objective was to implement an end-to-end machine learning pipeline for each task—covering data preprocessing, exploratory analysis, model selection, and performance evaluation.

## Repository Structure

* `Spring2025ProjectML.ipynb`: Jupyter Notebook containing the full codebase for data ingestion, processing, and modeling.
* `README.md`: Summary of methodology and key findings.

---

## Task 1: Regression (Student Grade Prediction)

**Objective:** Predict final student grades based on demographic and academic history.

### Data & Preprocessing

* **Dataset:** Student Performance Data (395 samples, 31 features).
* **Cleaning:** Addressed 15 null values. I used the inner 90% range to detect outliers rather than the standard interquartile range to preserve critical data points. Notably, I retained outliers in the `Fjob_health` category as they represented valid, significant signals.
* **Feature Engineering:** Features were scaled specifically for Lasso regression; other models used raw feature sets where appropriate.

### Analysis & Modeling

Exploratory analysis indicated that **past class failures** and **mother's education (Medu)** had the strongest initial correlation with performance. However, feature importance analysis from the Decision Tree model identified **absences** and **failures** as the primary drivers of the target variable.

I evaluated four distinct model architectures:

1. **Lasso Regression:** Optimized  at 0.188 (CV).
2. **Decision Tree:** Pruned to avoid overfitting (208  8 terminal nodes); optimal  = 0.578.
3. **Boosting:** Optimized at 37 estimators.
4. **Bagging:** Optimized at 44 estimators.

### Key Results

| Model | R² Score | Performance Notes |
| --- | --- | --- |
| **Bagging** | **0.467** | **Best Performance.** |
| Decision Tree | 0.358 | Significant pruning required to generalize. |
| Boosting | 0.276 | Underperformed relative to Bagging. |
| Lasso | 0.229 | MSE: 19.5. |

**Conclusion:** The Bagging ensemble provided the most robust predictions. The analysis confirms that behavioral metrics (attendance and prior failures) are stronger predictors of success than demographic factors alone.

---

## Task 2: Classification (Obesity Level Estimation)

**Objective:** Classify individuals into obesity levels based on dietary habits and physical condition.

### Data & Preprocessing

* **Dataset:** Obesity Level Estimation (2112 samples, 17 features).
* **Cleaning:** Removed 6 records containing null values. I analyzed outliers in `Weight`, `FCVS` (vegetable consumption), and `CH2O` (water intake) but determined they were valid extreme cases and retained them to maintain model robustness.

### Modeling Status

* *Pending final evaluation of Logistic Regression and KNN performance.*

---

## Task 3: Clustering (California Wildfire Analysis)

**Objective:** Uncover patterns in wildfire occurrences using unsupervised learning techniques.

### Methodology

* Applied **KMeans** and **Hierarchical Clustering** to identify geographical or intensity-based groupings in the data.

### Modeling Status

* *Analysis in progress.*

---

## Technical Stack

* **Language:** Python 3.x
* **Core Libraries:** `pandas`, `numpy`, `scikit-learn`
* **Visualization:** `matplotlib`, `seaborn`

## Usage

To replicate this analysis:

1. Clone the repository.
2. Ensure source datasets are located in the root directory.
3. Run the `Spring2025ProjectML.ipynb` notebook.
