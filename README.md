# Machine Learning Project: Spring 2025

## Project Overview

This repository documents the application of statistical learning methods to three distinct real-world problems: regression, classification, and clustering. The objective was to implement an end-to-end machine learning pipeline for each task covering data preprocessing, exploratory analysis, model selection, and performance evaluation.

## Repository Structure

* `Spring2025ProjectML.ipynb`: Jupyter Notebook containing the full codebase for data ingestion, processing, and modeling.
* `README.md`: Summary of methodology and key findings.

---

## Task 1: Regression (Student Grade Prediction)

**Objective:** Predict final student grades (`G3`) based on demographic and academic history.

### Data & Preprocessing

* **Dataset:** Student Performance Data (395 samples, 31 features).
* **Cleaning:** Handled missing values and used the 5th-95th percentile range to detect and handle outliers, preserving critical data points. Converted categorical variables into numerical values using one-hot encoding.
* **Feature Engineering:** Features were standardized for linear models; ensemble methods used raw feature sets where appropriate.

### Analysis & Modeling

Exploratory analysis and Decision Tree feature importance identified **past class failures** and **absences** as the primary drivers of the target variable.

I evaluated four distinct model architectures:

1. **Multiple Linear Regression:** Baseline model (MSE: 17.24).
2. **Lasso Regression:** Optimized  via cross-validation. (R²: ~0.21).
3. **Decision Tree:** Required aggressive pruning (cost complexity pruning) to prevent overfitting, reducing the tree to 3 terminal nodes.
4. **Ensemble Methods (Bagging & Boosting):** Tuned estimators to maximize predictive power.

### Key Results

| Model | R² Score | Performance Notes |
| --- | --- | --- |
| **Bagging** | **0.36** | **Best Performance.** |
| Boosting | 0.22 | performed comparably to Lasso. |
| Lasso | 0.21 | Effective for feature selection but lower predictive power. |

**Conclusion:** The Bagging ensemble provided the most robust predictions, outperforming individual decision trees and linear regularization methods.

---

## Task 2: Classification (Obesity Level Estimation)

**Objective:** Classify individuals into obesity levels (7 categories) based on dietary habits and physical condition.

### Data & Preprocessing

* **Dataset:** Obesity Level Estimation (2112 samples, 17 features).
* **Cleaning:** Addressed null values and analyzed outliers in `Weight`, `FCVS` (vegetable consumption), and `CH2O`. Target variables were encoded ordinally, and nominal features were dummified.
* **Analysis:** Chi-square contingency tests revealed significant associations between obesity levels and features like `Family History`, `FAVC` (high-caloric food consumption), and `Transportation Method`.

### Modeling & Results

I compared a distance-based algorithm against a linear probabilistic model:

1. **Multinomial Logistic Regression:**
* Hyperparameters tuned using `GridSearchCV`.
* **Accuracy:** **96%**
* **Insight:** Proved highly effective at distinguishing between the 7 distinct weight classes.


2. **K-Nearest Neighbors (KNN):**
* Optimized at `k=5`.
* **Accuracy:** 81%
* **Insight:** Struggled slightly with the high dimensionality compared to Logistic Regression.



**Conclusion:** Logistic Regression was the superior model for this dataset, achieving near-perfect classification performance.

---

## Task 3: Clustering (California Wildfire Analysis)

**Objective:** Uncover patterns in wildfire occurrences and damage severity using unsupervised learning techniques.

### Data & Preprocessing

* **Dataset:** California Wildfire Incidents.
* **Cleaning:** Extensive cleaning required. Dropped columns with >50% missing data, filled missing numerical values, and standardized the dataset. Mapped damage descriptions to an ordinal scale.
* **Dimensionality Reduction:** Applied **PCA (Principal Component Analysis)** to visualize the high-dimensional data in 2D space and analyze variance.

### Clustering Results

1. **KMeans Clustering:**
* Utilized the **Elbow Method** to determine the optimal number of clusters.
* Successfully grouped incidents based on geographical location and damage severity.


2. **Hierarchical Clustering:**
* Generated dendrograms using multiple linkage methods (`ward`, `complete`, `single`, `average`).
* The `ward` linkage provided the distinct clusters, aligning well with the structure observed in the KMeans analysis.



---

## Technical Stack

* **Language:** Python 3.x
* **Core Libraries:** `pandas`, `numpy`, `scikit-learn`, `statsmodels`, `scipy`
* **Visualization:** `matplotlib`, `seaborn`

## Usage

To replicate this analysis:

1. Clone the repository.
2. Ensure source datasets are located in the root directory.
3. Run the `Spring2025ProjectML.ipynb` notebook.
