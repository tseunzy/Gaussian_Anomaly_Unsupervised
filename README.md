# Credit Card Fraud Detection with Gaussian Anomaly Detection and XGBoost

This project explores credit card fraud detection using two different approaches:

- Gaussian anomaly detection as an unsupervised baseline
- XGBoost as a supervised comparison model

The main analysis is contained in the notebook `Gaussian_anomaly(1).ipynb`.

## Project Overview

Fraud detection is a class-imbalanced problem where fraudulent transactions are rare compared with normal transactions. This project focuses on building a practical workflow that:

- prepares and explores the data
- handles skewed feature distributions
- trains a Gaussian anomaly detector on normal transactions only
- evaluates the anomaly model on validation and test data
- compares the baseline against XGBoost

## Files

- `Gaussian_anomaly(1).ipynb` — main notebook for data preparation, anomaly detection, error analysis, and XGBoost comparison
- `creditcard.csv` — [KAGGLE DATASET](https://www.kaggle.com/datasets/whenamancodes/fraud-detection/data)

## Dataset

The notebook expects the credit card fraud dataset in the repository root as:

- `creditcard.csv`

[DATASET LINK](https://www.kaggle.com/datasets/whenamancodes/fraud-detection/data)

The dataset includes anonymized PCA-style features `V1` to `V28`, plus `Time`, `Amount`, and the target column `Class`.

## Workflow

### 1. Data loading and preprocessing

- load the dataset with pandas
- inspect class distribution and missing values
- drop `Time`
- transform `Amount` into `Amount_log`

### 2. Gaussian anomaly detection

- split normal transactions into train, validation, and test subsets
- add fraud samples into validation and test sets
- measure skewness on training features
- apply Yeo-Johnson transformation to heavily skewed columns
- exclude `V6` from transformation because it becomes more skewed after transformation
- standardize the transformed features with `StandardScaler`
- estimate Gaussian parameters from the normal-only training set
- score validation and test examples with Gaussian log-density
- choose the anomaly threshold on validation data
- evaluate on test data

### 3. Error analysis

- identify false positives and false negatives
- inspect average feature behavior by error type

### 4. XGBoost comparison

- build a supervised XGBoost classifier on the labeled dataset
- split the labeled data into train, validation, and test sets
- use `RandomizedSearchCV` for hyperparameter tuning
- tune the classification threshold on the validation set only
- evaluate once on the untouched test set

## Key Learning Points

- Gaussian anomaly detection is useful as an unsupervised fraud-detection baseline
- feature skewness matters when the model assumes Gaussian-like behavior
- threshold tuning should be done on validation data, not the test set
- XGBoost generally performs better when labeled fraud examples are available
- evaluation should focus on precision, recall, and F1 rather than accuracy alone

## Requirements

Typical Python packages used in this notebook:

- `numpy`
- `pandas`
- `matplotlib`
- `seaborn`
- `scikit-learn`
- `xgboost`

## How To Run

1. Place `creditcard.csv` in the project root.
2. Open `Gaussian_anomaly(1).ipynb`.
3. Download the dataset from https://www.kaggle.com/datasets/whenamancodes/fraud-detection/data
3. Run the notebook from top to bottom in order.
4. Avoid skipping cells because later sections depend on variables created earlier.

## Notes

- The Gaussian model is intentionally used as a baseline and may underperform because of strong distribution assumptions.
- Some transformed features may still remain skewed even after Yeo-Johnson correction.
- The XGBoost section should use validation data for threshold selection and test data only for final reporting.
