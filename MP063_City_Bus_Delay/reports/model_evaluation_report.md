# MP063 - Machine Learning Model Evaluation Report

**Author:** Sagar Goyal (ML Engineering Lead, Group G192)  
**Project:** MP063 — City Bus Delay Prediction (Nagpur)  
**Evaluation Metric:** Mean Absolute Error (MAE in minutes)  
**Dataset SHA-256:** `ebf413b22ab52d4580f7d9eb705f903db91e3780809599e85c208d56378750ae`  

---

## 1. Executive Summary & Model Selection

Three Scikit-learn regression models were systematically trained and tracked using MLflow (`sqlite:///mlflow.db`) in experiment `G192_MP063_City_Bus_Delay`.

The models were evaluated on an 80/20 train/validation split (800 train records, 200 validation records) derived from `data/processed/train_clean.csv`.

* **Selected Champion Model:** `HistGradientBoostingRegressor`
* **Validation MAE:** **1.2229 minutes**
* **MLflow Registered Model:** `G192_MP063_BusDelay`
* **Model Version:** `v1`
* **Assigned Alias:** `@champion`
* **Exported Artifact:** `models\model.joblib`

---

## 2. Model Performance Comparison

| Model Architecture | Train MAE | Val MAE (Primary) | Val RMSE | Val R² | MLflow Run ID |
|---|---|---|---|---|---|
| `RandomForestRegressor` | 0.6294 min | **1.4044 min** | 1.7206 min | 0.9166 | `a259f59c...` |
| `ExtraTreesRegressor` | 0.5073 min | **1.5145 min** | 1.8303 min | 0.9056 | `caf42831...` |
| `HistGradientBoostingRegressor` (Champion) | 0.3735 min | **1.2229 min** | 1.5211 min | 0.9348 | `43b53a1d...` |

### Selection Justification:
`HistGradientBoostingRegressor` achieved the lowest validation MAE of **1.2229 minutes**, demonstrating superior generalization on unseen bus trips without overfitting. It natively captures non-linear interactions between weather conditions, route length, and traffic peak hours.

---

## 3. Prediction-Time Feature Safety & Leakage Prevention

The model pipeline strictly adheres to the 8 production fields defined in `PROBLEM.md` Section 5:
- **Numerical (6):** `route_km`, `stops`, `passengers`, `hour`, `temp_c`, `driver_exp_yr`
- **Categorical (2):** `weather`, `depot_zone`

**Target Leakage Safeguard:**
- The column `delay_logged_by_conductor` is strictly excluded from all preprocessing transformers and estimator features.
- Identifiers (`record_id`) and timestamp (`date`) are excluded from feature vectors.
- Preprocessing implements median imputation for numerical features and most_frequent imputation with safe `handle_unknown='ignore'` OneHotEncoding for categorical features.

---

## 4. API Inference Compatibility Verification

The exported inference pipeline (`models/model.joblib`) was verified against the expected schema:
* Valid 8-field inference: **Passed**
* Graceful missing-value imputation: **Passed**
* Robustness to previously unseen categories: **Passed**
