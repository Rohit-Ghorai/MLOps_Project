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

---

## 5. Drift Monitoring, Retraining (v2), and Champion Promotion

Following two months of production operation, the arrival of `later.csv` (300 trips from July–August 2026) initiated the monitoring pipeline (`src/model_pipeline/monitoring.py`):

1. **Population Stability Index (PSI):**
   - Evaluated across all 6 numeric features using 10 quantile bins between `train_clean.csv` and `later.csv`.
   - **Most Drifted Feature:** `stops` ($\text{PSI} = 6.9235$, caused by substantial route network expansion in July/August).
   - Other features exhibited minimal drift ($\text{PSI} \le 0.057$).

2. **Subgroup Performance Audit (`depot_zone`):**
   - v1 MAE on `later.csv` by depot zone:
     - **North:** 11.9857 min (Worst served)
     - **South:** 3.7840 min
     - **East:** 4.1818 min

3. **v2 Retraining & Holdout Evaluation:**
   - Chronological split of `later.csv` sorted by date: first 70% (210 trips) combined with `train_clean.csv` for retraining; last 30% (90 trips) strictly held out for model evaluation.
   - **v1 MAE on evaluation check split:** 9.1170 min
   - **v2 MAE on evaluation check split:** 2.9100 min
   - **Improvement:** **68.08%** reduction in MAE (well exceeding the client's 10% threshold).

4. **Promotion Decision:**
   - **Decision:** `promote`
   - Model v2 was registered in MLflow (`runs:/69e4590695174944939ba87c4b80ae99/model`) as **Version 2**.
   - Alias `@champion` was assigned to **Version 2**.
   - Production artifact `models/model.joblib` was updated with the promoted Version 2 pipeline.
