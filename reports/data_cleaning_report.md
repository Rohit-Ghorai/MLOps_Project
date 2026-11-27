# MP063 - Data Cleaning & Validation Report (Nagpur City Bus Delay)

**Author:** Data Lead (Group G192)  
**Project:** MP063 — City Bus Delay Prediction (Nagpur)  
**Date Generated:** 2026-11-18  
**Status:** Cleaned and Validated (All Checks Green)  

---

## 1. Overview and Objectives

The objective of Step 4 is to translate the audit findings from [`reports/data_issues_report.md`](file:///c:/Users/rohit/Desktop/MP063_City_Bus_Delay/reports/data_issues_report.md) into a reproducible, deterministic cleaning and validation pipeline in [`src/data_pipeline/cleaning.py`](file:///c:/Users/rohit/Desktop/MP063_City_Bus_Delay/src/data_pipeline/cleaning.py) and [`src/data_pipeline/validation.py`](file:///c:/Users/rohit/Desktop/MP063_City_Bus_Delay/src/data_pipeline/validation.py).

The raw training dataset (`data/raw/train.csv`) had two concrete corruption patterns and one severe data leakage hazard:
1. **Duplicate Records:** 46 duplicate rows artificially inflating sample weights.
2. **Target Typos:** 15 rows where `delay_min` had an extra zero typed (10× too large, jumping from normal values <= 34.4 min up to 94.0–271.0 min).
3. **Target Leakage:** `delay_logged_by_conductor` recorded *after* trip completion, holding near-perfect correlation (r = 0.9986) with the delay.

All cleaning operations were executed programmatically without altering the original raw datasets.

---

## 2. Cleaning Operations and Statistical Reasoning

### Operation 1: Target Typo Correction (10× Divider)
* **Diagnosis:** In `train.csv`, 15 records had `delay_min > 50.0`. A massive 59.6-minute void separated normal delays (max 34.4 min) from the lowest typo value (94.0 min).
* **Cross-Validation with Conductor's Sheet:** Comparing `delay_min / 10.0` with `delay_logged_by_conductor` confirmed that every single outlier was caused by an accidental extra zero at data entry:
  * Record `R00017`: raw `163.0` → divided by 10 = `16.3` min (conductor logged: `16.0` min)
  * Record `R00189`: raw `169.0` → divided by 10 = `16.9` min (conductor logged: `16.9` min)
  * Record `R00335`: raw `158.0` → divided by 10 = `15.8` min (conductor logged: `15.9` min)
  * Record `R00498`: raw `233.0` → divided by 10 = `23.3` min (conductor logged: `23.1` min)
  * Record `R00853`: raw `94.0` → divided by 10 = `9.4` min (conductor logged: `9.9` min)
* **Action:**
  ```python
  df.loc[df["delay_min"] > 50.0, "delay_min"] = df.loc[df["delay_min"] > 50.0, "delay_min"] / 10.0
  ```
* **Result:** All 15 typo rows corrected. Post-correction `delay_min` spans from **3.70 to 34.40 minutes** (mean: **18.18 minutes**).

### Operation 2: Duplicate Row Pruning
* **Diagnosis:** Exactly 46 duplicate rows existed where all 12 columns (including `record_id`) were identical copies of prior entries. One typo record (`R00853`, row 891 and 892) was also duplicated.
* **Action:**
  ```python
  df_cleaned = df.drop_duplicates(keep="first").reset_index(drop=True)
  ```
* **Result:** 46 duplicate rows removed. Training set reduced from 1,046 to **exactly 1,000 unique, clean records**.

### Operation 3: Target Leakage Safeguard
* **Diagnosis:** The column `delay_logged_by_conductor` is completed only *after* the trip concludes. Because the Nagpur City Bus Service prediction API must estimate delays at the *start* of the trip, this feature is absent from production inputs (`holdout_features.csv` and `POST /predict`).
* **Design Decision:**
  * In [`data/processed/train_clean.csv`](file:///c:/Users/rohit/Desktop/MP063_City_Bus_Delay/data/processed/train_clean.csv) and [`data/processed/later_clean.csv`](file:///c:/Users/rohit/Desktop/MP063_City_Bus_Delay/data/processed/later_clean.csv), `delay_logged_by_conductor` is preserved alongside `record_id` and `date` for auditing, verification, and comparison.
  * In the codebase, `MODEL_FEATURES` explicitly designates the canonical 8 input features:
    ```python
    MODEL_FEATURES = [
        "route_km", "stops", "passengers", "hour",
        "temp_c", "driver_exp_yr", "weather", "depot_zone"
    ]
    ```
  * A helper function `get_model_features(df)` is provided for the Model Lead to extract feature matrix `X` and target `y` without risk of leakage.

### Operation 4: Consistent Treatment of `later.csv`
* **Diagnosis:** `later.csv` (300 rows) represents clean production data arriving two months later. It has 0 duplicates, 0 nulls, and valid delay values (13.3 to 58.0 min).
* **Action:** Verified schema consistency and saved to [`data/processed/later_clean.csv`](file:///c:/Users/rohit/Desktop/MP063_City_Bus_Delay/data/processed/later_clean.csv) with zero rows dropped and zero targets altered.

---

## 3. Cleaned Dataset Specifications

| Metric | Raw `train.csv` | Cleaned `train_clean.csv` | Raw `later.csv` | Cleaned `later_clean.csv` |
|---|---|---|---|---|
| **Row Count** | 1,046 | **1,000** (-46 duplicates) | 300 | **300** |
| **Column Count** | 12 | **12** | 12 | **12** |
| **Duplicate Rows** | 46 | **0** | 0 | **0** |
| **Null Values** | 0 | **0** | 0 | **0** |
| **Target Min** | 3.70 min | **3.70 min** | 13.30 min | **13.30 min** |
| **Target Max** | 271.00 min | **34.40 min** | 58.00 min | **58.00 min** |
| **Target Mean** | 20.53 min | **18.18 min** | 32.74 min | **32.74 min** |
| **Target Outliers (>50 min)** | 15 | **0** | 15 (natural drift) | 15 (natural drift) |

---

## 4. Validation Results (`src.data_pipeline.validation`)

All programmatic validation assertions passed (Exit Code 0):

```text
=================================================================
DATA VALIDATION SUITE (MP063)
=================================================================

[RAW FILE INTEGRITY CHECK]
  data/raw/train.csv: PASSED (SHA-256 MATCH)
  data/raw/later.csv: PASSED (SHA-256 MATCH)
  data/raw/holdout_features.csv: PASSED (SHA-256 MATCH)

[TRAIN CLEAN VALIDATION (1000 rows)]
  row_count_matches_expected: True
  zero_duplicate_rows: True
  zero_duplicate_record_ids: True
  zero_null_values: True
  target_column_exists: True
  target_is_numeric: True
  zero_target_typos_above_50: True
  target_min: 3.7
  target_max: 34.4
  target_mean: 18.18
  all_expected_columns_present: True
  stops_le_70: True
  route_km_ge_1_5: True

[LATER CLEAN VALIDATION (300 rows)]
  row_count_matches_expected: True
  zero_duplicate_rows: True
  zero_duplicate_record_ids: True
  zero_null_values: True
  target_column_exists: True
  target_is_numeric: True
  target_min: 13.3
  target_max: 58.0
  target_mean: 32.74
  all_expected_columns_present: True
  stops_le_70: True
  route_km_ge_1_5: True

[MODEL FEATURE LEAKAGE SAFETY]
  leakage_column: delay_logged_by_conductor
  is_excluded_from_model_features: True
  matches_api_request_schema: True
  feature_count: 8

=================================================================
OVERALL VALIDATION STATUS: ALL CHECKS PASSED (GREEN)
=================================================================
```

---

## 5. Raw Data Immutability

The cryptographic SHA-256 fingerprints of the original raw datasets remain unchanged:

* `data/raw/train.csv`: `ebf413b22ab52d4580f7d9eb705f903db91e3780809599e85c208d56378750ae` (Verified)
* `data/raw/later.csv`: `efe02164027ceb50979f12544b3a6bfeb51072ec7a6f783eeaf3352b117b64ce` (Verified)
* `data/raw/holdout_features.csv`: `feb6d831601bef952d5777ec1784c7536ab51712296f07e2caa646b1b519ffd5` (Verified)
