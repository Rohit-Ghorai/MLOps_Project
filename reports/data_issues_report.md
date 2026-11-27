# MP063 - Data Issues Audit Report (Nagpur City Bus Delay)

**Author:** Data Lead (Group G192)
**Problem:** City Bus Delay Prediction (Nagpur)
**Status:** Audit Completed

---

## Executive Summary of Findings

| Issue Type | Present? | Affected Column / Detail | Count / Value | Evidence Summary |
|---|---|---|---|---|
| `leaking_column` | **TRUE** | `delay_logged_by_conductor` | r = 0.9986 | Omitted from holdout; conductor logs after trip ends (r = 0.9986 with target) |
| `unit_mixup` | FALSE | None | None | Mean ratio across depots is <= 1.10 for all numeric features |
| `sentinel_value` | FALSE | None | None | No negative values or extreme placeholders (-1, -999, 9999) found |
| `duplicate_rows` | **TRUE** | `All columns` | 46 rows | Exact identical duplicates across all 12 columns |
| `stuck_sensor` | FALSE | None | None | Max consecutive run across distinct dates is 2 dates |
| `target_typos` | **TRUE** | `delay_min` | 15 rows (14 unique IDs) | delay_min > 50 min sitting above a 59.6 min gap; matches conductor delay when divided by 10 |
| `spelling_variants` | FALSE | None | None | Weather (3) and Depot (3) categories perfectly match clean later.csv |

---

## Detailed Evidence for Each Data Problem

### 1. Leaking Column (`leaking_column`)
- **Status:** `present: true`
- **Column:** `delay_logged_by_conductor`
- **Reasoning & Proof:** 'delay_logged_by_conductor' is logged by the conductor after the trip completes. It is absent from holdout_features.csv and has a 0.9986 correlation with delay_min (mean absolute error of only 0.24 minutes).
- **Metrics:** Pearson correlation = `0.9986`, Mean Absolute Difference = `0.2412 min`.

### 2. Unit Mixup (`unit_mixup`)
- **Status:** `present: false`
- **Column:** `null`
- **Group Value:** `null`
- **Factor:** `null`
- **Reasoning & Proof:** All numeric features exhibit near-identical distributions across North, South, and East depots (max mean ratio <= 1.10). No conversion factor (10, 12, 60, 100, 1000) exists.

### 3. Sentinel Value (`sentinel_value`)
- **Status:** `present: false`
- **Column:** `null`
- **Value:** `null`
- **Count:** `null`
- **Reasoning & Proof:** No sentinel numbers (-1, -999, 999, 9999) found. All numeric values fall strictly within expected real-world bus service operating ranges.

### 4. Duplicate Rows (`duplicate_rows`)
- **Status:** `present: true`
- **Count:** `46` (extra duplicate copies)
- **Reasoning & Proof:** train.csv contains 46 duplicate rows where all 12 columns (including record_id) are identical copies of existing records.

### 5. Stuck Sensor (`stuck_sensor`)
- **Status:** `present: false`
- **Column:** `null`
- **Start Date:** `null`
- **End Date:** `null`
- **Reasoning & Proof:** No stuck sensor found. The longest run of identical values for any sensor spans at most 2 distinct dates (3 records max), consistent with random occurrence.

### 6. Target Typos (`target_typos`)
- **Status:** `present: true`
- **Count:** `15` rows (or `14` after deduplication)
- **Reasoning & Proof:** Found 15 rows with delay_min > 50 min (values from 94.0 to 271.0). A massive gap of 59.6 min exists between 34.4 and 94.0. Dividing these values by 10 matches delay_logged_by_conductor almost exactly, confirming a 10x typing mistake.

**Sample Reconciliations with Conductor's Log:**

| record_id   |   raw_delay_min |   corrected_div_10 |   conductor_logged |
|:------------|----------------:|-------------------:|-------------------:|
| R00017      |             163 |               16.3 |               16   |
| R00189      |             169 |               16.9 |               16.9 |
| R00335      |             158 |               15.8 |               15.9 |
| R00367      |             149 |               14.9 |               14.9 |
| R00498      |             233 |               23.3 |               23.1 |


### 7. Spelling Variants (`spelling_variants`)
- **Status:** `present: false`
- **Column:** `null`
- **True Levels:** `null`
- **Reasoning & Proof:** Categorical columns (weather, depot_zone) have clean, standard spellings. Train levels match later.csv exactly with no casing or whitespace issues.
