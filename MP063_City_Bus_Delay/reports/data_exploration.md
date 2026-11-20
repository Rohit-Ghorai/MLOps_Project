# MP063 - Dataset Exploration Report (Nagpur City Bus Delay)

**Author:** Data Lead (Group G192)
**Problem:** City Bus Delay Prediction (Nagpur)
**Date Generated:** 2026-11-18

---

## 1. Overview of Datasets

Three datasets provided by Nagpur City Bus Service were analyzed:

| dataset   |   rows |   columns |   total_nulls |   duplicate_rows |   unique_record_ids |
|:----------|-------:|----------:|--------------:|-----------------:|--------------------:|
| train     |   1046 |        12 |             0 |               46 |                1000 |
| later     |    300 |        12 |             0 |                0 |                 300 |
| holdout   |    250 |        10 |             0 |                0 |                 250 |


### Date Ranges & Temporal Coverage

| dataset   | start_date   | end_date   |   total_days_span |   unique_dates |
|:----------|:-------------|:-----------|------------------:|---------------:|
| train     | 2026-01-01   | 2026-06-30 |               181 |            180 |
| later     | 2026-07-01   | 2026-08-31 |                62 |             61 |
| holdout   | 2026-09-01   | 2026-09-30 |                30 |             30 |


## 2. Column Schemas & Missing Values

### train.csv (1046 rows × 12 columns)

| column                    | dtype   |   non_null_count |   null_count |   null_pct |   unique_values |
|:--------------------------|:--------|-----------------:|-------------:|-----------:|----------------:|
| record_id                 | str     |             1046 |            0 |          0 |            1000 |
| date                      | str     |             1046 |            0 |          0 |             180 |
| route_km                  | float64 |             1046 |            0 |          0 |             247 |
| stops                     | int64   |             1046 |            0 |          0 |              36 |
| passengers                | int64   |             1046 |            0 |          0 |              81 |
| hour                      | int64   |             1046 |            0 |          0 |              17 |
| temp_c                    | float64 |             1046 |            0 |          0 |             343 |
| driver_exp_yr             | int64   |             1046 |            0 |          0 |              25 |
| weather                   | str     |             1046 |            0 |          0 |               3 |
| depot_zone                | str     |             1046 |            0 |          0 |               3 |
| delay_logged_by_conductor | float64 |             1046 |            0 |          0 |             241 |
| delay_min                 | float64 |             1046 |            0 |          0 |             254 |


> **Key Observation:** There are **zero missing (null) values** across all three datasets.
 However, `train.csv` contains **46 duplicate rows**.

## 3. Categorical Variables (`weather` & `depot_zone`)

| Dataset | Categorical Column | Levels & Frequencies |
|---|---|---|
| `train.csv` | `weather` | `clear`: 432, `fog`: 307, `rain`: 307 |
| `train.csv` | `depot_zone` | `North`: 626, `South`: 221, `East`: 199 |
| `later.csv` | `weather` | `clear`: 132, `rain`: 85, `fog`: 83 |
| `later.csv` | `depot_zone` | `North`: 168, `East`: 70, `South`: 62 |
| `holdout.csv` | `weather` | `clear`: 102, `rain`: 81, `fog`: 67 |
| `holdout.csv` | `depot_zone` | `North`: 143, `South`: 56, `East`: 51 |

> **Key Observation:** Both `weather` and `depot_zone` have clean, consistent categories
 across all datasets (clear, fog, rain; North, South, East) with no casing or whitespace inconsistencies.

## 4. Numerical Features Summary (train.csv)

|                           |   mean |   std |   min |   q25 |   median |   q75 |   max |   skewness |
|:--------------------------|-------:|------:|------:|------:|---------:|------:|------:|-----------:|
| route_km                  |  15.91 |  7.33 |   3   |   9.4 |    16.6  | 22.3  |  28   |      -0.08 |
| stops                     |  22.2  | 10.32 |   5   |  13   |    22    | 31    |  40   |       0.06 |
| passengers                |  50.04 | 23.19 |  10   |  30   |    49    | 71    |  90   |       0.01 |
| hour                      |  14.23 |  4.61 |   6   |  10   |    14    | 18    |  22   |      -0.04 |
| temp_c                    |  26.03 | 10.61 |   8   |  17   |    26.25 | 35.2  |  44   |      -0.01 |
| driver_exp_yr             |  13.13 |  6.82 |   1   |   7   |    13    | 19    |  25   |       0.05 |
| delay_logged_by_conductor |  18.12 |  5.74 |   3.4 |  13.8 |    17.4  | 22.38 |  34.6 |       0.28 |
| delay_min                 |  20.53 | 21.93 |   3.7 |  13.9 |    17.6  | 22.5  | 271   |       8.46 |


## 5. Target Variable (`delay_min`) Distribution

- **Count:** 1046
- **Mean:** 20.53 min
- **Standard Deviation:** 21.93 min
- **Median (50%):** 17.6 min
- **Interquartile Range (25% - 75%):** 13.9 min – 22.5 min
- **Min / Max:** 3.7 min / 271.0 min
- **Rows with delay_min > 50 min:** 15 rows

### Quantile Distribution:

|   Quantile |   delay_min |
|-----------:|------------:|
|       0.01 |        7.2  |
|       0.05 |        9.5  |
|       0.25 |       13.9  |
|       0.5  |       17.6  |
|       0.75 |       22.5  |
|       0.9  |       26.45 |
|       0.95 |       28.6  |
|       0.99 |      153.95 |
|       1    |      271    |


> **Anomaly Flag:** 98.5% of trips have delays <= 34.27 minutes, but 15 rows jump abruptly
 to values between 94.0 and 271.0 minutes. This will be examined in the data issues detection.

## 6. Group-Wise Statistics by `depot_zone`

| depot_zone   |   ('route_km', 'mean') |   ('route_km', 'std') |   ('route_km', 'min') |   ('route_km', 'max') |   ('stops', 'mean') |   ('stops', 'std') |   ('stops', 'min') |   ('stops', 'max') |   ('passengers', 'mean') |   ('passengers', 'std') |   ('passengers', 'min') |   ('passengers', 'max') |   ('hour', 'mean') |   ('hour', 'std') |   ('hour', 'min') |   ('hour', 'max') |   ('temp_c', 'mean') |   ('temp_c', 'std') |   ('temp_c', 'min') |   ('temp_c', 'max') |   ('driver_exp_yr', 'mean') |   ('driver_exp_yr', 'std') |   ('driver_exp_yr', 'min') |   ('driver_exp_yr', 'max') |   ('delay_min', 'mean') |   ('delay_min', 'std') |   ('delay_min', 'min') |   ('delay_min', 'max') |
|:-------------|-----------------------:|----------------------:|----------------------:|----------------------:|--------------------:|-------------------:|-------------------:|-------------------:|-------------------------:|------------------------:|------------------------:|------------------------:|-------------------:|------------------:|------------------:|------------------:|---------------------:|--------------------:|--------------------:|--------------------:|----------------------------:|---------------------------:|---------------------------:|---------------------------:|------------------------:|-----------------------:|-----------------------:|-----------------------:|
| East         |                  15.66 |                  7.43 |                   3   |                    28 |               21.36 |              10.19 |                  5 |                 40 |                    47.68 |                   23.68 |                      10 |                      90 |              14.09 |              4.56 |                 6 |                22 |                25.82 |                9.96 |                 8   |                43.9 |                       12.31 |                       6.72 |                          1 |                         25 |                   20.28 |                  25.71 |                    5   |                    271 |
| North        |                  15.93 |                  7.33 |                   3.1 |                    28 |               22.3  |              10.34 |                  5 |                 40 |                    50.45 |                   23.01 |                      10 |                      89 |              14.25 |              4.6  |                 6 |                22 |                26.04 |               10.92 |                 8   |                44   |                       13.51 |                       6.9  |                          1 |                         25 |                   20.07 |                  17.77 |                    3.7 |                    270 |
| South        |                  16.1  |                  7.27 |                   3.1 |                    28 |               22.67 |              10.4  |                  5 |                 40 |                    51.02 |                   23.21 |                      11 |                      90 |              14.29 |              4.71 |                 6 |                22 |                26.18 |               10.31 |                 8.1 |                43.9 |                       12.81 |                       6.64 |                          1 |                         25 |                   22.07 |                  28.11 |                    3.7 |                    260 |


> **Unit Mixup Check:** Minimum, mean, and maximum values across North, South, and East depots
 are virtually identical for route_km (~15.6–16.1 km), stops (~21.3–22.7), and other metrics.
 No depot recorded values in alternative units (e.g. meters instead of km).
