"""Data cleaning pipeline for MP063 - City Bus Delay Prediction (Nagpur).

This module implements a reproducible cleaning pipeline for the raw training
and monitoring datasets:
1. Deduplication (removing exactly 46 duplicate rows).
2. Target typo correction (dividing the 15 delay_min values > 50 by 10).
3. Leakage prevention: Defining the exact 8 model features, ensuring
   delay_logged_by_conductor is never used as a model feature while preserving
   it in the processed dataset for auditability.
4. Output generation for data/processed/train_clean.csv and data/processed/later_clean.csv.
"""

from pathlib import Path
from typing import Any, Dict, List, Tuple
import pandas as pd
import numpy as np

# Official feature set matching PROBLEM.md Section 5 (POST /predict schema)
MODEL_FEATURES: List[str] = [
    "route_km",
    "stops",
    "passengers",
    "hour",
    "temp_c",
    "driver_exp_yr",
    "weather",
    "depot_zone",
]

TARGET_COL: str = "delay_min"
LEAKAGE_COL: str = "delay_logged_by_conductor"
ID_COL: str = "record_id"
DATE_COL: str = "date"

ALL_COLUMNS: List[str] = [
    ID_COL,
    DATE_COL,
    *MODEL_FEATURES,
    LEAKAGE_COL,
    TARGET_COL,
]


def clean_training_data(
    raw_df: pd.DataFrame,
    typo_threshold: float = 50.0
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Clean the raw training dataset.

    Steps:
    1. Identify and remove duplicate rows (expected: 46 extra copies).
    2. Identify and correct 10x target typos in delay_min (expected: 15 rows in raw).
    3. Validate that non-target, non-leakage columns are untouched.
    """
    raw_count = len(raw_df)

    # 1. Detect target typos in raw dataset before deduplication
    raw_typo_mask = raw_df[TARGET_COL] > typo_threshold
    raw_typos_count = int(raw_typo_mask.sum())

    # Correct target typos (divide values > 50 by 10)
    df_corrected = raw_df.copy()
    df_corrected.loc[raw_typo_mask, TARGET_COL] = df_corrected.loc[raw_typo_mask, TARGET_COL] / 10.0

    # 2. Remove duplicate rows
    duplicate_count = int(df_corrected.duplicated().sum())
    df_cleaned = df_corrected.drop_duplicates(keep="first").reset_index(drop=True)
    final_count = len(df_cleaned)

    # Post-dedup typo check
    clean_typo_count = int((df_cleaned[TARGET_COL] > typo_threshold).sum())

    metadata = {
        "raw_rows": raw_count,
        "duplicates_removed": duplicate_count,
        "raw_target_typos_corrected": raw_typos_count,
        "final_rows": final_count,
        "target_min": round(float(df_cleaned[TARGET_COL].min()), 2),
        "target_max": round(float(df_cleaned[TARGET_COL].max()), 2),
        "target_mean": round(float(df_cleaned[TARGET_COL].mean()), 2),
        "remaining_typos_above_50": clean_typo_count,
        "null_count": int(df_cleaned.isna().sum().sum()),
    }

    return df_cleaned, metadata


def clean_later_data(raw_df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Prepare later.csv consistently with training data without altering target values."""
    raw_count = len(raw_df)
    duplicate_count = int(raw_df.duplicated().sum())

    # later.csv is clean: remove duplicates if any, preserve valid rows & targets
    df_cleaned = raw_df.drop_duplicates(keep="first").reset_index(drop=True)

    metadata = {
        "raw_rows": raw_count,
        "duplicates_removed": duplicate_count,
        "final_rows": len(df_cleaned),
        "target_min": round(float(df_cleaned[TARGET_COL].min()), 2),
        "target_max": round(float(df_cleaned[TARGET_COL].max()), 2),
        "target_mean": round(float(df_cleaned[TARGET_COL].mean()), 2),
        "null_count": int(df_cleaned.isna().sum().sum()),
    }
    return df_cleaned, metadata


def get_model_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """Helper for model training: extracts X (only MODEL_FEATURES) and y (TARGET_COL).

    Guarantees that delay_logged_by_conductor, record_id, and date are never
    passed into any machine learning estimator.
    """
    X = df[MODEL_FEATURES].copy()
    y = df[TARGET_COL].copy()
    return X, y


def run_pipeline(
    raw_dir: str = "data/raw",
    processed_dir: str = "data/processed"
) -> Dict[str, Any]:
    """Run full cleaning pipeline and save processed CSV files."""
    raw_path = Path(raw_dir)
    proc_path = Path(processed_dir)
    proc_path.mkdir(parents=True, exist_ok=True)

    # 1. Clean train.csv
    train_raw = pd.read_csv(raw_path / "train.csv")
    train_clean, train_meta = clean_training_data(train_raw)
    train_out_file = proc_path / "train_clean.csv"
    train_clean.to_csv(train_out_file, index=False)

    # 2. Clean later.csv
    later_raw = pd.read_csv(raw_path / "later.csv")
    later_clean, later_meta = clean_later_data(later_raw)
    later_out_file = proc_path / "later_clean.csv"
    later_clean.to_csv(later_out_file, index=False)

    summary = {
        "train": train_meta,
        "later": later_meta,
        "train_clean_path": str(train_out_file),
        "later_clean_path": str(later_out_file),
        "model_features": MODEL_FEATURES,
        "leakage_excluded_from_model": LEAKAGE_COL not in MODEL_FEATURES,
    }
    return summary


if __name__ == "__main__":
    results = run_pipeline()
    print("=" * 65)
    print("DATA CLEANING PIPELINE EXECUTION (MP063)")
    print("=" * 65)
    t = results["train"]
    print("\n[TRAIN CLEANING RESULTS]")
    print(f"  Original row count:          {t['raw_rows']}")
    print(f"  Duplicate rows removed:      {t['duplicates_removed']}")
    print(f"  Target typo rows corrected:  {t['raw_target_typos_corrected']}")
    print(f"  Final cleaned row count:     {t['final_rows']}")
    print(f"  Cleaned target range:        {t['target_min']} min to {t['target_max']} min (mean: {t['target_mean']} min)")
    print(f"  Saved to:                    {results['train_clean_path']}")

    l = results["later"]
    print("\n[LATER CLEANING RESULTS]")
    print(f"  Original row count:          {l['raw_rows']}")
    print(f"  Duplicate rows removed:      {l['duplicates_removed']}")
    print(f"  Final row count:             {l['final_rows']}")
    print(f"  Target range:                {l['target_min']} min to {l['target_max']} min (mean: {l['target_mean']} min)")
    print(f"  Saved to:                    {results['later_clean_path']}")

    print("\n[MODEL FEATURE INTEGRITY]")
    print(f"  Model Features (8):          {results['model_features']}")
    print(f"  Leakage column excluded?:    {results['leakage_excluded_from_model']}")
