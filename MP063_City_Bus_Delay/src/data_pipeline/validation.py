"""Data validation module for MP063 - City Bus Delay Prediction (Nagpur).

This module validates:
1. Deduplication integrity (0 duplicate rows).
2. Target typo resolution in training data (0 values > 50 min in train_clean).
3. Null value absence (0 null values in processed datasets).
4. Schema consistency & expected column presence.
5. Target variable presence and valid numeric dtype.
6. Target leakage prevention: delay_logged_by_conductor excluded from MODEL_FEATURES.
7. Raw CSV immutability: SHA-256 hashes of original raw files match expected fingerprints.
8. Client business rules: route_km >= 1.5 and stops <= 70.
"""

from pathlib import Path
import hashlib
from typing import Any, Dict, List
import pandas as pd

from src.data_pipeline.cleaning import (
    MODEL_FEATURES,
    TARGET_COL,
    LEAKAGE_COL,
    ALL_COLUMNS,
)

# Expected SHA-256 hashes of the untouched original raw CSVs
EXPECTED_RAW_HASHES: Dict[str, str] = {
    "data/raw/train.csv": "ebf413b22ab52d4580f7d9eb705f903db91e3780809599e85c208d56378750ae",
    "data/raw/later.csv": "efe02164027ceb50979f12544b3a6bfeb51072ec7a6f783eeaf3352b117b64ce",
    "data/raw/holdout_features.csv": "feb6d831601bef952d5777ec1784c7536ab51712296f07e2caa646b1b519ffd5",
}


def compute_sha256(file_path: Path) -> str:
    """Compute the SHA-256 hash of a file."""
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()


def validate_raw_hashes() -> Dict[str, bool]:
    """Verify that original raw CSV files have not been modified."""
    status = {}
    for rel_path, expected_hash in EXPECTED_RAW_HASHES.items():
        p = Path(rel_path)
        if not p.exists():
            status[rel_path] = False
            continue
        actual_hash = compute_sha256(p)
        status[rel_path] = (actual_hash == expected_hash)
    return status


def validate_cleaned_dataset(
    df: pd.DataFrame,
    name: str = "train_clean",
    expected_rows: int = 1000,
    check_typos: bool = True
) -> Dict[str, Any]:
    """Run integrity and quality assertions on a cleaned dataset."""
    checks = {}

    # 1. Row count check
    actual_rows = len(df)
    checks["row_count_matches_expected"] = (actual_rows == expected_rows)

    # 2. Duplicate checks
    dup_rows = int(df.duplicated().sum())
    checks["zero_duplicate_rows"] = (dup_rows == 0)

    if "record_id" in df.columns:
        dup_ids = int(df["record_id"].duplicated().sum())
        checks["zero_duplicate_record_ids"] = (dup_ids == 0)

    # 3. Missing values check
    total_nulls = int(df.isna().sum().sum())
    checks["zero_null_values"] = (total_nulls == 0)

    # 4. Target variable checks
    if TARGET_COL in df.columns:
        checks["target_column_exists"] = True
        checks["target_is_numeric"] = pd.api.types.is_numeric_dtype(df[TARGET_COL])
        if check_typos:
            typos = int((df[TARGET_COL] > 50).sum())
            checks["zero_target_typos_above_50"] = (typos == 0)
        checks["target_min"] = round(float(df[TARGET_COL].min()), 2)
        checks["target_max"] = round(float(df[TARGET_COL].max()), 2)
        checks["target_mean"] = round(float(df[TARGET_COL].mean()), 2)
    else:
        checks["target_column_exists"] = False

    # 5. Schema and column presence
    checks["all_expected_columns_present"] = set(ALL_COLUMNS).issubset(df.columns)

    # 6. Client business rules from PROBLEM.md Section 5
    if "stops" in df.columns:
        checks["stops_le_70"] = bool((df["stops"] <= 70).all())
    if "route_km" in df.columns:
        checks["route_km_ge_1_5"] = bool((df["route_km"] >= 1.5).all())

    return checks


def validate_model_feature_safety() -> Dict[str, Any]:
    """Verify that data leakage column is strictly excluded from model feature set."""
    is_excluded = LEAKAGE_COL not in MODEL_FEATURES
    is_valid_schema = set(MODEL_FEATURES) == {
        "route_km", "stops", "passengers", "hour",
        "temp_c", "driver_exp_yr", "weather", "depot_zone"
    }
    return {
        "leakage_column": LEAKAGE_COL,
        "is_excluded_from_model_features": is_excluded,
        "matches_api_request_schema": is_valid_schema,
        "feature_count": len(MODEL_FEATURES),
    }


def run_full_validation() -> Dict[str, Any]:
    """Execute all validation checks and return results summary."""
    train_clean_path = Path("data/processed/train_clean.csv")
    later_clean_path = Path("data/processed/later_clean.csv")

    assert train_clean_path.exists(), f"Missing {train_clean_path}"
    assert later_clean_path.exists(), f"Missing {later_clean_path}"

    train_clean = pd.read_csv(train_clean_path)
    later_clean = pd.read_csv(later_clean_path)

    raw_hash_results = validate_raw_hashes()
    train_checks = validate_cleaned_dataset(train_clean, name="train_clean", expected_rows=1000, check_typos=True)
    later_checks = validate_cleaned_dataset(later_clean, name="later_clean", expected_rows=300, check_typos=False)
    feature_safety = validate_model_feature_safety()

    all_passed = (
        all(raw_hash_results.values()) and
        all(v is True for k, v in train_checks.items() if isinstance(v, bool)) and
        all(v is True for k, v in later_checks.items() if isinstance(v, bool)) and
        feature_safety["is_excluded_from_model_features"] and
        feature_safety["matches_api_request_schema"]
    )

    return {
        "all_passed": all_passed,
        "raw_hashes": raw_hash_results,
        "train_checks": train_checks,
        "later_checks": later_checks,
        "feature_safety": feature_safety,
    }


if __name__ == "__main__":
    results = run_full_validation()
    print("=" * 65)
    print("DATA VALIDATION SUITE (MP063)")
    print("=" * 65)

    print("\n[RAW FILE INTEGRITY CHECK]")
    for file_path, matches in results["raw_hashes"].items():
        status = "PASSED (SHA-256 MATCH)" if matches else "FAILED"
        print(f"  {file_path}: {status}")

    print("\n[TRAIN CLEAN VALIDATION (1000 rows)]")
    for check_name, val in results["train_checks"].items():
        print(f"  {check_name}: {val}")

    print("\n[LATER CLEAN VALIDATION (300 rows)]")
    for check_name, val in results["later_checks"].items():
        print(f"  {check_name}: {val}")

    print("\n[MODEL FEATURE LEAKAGE SAFETY]")
    for k, v in results["feature_safety"].items():
        print(f"  {k}: {v}")

    print("\n" + "=" * 65)
    if results["all_passed"]:
        print("OVERALL VALIDATION STATUS: ALL CHECKS PASSED (GREEN)")
    else:
        print("OVERALL VALIDATION STATUS: SOME CHECKS FAILED (RED)")
    print("=" * 65)
