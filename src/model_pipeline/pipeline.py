"""ML Feature Pipeline & Preprocessing Architecture for MP063.

Reuses the data definitions from src.data_pipeline.cleaning:
- Excludes delay_logged_by_conductor, record_id, date, and delay_min.
- Encapsulates ColumnTransformer and Pipeline for reproducible inference.
- Handles numerical missing values via median imputation.
- Handles categorical missing values via most_frequent imputation.
- Encodes categorical features via OneHotEncoder(handle_unknown='ignore').
"""

from pathlib import Path
from typing import Any, Dict, List, Optional
import yaml
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Import canonical column definitions established by Data Lead Rohit Ghorai
from src.data_pipeline.cleaning import (
    MODEL_FEATURES,
    TARGET_COL,
    LEAKAGE_COL,
    ID_COL,
    DATE_COL,
)

NUMERICAL_FEATURES: List[str] = [
    "route_km",
    "stops",
    "passengers",
    "hour",
    "temp_c",
    "driver_exp_yr",
]

CATEGORICAL_FEATURES: List[str] = [
    "weather",
    "depot_zone",
]


def load_params(params_path: Optional[str] = None) -> Dict[str, Any]:
    """Load project parameters from params.yaml using yaml.safe_load."""
    if params_path is None:
        candidates = [
            Path("params.yaml"),
            Path("MP063_City_Bus_Delay/params.yaml"),
            Path(__file__).resolve().parent.parent.parent / "params.yaml",
        ]
        for c in candidates:
            if c.exists():
                params_path = str(c)
                break
    if params_path is None or not Path(params_path).exists():
        raise FileNotFoundError(f"params.yaml not found at {params_path}")

    with open(params_path, "r", encoding="utf-8") as f:
        params = yaml.safe_load(f)
    return params


def build_preprocessing_pipeline() -> ColumnTransformer:
    """Build Scikit-learn ColumnTransformer for prediction-time features.

    Implements:
    - Numerical: Median imputation + Standard scaling.
    - Categorical: Most frequent imputation + OneHotEncoding with unseen level handling.
    """
    num_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    cat_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_transformer, NUMERICAL_FEATURES),
            ("cat", cat_transformer, CATEGORICAL_FEATURES),
        ],
        remainder="drop",  # Drops any extra columns like record_id or date if passed
    )

    return preprocessor


def build_full_pipeline(estimator: BaseEstimator) -> Pipeline:
    """Build an end-to-end inference Pipeline combining preprocessing and estimator.

    Ensures training and production inference (FastAPI POST /predict) execute
    the identical preprocessing graph.
    """
    preprocessor = build_preprocessing_pipeline()
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", estimator),
    ])
    return pipeline


def validate_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure dataframe has all 8 prediction features and exclude forbidden columns."""
    for col in MODEL_FEATURES:
        if col not in df.columns:
            raise ValueError(f"Required prediction feature '{col}' missing from input.")

    # Explicitly ensure leakage column is never returned
    clean_X = df[MODEL_FEATURES].copy()
    assert LEAKAGE_COL not in clean_X.columns, f"Leakage column '{LEAKAGE_COL}' detected in feature matrix!"
    assert TARGET_COL not in clean_X.columns, f"Target column '{TARGET_COL}' detected in feature matrix!"
    return clean_X
