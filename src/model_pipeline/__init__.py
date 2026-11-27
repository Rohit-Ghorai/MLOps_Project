"""Model pipeline package for MP063 - City Bus Delay Prediction (Nagpur).

Modules:
- pipeline: Feature preprocessing pipeline and schema definitions.
- train: Training 3 candidate models, evaluation, MLflow tracking, model registration, champion alias.
- evaluate: Detailed model evaluation, metrics generation, report production.
- predict: Batch prediction generation for holdout_features.csv.
- monitoring: Drift analysis (PSI), depot group MAE, v2 retraining, and promotion evaluation.
"""

from src.model_pipeline.pipeline import (
    MODEL_FEATURES,
    NUMERICAL_FEATURES,
    CATEGORICAL_FEATURES,
    TARGET_COL,
    LEAKAGE_COL,
    build_preprocessing_pipeline,
    build_full_pipeline,
)

__all__ = [
    "MODEL_FEATURES",
    "NUMERICAL_FEATURES",
    "CATEGORICAL_FEATURES",
    "TARGET_COL",
    "LEAKAGE_COL",
    "build_preprocessing_pipeline",
    "build_full_pipeline",
]
