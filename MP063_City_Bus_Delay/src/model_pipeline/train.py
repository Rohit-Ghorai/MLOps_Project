"""Model Training, MLflow Experiment Tracking, and Model Registration for MP063.

Implements:
1. Loading cleaned training data from data/processed/train_clean.csv.
2. 80/20 Train/Validation Split with reproducible random_state=42.
3. Training three Scikit-learn regression candidates:
   - RandomForestRegressor
   - ExtraTreesRegressor
   - HistGradientBoostingRegressor
4. MLflow Experiment Tracking with SQLite backend (sqlite:///mlflow.db):
   - Experiment: G192_MP063_City_Bus_Delay
   - Registered Model: G192_MP063_BusDelay
   - Parameter data_sha256 logged on every run.
   - Validation MAE logged as metric.
   - Cloudpickle serialization format for model logging.
5. Model Registration & Champion Alias Assignment:
   - Registers best-performing model as Version 1.
   - Sets alias '@champion' on Version 1.
6. Exports serialized inference pipeline to models/model.joblib.
"""

from pathlib import Path
import hashlib
import json
import os
from typing import Any, Dict, List, Optional, Tuple
import joblib
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
import numpy as np
import pandas as pd
from sklearn.ensemble import (
    ExtraTreesRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from src.model_pipeline.pipeline import (
    MODEL_FEATURES,
    TARGET_COL,
    LEAKAGE_COL,
    ID_COL,
    DATE_COL,
    build_full_pipeline,
    load_params,
    validate_feature_matrix,
)


def get_raw_train_sha256(raw_path: str = "data/raw/train.csv") -> str:
    """Calculate the exact SHA-256 hash of the untouched raw train.csv."""
    p = Path(raw_path)
    if not p.exists():
        p = Path(__file__).resolve().parent.parent.parent / raw_path
    if not p.exists():
        raise FileNotFoundError(f"Raw training file not found at {raw_path}")
    raw_bytes = open(p, "rb").read()
    return hashlib.sha256(raw_bytes).hexdigest()


def get_candidate_models(params: Dict[str, Any]) -> Dict[str, Any]:
    """Instantiate the 3 candidate regressors using params.yaml settings."""
    rf_params = params.get("models", {}).get("random_forest", {
        "n_estimators": 100, "max_depth": 10, "min_samples_split": 5, "random_state": 42
    })
    et_params = params.get("models", {}).get("extra_trees", {
        "n_estimators": 100, "max_depth": 10, "min_samples_split": 5, "random_state": 42
    })
    hgb_params = params.get("models", {}).get("hist_gradient_boosting", {
        "max_iter": 100, "max_depth": 6, "min_samples_leaf": 10, "random_state": 42
    })

    return {
        "RandomForestRegressor": RandomForestRegressor(**rf_params),
        "ExtraTreesRegressor": ExtraTreesRegressor(**et_params),
        "HistGradientBoostingRegressor": HistGradientBoostingRegressor(**hgb_params),
    }


def train_and_track_models(
    params_path: Optional[str] = None,
    export_artifact: bool = True,
) -> Dict[str, Any]:
    """Execute training of 3 candidate models, log to MLflow, and register champion."""
    params = load_params(params_path)

    # 1. Resolve paths
    train_clean_path = Path(params["paths"]["processed_train"])
    raw_train_path = Path(params["paths"]["raw_train"])
    model_dir = Path(params["paths"]["model_dir"])
    model_artifact_path = Path(params["paths"]["model_artifact"])
    reports_dir = Path(params["paths"]["reports_dir"])

    model_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    # 2. Verify raw train SHA-256
    data_sha256 = get_raw_train_sha256(str(raw_train_path))

    # 3. Load processed training dataset
    df = pd.read_csv(train_clean_path)
    X = validate_feature_matrix(df)
    y = df[params["data"]["target_col"]].copy()

    # 4. Train/Validation Split (80/20)
    test_size = float(params.get("split", {}).get("test_size", 0.20))
    random_state = int(params.get("split", {}).get("random_state", 42))

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # 5. Configure MLflow
    tracking_uri = params.get("mlflow", {}).get("tracking_uri", "sqlite:///mlflow.db")
    experiment_name = params.get("mlflow", {}).get("experiment_name", "G192_MP063_City_Bus_Delay")
    registered_model_name = params.get("mlflow", {}).get("registered_model_name", "G192_MP063_BusDelay")
    alias_name = params.get("mlflow", {}).get("alias_name", "champion")

    mlflow.set_tracking_uri(tracking_uri)
    experiment = mlflow.set_experiment(experiment_name)
    client = MlflowClient(tracking_uri=tracking_uri)

    candidates = get_candidate_models(params)
    run_results: List[Dict[str, Any]] = []

    best_model_name: Optional[str] = None
    best_val_mae: float = float("inf")
    best_run_id: Optional[str] = None
    best_pipeline = None

    # 6. Train and track each candidate
    for model_name, estimator in candidates.items():
        with mlflow.start_run(run_name=f"v1_{model_name}") as run:
            run_id = run.info.run_id

            # Build unified pipeline (preprocessor + estimator)
            pipeline = build_full_pipeline(estimator)

            # Fit pipeline on training split
            pipeline.fit(X_train, y_train)

            # Evaluate on train and validation splits
            train_preds = pipeline.predict(X_train)
            val_preds = pipeline.predict(X_val)

            train_mae = float(mean_absolute_error(y_train, train_preds))
            val_mae = float(mean_absolute_error(y_val, val_preds))
            val_rmse = float(np.sqrt(mean_squared_error(y_val, val_preds)))
            val_r2 = float(r2_score(y_val, val_preds))

            # Mandatory parameters and metrics
            mlflow.log_param("data_sha256", data_sha256)
            mlflow.log_param("model_name", model_name)
            mlflow.log_param("estimator_class", estimator.__class__.__name__)
            mlflow.log_param("train_samples", len(X_train))
            mlflow.log_param("val_samples", len(X_val))
            mlflow.log_param("features", MODEL_FEATURES)

            # Log estimator hyperparameters
            for hp_name, hp_val in estimator.get_params().items():
                if isinstance(hp_val, (int, float, str, bool)):
                    mlflow.log_param(f"model__{hp_name}", hp_val)

            # Log metrics (must contain 'mae')
            mlflow.log_metric("mae", val_mae)
            mlflow.log_metric("val_mae", val_mae)
            mlflow.log_metric("train_mae", train_mae)
            mlflow.log_metric("val_rmse", val_rmse)
            mlflow.log_metric("val_r2", val_r2)

            mlflow.set_tag("group_id", "G192")
            mlflow.set_tag("problem_id", "MP063")
            mlflow.set_tag("model_phase", "v1")

            # Log model artifact with cloudpickle serialization
            input_example = X_val.iloc[:2]
            mlflow.sklearn.log_model(
                sk_model=pipeline,
                artifact_path="model",
                input_example=input_example,
                serialization_format="cloudpickle",
            )

            result = {
                "model_name": model_name,
                "run_id": run_id,
                "train_mae": round(train_mae, 4),
                "val_mae": round(val_mae, 4),
                "val_rmse": round(val_rmse, 4),
                "val_r2": round(val_r2, 4),
            }
            run_results.append(result)

            if val_mae < best_val_mae:
                best_val_mae = val_mae
                best_model_name = model_name
                best_run_id = run_id
                best_pipeline = pipeline

    # 7. Model Registration for Champion V1
    model_uri = f"runs:/{best_run_id}/model"
    model_version = mlflow.register_model(
        model_uri=model_uri,
        name=registered_model_name,
        tags={"promoted_from": "v1_evaluation", "best_model": best_model_name},
    )

    # Assign champion alias
    client.set_registered_model_alias(
        name=registered_model_name,
        alias=alias_name,
        version=str(model_version.version),
    )

    # 8. Export Champion Model Artifact for API consumption
    if export_artifact and best_pipeline is not None:
        joblib.dump(best_pipeline, model_artifact_path)
        # Also preserve explicit v1 artifact for monitoring baseline comparison
        v1_path = model_dir / "model_v1.joblib"
        joblib.dump(best_pipeline, v1_path)

    summary = {
        "data_sha256": data_sha256,
        "experiment_name": experiment_name,
        "registered_model_name": registered_model_name,
        "runs": run_results,
        "best_model_name": best_model_name,
        "best_run_id": best_run_id,
        "best_val_mae": round(best_val_mae, 4),
        "champion_version": str(model_version.version),
        "champion_alias": alias_name,
        "artifact_path": str(model_artifact_path),
    }

    # Save summary report
    summary_path = reports_dir / "model_training_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return summary


if __name__ == "__main__":
    print("=" * 65)
    print("EXECUTING MODEL TRAINING & MLFLOW TRACKING (MP063)")
    print("=" * 65)
    summary = train_and_track_models()
    print(f"\n[EXPERIMENT DETAILS]")
    print(f"  Experiment:       {summary['experiment_name']}")
    print(f"  Data SHA-256:     {summary['data_sha256']}")
    print(f"  Registered Model: {summary['registered_model_name']}")
    print(f"  Champion Version: v{summary['champion_version']} (alias: @{summary['champion_alias']})")
    print(f"  Artifact Path:    {summary['artifact_path']}")

    print("\n[CANDIDATE COMPARISON]")
    for r in summary["runs"]:
        star = " <--- BEST [CHAMPION]" if r["model_name"] == summary["best_model_name"] else ""
        print(f"  - {r['model_name']:<30} | Val MAE: {r['val_mae']:.4f} min | Run ID: {r['run_id']}{star}")
    print("=" * 65)
