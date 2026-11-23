"""Pytest suite for MP063 Machine Learning Pipeline.

Tests:
1. Data cleaning integration and strict leakage prevention.
2. Feature preprocessing pipeline robustness (missing values, unseen categories).
3. MLflow tracking integrity (SQLite backend, experiment, >=3 runs, data_sha256, MAE metrics).
4. Model registration and @champion alias resolution.
5. Saved inference artifact loading and prediction with 8 API fields.
6. Holdout predictions CSV structure and constraints.
7. Findings.yaml tracking and monitoring fields.
"""

from pathlib import Path
import hashlib
import joblib
import mlflow
from mlflow.tracking import MlflowClient
import numpy as np
import pandas as pd
import pytest
import yaml

from src.model_pipeline.pipeline import (
    MODEL_FEATURES,
    NUMERICAL_FEATURES,
    CATEGORICAL_FEATURES,
    TARGET_COL,
    LEAKAGE_COL,
    build_preprocessing_pipeline,
    build_full_pipeline,
    load_params,
)


@pytest.fixture
def params():
    return load_params()


def test_leakage_exclusion_and_feature_schema():
    """Verify that conductor delay is strictly excluded and exactly 8 features are defined."""
    assert LEAKAGE_COL not in MODEL_FEATURES, "Leakage column must not be in MODEL_FEATURES!"
    assert TARGET_COL not in MODEL_FEATURES, "Target column must not be in MODEL_FEATURES!"
    assert len(MODEL_FEATURES) == 8, f"Expected 8 prediction features, got {len(MODEL_FEATURES)}"
    assert len(NUMERICAL_FEATURES) == 6
    assert len(CATEGORICAL_FEATURES) == 2
    assert set(NUMERICAL_FEATURES + CATEGORICAL_FEATURES) == set(MODEL_FEATURES)


def test_pipeline_missing_values_and_unseen_categories():
    """Test that preprocessing pipeline handles nulls and unseen categories gracefully."""
    preprocessor = build_preprocessing_pipeline()

    # Synthetic training batch
    train_df = pd.DataFrame({
        "route_km": [5.0, 10.0, 15.0, 20.0],
        "stops": [10, 15, 20, 25],
        "passengers": [20, 30, 40, 50],
        "hour": [8, 12, 16, 20],
        "temp_c": [25.0, 30.0, 35.0, 28.0],
        "driver_exp_yr": [1, 5, 10, 15],
        "weather": ["clear", "rain", "fog", "clear"],
        "depot_zone": ["North", "South", "East", "North"],
    })
    preprocessor.fit(train_df)

    # Inference batch with missing values and brand new categories
    test_df = pd.DataFrame({
        "route_km": [np.nan],
        "stops": [18],
        "passengers": [np.nan],
        "hour": [14],
        "temp_c": [32.0],
        "driver_exp_yr": [4],
        "weather": ["snow_storm"],  # Unseen
        "depot_zone": ["West_Zone"], # Unseen
    })

    transformed = preprocessor.transform(test_df)
    assert transformed.shape[0] == 1
    assert not np.isnan(transformed).any(), "NaN found in transformed feature matrix!"


def test_model_artifact_loading_and_inference(params):
    """Test that models/model.joblib loads cleanly and predicts on 8-field input."""
    model_path = Path(params["paths"]["model_artifact"])
    assert model_path.exists(), f"Champion model artifact missing at {model_path}"

    model = joblib.load(model_path)

    # Exactly the 8 fields sent by FastAPI POST /predict
    input_payload = pd.DataFrame([{
        "route_km": 15.5,
        "stops": 22,
        "passengers": 38,
        "hour": 10,
        "temp_c": 29.0,
        "driver_exp_yr": 7,
        "weather": "clear",
        "depot_zone": "South",
    }])

    pred = model.predict(input_payload)
    assert len(pred) == 1
    assert float(pred[0]) > 0.0, f"Predicted delay must be positive, got {pred[0]}"


def test_mlflow_tracking_and_registry(params):
    """Verify MLflow runs, logged params, data_sha256, and champion alias."""
    tracking_uri = params["mlflow"]["tracking_uri"]
    exp_name = params["mlflow"]["experiment_name"]
    model_name = params["mlflow"]["registered_model_name"]

    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient(tracking_uri=tracking_uri)

    experiment = client.get_experiment_by_name(exp_name)
    assert experiment is not None, f"MLflow experiment '{exp_name}' not found"

    runs = client.search_runs(experiment_ids=[experiment.experiment_id])
    assert len(runs) >= 3, f"Expected at least 3 MLflow runs, found {len(runs)}"

    raw_path = Path(params["paths"]["raw_train"])
    expected_sha256 = hashlib.sha256(open(raw_path, "rb").read()).hexdigest()

    for r in runs:
        params_dict = r.data.params
        metrics_dict = r.data.metrics
        assert "data_sha256" in params_dict, f"Run {r.info.run_id} missing data_sha256 param"
        assert params_dict["data_sha256"] == expected_sha256, f"Run {r.info.run_id} hash mismatch"
        has_mae = any("mae" in k.lower() for k in metrics_dict.keys())
        assert has_mae, f"Run {r.info.run_id} missing MAE metric"

    registered_model = client.get_registered_model(model_name)
    assert registered_model is not None, f"Registered model '{model_name}' not found"
    assert "champion" in registered_model.aliases, "Alias '@champion' not found on registered model"

    champion_version = client.get_model_version_by_alias(model_name, "champion")
    assert champion_version is not None
    assert int(champion_version.version) >= 1


def test_holdout_predictions_file(params):
    """Verify that predictions/holdout_predictions.csv adheres to exact constraints."""
    pred_path = Path(params["paths"]["holdout_predictions"])
    assert pred_path.exists(), f"Holdout predictions file missing at {pred_path}"

    df_preds = pd.read_csv(pred_path)
    assert len(df_preds) == 250, f"Expected 250 rows in holdout predictions, found {len(df_preds)}"
    assert list(df_preds.columns) == ["record_id", "prediction"], f"Columns mismatch: {list(df_preds.columns)}"
    assert not df_preds["prediction"].isna().any(), "Holdout predictions contain NaN values"

    # Compare with holdout_features.csv record IDs
    raw_holdout = pd.read_csv(params["paths"]["raw_holdout"])
    assert df_preds["record_id"].tolist() == raw_holdout["record_id"].tolist(), "record_id values or ordering altered!"


def test_findings_yaml_populated(params):
    """Verify that reports/findings.yaml tracking and monitoring fields are filled."""
    findings_path = Path(params["paths"]["findings_file"])
    assert findings_path.exists(), f"findings.yaml missing at {findings_path}"

    with open(findings_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    # Check tracking
    tracking = data.get("tracking", {})
    assert tracking.get("experiment_name") == "G192_MP063_City_Bus_Delay"
    assert tracking.get("best_run_id") is not None
    assert tracking.get("registered_model_name") == "G192_MP063_BusDelay"
    assert tracking.get("champion_version") is not None

    # Check monitoring
    monitoring = data.get("monitoring", {})
    assert monitoring.get("most_drifted_feature") == "stops"
    assert monitoring.get("worst_served_group") == "North"
    assert monitoring.get("decision") in ["promote", "keep"]
