"""Monitoring, Drift Detection (PSI), and Retraining Pipeline for MP063.

Implements requirements from PROBLEM.md Section 7 and STUDENT_HANDBOOK.md Section 10:
1. Population Stability Index (PSI) computation for all numeric columns (10 bins).
2. Group MAE evaluation of v1 across depot_zone categories on later.csv.
3. Chronological 70/30 split on later.csv (sorted by date).
4. Retraining v2 on train_clean.csv + first 70% of later.csv.
5. Evaluation of v1 vs v2 on the last 30% of later.csv.
6. Promotion Rule application: promote v2 if MAE improvement >= 10%.
7. MLflow registration of v2 and @champion alias management.
8. Population of reports/findings.yaml tracking and monitoring fields.
"""

from pathlib import Path
import json
from typing import Any, Dict, List, Optional, Tuple
import joblib
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
import yaml

from src.model_pipeline.pipeline import (
    MODEL_FEATURES,
    NUMERICAL_FEATURES,
    CATEGORICAL_FEATURES,
    TARGET_COL,
    DATE_COL,
    build_full_pipeline,
    load_params,
    validate_feature_matrix,
)
from src.model_pipeline.train import get_raw_train_sha256, get_candidate_models


def calculate_psi(expected: pd.Series, actual: pd.Series, num_bins: int = 10) -> float:
    """Calculate Population Stability Index (PSI) using quantile binning on expected data."""
    quantiles = np.linspace(0, 1, num_bins + 1)
    bin_edges = np.percentile(expected.dropna(), quantiles * 100)
    bin_edges = np.unique(bin_edges)
    if len(bin_edges) < 2:
        return 0.0

    bin_edges[0] = -np.inf
    bin_edges[-1] = np.inf

    exp_counts, _ = np.histogram(expected, bins=bin_edges)
    act_counts, _ = np.histogram(actual, bins=bin_edges)

    exp_pct = np.where(exp_counts == 0, 1e-4, exp_counts) / len(expected)
    act_pct = np.where(act_counts == 0, 1e-4, act_counts) / len(actual)

    psi = np.sum((act_pct - exp_pct) * np.log(act_pct / exp_pct))
    return float(psi)


def run_monitoring_and_retraining(
    params_path: Optional[str] = None,
    update_findings: bool = True,
) -> Dict[str, Any]:
    """Execute complete drift monitoring, v2 retraining, and promotion workflow."""
    params = load_params(params_path)

    train_clean_path = Path(params["paths"]["processed_train"])
    later_clean_path = Path(params["paths"]["processed_later"])
    raw_train_path = Path(params["paths"]["raw_train"])
    model_artifact_path = Path(params["paths"]["model_artifact"])
    findings_path = Path(params["paths"]["findings_file"])

    train_df = pd.read_csv(train_clean_path)
    later_df = pd.read_csv(later_clean_path)

    data_sha256 = get_raw_train_sha256(str(raw_train_path))

    # 1. Compute PSI for all 6 numeric columns (train vs later)
    psi_dict = {}
    for col in NUMERICAL_FEATURES:
        val = calculate_psi(train_df[col], later_df[col], num_bins=10)
        psi_dict[col] = round(val, 4)

    most_drifted = max(psi_dict, key=psi_dict.get)

    # 2. Load v1 model (strictly trained on train_clean.csv only)
    v1_artifact_path = Path(params["paths"]["model_dir"]) / "model_v1.joblib"
    if v1_artifact_path.exists():
        v1_model = joblib.load(v1_artifact_path)
    elif model_artifact_path.exists():
        v1_model = joblib.load(model_artifact_path)
    else:
        raise FileNotFoundError("Model artifact not found. Run train.py first.")

    # Compute v1 MAE on later.csv per depot_zone
    group_maes = {}
    for zone in sorted(later_df["depot_zone"].unique()):
        subset = later_df[later_df["depot_zone"] == zone]
        sub_preds = v1_model.predict(subset[MODEL_FEATURES])
        sub_mae = float(mean_absolute_error(subset[TARGET_COL], sub_preds))
        group_maes[zone] = round(sub_mae, 4)

    worst_served_group = max(group_maes, key=group_maes.get)

    # 3. Chronological split of later.csv (70% train, 30% check)
    later_sorted = later_df.sort_values(DATE_COL).reset_index(drop=True)
    split_ratio = float(params.get("monitoring", {}).get("later_train_ratio", 0.70))
    split_idx = int(len(later_sorted) * split_ratio)

    later_train = later_sorted.iloc[:split_idx]
    later_check = later_sorted.iloc[split_idx:]

    # 4. Retrain v2 on train_clean.csv + first 70% of later.csv
    combined_train = pd.concat([train_df, later_train], ignore_index=True)
    X_v2 = combined_train[MODEL_FEATURES]
    y_v2 = combined_train[TARGET_COL]

    # Instantiate best model architecture for v2 (HistGradientBoostingRegressor)
    candidates = get_candidate_models(params)
    v2_estimator = candidates["HistGradientBoostingRegressor"]
    v2_pipeline = build_full_pipeline(v2_estimator)

    tracking_uri = params.get("mlflow", {}).get("tracking_uri", "sqlite:///mlflow.db")
    experiment_name = params.get("mlflow", {}).get("experiment_name", "G192_MP063_City_Bus_Delay")
    registered_model_name = params.get("mlflow", {}).get("registered_model_name", "G192_MP063_BusDelay")
    alias_name = params.get("mlflow", {}).get("alias_name", "champion")

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)
    client = MlflowClient(tracking_uri=tracking_uri)

    with mlflow.start_run(run_name="v2_retrained_HistGradientBoosting") as run:
        v2_run_id = run.info.run_id
        v2_pipeline.fit(X_v2, y_v2)

        # 5. Compare v1 and v2 on the last 30% of later.csv
        v1_preds_check = v1_model.predict(later_check[MODEL_FEATURES])
        v2_preds_check = v2_pipeline.predict(later_check[MODEL_FEATURES])

        mae_v1_check = round(float(mean_absolute_error(later_check[TARGET_COL], v1_preds_check)), 4)
        mae_v2_check = round(float(mean_absolute_error(later_check[TARGET_COL], v2_preds_check)), 4)

        improvement = (mae_v1_check - mae_v2_check) / mae_v1_check
        promo_threshold = float(params.get("monitoring", {}).get("promotion_improvement_threshold", 0.10))
        decision = "promote" if improvement >= promo_threshold else "keep"

        mlflow.log_param("data_sha256", data_sha256)
        mlflow.log_param("model_name", "v2_retrained_HistGradientBoosting")
        mlflow.log_param("combined_train_samples", len(combined_train))
        mlflow.log_param("check_samples", len(later_check))
        mlflow.log_metric("mae", mae_v2_check)
        mlflow.log_metric("mae_v1_on_check", mae_v1_check)
        mlflow.log_metric("mae_v2_on_check", mae_v2_check)
        mlflow.log_metric("improvement_ratio", float(improvement))
        mlflow.set_tag("decision", decision)
        mlflow.set_tag("model_phase", "v2")

        mlflow.sklearn.log_model(
            sk_model=v2_pipeline,
            artifact_path="model",
            input_example=later_check[MODEL_FEATURES].iloc[:2],
            serialization_format="cloudpickle",
        )

    # 6. Register v2 in MLflow Registry
    v2_model_uri = f"runs:/{v2_run_id}/model"
    v2_version = mlflow.register_model(
        model_uri=v2_model_uri,
        name=registered_model_name,
        tags={"promoted_from": "v2_monitoring_retrain", "decision": decision},
    )

    # Save explicit v2 artifact
    v2_artifact_path = Path(params["paths"]["model_dir"]) / "model_v2.joblib"
    joblib.dump(v2_pipeline, v2_artifact_path)

    # If promoted, update @champion alias to v2
    if decision == "promote":
        client.set_registered_model_alias(
            name=registered_model_name,
            alias=alias_name,
            version=str(v2_version.version),
        )
        # Update champion model artifact
        joblib.dump(v2_pipeline, model_artifact_path)
        champion_version_final = str(v2_version.version)
    else:
        champion_version_final = "1"

    # 7. Update findings.yaml
    if update_findings and findings_path.exists():
        with open(findings_path, "r", encoding="utf-8") as f:
            findings_data = yaml.safe_load(f)

        # Get best v1 run id from previous training
        training_summary_path = Path(params["paths"]["reports_dir"]) / "model_training_summary.json"
        best_v1_run_id = None
        if training_summary_path.exists():
            with open(training_summary_path, "r", encoding="utf-8") as f:
                ts = json.load(f)
                best_v1_run_id = ts.get("best_run_id")

        findings_data["tracking"] = {
            "experiment_name": experiment_name,
            "best_run_id": best_v1_run_id,
            "registered_model_name": registered_model_name,
            "champion_version": int(champion_version_final),
        }

        findings_data["monitoring"] = {
            "psi": psi_dict,
            "most_drifted_feature": most_drifted,
            "mae_by_group": {
                "North": group_maes.get("North"),
                "South": group_maes.get("South"),
                "East": group_maes.get("East"),
            },
            "worst_served_group": worst_served_group,
            "mae_v1_on_check": mae_v1_check,
            "mae_v2_on_check": mae_v2_check,
            "decision": decision,
        }

        with open(findings_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(findings_data, f, sort_keys=False, default_flow_style=False)

    return {
        "psi": psi_dict,
        "most_drifted_feature": most_drifted,
        "group_maes": group_maes,
        "worst_served_group": worst_served_group,
        "mae_v1_on_check": mae_v1_check,
        "mae_v2_on_check": mae_v2_check,
        "improvement_pct": round(improvement * 100, 2),
        "decision": decision,
        "v2_run_id": v2_run_id,
        "v2_version": str(v2_version.version),
        "champion_version": champion_version_final,
    }


if __name__ == "__main__":
    res = run_monitoring_and_retraining()
    print("=" * 65)
    print("MONITORING & RETRAINING RESULTS (MP063)")
    print("=" * 65)
    print("\n[PSI VALUES (10 bins)]")
    for col, val in res["psi"].items():
        print(f"  {col:<15}: {val:.4f}")
    print(f"  Most drifted feature: {res['most_drifted_feature']}")

    print("\n[V1 MAE BY DEPOT ZONE ON LATER.CSV]")
    for zone, val in res["group_maes"].items():
        print(f"  {zone:<15}: {val:.4f} min")
    print(f"  Worst served group:   {res['worst_served_group']}")

    print("\n[RETRAINING & PROMOTION CHECK]")
    print(f"  MAE v1 on check (last 30%): {res['mae_v1_on_check']:.4f} min")
    print(f"  MAE v2 on check (last 30%): {res['mae_v2_on_check']:.4f} min")
    print(f"  Improvement:                {res['improvement_pct']}%")
    print(f"  Decision:                   {res['decision'].upper()}")
    print(f"  Final Champion Version:     v{res['champion_version']}")
    print("=" * 65)
