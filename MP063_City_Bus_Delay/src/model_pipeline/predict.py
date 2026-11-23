"""Holdout Prediction Generation for MP063.

Loads the champion model artifact (models/model.joblib) and predicts
delay_min for all records in data/raw/holdout_features.csv.

Outputs predictions/holdout_predictions.csv with exactly two columns:
record_id,prediction.
"""

from pathlib import Path
from typing import Optional
import joblib
import pandas as pd

from src.model_pipeline.pipeline import (
    MODEL_FEATURES,
    ID_COL,
    load_params,
    validate_feature_matrix,
)


def generate_holdout_predictions(
    holdout_path: Optional[str] = None,
    model_path: Optional[str] = None,
    output_path: Optional[str] = None,
    params_path: Optional[str] = None,
) -> pd.DataFrame:
    """Generate and validate final holdout predictions using champion model."""
    params = load_params(params_path)

    if holdout_path is None:
        holdout_path = params["paths"]["raw_holdout"]
    if model_path is None:
        model_path = params["paths"]["model_artifact"]
    if output_path is None:
        output_path = params["paths"]["holdout_predictions"]

    h_path = Path(holdout_path)
    m_path = Path(model_path)
    o_path = Path(output_path)

    if not h_path.exists():
        h_path = Path(__file__).resolve().parent.parent.parent / holdout_path
    if not m_path.exists():
        m_path = Path(__file__).resolve().parent.parent.parent / model_path

    assert h_path.exists(), f"Holdout features file not found: {h_path}"
    assert m_path.exists(), f"Champion model artifact not found: {m_path}"

    # 1. Load holdout features
    holdout_df = pd.read_csv(h_path)
    expected_rows = len(holdout_df)
    record_ids = holdout_df[ID_COL].tolist()

    # 2. Extract feature matrix
    X_holdout = validate_feature_matrix(holdout_df)

    # 3. Load champion model
    model = joblib.load(m_path)

    # 4. Predict
    predictions = model.predict(X_holdout)

    # 5. Format output dataframe
    results_df = pd.DataFrame({
        "record_id": record_ids,
        "prediction": [round(float(p), 4) for p in predictions],
    })

    # 6. Strict validation
    assert len(results_df) == expected_rows, f"Expected {expected_rows} predictions, got {len(results_df)}"
    assert list(results_df.columns) == ["record_id", "prediction"], f"Invalid columns: {results_df.columns}"
    assert results_df["record_id"].tolist() == record_ids, "Record ID ordering or values corrupted!"
    assert not results_df["prediction"].isna().any(), "NaN found in predictions!"
    assert not (results_df["prediction"] <= 0).all(), "Predictions are completely non-positive!"

    # 7. Write to CSV
    o_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(o_path, index=False)

    return results_df


if __name__ == "__main__":
    preds = generate_holdout_predictions()
    print("=" * 65)
    print("HOLDOUT PREDICTIONS GENERATED SUCCESSFULLY (MP063)")
    print("=" * 65)
    print(f"  Rows generated:    {len(preds)}")
    print(f"  Columns:           {list(preds.columns)}")
    print(f"  Prediction range:  {preds['prediction'].min():.2f} min to {preds['prediction'].max():.2f} min")
    print(f"  Prediction mean:   {preds['prediction'].mean():.2f} min")
    print(f"  Head 5:\n{preds.head(5)}")
    print("=" * 65)
