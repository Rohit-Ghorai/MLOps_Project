"""Tests for the FastAPI prediction service."""

import json

from fastapi.testclient import TestClient

from app.main import create_app


class StubPredictor:
    def predict(self, features):
        assert list(features.columns) == [
            "route_km",
            "stops",
            "passengers",
            "hour",
            "temp_c",
            "driver_exp_yr",
            "weather",
            "depot_zone",
        ]
        return [12.34567]


def make_client() -> TestClient:
    return TestClient(create_app(predictor=StubPredictor()))


def test_health_returns_ok():
    with make_client() as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_predict_returns_prediction_and_json_log(capfd):
    with make_client() as client:
        response = client.post(
            "/predict",
            json={
                "route_km": 15.5,
                "stops": 22,
                "passengers": 38,
                "hour": 10,
                "temp_c": 29.0,
                "driver_exp_yr": 7,
                "weather": "clear",
                "depot_zone": "South",
            },
        )

    assert response.status_code == 200
    assert response.json() == {"prediction": 12.3457}
    log_entry = json.loads(capfd.readouterr().out.strip())
    assert log_entry["event"] == "prediction"
    assert log_entry["prediction"] == 12.3457


def test_predict_rejects_client_rule_violations_and_extra_fields():
    with make_client() as client:
        invalid_route = client.post(
            "/predict",
            json={
                "route_km": 1.49,
                "stops": 22,
                "passengers": 38,
                "hour": 10,
                "temp_c": 29.0,
                "driver_exp_yr": 7,
                "weather": "clear",
                "depot_zone": "South",
            },
        )
        invalid_stops = client.post(
            "/predict",
            json={
                "route_km": 15.5,
                "stops": 71,
                "passengers": 38,
                "hour": 10,
                "temp_c": 29.0,
                "driver_exp_yr": 7,
                "weather": "clear",
                "depot_zone": "South",
            },
        )
        extra_field = client.post(
            "/predict",
            json={
                "route_km": 15.5,
                "stops": 22,
                "passengers": 38,
                "hour": 10,
                "temp_c": 29.0,
                "driver_exp_yr": 7,
                "weather": "clear",
                "depot_zone": "South",
                "record_id": "not-accepted",
            },
        )

    assert invalid_route.status_code == 422
    assert invalid_stops.status_code == 422
    assert extra_field.status_code == 422


def test_predict_with_real_model():
    """Integration test: Verify the real champion model artifact produces predictions."""
    from pathlib import Path
    model_path = Path("models/model.joblib")
    if not model_path.exists():
        model_path = Path(__file__).resolve().parent.parent / "models/model.joblib"

    if model_path.exists():
        with TestClient(create_app(model_path=model_path)) as client:
            response = client.post(
                "/predict",
                json={
                    "route_km": 15.5,
                    "stops": 22,
                    "passengers": 38,
                    "hour": 10,
                    "temp_c": 29.0,
                    "driver_exp_yr": 7.0,
                    "weather": "clear",
                    "depot_zone": "South",
                },
            )
            assert response.status_code == 200
            data = response.json()
            assert "prediction" in data
            assert isinstance(data["prediction"], float)
            assert data["prediction"] > 0.0


def test_predict_missing_required_fields_rejected():
    """Verify that requests missing any of the 8 required fields return HTTP 422."""
    with make_client() as client:
        # Missing weather and depot_zone
        response = client.post(
            "/predict",
            json={
                "route_km": 15.5,
                "stops": 22,
                "passengers": 38,
                "hour": 10,
                "temp_c": 29.0,
                "driver_exp_yr": 7.0,
            },
        )
        assert response.status_code == 422
