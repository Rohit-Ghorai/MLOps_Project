"""FastAPI prediction service for MP063 City Bus Delay."""

from contextlib import asynccontextmanager
from datetime import datetime, timezone
import json
import logging
import os
from pathlib import Path
import sys
from typing import Any, Protocol

import joblib
import pandas as pd
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, ConfigDict, Field


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = Path(os.environ.get("MODEL_PATH", PROJECT_ROOT / "models/model.joblib"))
FRONTEND_PATH = Path(__file__).resolve().parent / "frontend.html"


class PredictionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    route_km: float = Field(ge=1.5)
    stops: int = Field(le=70)
    passengers: int
    hour: int
    temp_c: float
    driver_exp_yr: float
    weather: str
    depot_zone: str


class PredictionResponse(BaseModel):
    prediction: float


class Predictor(Protocol):
    def predict(self, features: pd.DataFrame) -> Any: ...


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        entry = {
            "timestamp": datetime.fromtimestamp(record.created, timezone.utc).isoformat(),
            "level": record.levelname,
            "message": record.getMessage(),
        }
        for field in ("event", "prediction"):
            if hasattr(record, field):
                entry[field] = getattr(record, field)
        return json.dumps(entry, separators=(",", ":"))


class StdoutHandler(logging.StreamHandler):
    def emit(self, record: logging.LogRecord) -> None:
        self.stream = sys.stdout
        super().emit(record)


logger = logging.getLogger("mp063.api")
logger.setLevel(logging.INFO)
logger.propagate = False
if not logger.handlers:
    stream_handler = StdoutHandler()
    stream_handler.setFormatter(JsonFormatter())
    logger.addHandler(stream_handler)


def load_model(model_path: Path) -> Predictor:
    if not model_path.is_file():
        raise FileNotFoundError(f"Model artifact not found: {model_path}")
    return joblib.load(model_path)


def create_app(
    model_path: Path = MODEL_PATH,
    predictor: Predictor | None = None,
) -> FastAPI:
    @asynccontextmanager
    async def lifespan(application: FastAPI):
        application.state.model = predictor if predictor is not None else load_model(model_path)
        yield

    application = FastAPI(title="MP063 City Bus Delay API", lifespan=lifespan)

    @application.get("/", response_class=HTMLResponse)
    def home() -> HTMLResponse:
        if FRONTEND_PATH.is_file():
            return HTMLResponse(content=FRONTEND_PATH.read_text(encoding="utf-8"))
        return HTMLResponse(content="<h1>Nagpur City Bus Delay API (MP063)</h1><p>Visit <a href='/docs'>/docs</a></p>")

    @application.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @application.post("/predict", response_model=PredictionResponse)
    def predict(payload: PredictionRequest, request: Request) -> PredictionResponse:
        features = pd.DataFrame([payload.model_dump()])
        model: Predictor = request.app.state.model
        prediction = float(model.predict(features)[0])
        logger.info(
            "Prediction generated",
            extra={"event": "prediction", "prediction": round(prediction, 4)},
        )
        return PredictionResponse(prediction=round(prediction, 4))

    return application


app = create_app()
