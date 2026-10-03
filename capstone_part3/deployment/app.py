"""FastAPI application for traffic volume prediction.

At startup, tries to load the production model from MLflow registry.
Falls back to training a GradientBoostingRegressor on featured data.
"""

import json
import logging
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from sklearn.ensemble import GradientBoostingRegressor

logger = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).resolve().parent
PART3_DIR = SCRIPT_DIR.parent
REPO_ROOT = PART3_DIR.parent

# Add capstone_part3 to path for data_loader
sys.path.insert(0, str(PART3_DIR))
from data_loader import load_featured_data, get_feature_columns

# Global model state
model: Optional[object] = None
feature_cols: Optional[list] = None
quartile_thresholds: Optional[dict] = None
model_source: str = "unknown"


class PredictRequest(BaseModel):
    hour: int = Field(..., ge=0, le=23)
    day_of_week: int = Field(..., ge=0, le=6)
    is_weekend: int = Field(..., ge=0, le=1)
    temp: float
    rain_1h: float = Field(..., ge=0.0)
    snow_1h: float = Field(..., ge=0.0)
    clouds_all: int = Field(..., ge=0, le=100)
    weather_main: str


class PredictResponse(BaseModel):
    predicted_traffic_volume: float
    congestion_level: str


class HealthResponse(BaseModel):
    status: str
    model_type: str
    model_source: str


def _load_mlflow_model():
    """Try to load the production model from MLflow registry."""
    global model, model_source

    mlflow_dir = PART3_DIR / "mlflow_logs"
    db_path = mlflow_dir / "mlflow.db"
    if not db_path.exists():
        logger.info("MLflow DB not found at %s, skipping registry load", db_path)
        return False

    try:
        import mlflow
        tracking_uri = f"sqlite:///{db_path}"
        mlflow.set_tracking_uri(tracking_uri)
        model = mlflow.pyfunc.load_model("models:/traffic_volume_regressor@production")
        model_source = "mlflow_registry"
        logger.info("Loaded production model from MLflow registry")
        return True
    except Exception as e:
        logger.warning("Failed to load MLflow model: %s", e)
        return False


def _load_joblib_model():
    """Load production model from the portable joblib folder."""
    global model, model_source
    import joblib

    model_path = PART3_DIR / "models" / "production_model" / "gradient_boosting.joblib"
    if not model_path.exists():
        logger.info("Joblib model not found at %s", model_path)
        return False

    model = joblib.load(model_path)
    model_source = "joblib_folder"
    logger.info("Loaded production model from %s", model_path)
    return True


def _train_fallback_model():
    """Train a GBR on featured data as last-resort fallback."""
    global model, model_source

    df = load_featured_data()
    fcols = get_feature_columns(df)
    X = df[fcols].values
    y = df["traffic_volume"].values

    gbr = GradientBoostingRegressor(n_estimators=200, max_depth=5, random_state=42)
    logger.info("Training fallback GBR on %d samples, %d features", X.shape[0], X.shape[1])
    gbr.fit(X, y)

    model = gbr
    model_source = "fallback_trained"
    logger.info("Fallback model training complete")


def _init_model():
    """Initialize model and feature columns at startup."""
    global feature_cols, quartile_thresholds

    df = load_featured_data()
    feature_cols = get_feature_columns(df)

    # Compute quartile thresholds for congestion classification
    q25 = float(df["traffic_volume"].quantile(0.25))
    q50 = float(df["traffic_volume"].quantile(0.50))
    q75 = float(df["traffic_volume"].quantile(0.75))
    quartile_thresholds = {"q25": q25, "q50": q50, "q75": q75}

    if not _load_mlflow_model():
        if not _load_joblib_model():
            _train_fallback_model()


def _build_feature_row(request: PredictRequest) -> np.ndarray:
    """Build a feature vector from a prediction request."""
    hour = request.hour
    dow = request.day_of_week
    row = {
        "hour_sin": np.sin(2 * np.pi * hour / 24),
        "hour_cos": np.cos(2 * np.pi * hour / 24),
        "dow_sin": np.sin(2 * np.pi * dow / 7),
        "dow_cos": np.cos(2 * np.pi * dow / 7),
        "is_weekend": request.is_weekend,
        "is_holiday": 0,
        "is_low_visibility": 1 if request.weather_main in ("Fog", "Mist", "Haze", "Smoke") else 0,
        "weather_severity": {"Clear": 0, "Clouds": 1, "Mist": 2, "Haze": 2, "Drizzle": 2,
                             "Rain": 3, "Fog": 3, "Snow": 4, "Thunderstorm": 4,
                             "Squall": 4, "Smoke": 4}.get(request.weather_main, 2),
        "temp": request.temp,
        "rain_1h": request.rain_1h,
        "snow_1h": request.snow_1h,
        "clouds_all": request.clouds_all,
    }
    # Add weather one-hot columns (set matching one to 1, rest to 0)
    for col in feature_cols:
        if col.startswith("weather_") and col not in row:
            expected_weather = col.replace("weather_", "")
            row[col] = 1 if request.weather_main == expected_weather else 0

    return np.array([[row.get(c, 0) for c in feature_cols]])


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load or train model on startup."""
    _init_model()
    yield


app = FastAPI(title="Traffic Volume Prediction API", lifespan=lifespan)


@app.get("/health", response_model=HealthResponse)
async def health():
    """Return service health status."""
    return HealthResponse(
        status="healthy",
        model_type=type(model).__name__ if model else "none",
        model_source=model_source,
    )


@app.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest):
    """Predict traffic volume and congestion level."""
    if model is None or feature_cols is None or quartile_thresholds is None:
        raise HTTPException(status_code=503, detail="Model not ready")

    try:
        X = _build_feature_row(request)

        if model_source == "mlflow_registry":
            prediction = float(model.predict(pd.DataFrame(X, columns=feature_cols))[0])
        else:
            prediction = float(model.predict(X)[0])  # joblib or fallback sklearn

        prediction = max(prediction, 0.0)

        if prediction <= quartile_thresholds["q25"]:
            congestion = "low"
        elif prediction <= quartile_thresholds["q50"]:
            congestion = "moderate"
        elif prediction <= quartile_thresholds["q75"]:
            congestion = "high"
        else:
            congestion = "very_high"

        return PredictResponse(
            predicted_traffic_volume=round(prediction, 2),
            congestion_level=congestion,
        )

    except Exception as e:
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/status")
async def monitoring_status():
    """Return the latest monitoring status from monitoring_status.json."""
    status_path = PART3_DIR / "reports" / "monitoring_status.json"
    try:
        with open(status_path) as f:
            return json.load(f)
    except FileNotFoundError:
        return {"overall_status": "UNKNOWN", "message": "Run monitoring.py first"}
