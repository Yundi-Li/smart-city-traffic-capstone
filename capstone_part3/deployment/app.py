"""FastAPI application for traffic volume prediction using GradientBoostingRegressor."""

import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import OneHotEncoder

logger = logging.getLogger(__name__)

# Global model state
model: Optional[GradientBoostingRegressor] = None
encoder: Optional[OneHotEncoder] = None
weather_categories: Optional[list] = None
quartile_thresholds: Optional[dict] = None

FEATURE_COLS = ["hour", "day_of_week", "is_weekend", "temp", "rain_1h", "snow_1h", "clouds_all"]


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


def _build_features(df: pd.DataFrame, encoder: OneHotEncoder, fit: bool = False) -> np.ndarray:
    """Build the feature matrix from a dataframe containing the raw columns."""
    numeric = df[FEATURE_COLS].values
    weather = df[["weather_main"]].values
    if fit:
        encoded = encoder.fit_transform(weather).toarray()
    else:
        encoded = encoder.transform(weather).toarray()
    return np.hstack([numeric, encoded])


def train_model() -> None:
    """Train a GradientBoostingRegressor on the traffic dataset."""
    global model, encoder, weather_categories, quartile_thresholds

    repo_root = Path(__file__).resolve().parents[2]
    data_path = repo_root / "data" / "Metro_Interstate_Traffic_Volume.csv"

    logger.info("Loading dataset from %s", data_path)
    df = pd.read_csv(data_path)

    # Feature engineering
    df["date_time"] = pd.to_datetime(df["date_time"])
    df["hour"] = df["date_time"].dt.hour
    df["day_of_week"] = df["date_time"].dt.dayofweek
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

    # Compute quartile thresholds for congestion classification
    q25 = float(df["traffic_volume"].quantile(0.25))
    q50 = float(df["traffic_volume"].quantile(0.50))
    q75 = float(df["traffic_volume"].quantile(0.75))
    quartile_thresholds = {"q25": q25, "q50": q50, "q75": q75}
    logger.info("Quartile thresholds: %s", quartile_thresholds)

    # One-hot encode weather_main
    encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    X = _build_features(df, encoder, fit=True)
    y = df["traffic_volume"].values

    weather_categories = list(encoder.categories_[0])

    model = GradientBoostingRegressor(n_estimators=200, max_depth=5, random_state=42)
    logger.info("Training GradientBoostingRegressor (%d samples, %d features)", X.shape[0], X.shape[1])
    model.fit(X, y)
    logger.info("Model training complete")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Train model on startup."""
    train_model()
    yield


app = FastAPI(title="Traffic Volume Prediction API", lifespan=lifespan)


@app.get("/health", response_model=HealthResponse)
async def health():
    """Return service health status."""
    return HealthResponse(status="healthy", model_type="GradientBoostingRegressor")


@app.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest):
    """Predict traffic volume and congestion level."""
    if model is None or encoder is None or quartile_thresholds is None:
        raise HTTPException(status_code=503, detail="Model not ready")

    try:
        input_df = pd.DataFrame([{
            "hour": request.hour,
            "day_of_week": request.day_of_week,
            "is_weekend": request.is_weekend,
            "temp": request.temp,
            "rain_1h": request.rain_1h,
            "snow_1h": request.snow_1h,
            "clouds_all": request.clouds_all,
            "weather_main": request.weather_main,
        }])

        X = _build_features(input_df, encoder, fit=False)
        prediction = float(model.predict(X)[0])
        prediction = max(prediction, 0.0)

        # Classify congestion based on quartile thresholds
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
