"""
advanced_ai.py - Advanced AI Model Training with MLflow Experiment Tracking
==========================================================================

Why MLflow was chosen:
    MLflow is an open-source platform for managing the end-to-end machine learning
    lifecycle. It was selected for this capstone for several reasons:

    1. **Reproducibility** - Every experiment run records the exact hyperparameters,
       metrics, and code version used, making it trivial to reproduce any result.

    2. **Model Comparison** - MLflow's tracking API lets us compare runs side-by-side.
       Here we train three model families (Random Forest, Gradient Boosting, Neural
       Network) and need a structured way to evaluate them.

    3. **Artifact Management** - Trained models, plots, and metadata are stored
       alongside the run that produced them, avoiding orphaned model files.

    4. **Standardised API** - The same log_param / log_metric pattern works regardless
       of the underlying framework, reducing cognitive overhead.

How it is implemented:
    - A local file-based tracking URI (capstone_part3/mlflow_logs/) stores all data.
    - Each model is trained inside its own mlflow.start_run() context.
    - Hyperparameters are logged with mlflow.log_param, metrics with mlflow.log_metric,
      and serialised models with the appropriate log_model call.

Limitations:
    - Adds disk I/O overhead to each run.
    - New users must learn MLflow concepts (experiments, runs, artifacts).
    - Local file store only; collaborative teams need a remote tracking server.
    - No Model Registry used here; that would be the next step for production.
"""

import logging
import os
import warnings

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=FutureWarning)

logger = logging.getLogger(__name__)

DATA_PATH = os.path.join("capstone_part2", "cleaned_traffic.csv")
MLFLOW_TRACKING_DIR = os.path.join("capstone_part3", "mlflow_logs")
EXPERIMENT_NAME = "traffic_volume_prediction"


def load_data(path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(path, parse_dates=["date_time"])
        logger.info("Loaded dataset: %d rows, %d columns", len(df), len(df.columns))
        return df
    except FileNotFoundError:
        logger.error("Dataset not found at %s", path)
        raise


def engineer_features(df: pd.DataFrame):
    df = df.copy()
    df["hour"] = df["date_time"].dt.hour
    df["day_of_week"] = df["date_time"].dt.dayofweek
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)
    weather_dummies = pd.get_dummies(df["weather_main"], prefix="weather")
    df = pd.concat([df, weather_dummies], axis=1)
    df["is_holiday"] = (df["holiday"] != "None").astype(int)

    numeric_features = ["temp", "rain_1h", "snow_1h", "clouds_all"]
    time_features = ["hour", "day_of_week", "is_weekend",
                     "hour_sin", "hour_cos", "dow_sin", "dow_cos"]
    weather_features = [c for c in df.columns if c.startswith("weather_")
                        and c not in ("weather_main", "weather_description")]
    other_features = ["is_holiday"]
    feature_cols = numeric_features + time_features + weather_features + other_features
    logger.info("Total feature count: %d", len(feature_cols))
    return df, feature_cols


def _evaluate(y_true, y_pred):
    return {"mae": mean_absolute_error(y_true, y_pred), "r2": r2_score(y_true, y_pred)}


REGISTERED_MODEL_NAME = "traffic_volume_regressor"


def train_sklearn_model(model, model_name, X_train, X_test, y_train, y_test, params,
                        register_as=None, alias=None):
    with mlflow.start_run(run_name=model_name):
        for key, value in params.items():
            mlflow.log_param(key, value)
        logger.info("Training %s ...", model_name)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        metrics = _evaluate(y_test, y_pred)
        for k, v in metrics.items():
            mlflow.log_metric(k, v)

        if register_as:
            model_info = mlflow.sklearn.log_model(
                model, artifact_path="model",
                registered_model_name=register_as,
                skops_trusted_types=["sklearn.tree._tree.Tree",
                                     "numpy.float64", "numpy.int64"],
            )
            logger.info("Registered %s as %s version %s",
                        model_name, register_as, model_info.registered_model_version)
            if alias:
                client = mlflow.tracking.MlflowClient()
                client.set_registered_model_alias(
                    register_as, alias, model_info.registered_model_version
                )
                logger.info("Set alias '%s' on version %s", alias, model_info.registered_model_version)
        else:
            mlflow.sklearn.log_model(model, artifact_path="model",
                                     skops_trusted_types=["sklearn.tree._tree.Tree",
                                                          "numpy.float64", "numpy.int64"])

        logger.info("%s - MAE: %.2f, R2: %.4f", model_name, metrics["mae"], metrics["r2"])
    return metrics


class SimpleNet(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 1),
        )
    def forward(self, x):
        return self.net(x).squeeze(-1)


def train_pytorch_model(X_train_scaled, X_test_scaled, y_train, y_test, params):
    model_name = "PyTorchNeuralNet"
    with mlflow.start_run(run_name=model_name):
        for key, value in params.items():
            mlflow.log_param(key, value)

        logger.info("Training %s ...", model_name)
        model = SimpleNet(X_train_scaled.shape[1])
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.MSELoss()

        X_t = torch.FloatTensor(X_train_scaled)
        y_t = torch.FloatTensor(y_train)
        loader = DataLoader(TensorDataset(X_t, y_t),
                            batch_size=params.get("batch_size", 64), shuffle=True)

        epochs = params.get("epochs", 50)
        for epoch in range(epochs):
            model.train()
            for xb, yb in loader:
                optimizer.zero_grad()
                loss = criterion(model(xb), yb)
                loss.backward()
                optimizer.step()

        model.eval()
        with torch.no_grad():
            y_pred = model(torch.FloatTensor(X_test_scaled)).numpy()
        metrics = _evaluate(y_test, y_pred)

        for k, v in metrics.items():
            mlflow.log_metric(k, v)

        logger.info("%s - MAE: %.2f, R2: %.4f", model_name, metrics["mae"], metrics["r2"])
    return metrics


def print_summary(results: dict) -> None:
    header = f"{'Model':<30} {'MAE':>12} {'R2':>12}"
    sep = "-" * len(header)
    print(f"\n{sep}")
    print("  MLflow Experiment Summary")
    print(sep)
    print(header)
    print(sep)
    for name, m in results.items():
        print(f"{name:<30} {m['mae']:>12.2f} {m['r2']:>12.4f}")
    print(sep)
    print(f"Tracking URI: {mlflow.get_tracking_uri()}")
    print(f"Experiment  : {EXPERIMENT_NAME}")
    print(sep + "\n")


def main():
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s")

    os.makedirs(MLFLOW_TRACKING_DIR, exist_ok=True)
    db_path = os.path.abspath(os.path.join(MLFLOW_TRACKING_DIR, "mlflow.db"))
    tracking_uri = f"sqlite:///{db_path}"
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(EXPERIMENT_NAME)
    logger.info("MLflow tracking URI: %s", tracking_uri)

    df = load_data(DATA_PATH)
    df, feature_cols = engineer_features(df)

    X = df[feature_cols].values
    y = df["traffic_volume"].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
    logger.info("Train size: %d, Test size: %d", len(X_train), len(X_test))

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    results = {}

    rf_params = {"n_estimators": 200, "max_depth": 15, "min_samples_split": 5, "random_state": 42}
    results["RandomForestRegressor"] = train_sklearn_model(
        RandomForestRegressor(**rf_params), "RandomForestRegressor",
        X_train, X_test, y_train, y_test, rf_params,
        register_as=REGISTERED_MODEL_NAME)

    gb_params = {"n_estimators": 200, "max_depth": 5, "learning_rate": 0.1,
                 "min_samples_split": 5, "random_state": 42}
    results["GradientBoostingRegressor"] = train_sklearn_model(
        GradientBoostingRegressor(**gb_params), "GradientBoostingRegressor",
        X_train, X_test, y_train, y_test, gb_params,
        register_as=REGISTERED_MODEL_NAME, alias="production")

    nn_params = {"epochs": 50, "batch_size": 64, "optimizer": "adam",
                 "hidden_layers": "64-32", "activation": "relu"}
    results["PyTorchNeuralNet"] = train_pytorch_model(
        X_train_scaled, X_test_scaled, y_train, y_test, nn_params)

    print_summary(results)


if __name__ == "__main__":
    main()
