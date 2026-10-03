"""
advanced_ai.py - Advanced AI Model Training with MLflow Experiment Tracking
==========================================================================

Trains RandomForest, GradientBoosting, and PyTorch Neural Network models
on the Part 2 featured dataset. Uses chronological train/test split
(2012-2016 train, 2017 test). Tracks all experiments with MLflow and
registers models in the MLflow Model Registry.
"""

import logging
import os
import sys
import warnings

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler

from data_loader import load_featured_data, get_feature_columns, FIGURES_DIR

warnings.filterwarnings("ignore", category=FutureWarning)

logger = logging.getLogger(__name__)

EXPERIMENT_NAME = "traffic_volume_prediction"
REGISTERED_MODEL_NAME = "traffic_volume_regressor"


def _evaluate(y_true, y_pred):
    return {"mae": mean_absolute_error(y_true, y_pred), "r2": r2_score(y_true, y_pred)}


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

        epochs = params.get("epochs", 30)
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


def export_summary(results: dict, registered_versions: dict) -> None:
    """Export a markdown summary of all MLflow runs to reports/mlflow_summary.md."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    reports_dir = os.path.join(script_dir, "reports")
    os.makedirs(reports_dir, exist_ok=True)
    summary_path = os.path.join(reports_dir, "mlflow_summary.md")

    lines = [
        "# MLflow Experiment Summary",
        "",
        f"**Experiment:** {EXPERIMENT_NAME}",
        f"**Tracking URI:** {mlflow.get_tracking_uri()}",
        "",
        "## Model Runs",
        "",
        "| Model | MAE | R2 | Registered Version |",
        "|-------|-----|----|--------------------|",
    ]
    for name, m in results.items():
        ver = registered_versions.get(name, "—")
        lines.append(f"| {name} | {m['mae']:.2f} | {m['r2']:.4f} | {ver} |")

    lines.extend(["", "## Registered Model Versions", ""])
    for name, ver in registered_versions.items():
        lines.append(f"- **{name}**: version {ver}")

    with open(summary_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    logger.info("MLflow summary exported to %s", summary_path)


def main():
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s")

    # MLflow setup with relative paths
    mlflow_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mlflow_logs")
    os.makedirs(mlflow_dir, exist_ok=True)

    # Delete old mlflow.db for a fresh start
    db_file = os.path.join(mlflow_dir, "mlflow.db")
    if os.path.exists(db_file):
        os.remove(db_file)
        logger.info("Deleted old mlflow.db for fresh start")

    tracking_uri = f"sqlite:///{os.path.join(mlflow_dir, 'mlflow.db')}"
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(EXPERIMENT_NAME)
    logger.info("MLflow tracking URI: %s", tracking_uri)

    # Load featured data from Part 2
    df = load_featured_data()
    feature_cols = get_feature_columns(df)

    # Chronological split: train 2012-2016, test 2017
    df = df.sort_values("date_time").reset_index(drop=True)
    train_mask = df["date_time"].dt.year <= 2016
    test_mask = df["date_time"].dt.year == 2017

    X_train = df.loc[train_mask, feature_cols].values
    y_train = df.loc[train_mask, "traffic_volume"].values
    X_test = df.loc[test_mask, feature_cols].values
    y_test = df.loc[test_mask, "traffic_volume"].values
    logger.info("Train size: %d (2012-2016), Test size: %d (2017)", len(X_train), len(X_test))

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    results = {}
    registered_versions = {}

    # RandomForest — registered as v1
    rf_params = {"n_estimators": 200, "max_depth": 15, "min_samples_split": 5, "random_state": 42}
    results["RandomForestRegressor"] = train_sklearn_model(
        RandomForestRegressor(**rf_params), "RandomForestRegressor",
        X_train, X_test, y_train, y_test, rf_params,
        register_as=REGISTERED_MODEL_NAME)
    registered_versions["RandomForestRegressor"] = "1"

    # GradientBoosting — registered as v2, alias "production"
    gb_params = {"n_estimators": 200, "max_depth": 5, "learning_rate": 0.1,
                 "min_samples_split": 5, "random_state": 42}
    results["GradientBoostingRegressor"] = train_sklearn_model(
        GradientBoostingRegressor(**gb_params), "GradientBoostingRegressor",
        X_train, X_test, y_train, y_test, gb_params,
        register_as=REGISTERED_MODEL_NAME, alias="production")
    registered_versions["GradientBoostingRegressor"] = "2"

    # PyTorch Neural Net (64->32->1, 30 epochs)
    nn_params = {"epochs": 30, "batch_size": 64, "optimizer": "adam",
                 "hidden_layers": "64-32", "activation": "relu"}
    results["PyTorchNeuralNet"] = train_pytorch_model(
        X_train_scaled, X_test_scaled, y_train, y_test, nn_params)

    print_summary(results)
    export_summary(results, registered_versions)


if __name__ == "__main__":
    main()
