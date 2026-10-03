"""
advanced_ai.py - MLflow Experiment Tracking for All Models
==========================================================

Trains and tracks ALL models (regression + classification) with MLflow.
Chronological split: train 2012-2016, test 2017.
Registers traffic_volume_regressor: v1=LinearRegression, v2=GradientBoosting (production).
Saves production model to a portable folder for deployment fallback.
"""

import logging
import os
import sys
import warnings
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (mean_absolute_error, r2_score, accuracy_score,
                             precision_score, recall_score, f1_score, roc_auc_score)
from sklearn.preprocessing import StandardScaler

from data_loader import load_featured_data, get_feature_columns, add_proxy_label, SEVERE_WEATHER

warnings.filterwarnings("ignore", category=FutureWarning)
logger = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).resolve().parent
MLFLOW_DIR = SCRIPT_DIR / "mlflow_logs"
ARTIFACTS_DIR = MLFLOW_DIR / "artifacts"
MODELS_DIR = SCRIPT_DIR / "models" / "production_model"
EXPERIMENT_NAME = "traffic_volume_prediction"
REGISTERED_MODEL_NAME = "traffic_volume_regressor"


def _eval_reg(y_true, y_pred):
    return {"mae": float(mean_absolute_error(y_true, y_pred)),
            "r2": float(r2_score(y_true, y_pred))}


def _eval_cls(y_true, y_pred, y_prob):
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
    }


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


def main():
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s")

    # Clean slate
    db_file = MLFLOW_DIR / "mlflow.db"
    if db_file.exists():
        db_file.unlink()
    MLFLOW_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    tracking_uri = f"sqlite:///{db_file}"
    mlflow.set_tracking_uri(tracking_uri)

    # Create experiment with artifact_location inside repo
    artifact_loc = f"file:{ARTIFACTS_DIR}"
    client = mlflow.tracking.MlflowClient()
    try:
        exp_id = client.create_experiment(EXPERIMENT_NAME, artifact_location=artifact_loc)
    except mlflow.exceptions.MlflowException:
        exp = client.get_experiment_by_name(EXPERIMENT_NAME)
        exp_id = exp.experiment_id
    mlflow.set_experiment(EXPERIMENT_NAME)
    logger.info("MLflow tracking URI: %s", tracking_uri)
    logger.info("Artifact location: %s", artifact_loc)

    # Load data
    df = load_featured_data()
    df = add_proxy_label(df)
    feature_cols = get_feature_columns(df)

    df = df.sort_values("date_time").reset_index(drop=True)
    train_mask = (df["date_time"].dt.year <= 2016).values
    test_mask = (df["date_time"].dt.year == 2017).values

    X_train, X_test = df.loc[train_mask, feature_cols].values, df.loc[test_mask, feature_cols].values
    y_reg_train = df.loc[train_mask, "traffic_volume"].values
    y_reg_test = df.loc[test_mask, "traffic_volume"].values
    y_cls_train = df.loc[train_mask, "high_risk"].values
    y_cls_test = df.loc[test_mask, "high_risk"].values

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    logger.info("Train: %d (2012-2016), Test: %d (2017)", train_mask.sum(), test_mask.sum())

    results = {}
    registered = {}
    skops_types = ["sklearn.tree._tree.Tree", "numpy.float64", "numpy.int64"]

    # === REGRESSION MODELS ===

    # LinearRegression — registered as v1 (baseline)
    with mlflow.start_run(run_name="LinearRegression"):
        lr = LinearRegression()
        mlflow.log_param("model_type", "LinearRegression")
        lr.fit(X_train, y_reg_train)
        m = _eval_reg(y_reg_test, lr.predict(X_test))
        for k, v in m.items():
            mlflow.log_metric(k, v)
        info = mlflow.sklearn.log_model(lr, artifact_path="model",
                                        registered_model_name=REGISTERED_MODEL_NAME)
        registered["LinearRegression"] = info.registered_model_version
        logger.info("LinearRegression — MAE: %.2f, R²: %.4f (registered v%s)",
                    m["mae"], m["r2"], info.registered_model_version)
    results["LinearRegression"] = m

    # GradientBoosting — registered as v2, alias "production"
    with mlflow.start_run(run_name="GradientBoostingRegressor"):
        gb = GradientBoostingRegressor(n_estimators=200, max_depth=5, learning_rate=0.1, random_state=42)
        for k, v in {"n_estimators": 200, "max_depth": 5, "learning_rate": 0.1}.items():
            mlflow.log_param(k, v)
        gb.fit(X_train, y_reg_train)
        m = _eval_reg(y_reg_test, gb.predict(X_test))
        for k, v in m.items():
            mlflow.log_metric(k, v)
        info = mlflow.sklearn.log_model(gb, artifact_path="model",
                                        registered_model_name=REGISTERED_MODEL_NAME,
                                        skops_trusted_types=skops_types)
        ver = info.registered_model_version
        client.set_registered_model_alias(REGISTERED_MODEL_NAME, "production", ver)
        registered["GradientBoostingRegressor"] = ver
        logger.info("GradientBoosting — MAE: %.2f, R²: %.4f (registered v%s, alias=production)",
                    m["mae"], m["r2"], ver)
    results["GradientBoostingRegressor"] = m

    # RandomForest (regression, not registered)
    with mlflow.start_run(run_name="RandomForestRegressor"):
        rf_reg = RandomForestRegressor(n_estimators=200, max_depth=15, random_state=42)
        mlflow.log_param("n_estimators", 200)
        mlflow.log_param("max_depth", 15)
        rf_reg.fit(X_train, y_reg_train)
        m = _eval_reg(y_reg_test, rf_reg.predict(X_test))
        for k, v in m.items():
            mlflow.log_metric(k, v)
        mlflow.sklearn.log_model(rf_reg, artifact_path="model", skops_trusted_types=skops_types)
        logger.info("RandomForest (reg) — MAE: %.2f, R²: %.4f", m["mae"], m["r2"])
    results["RandomForestRegressor"] = m

    # PyTorch NN
    with mlflow.start_run(run_name="PyTorchNeuralNet"):
        for k, v in {"epochs": 30, "batch_size": 64, "hidden": "64-32"}.items():
            mlflow.log_param(k, v)
        net = SimpleNet(X_train_s.shape[1])
        opt = torch.optim.Adam(net.parameters(), lr=0.001)
        crit = nn.MSELoss()
        loader = DataLoader(TensorDataset(torch.FloatTensor(X_train_s), torch.FloatTensor(y_reg_train)),
                            batch_size=64, shuffle=True)
        for _ in range(30):
            net.train()
            for xb, yb in loader:
                opt.zero_grad(); loss = crit(net(xb), yb); loss.backward(); opt.step()
        net.eval()
        with torch.no_grad():
            pred = net(torch.FloatTensor(X_test_s)).numpy()
        m = _eval_reg(y_reg_test, pred)
        for k, v in m.items():
            mlflow.log_metric(k, v)
        logger.info("PyTorchNN — MAE: %.2f, R²: %.4f", m["mae"], m["r2"])
    results["PyTorchNeuralNet"] = m

    # === CLASSIFICATION MODELS ===

    # LogisticRegression (classification)
    with mlflow.start_run(run_name="LogisticRegression_cls"):
        lr_cls = LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced")
        mlflow.log_param("model_type", "LogisticRegression")
        mlflow.log_param("class_weight", "balanced")
        lr_cls.fit(X_train_s, y_cls_train)
        y_pred = lr_cls.predict(X_test_s)
        y_prob = lr_cls.predict_proba(X_test_s)[:, 1]
        m = _eval_cls(y_cls_test, y_pred, y_prob)
        for k, v in m.items():
            mlflow.log_metric(k, v)
        mlflow.sklearn.log_model(lr_cls, artifact_path="model")
        logger.info("LogisticRegression (cls) — Accuracy: %.4f, F1: %.4f, AUC: %.4f",
                    m["accuracy"], m["f1"], m["roc_auc"])
    results["LogisticRegression_cls"] = m

    # RandomForest (classification)
    with mlflow.start_run(run_name="RandomForestClassifier"):
        rf_cls = RandomForestClassifier(n_estimators=100, random_state=42, class_weight="balanced")
        mlflow.log_param("n_estimators", 100)
        mlflow.log_param("class_weight", "balanced")
        rf_cls.fit(X_train, y_cls_train)
        y_pred = rf_cls.predict(X_test)
        y_prob = rf_cls.predict_proba(X_test)[:, 1]
        m = _eval_cls(y_cls_test, y_pred, y_prob)
        for k, v in m.items():
            mlflow.log_metric(k, v)
        mlflow.sklearn.log_model(rf_cls, artifact_path="model", skops_trusted_types=skops_types)
        logger.info("RandomForest (cls) — Accuracy: %.4f, F1: %.4f, AUC: %.4f",
                    m["accuracy"], m["f1"], m["roc_auc"])
    results["RandomForestClassifier"] = m

    # === SAVE PRODUCTION MODEL TO PORTABLE FOLDER ===
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    prod_path = MODELS_DIR / "gradient_boosting.joblib"
    joblib.dump(gb, prod_path)
    # Also save feature columns and scaler
    joblib.dump(feature_cols, MODELS_DIR / "feature_cols.joblib")
    joblib.dump(scaler, MODELS_DIR / "scaler.joblib")
    logger.info("Production model saved to %s", prod_path)

    # === PRINT & EXPORT ===
    print("\n" + "=" * 80)
    print("  MLflow Experiment Summary — All Models")
    print("=" * 80)
    print(f"{'Model':<30} {'MAE/Acc':>12} {'R²/AUC':>12} {'Registry':>12}")
    print("-" * 80)
    for name, m in results.items():
        if "mae" in m:
            print(f"{name:<30} {m['mae']:>12.2f} {m['r2']:>12.4f} {registered.get(name, '—'):>12}")
        else:
            print(f"{name:<30} {m['accuracy']:>12.4f} {m['roc_auc']:>12.4f} {'—':>12}")
    print("=" * 80)

    # Export summary
    reports_dir = SCRIPT_DIR / "reports"
    reports_dir.mkdir(exist_ok=True)
    lines = [
        "# MLflow Experiment Summary",
        "",
        f"**Experiment:** {EXPERIMENT_NAME}",
        f"**Tracking URI:** sqlite:///capstone_part3/mlflow_logs/mlflow.db",
        f"**Artifact location:** capstone_part3/mlflow_logs/artifacts/",
        "",
        "## Regression Runs",
        "",
        "| Model | MAE | R² | Registry |",
        "|-------|-----|-----|----------|",
    ]
    for name in ["LinearRegression", "GradientBoostingRegressor", "RandomForestRegressor", "PyTorchNeuralNet"]:
        m = results[name]
        ver = registered.get(name, "—")
        alias = " (production)" if name == "GradientBoostingRegressor" else ""
        lines.append(f"| {name} | {m['mae']:.2f} | {m['r2']:.4f} | v{ver}{alias} |" if ver != "—"
                     else f"| {name} | {m['mae']:.2f} | {m['r2']:.4f} | — |")

    lines.extend([
        "",
        "## Classification Runs",
        "",
        "| Model | Accuracy | Precision | Recall | F1 | ROC AUC |",
        "|-------|----------|-----------|--------|-----|---------|",
    ])
    for name in ["LogisticRegression_cls", "RandomForestClassifier"]:
        m = results[name]
        lines.append(f"| {name} | {m['accuracy']:.4f} | {m['precision']:.4f} | {m['recall']:.4f} | {m['f1']:.4f} | {m['roc_auc']:.4f} |")

    lines.extend([
        "",
        "## Model Registry",
        "",
        f"- `{REGISTERED_MODEL_NAME}` v{registered['LinearRegression']} = LinearRegression (baseline)",
        f"- `{REGISTERED_MODEL_NAME}` v{registered['GradientBoostingRegressor']} = GradientBoosting (alias: **production**)",
        "",
        "## Portable Fallback",
        "",
        "Production model also saved to `capstone_part3/models/production_model/` via joblib for deployment portability.",
    ])

    with open(reports_dir / "mlflow_summary.md", "w") as f:
        f.write("\n".join(lines) + "\n")
    logger.info("Summary exported to reports/mlflow_summary.md")


if __name__ == "__main__":
    main()
