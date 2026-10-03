"""
Deep Learning Module for Traffic Volume Prediction

Builds a feedforward neural network (PyTorch) to predict Metro Interstate
traffic volume and uses SHAP on a surrogate GradientBoostingRegressor for
feature-importance explainability.

SHAP is applied to a GradientBoostingRegressor instead of the neural network
directly because:
1. Neural networks are opaque — per-feature attribution is expensive and
   approximate (DeepExplainer / GradientExplainer can be unstable).
2. TreeExplainer is exact and fast for tree-based models.
3. GBR achieves comparable accuracy on tabular data, so feature-importance
   insights transfer well.
"""

import logging
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "Metro_Interstate_Traffic_Volume.csv")
FIGURES_DIR = os.path.join(BASE_DIR, "figures")


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["date_time"] = pd.to_datetime(df["date_time"])
    df["hour"] = df["date_time"].dt.hour
    df["day_of_week"] = df["date_time"].dt.dayofweek
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["day_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["day_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)
    df["is_holiday"] = (df["holiday"] != "None").astype(int)
    weather_dummies = pd.get_dummies(df["weather_main"], prefix="weather")
    df = pd.concat([df, weather_dummies], axis=1)
    drop_cols = ["holiday", "weather_main", "weather_description", "date_time",
                 "hour", "day_of_week"]
    df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True)
    return df


def prepare_data(df: pd.DataFrame):
    target = "traffic_volume"
    feature_cols = [c for c in df.columns if c != target]
    X = df[feature_cols].values.astype(np.float32)
    y = df[target].values.astype(np.float32)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test, y_train, y_test, feature_cols, scaler


class TrafficNet(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def train_nn(model, X_train, y_train, epochs=50, batch_size=256, patience=10):
    val_size = int(0.2 * len(X_train))
    X_val = torch.FloatTensor(X_train[-val_size:])
    y_val = torch.FloatTensor(y_train[-val_size:])
    X_t = torch.FloatTensor(X_train[:-val_size])
    y_t = torch.FloatTensor(y_train[:-val_size])

    loader = DataLoader(TensorDataset(X_t, y_t), batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    train_losses, val_losses, train_maes, val_maes = [], [], [], []
    best_val_loss = float("inf")
    best_state = None
    wait = 0

    for epoch in range(epochs):
        model.train()
        epoch_loss, epoch_mae, n = 0.0, 0.0, 0
        for xb, yb in loader:
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(xb)
            epoch_mae += (pred - yb).abs().sum().item()
            n += len(xb)
        train_losses.append(epoch_loss / n)
        train_maes.append(epoch_mae / n)

        model.eval()
        with torch.no_grad():
            val_pred = model(X_val)
            vl = criterion(val_pred, y_val).item()
            vm = (val_pred - y_val).abs().mean().item()
        val_losses.append(vl)
        val_maes.append(vm)

        if vl < best_val_loss:
            best_val_loss = vl
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                logger.info("Early stopping at epoch %d", epoch + 1)
                break

        if (epoch + 1) % 10 == 0:
            logger.info("Epoch %d/%d — train_loss=%.1f val_loss=%.1f val_mae=%.1f",
                        epoch + 1, epochs, train_losses[-1], vl, vm)

    if best_state:
        model.load_state_dict(best_state)

    return {"loss": train_losses, "val_loss": val_losses,
            "mae": train_maes, "val_mae": val_maes}


def plot_training_history(history, save_path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(history["loss"], label="Train Loss")
    axes[0].plot(history["val_loss"], label="Val Loss")
    axes[0].set_title("Training & Validation Loss (MSE)")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("MSE")
    axes[0].legend()

    axes[1].plot(history["mae"], label="Train MAE")
    axes[1].plot(history["val_mae"], label="Val MAE")
    axes[1].set_title("Training & Validation MAE")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("MAE")
    axes[1].legend()

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    logger.info("Training history plot saved to %s", save_path)


def shap_explainability(X_train, X_test, y_train, y_test, feature_names, save_path):
    logger.info("Training GradientBoostingRegressor for SHAP explainability")
    gbr = GradientBoostingRegressor(
        n_estimators=200, max_depth=5, learning_rate=0.1, random_state=42
    )
    gbr.fit(X_train, y_train)

    explainer = shap.TreeExplainer(gbr)
    shap_values = explainer.shap_values(X_test[:1000])

    shap.summary_plot(shap_values, X_test[:1000], feature_names=feature_names, show=False)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close("all")
    logger.info("SHAP summary plot saved to %s", save_path)

    gbr_preds = gbr.predict(X_test)
    gbr_mae = mean_absolute_error(y_test, gbr_preds)
    gbr_r2 = r2_score(y_test, gbr_preds)
    return gbr, gbr_mae, gbr_r2


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(name)s  %(levelname)s  %(message)s",
    )

    logger.info("Loading dataset from %s", DATA_PATH)
    try:
        df = pd.read_csv(DATA_PATH)
    except FileNotFoundError:
        logger.error("Dataset not found at %s", DATA_PATH)
        raise

    logger.info("Dataset shape: %s", df.shape)
    df = engineer_features(df)
    X_train, X_test, y_train, y_test, feature_names, scaler = prepare_data(df)
    logger.info("Train size: %d | Test size: %d", len(X_train), len(X_test))

    model = TrafficNet(X_train.shape[1])
    logger.info("Model architecture:\n%s", model)

    history = train_nn(model, X_train, y_train)
    plot_training_history(history, os.path.join(FIGURES_DIR, "nn_training_history.png"))

    model.eval()
    with torch.no_grad():
        y_pred = model(torch.FloatTensor(X_test)).numpy()
    nn_mae = mean_absolute_error(y_test, y_pred)
    nn_r2 = r2_score(y_test, y_pred)

    print("\n===== Neural Network Evaluation =====")
    print(f"  MAE : {nn_mae:.2f}")
    print(f"  R²  : {nn_r2:.4f}")

    shap_path = os.path.join(FIGURES_DIR, "shap_summary.png")
    gbr, gbr_mae, gbr_r2 = shap_explainability(
        X_train, X_test, y_train, y_test, feature_names, shap_path
    )

    print("\n===== GBR Surrogate Evaluation =====")
    print(f"  MAE : {gbr_mae:.2f}")
    print(f"  R²  : {gbr_r2:.4f}")
    print("\nSHAP explainability applied to GBR surrogate.")
    print("See figures/shap_summary.png for feature importance.")


if __name__ == "__main__":
    main()
