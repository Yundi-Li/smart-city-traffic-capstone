"""
Deep Learning for Traffic Volume Prediction with SHAP Explainability.

Uses PyTorch feedforward neural network and SHAP on a GBR surrogate.
SHAP targets the GBR because TreeExplainer is exact and fast, whereas
neural net explainers (DeepExplainer) can be unstable on tabular data.
"""

import logging
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import shap
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_loader import load_featured_data, get_feature_columns, FIGURES_DIR

logger = logging.getLogger(__name__)


class TrafficNet(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def train_nn(model, X_train, y_train, epochs=30, batch_size=256, patience=10):
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

    return {"loss": train_losses, "val_loss": val_losses, "mae": train_maes, "val_mae": val_maes}


def plot_training_history(history, save_path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(history["loss"], label="Train Loss")
    axes[0].plot(history["val_loss"], label="Val Loss")
    axes[0].set_title("Training & Validation Loss (MSE)")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("MSE"); axes[0].legend()
    axes[1].plot(history["mae"], label="Train MAE")
    axes[1].plot(history["val_mae"], label="Val MAE")
    axes[1].set_title("Training & Validation MAE")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("MAE"); axes[1].legend()
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    logger.info("Training history plot saved to %s", save_path)


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(name)s  %(levelname)s  %(message)s")

    df = load_featured_data()
    feature_cols = get_feature_columns(df)
    X = df[feature_cols].values.astype(np.float32)
    y = df["traffic_volume"].values.astype(np.float32)

    # Chronological split: train 2012-2016, test 2017
    years = df["date_time"].dt.year
    train_mask = (years <= 2016).values
    test_mask = (years == 2017).values
    X_train, X_test = X[train_mask], X[test_mask]
    y_train, y_test = y[train_mask], y[test_mask]

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    logger.info("Train: %d, Test (2017): %d", len(X_train), len(X_test))

    model = TrafficNet(X_train.shape[1])
    logger.info("Model: %s", model)
    history = train_nn(model, X_train, y_train, epochs=30)
    plot_training_history(history, os.path.join(FIGURES_DIR, "nn_training_history.png"))

    model.eval()
    with torch.no_grad():
        y_pred = model(torch.FloatTensor(X_test)).numpy()
    nn_mae = mean_absolute_error(y_test, y_pred)
    nn_r2 = r2_score(y_test, y_pred)
    print("\n===== Neural Network Evaluation =====")
    print(f"  MAE : {nn_mae:.2f}")
    print(f"  R²  : {nn_r2:.4f}")

    # SHAP on GBR surrogate (sample 1000 rows for speed)
    logger.info("Training GBR surrogate for SHAP explainability")
    gbr = GradientBoostingRegressor(n_estimators=200, max_depth=5, learning_rate=0.1, random_state=42)
    gbr.fit(X_train, y_train)
    gbr_pred = gbr.predict(X_test)
    gbr_mae = mean_absolute_error(y_test, gbr_pred)
    gbr_r2 = r2_score(y_test, gbr_pred)

    explainer = shap.TreeExplainer(gbr)
    shap_sample = X_test[:1000]
    shap_values = explainer.shap_values(shap_sample)
    shap.summary_plot(shap_values, shap_sample, feature_names=feature_cols, show=False)
    shap_path = os.path.join(FIGURES_DIR, "shap_summary.png")
    plt.tight_layout()
    plt.savefig(shap_path, dpi=150, bbox_inches="tight")
    plt.close("all")
    logger.info("SHAP summary plot saved to %s", shap_path)

    print("\n===== GBR Surrogate Evaluation =====")
    print(f"  MAE : {gbr_mae:.2f}")
    print(f"  R²  : {gbr_r2:.4f}")
    print("\nSHAP applied to GBR surrogate (1,000-row sample).")


if __name__ == "__main__":
    main()
