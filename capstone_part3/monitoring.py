"""
Model monitoring module for Metro Interstate Traffic Volume prediction.

Simulates production monitoring by checking for prediction error drift
and feature distribution drift between training and test data.
"""

import logging
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)


def load_and_engineer_features(data_path: str) -> pd.DataFrame:
    """Load dataset and engineer features for modeling."""
    logger.info("Loading data from %s", data_path)
    df = pd.read_csv(data_path)
    df["date_time"] = pd.to_datetime(df["date_time"])

    # Time features
    df["hour"] = df["date_time"].dt.hour
    df["day_of_week"] = df["date_time"].dt.dayofweek
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

    # Cyclical encodings
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)

    # One-hot encode weather_main
    weather_dummies = pd.get_dummies(df["weather_main"], prefix="weather")
    df = pd.concat([df, weather_dummies], axis=1)

    # Sort by date for chronological splitting later
    df = df.sort_values("date_time").reset_index(drop=True)

    logger.info("Feature engineering complete. Shape: %s", df.shape)
    return df


def get_feature_columns(df: pd.DataFrame) -> list:
    """Return the list of feature columns used for modeling."""
    base_features = [
        "temp", "rain_1h", "snow_1h", "clouds_all",
        "hour", "day_of_week", "is_weekend",
        "hour_sin", "hour_cos", "dow_sin", "dow_cos",
    ]
    weather_cols = [c for c in df.columns if c.startswith("weather_") and c not in ("weather_main", "weather_description")]
    return base_features + weather_cols


def train_model(X_train: pd.DataFrame, y_train: pd.Series):
    """Train a GradientBoostingRegressor."""
    logger.info("Training GradientBoostingRegressor on %d samples", len(X_train))
    model = GradientBoostingRegressor(random_state=42)
    model.fit(X_train, y_train)
    return model


def check_prediction_error_drift(
    y_true: np.ndarray, y_pred: np.ndarray, threshold: float = 0.20
) -> dict:
    """
    Split test predictions chronologically in half and compare MAE
    between the first and second halves.

    Returns a dict with status, first_half_mae, second_half_mae, and pct_change.
    """
    mid = len(y_true) // 2
    errors = np.abs(y_true - y_pred)

    mae_first = errors[:mid].mean()
    mae_second = errors[mid:].mean()

    if mae_first == 0:
        pct_change = float("inf") if mae_second > 0 else 0.0
    else:
        pct_change = (mae_second - mae_first) / mae_first

    status = "ALERT" if pct_change > threshold else "PASS"

    logger.info(
        "Prediction error drift: first_half_mae=%.2f, second_half_mae=%.2f, "
        "pct_change=%.2f%%, status=%s",
        mae_first, mae_second, pct_change * 100, status,
    )
    return {
        "status": status,
        "first_half_mae": mae_first,
        "second_half_mae": mae_second,
        "pct_change": pct_change,
    }


def check_feature_drift(
    train_data: pd.DataFrame,
    test_data: pd.DataFrame,
    features: list,
    alpha: float = 0.05,
) -> list:
    """
    Run KS test on key features between training and test sets.

    Returns a list of dicts with feature name, statistic, p_value, and status.
    """
    results = []
    for feat in features:
        stat, p_value = ks_2samp(train_data[feat].dropna(), test_data[feat].dropna())
        status = "ALERT" if p_value < alpha else "PASS"
        results.append({
            "feature": feat,
            "ks_statistic": stat,
            "p_value": p_value,
            "status": status,
        })
        logger.info(
            "Feature drift [%s]: KS=%.4f, p=%.4f, status=%s",
            feat, stat, p_value, status,
        )
    return results


def generate_report(error_drift: dict, feature_drift: list) -> str:
    """Generate a formatted monitoring report string."""
    lines = [
        "=" * 60,
        "MODEL MONITORING REPORT",
        "=" * 60,
        "",
        "--- Prediction Error Drift ---",
        f"  First half MAE:  {error_drift['first_half_mae']:.2f}",
        f"  Second half MAE: {error_drift['second_half_mae']:.2f}",
        f"  Change:          {error_drift['pct_change'] * 100:+.2f}%",
        f"  Status:          [{error_drift['status']}]",
        "",
        "--- Feature Distribution Drift (KS Test) ---",
    ]
    for fd in feature_drift:
        lines.append(
            f"  {fd['feature']:20s}  KS={fd['ks_statistic']:.4f}  "
            f"p={fd['p_value']:.4f}  [{fd['status']}]"
        )
    lines.append("")
    lines.append("=" * 60)
    return "\n".join(lines)


def save_drift_visualization(
    train_data: pd.DataFrame,
    test_data: pd.DataFrame,
    features: list,
    save_path: str,
) -> None:
    """Save histograms comparing train vs test distributions for key features."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()

    for i, feat in enumerate(features):
        ax = axes[i]
        ax.hist(
            train_data[feat].dropna(), bins=50, alpha=0.5, label="Train", density=True
        )
        ax.hist(
            test_data[feat].dropna(), bins=50, alpha=0.5, label="Test", density=True
        )
        ax.set_title(f"{feat} Distribution")
        ax.set_xlabel(feat)
        ax.set_ylabel("Density")
        ax.legend()

    plt.suptitle("Feature Distribution Drift: Train vs Test", fontsize=14)
    plt.tight_layout()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Drift visualization saved to %s", save_path)


def run_monitoring() -> None:
    """Execute the full monitoring pipeline."""
    # Paths relative to project root
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base_dir, "capstone_part2", "cleaned_traffic.csv")
    report_path = os.path.join(base_dir, "capstone_part3", "reports", "monitoring_report.txt")
    figure_path = os.path.join(base_dir, "figures", "monitoring_drift.png")

    # 1. Load and engineer features
    df = load_and_engineer_features(data_path)
    feature_cols = get_feature_columns(df)

    X = df[feature_cols]
    y = df["traffic_volume"]

    # 2. Train/test split (80/20, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 3. Train model
    model = train_model(X_train, y_train)

    # 4. Generate predictions on holdout set (keep chronological order for drift)
    #    Re-sort test set by original index to preserve chronological order
    test_idx_sorted = X_test.index.sort_values()
    X_test_sorted = X_test.loc[test_idx_sorted]
    y_test_sorted = y_test.loc[test_idx_sorted]

    y_pred = model.predict(X_test_sorted)

    # 5. Prediction error drift check
    error_drift = check_prediction_error_drift(
        y_test_sorted.values, y_pred, threshold=0.20
    )

    # 6. Feature distribution drift check
    drift_features = ["temp", "traffic_volume", "clouds_all", "rain_1h"]
    feature_drift = check_feature_drift(
        df.loc[X_train.index], df.loc[X_test.index], drift_features
    )

    # 7. Generate and print report
    report = generate_report(error_drift, feature_drift)
    print(report)

    # 8. Save report
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, "w") as f:
        f.write(report)
    logger.info("Monitoring report saved to %s", report_path)

    # 9. Save drift visualization
    save_drift_visualization(
        df.loc[X_train.index], df.loc[X_test.index], drift_features, figure_path
    )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    try:
        run_monitoring()
    except FileNotFoundError as e:
        logger.error("Data file not found: %s", e)
        raise
    except Exception as e:
        logger.error("Monitoring failed: %s", e)
        raise
