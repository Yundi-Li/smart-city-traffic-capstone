"""
Supervised Machine Learning for Traffic Volume Prediction and Accident-Risk Classification.

Uses Metro Interstate Traffic Volume dataset to:
1. Classify high-risk traffic conditions (LogisticRegression, RandomForestClassifier)
2. Predict traffic volume (LinearRegression, GradientBoostingRegressor)
"""

import logging
import os

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "capstone_part2", "cleaned_traffic.csv")
FIGURES_DIR = os.path.join(BASE_DIR, "figures")

SEVERE_WEATHER = [
    "Thunderstorm", "Squall", "Fog", "Smoke", "Haze", "Mist", "Snow", "Rain",
]
LOW_VISIBILITY_WEATHER = ["Fog", "Mist", "Haze", "Smoke"]


def load_data(path: str) -> pd.DataFrame:
    """Load the traffic volume CSV dataset."""
    logger.info("Loading data from %s", path)
    try:
        df = pd.read_csv(path)
        logger.info("Loaded %d rows, %d columns", len(df), len(df.columns))
        return df
    except FileNotFoundError:
        logger.error("Data file not found: %s", path)
        raise
    except Exception as exc:
        logger.error("Error loading data: %s", exc)
        raise


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """Engineer features from the raw dataset."""
    logger.info("Preparing features")
    df = df.copy()

    try:
        df["date_time"] = pd.to_datetime(df["date_time"])
    except Exception as exc:
        logger.error("Failed to parse date_time column: %s", exc)
        raise

    # Time features
    df["hour"] = df["date_time"].dt.hour
    df["day_of_week"] = df["date_time"].dt.dayofweek
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

    # Cyclical encodings
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["day_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["day_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)

    # Holiday binary flag
    df["is_holiday"] = (df["holiday"] != "None").astype(int)

    # Congestion category from traffic_volume quartiles
    df["congestion_category"] = pd.qcut(
        df["traffic_volume"],
        q=4,
        labels=["Low", "Medium", "High", "Severe"],
    )

    # is_low_visibility feature: clouds_all > 80 and weather is fog/mist/haze/smoke
    df["is_low_visibility"] = (
        (df["clouds_all"] > 80) & df["weather_main"].isin(LOW_VISIBILITY_WEATHER)
    ).astype(int)

    # Proxy accident-risk label (per capstone instructions):
    # high_risk = high/severe congestion AND (severe weather OR low visibility)
    is_high_congestion = df["congestion_category"].isin(["High", "Severe"])
    risky_weather = df["weather_main"].isin(SEVERE_WEATHER) | (df["is_low_visibility"] == 1)
    df["high_risk"] = (is_high_congestion & risky_weather).astype(int)

    # One-hot encode weather_main
    weather_dummies = pd.get_dummies(df["weather_main"], prefix="weather")
    df = pd.concat([df, weather_dummies], axis=1)

    logger.info("Feature preparation complete. Shape: %s", df.shape)
    return df


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """Return the list of feature column names."""
    base_features = [
        "hour_sin", "hour_cos", "day_sin", "day_cos",
        "is_weekend", "is_holiday", "is_low_visibility",
        "temp", "rain_1h", "snow_1h", "clouds_all",
    ]
    weather_cols = [c for c in df.columns if c.startswith("weather_") and c not in ("weather_main", "weather_description")]
    return base_features + weather_cols


def train_classifiers(X_train, X_test, y_train, y_test, feature_names):
    """Train and evaluate classification models for high_risk prediction."""
    logger.info("Training classification models")

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    models = {
        "LogisticRegression": LogisticRegression(
            max_iter=1000, random_state=42, class_weight="balanced"
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=100, random_state=42, class_weight="balanced", n_jobs=-1
        ),
    }

    results = {}
    for name, model in models.items():
        logger.info("Training %s", name)
        try:
            data = X_train_scaled if name == "LogisticRegression" else X_train
            test_data = X_test_scaled if name == "LogisticRegression" else X_test

            model.fit(data, y_train)
            y_pred = model.predict(test_data)
            y_prob = model.predict_proba(test_data)[:, 1]

            results[name] = {
                "accuracy": accuracy_score(y_test, y_pred),
                "precision": precision_score(y_test, y_pred, zero_division=0),
                "recall": recall_score(y_test, y_pred, zero_division=0),
                "f1": f1_score(y_test, y_pred, zero_division=0),
                "roc_auc": roc_auc_score(y_test, y_prob),
                "confusion_matrix": confusion_matrix(y_test, y_pred),
                "model": model,
            }
            logger.info("%s ROC AUC: %.4f", name, results[name]["roc_auc"])
        except Exception as exc:
            logger.error("Error training %s: %s", name, exc)
            raise

    # Save confusion matrices
    try:
        _save_confusion_matrices(results)
    except Exception as exc:
        logger.warning("Could not save confusion matrix figures: %s", exc)

    # Save feature importance for RandomForest
    try:
        _save_feature_importance(
            results["RandomForest"]["model"], feature_names, "classification"
        )
    except Exception as exc:
        logger.warning("Could not save feature importance figure: %s", exc)

    return results


def train_regressors(X_train, X_test, y_train, y_test, feature_names):
    """Train and evaluate regression models for traffic_volume prediction."""
    logger.info("Training regression models")

    models = {
        "LinearRegression": LinearRegression(),
        "GradientBoosting": GradientBoostingRegressor(
            n_estimators=200, max_depth=5, random_state=42, learning_rate=0.1
        ),
    }

    results = {}
    for name, model in models.items():
        logger.info("Training %s", name)
        try:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            results[name] = {
                "mae": mean_absolute_error(y_test, y_pred),
                "r2": r2_score(y_test, y_pred),
                "model": model,
            }
            logger.info("%s R²: %.4f", name, results[name]["r2"])
        except Exception as exc:
            logger.error("Error training %s: %s", name, exc)
            raise

    # Save feature importance for GradientBoosting
    try:
        _save_feature_importance(
            results["GradientBoosting"]["model"], feature_names, "regression"
        )
    except Exception as exc:
        logger.warning("Could not save feature importance figure: %s", exc)

    return results


def _save_confusion_matrices(results: dict) -> None:
    """Save confusion matrix plots for each classifier."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(FIGURES_DIR, exist_ok=True)

    for name, res in results.items():
        fig, ax = plt.subplots(figsize=(6, 5))
        cm = res["confusion_matrix"]
        im = ax.imshow(cm, cmap="Blues")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title(f"Confusion Matrix — {name}")
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["Low Risk", "High Risk"])
        ax.set_yticklabels(["Low Risk", "High Risk"])

        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                        color="white" if cm[i, j] > cm.max() / 2 else "black")

        fig.colorbar(im)
        fig.tight_layout()
        path = os.path.join(FIGURES_DIR, f"confusion_matrix_{name.lower()}.png")
        fig.savefig(path, dpi=150)
        plt.close(fig)
        logger.info("Saved confusion matrix to %s", path)


def _save_feature_importance(model, feature_names: list[str], task: str) -> None:
    """Save a feature importance bar chart."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(FIGURES_DIR, exist_ok=True)

    importances = model.feature_importances_
    indices = np.argsort(importances)[-15:]  # top 15

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(range(len(indices)), importances[indices])
    ax.set_yticks(range(len(indices)))
    ax.set_yticklabels([feature_names[i] for i in indices])
    ax.set_xlabel("Importance")
    ax.set_title(f"Top 15 Feature Importances — {task.title()}")
    fig.tight_layout()

    path = os.path.join(FIGURES_DIR, f"feature_importance_{task}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info("Saved feature importance to %s", path)


def print_classification_results(results: dict) -> None:
    """Print a comparison table for classification models."""
    header = f"{'Model':<25} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1':>10} {'ROC AUC':>10}"
    print("\n" + "=" * 80)
    print("CLASSIFICATION RESULTS — High-Risk Prediction")
    print("=" * 80)
    print(header)
    print("-" * 80)
    for name, res in results.items():
        print(
            f"{name:<25} {res['accuracy']:>10.4f} {res['precision']:>10.4f} "
            f"{res['recall']:>10.4f} {res['f1']:>10.4f} {res['roc_auc']:>10.4f}"
        )
    print("=" * 80)


def print_regression_results(results: dict) -> None:
    """Print a comparison table for regression models."""
    header = f"{'Model':<25} {'MAE':>12} {'R²':>10}"
    print("\n" + "=" * 60)
    print("REGRESSION RESULTS — Traffic Volume Prediction")
    print("=" * 60)
    print(header)
    print("-" * 60)
    for name, res in results.items():
        print(f"{name:<25} {res['mae']:>12.2f} {res['r2']:>10.4f}")
    print("=" * 60)


def main() -> None:
    """Run the full supervised ML pipeline."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s — %(name)s — %(levelname)s — %(message)s",
    )

    # Load and prepare data
    df = load_data(DATA_PATH)
    df = prepare_features(df)

    feature_cols = get_feature_columns(df)
    X = df[feature_cols].values
    y_classification = df["high_risk"].values
    y_regression = df["traffic_volume"].values

    logger.info("Feature columns (%d): %s", len(feature_cols), feature_cols)
    logger.info(
        "High-risk distribution: %d positive (%.1f%%)",
        y_classification.sum(),
        100 * y_classification.mean(),
    )

    # Train/test split
    X_train, X_test, y_cls_train, y_cls_test, y_reg_train, y_reg_test = (
        train_test_split(
            X, y_classification, y_regression,
            test_size=0.2, random_state=42,
        )
    )

    logger.info("Train size: %d, Test size: %d", len(X_train), len(X_test))

    # Classification
    cls_results = train_classifiers(
        X_train, X_test, y_cls_train, y_cls_test, feature_cols
    )
    print_classification_results(cls_results)

    # Regression
    reg_results = train_regressors(
        X_train, X_test, y_reg_train, y_reg_test, feature_cols
    )
    print_regression_results(reg_results)


if __name__ == "__main__":
    main()
