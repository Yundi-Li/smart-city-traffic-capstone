"""
feature_engineering.py - Create ML-ready features from cleaned traffic data.
"""

import logging
import os

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

CLEANED_DATA_PATH = os.path.join("capstone_part2", "cleaned_traffic.csv")
FEATURED_DATA_PATH = os.path.join("capstone_part2", "featured_traffic.csv")


def load_cleaned_data() -> pd.DataFrame:
    """Load the cleaned dataset."""
    try:
        df = pd.read_csv(CLEANED_DATA_PATH, parse_dates=["date_time"])
        logger.info("Loaded cleaned data: %d rows, %d columns", len(df), len(df.columns))
        return df
    except Exception:
        logger.error("Failed to load cleaned data from %s", CLEANED_DATA_PATH, exc_info=True)
        raise


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extract time-based features including cyclical encodings."""
    df["hour"] = df["date_time"].dt.hour
    df["day_of_week"] = df["date_time"].dt.dayofweek  # 0=Monday
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

    # Cyclical encoding
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)

    logger.info("Added time features: hour, day_of_week, is_weekend, cyclical encodings")
    return df


def add_weather_features(df: pd.DataFrame) -> pd.DataFrame:
    """One-hot encode weather_main, create rain/snow indicators."""
    # One-hot encode weather_main
    weather_dummies = pd.get_dummies(df["weather_main"], prefix="weather")
    df = pd.concat([df, weather_dummies], axis=1)

    # Binary indicators
    df["is_rainy"] = df["weather_main"].isin(["Rain", "Drizzle", "Thunderstorm"]).astype(int)
    df["is_snowy"] = (df["weather_main"] == "Snow").astype(int)

    logger.info("Added %d one-hot weather columns, is_rainy, is_snowy", len(weather_dummies.columns))
    return df


def add_numerical_features(df: pd.DataFrame) -> pd.DataFrame:
    """Min-max scale temp and traffic_volume."""
    for col in ["temp", "traffic_volume"]:
        col_min = df[col].min()
        col_max = df[col].max()
        df[f"{col}_norm"] = (df[col] - col_min) / (col_max - col_min)
        logger.debug("Min-max scaled %s: min=%.2f, max=%.2f", col, col_min, col_max)

    logger.info("Added normalized temp and traffic_volume features")
    return df


def add_congestion_category(df: pd.DataFrame) -> pd.DataFrame:
    """Create congestion category based on traffic_volume quartiles."""
    q1 = df["traffic_volume"].quantile(0.25)
    q2 = df["traffic_volume"].quantile(0.50)
    q3 = df["traffic_volume"].quantile(0.75)

    logger.debug("Quartile thresholds - Q1: %.1f, Q2: %.1f, Q3: %.1f", q1, q2, q3)

    conditions = [
        df["traffic_volume"] <= q1,
        (df["traffic_volume"] > q1) & (df["traffic_volume"] <= q2),
        (df["traffic_volume"] > q2) & (df["traffic_volume"] <= q3),
        df["traffic_volume"] > q3,
    ]
    labels = ["Low", "Medium", "High", "Severe"]
    df["congestion"] = np.select(conditions, labels, default="Medium")

    logger.info("Added congestion category: %s", dict(df["congestion"].value_counts()))
    return df


def run_feature_engineering() -> None:
    """Execute the full feature engineering pipeline."""
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s | %(levelname)-8s | %(module)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    logger.info("Starting feature engineering")

    try:
        df = load_cleaned_data()
        shape_before = df.shape

        df = add_time_features(df)
        df = add_weather_features(df)
        df = add_numerical_features(df)
        df = add_congestion_category(df)

        logger.info("Dataset shape: before=%s, after=%s", shape_before, df.shape)

        df.to_csv(FEATURED_DATA_PATH, index=False)
        logger.info("Featured data saved to %s", FEATURED_DATA_PATH)
    except Exception:
        logger.error("Feature engineering failed", exc_info=True)
        raise


if __name__ == "__main__":
    run_feature_engineering()
