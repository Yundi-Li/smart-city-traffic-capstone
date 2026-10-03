"""
feature_engineering.py - Create ML-ready features from cleaned traffic data.
"""

import argparse
import logging
import os

import numpy as np
import pandas as pd

import sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); from logging_config import setup_logging

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

    # Holiday flag: the holiday column only labels the midnight row, so flag
    # ALL hours on any date that has a holiday label.
    holiday_dates = df.loc[df["holiday"] != "No Holiday", "date_time"].dt.date.unique()
    df["is_holiday"] = df["date_time"].dt.date.isin(holiday_dates).astype(int)
    n_holiday = df["is_holiday"].sum()
    logger.info("Added is_holiday: %d rows flagged across %d holiday dates", n_holiday, len(holiday_dates))

    logger.info("Added time features: hour, day_of_week, is_weekend, cyclical encodings, is_holiday")
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
    """Min-max scale continuous input variables: temp, clouds_all, rain_1h."""
    for col in ["temp", "clouds_all", "rain_1h"]:
        col_min = df[col].min()
        col_max = df[col].max()
        df[f"{col}_norm"] = (df[col] - col_min) / (col_max - col_min)
        logger.debug("Min-max scaled %s: min=%.4f, max=%.4f", col, col_min, col_max)

    logger.info("Added normalized features: temp_norm, clouds_all_norm, rain_1h_norm")
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


def run_feature_engineering(debug: bool = False) -> None:
    """Execute the full feature engineering pipeline."""
    setup_logging(debug=debug)

    logger.info("Starting feature engineering")

    try:
        df = load_cleaned_data()
        shape_before = df.shape
        logger.info("Shape before feature engineering: %s", shape_before)

        df = add_time_features(df)
        df = add_weather_features(df)
        df = add_numerical_features(df)
        df = add_congestion_category(df)

        logger.info("Shape after feature engineering: %s", df.shape)

        df.to_csv(FEATURED_DATA_PATH, index=False)
        logger.info("Featured data saved to %s", FEATURED_DATA_PATH)
    except Exception:
        logger.error("Feature engineering failed", exc_info=True)
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Feature engineering for I-94 traffic data")
    parser.add_argument("--debug", action="store_true", help="Enable DEBUG logging")
    args = parser.parse_args()
    run_feature_engineering(debug=args.debug)
