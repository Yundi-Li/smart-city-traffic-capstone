"""
pipeline.py - Main data cleaning and validation pipeline for I-94 traffic data.

Run standalone: python capstone_part2/pipeline.py [--debug]
"""

import argparse
import logging
import os
import sys

import pandas as pd
import numpy as np

import sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); from logging_config import setup_logging

logger = logging.getLogger(__name__)

EXPECTED_COLUMNS = [
    "holiday", "temp", "rain_1h", "snow_1h", "clouds_all",
    "weather_main", "weather_description", "date_time", "traffic_volume",
]

RAW_DATA_PATH = os.path.join("data", "Metro_Interstate_Traffic_Volume.csv")
OUTPUT_DIR = "capstone_part2"
CLEANED_DATA_PATH = os.path.join(OUTPUT_DIR, "cleaned_traffic.csv")


def load_data(path: str) -> pd.DataFrame:
    """Load the raw CSV and log basic stats."""
    try:
        df = pd.read_csv(path)
        logger.info("Loaded %d rows and %d columns from %s", len(df), len(df.columns), path)
        return df
    except FileNotFoundError:
        logger.error("Data file not found: %s", path, exc_info=True)
        raise
    except Exception:
        logger.error("Failed to load data from %s", path, exc_info=True)
        raise


def validate_schema(df: pd.DataFrame) -> None:
    """Check that all expected columns are present."""
    missing = set(EXPECTED_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Missing expected columns: {missing}")
    logger.info("Schema validation passed - all %d expected columns present", len(EXPECTED_COLUMNS))


def standardise_weather(df: pd.DataFrame) -> pd.DataFrame:
    """Standardise inconsistent categorical values in weather columns."""
    before = df.copy()

    # Strip whitespace and title-case weather_main
    df["weather_main"] = df["weather_main"].str.strip().str.title()
    # Lowercase and strip weather_description
    df["weather_description"] = df["weather_description"].str.strip().str.lower()
    # Standardise holiday - replace NaN / 'None' with 'No Holiday'
    df["holiday"] = df["holiday"].fillna("No Holiday")
    df.loc[df["holiday"] == "None", "holiday"] = "No Holiday"

    changed_main = (before["weather_main"] != df["weather_main"]).sum()
    changed_desc = (before["weather_description"] != df["weather_description"]).sum()
    total_changed = changed_main + changed_desc
    if total_changed > 0:
        logger.warning(
            "Standardised weather values: %d weather_main, %d weather_description entries adjusted",
            changed_main, changed_desc,
        )
    else:
        logger.info("No weather values needed standardisation")
    return df


def parse_datetime(df: pd.DataFrame) -> pd.DataFrame:
    """Parse and validate date_time column."""
    try:
        df["date_time"] = pd.to_datetime(df["date_time"], format="%Y-%m-%d %H:%M:%S")
        logger.info("Parsed date_time column successfully")
    except Exception:
        logger.error("Failed to parse date_time column", exc_info=True)
        raise
    return df


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Remove exact duplicate rows and deduplicate by timestamp."""
    # Step 1: exact duplicates
    exact_dup_count = df.duplicated().sum()
    if exact_dup_count > 0:
        logger.warning("Removing %d exact duplicate rows", exact_dup_count)
        df = df.drop_duplicates().reset_index(drop=True)
    else:
        logger.info("No exact duplicate rows found")

    # Step 2: duplicate timestamps - keep first occurrence
    ts_dup_mask = df.duplicated(subset=["date_time"], keep="first")
    ts_dup_count = ts_dup_mask.sum()
    if ts_dup_count > 0:
        logger.warning(
            "Removing %d rows with duplicate date_time values (keeping first occurrence)",
            ts_dup_count,
        )
        df = df[~ts_dup_mask].reset_index(drop=True)

    return df


def detect_and_impute_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """Detect physically implausible values and impute with monthly medians."""
    df["month"] = df["date_time"].dt.month

    # Detect outliers
    temp_outlier_mask = df["temp"] == 0  # 0 Kelvin is physically impossible
    rain_outlier_mask = df["rain_1h"] > 9000  # > 9000mm is implausible

    temp_outlier_count = temp_outlier_mask.sum()
    rain_outlier_count = rain_outlier_mask.sum()

    logger.warning("Detected %d rows with 0 Kelvin temperature", temp_outlier_count)
    logger.warning("Detected %d rows with rain_1h > 9000mm", rain_outlier_count)

    # Compute per-month medians using a loop (not global)
    monthly_temp_median = {}
    monthly_rain_median = {}
    for month in range(1, 13):
        month_mask = df["month"] == month
        valid_temp = df.loc[month_mask & ~temp_outlier_mask, "temp"]
        valid_rain = df.loc[month_mask & ~rain_outlier_mask, "rain_1h"]
        monthly_temp_median[month] = valid_temp.median()
        monthly_rain_median[month] = valid_rain.median()

    # Impute outliers
    for month in range(1, 13):
        month_mask = df["month"] == month
        df.loc[month_mask & temp_outlier_mask, "temp"] = monthly_temp_median[month]
        df.loc[month_mask & rain_outlier_mask, "rain_1h"] = monthly_rain_median[month]

    logger.warning(
        "Imputed %d temp outliers and %d rain outliers with monthly medians",
        temp_outlier_count, rain_outlier_count,
    )

    df = df.drop(columns=["month"])
    return df


def run_pipeline(debug: bool = False) -> None:
    """Execute the full cleaning pipeline."""
    setup_logging(debug=debug)
    logger.info("Starting data cleaning pipeline")

    try:
        df = load_data(RAW_DATA_PATH)
        validate_schema(df)
        df = standardise_weather(df)
        df = parse_datetime(df)
        df = remove_duplicates(df)
        df = detect_and_impute_outliers(df)

        os.makedirs(OUTPUT_DIR, exist_ok=True)
        df.to_csv(CLEANED_DATA_PATH, index=False)
        logger.info("Cleaned data saved to %s (%d rows, %d columns)",
                     CLEANED_DATA_PATH, len(df), len(df.columns))
        logger.info("Pipeline completed successfully")
    except Exception:
        logger.error("Pipeline failed", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="I-94 traffic data cleaning pipeline")
    parser.add_argument("--debug", action="store_true", help="Enable DEBUG logging")
    args = parser.parse_args()
    run_pipeline(debug=args.debug)
