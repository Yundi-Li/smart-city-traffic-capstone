"""
Shared data-loading utility for Part 3 scripts.

All Part 3 scripts load the featured dataset from Part 2, which includes
cleaned data (deduplicated, outlier-imputed) plus all engineered features
(time, weather, cyclical encodings, is_holiday, is_low_visibility,
weather_severity, congestion_category, normalised numerics).
"""

import logging
import os

import pandas as pd

logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FEATURED_PATH = os.path.join(BASE_DIR, "capstone_part2", "featured_traffic.csv")
FIGURES_DIR = os.path.join(BASE_DIR, "figures")

SEVERE_WEATHER = [
    "Thunderstorm", "Squall", "Fog", "Smoke", "Haze", "Mist", "Snow", "Rain",
]


def load_featured_data() -> pd.DataFrame:
    """Load the Part 2 featured dataset."""
    try:
        df = pd.read_csv(FEATURED_PATH, parse_dates=["date_time"])
        logger.info("Loaded featured data: %d rows, %d columns", len(df), len(df.columns))
        return df
    except FileNotFoundError:
        logger.error("Featured data not found at %s — run Part 2 pipeline first", FEATURED_PATH)
        raise


def get_feature_columns(df: pd.DataFrame) -> list:
    """Return feature column names suitable for ML (no target, no IDs, no strings)."""
    base = [
        "hour_sin", "hour_cos", "dow_sin", "dow_cos",
        "is_weekend", "is_holiday", "is_low_visibility", "weather_severity",
        "temp", "rain_1h", "snow_1h", "clouds_all",
    ]
    weather_ohe = [c for c in df.columns if c.startswith("weather_")
                   and c not in ("weather_main", "weather_description", "weather_severity")]
    return [c for c in base + weather_ohe if c in df.columns]


def add_proxy_label(df: pd.DataFrame) -> pd.DataFrame:
    """Add the proxy accident-risk label per capstone instructions."""
    is_high = df["congestion_category"].isin(["High", "Severe"])
    risky = df["weather_main"].isin(SEVERE_WEATHER) | (df["is_low_visibility"] == 1)
    df["high_risk"] = (is_high & risky).astype(int)
    logger.info("Proxy high_risk label: %d positive (%.1f%%)",
                df["high_risk"].sum(), 100 * df["high_risk"].mean())
    return df
