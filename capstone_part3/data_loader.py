"""
Shared data-loading utility for Part 3 scripts.

All Part 3 scripts should use the cleaned data from Part 2 so that
outlier fixes (0 K temperatures, 9,831 mm rain) and timestamp
deduplication carry forward.
"""

import logging
import os

import pandas as pd

logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLEANED_PATH = os.path.join(BASE_DIR, "capstone_part2", "cleaned_traffic.csv")
FEATURED_PATH = os.path.join(BASE_DIR, "capstone_part2", "featured_traffic.csv")
RAW_PATH = os.path.join(BASE_DIR, "data", "Metro_Interstate_Traffic_Volume.csv")


def load_cleaned_data() -> pd.DataFrame:
    """Load the Part 2 cleaned dataset (40,575 rows, deduplicated and imputed)."""
    try:
        df = pd.read_csv(CLEANED_PATH, parse_dates=["date_time"])
        logger.info("Loaded cleaned data from Part 2: %d rows, %d columns", len(df), len(df.columns))
        return df
    except FileNotFoundError:
        logger.warning("Cleaned data not found at %s — falling back to raw CSV", CLEANED_PATH)
        df = pd.read_csv(RAW_PATH, parse_dates=["date_time"])
        logger.info("Loaded raw data: %d rows, %d columns", len(df), len(df.columns))
        return df


def load_featured_data() -> pd.DataFrame:
    """Load the Part 2 featured dataset (40,575 rows, 33 columns)."""
    try:
        df = pd.read_csv(FEATURED_PATH, parse_dates=["date_time"])
        logger.info("Loaded featured data from Part 2: %d rows, %d columns", len(df), len(df.columns))
        return df
    except FileNotFoundError:
        logger.warning("Featured data not found at %s — falling back to cleaned data", FEATURED_PATH)
        return load_cleaned_data()
