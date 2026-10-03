"""
Travel Timing Recommendation System

Analyzes historical Metro Interstate traffic patterns to recommend
optimal travel times based on day type, weather conditions, and hour of day.
"""

import logging
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

logger = logging.getLogger(__name__)

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "capstone_part2", "cleaned_traffic.csv")
FIGURES_DIR = os.path.join(os.path.dirname(__file__), "..", "figures")

WEATHER_CATEGORY_MAP = {
    "Clear": "Clear",
    "Clouds": "Cloudy",
    "Rain": "Rain",
    "Drizzle": "Rain",
    "Snow": "Snow",
    "Mist": "Fog",
    "Haze": "Fog",
    "Fog": "Fog",
    "Smoke": "Fog",
    "Thunderstorm": "Storm",
    "Squall": "Storm",
}


def load_and_prepare_data():
    """Load the traffic CSV and engineer time/weather features.

    Returns:
        pd.DataFrame with added columns: hour, day_of_week, is_weekend, weather_category.

    Raises:
        FileNotFoundError: If the CSV file is missing.
        ValueError: If required columns are absent.
    """
    logger.info("Loading data from %s", DATA_PATH)
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Dataset not found at {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)

    required = {"date_time", "traffic_volume", "weather_main"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df["date_time"] = pd.to_datetime(df["date_time"])
    df["hour"] = df["date_time"].dt.hour
    df["day_of_week"] = df["date_time"].dt.dayofweek  # 0=Mon, 6=Sun
    df["is_weekend"] = df["day_of_week"] >= 5
    df["day_type"] = df["is_weekend"].map({True: "weekend", False: "weekday"})
    df["weather_category"] = df["weather_main"].map(WEATHER_CATEGORY_MAP).fillna("Other")

    logger.info("Loaded %d records spanning %s to %s", len(df),
                df["date_time"].min().date(), df["date_time"].max().date())
    return df


def get_traffic_stats(df, day_type, weather_condition):
    """Return hourly traffic statistics for a given day type and weather.

    Args:
        df: Prepared DataFrame from load_and_prepare_data().
        day_type: 'weekday' or 'weekend'.
        weather_condition: One of Clear, Cloudy, Rain, Snow, Fog, Storm, Other.

    Returns:
        pd.DataFrame indexed by hour with columns: mean, std, p25, median, p75, count.

    Raises:
        ValueError: If no data matches the filters.
    """
    mask = (df["day_type"] == day_type) & (df["weather_category"] == weather_condition)
    subset = df.loc[mask]

    if subset.empty:
        raise ValueError(f"No data for day_type='{day_type}', weather='{weather_condition}'")

    stats = subset.groupby("hour")["traffic_volume"].agg(
        mean="mean",
        std="std",
        p25=lambda x: np.percentile(x, 25),
        median="median",
        p75=lambda x: np.percentile(x, 75),
        count="count",
    ).sort_index()

    logger.info("Computed stats for %s/%s: %d hours covered", day_type, weather_condition, len(stats))
    return stats


def _format_hour(hour):
    """Convert 24-hour int to readable string like '10:00 AM'."""
    if hour == 0:
        return "12:00 AM"
    elif hour < 12:
        return f"{hour}:00 AM"
    elif hour == 12:
        return "12:00 PM"
    else:
        return f"{hour - 12}:00 PM"


def _format_hour_range(hour):
    """Return a range string like '10:00-11:00 AM'."""
    start = _format_hour(hour)
    end = _format_hour((hour + 1) % 24)
    return f"{start} – {end}"


def recommend_travel_time(df, day_type, weather_condition, top_n=3):
    """Recommend the best travel windows based on lowest historical volume.

    Args:
        df: Prepared DataFrame.
        day_type: 'weekday' or 'weekend'.
        weather_condition: Weather category string.
        top_n: Number of recommendations to return.

    Returns:
        list[dict] with keys: hour, mean_volume, recommendation (plain-language string).
    """
    try:
        stats = get_traffic_stats(df, day_type, weather_condition)
    except ValueError:
        logger.warning("Cannot recommend: no data for %s/%s", day_type, weather_condition)
        return []

    best = stats.nsmallest(top_n, "mean")
    recommendations = []

    for hour, row in best.iterrows():
        rec_text = (
            f"For a {day_type} journey in {weather_condition.lower()} weather, "
            f"consider travelling between {_format_hour_range(hour)} "
            f"when historical volumes average {row['mean']:,.0f} vehicles/hour."
        )
        recommendations.append({
            "hour": int(hour),
            "mean_volume": round(row["mean"], 1),
            "recommendation": rec_text,
        })

    logger.info("Generated %d recommendations for %s/%s", len(recommendations), day_type, weather_condition)
    return recommendations


def get_peak_hours(df, day_type, top_n=3):
    """Return the peak congestion hours to avoid.

    Args:
        df: Prepared DataFrame.
        day_type: 'weekday' or 'weekend'.
        top_n: Number of peak hours.

    Returns:
        list[dict] with keys: hour, mean_volume, warning (plain-language string).
    """
    mask = df["day_type"] == day_type
    subset = df.loc[mask]

    if subset.empty:
        logger.warning("No data for day_type='%s'", day_type)
        return []

    hourly = subset.groupby("hour")["traffic_volume"].mean().sort_values(ascending=False)
    peaks = []

    for hour, vol in hourly.head(top_n).items():
        warning = (
            f"Avoid travelling between {_format_hour_range(hour)} on {day_type}s — "
            f"average volume reaches {vol:,.0f} vehicles/hour."
        )
        peaks.append({
            "hour": int(hour),
            "mean_volume": round(vol, 1),
            "warning": warning,
        })

    logger.info("Identified %d peak hours for %s", len(peaks), day_type)
    return peaks


def _save_heatmap(df):
    """Save a heatmap of traffic volume by hour and day of week."""
    os.makedirs(FIGURES_DIR, exist_ok=True)

    day_labels = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    pivot = df.pivot_table(values="traffic_volume", index="hour", columns="day_of_week", aggfunc="mean")
    pivot.columns = [day_labels[i] for i in pivot.columns]

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(pivot, cmap="YlOrRd", fmt=",.0f", annot=True, linewidths=0.5, ax=ax)
    ax.set_title("Average Traffic Volume by Hour and Day of Week", fontsize=14)
    ax.set_xlabel("Day of Week")
    ax.set_ylabel("Hour of Day")

    out_path = os.path.join(FIGURES_DIR, "traffic_heatmap_recommendations.png")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info("Heatmap saved to %s", out_path)


def generate_full_report(df):
    """Generate and print travel recommendations for all day_type/weather combos.

    Also saves the traffic heatmap figure.
    """
    print("=" * 70)
    print("SMART CITY TRAFFIC — TRAVEL TIMING RECOMMENDATIONS")
    print("=" * 70)

    _save_heatmap(df)

    for day_type in ("weekday", "weekend"):
        print(f"\n{'─' * 70}")
        print(f"  PEAK HOURS TO AVOID — {day_type.upper()}")
        print(f"{'─' * 70}")
        peaks = get_peak_hours(df, day_type)
        for p in peaks:
            print(f"  ⚠  {p['warning']}")

        weather_cats = sorted(df.loc[df["day_type"] == day_type, "weather_category"].unique())
        for weather in weather_cats:
            print(f"\n{'─' * 70}")
            print(f"  BEST TRAVEL WINDOWS — {day_type.upper()} / {weather.upper()}")
            print(f"{'─' * 70}")
            recs = recommend_travel_time(df, day_type, weather)
            if not recs:
                print("  (insufficient data)")
            for r in recs:
                print(f"  ✓  {r['recommendation']}")

    print(f"\n{'=' * 70}")
    print("Heatmap saved to figures/traffic_heatmap_recommendations.png")
    print("=" * 70)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")

    try:
        data = load_and_prepare_data()
        generate_full_report(data)
    except FileNotFoundError as exc:
        logger.error("Data file missing: %s", exc)
    except Exception as exc:
        logger.error("Unexpected error: %s", exc, exc_info=True)
