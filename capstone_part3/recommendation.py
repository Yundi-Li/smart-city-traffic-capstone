"""
Travel Timing Recommendation System

Analyzes historical Metro Interstate traffic patterns to recommend
optimal travel times based on day type, weather conditions, and hour of day.
Uses the Part 2 featured dataset via data_loader.
"""

import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from data_loader import load_featured_data, FIGURES_DIR

logger = logging.getLogger(__name__)

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


def _prepare(df: pd.DataFrame) -> pd.DataFrame:
    """Add weather_category and day_type columns from featured data."""
    df = df.copy()
    df["weather_category"] = df["weather_main"].map(WEATHER_CATEGORY_MAP).fillna("Other")
    df["day_type"] = df["is_weekend"].map({1: "weekend", 0: "weekday", True: "weekend", False: "weekday"})
    return df


def get_traffic_stats(df, day_type, weather_condition):
    """Return hourly traffic statistics for a given day type and weather."""
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
    if hour == 0:
        return "12:00 AM"
    elif hour < 12:
        return f"{hour}:00 AM"
    elif hour == 12:
        return "12:00 PM"
    else:
        return f"{hour - 12}:00 PM"


def _format_hour_range(hour):
    start = _format_hour(hour)
    end = _format_hour((hour + 1) % 24)
    return f"{start} – {end}"


def recommend_travel_time(df, day_type, weather_condition, top_n=3):
    """Recommend the best travel windows (06:00-22:00) based on lowest historical volume."""
    try:
        stats = get_traffic_stats(df, day_type, weather_condition)
    except ValueError:
        logger.warning("Cannot recommend: no data for %s/%s", day_type, weather_condition)
        return []

    # Restrict to 06:00-22:00
    realistic = stats.loc[(stats.index >= 6) & (stats.index <= 22)]
    if realistic.empty:
        realistic = stats

    best = realistic.nsmallest(top_n, "mean")
    peak_hour = realistic["mean"].idxmax()
    peak_volume = realistic["mean"].max()

    recommendations = []
    for hour, row in best.iterrows():
        pct_below_peak = (1 - row["mean"] / peak_volume) * 100 if peak_volume > 0 else 0
        rec_text = (
            f"For a {day_type} journey in {weather_condition.lower()} weather, "
            f"consider travelling between {_format_hour_range(hour)}, "
            f"when traffic is typically ~{row['mean']:,.0f} vehicles/hour "
            f"({pct_below_peak:.0f}% below the {_format_hour(peak_hour)} peak)."
        )
        recommendations.append({
            "hour": int(hour),
            "mean_volume": round(row["mean"], 1),
            "recommendation": rec_text,
        })

    logger.info("Generated %d recommendations for %s/%s", len(recommendations), day_type, weather_condition)
    return recommendations


def get_peak_hours(df, day_type, top_n=3):
    """Return the peak congestion hours to avoid."""
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
        peaks.append({"hour": int(hour), "mean_volume": round(vol, 1), "warning": warning})

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
    """Generate and print travel recommendations for all day_type/weather combos."""
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
            print(f"  !!  {p['warning']}")

        weather_cats = sorted(df.loc[df["day_type"] == day_type, "weather_category"].unique())
        for weather in weather_cats:
            print(f"\n{'─' * 70}")
            print(f"  BEST TRAVEL WINDOWS — {day_type.upper()} / {weather.upper()}")
            print(f"{'─' * 70}")
            recs = recommend_travel_time(df, day_type, weather)
            if not recs:
                print("  (insufficient data)")
            for r in recs:
                print(f"  ->  {r['recommendation']}")

    print(f"\n{'=' * 70}")
    print("Heatmap saved to figures/traffic_heatmap_recommendations.png")
    print("=" * 70)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")

    try:
        data = load_featured_data()
        data = _prepare(data)
        generate_full_report(data)
    except FileNotFoundError as exc:
        logger.error("Data file missing: %s", exc)
    except Exception as exc:
        logger.error("Unexpected error: %s", exc, exc_info=True)
