"""
visualizations.py - Generate traffic analysis visualizations.
"""

import argparse
import logging
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); from logging_config import setup_logging

logger = logging.getLogger(__name__)

FEATURED_DATA_PATH = os.path.join("capstone_part2", "featured_traffic.csv")
FIGURES_DIR = "figures"
PART2_FIGURES_DIR = os.path.join("capstone_part2", "figures")


def load_data() -> pd.DataFrame:
    """Load featured dataset."""
    try:
        df = pd.read_csv(FEATURED_DATA_PATH, parse_dates=["date_time"])
        logger.info("Loaded featured data: %d rows", len(df))
        return df
    except Exception:
        logger.error("Failed to load featured data", exc_info=True)
        raise


def _save_figure(fig: plt.Figure, filename: str) -> None:
    """Save figure to both figures/ and capstone_part2/figures/."""
    for directory in (FIGURES_DIR, PART2_FIGURES_DIR):
        path = os.path.join(directory, filename)
        fig.savefig(path, dpi=150, bbox_inches="tight")
        logger.info("Saved figure: %s", path)


def plot_traffic_by_hour(df: pd.DataFrame) -> None:
    """Average traffic by hour of day - weekday vs weekend."""
    fig, ax = plt.subplots(figsize=(10, 6))

    weekday = df[df["is_weekend"] == 0].groupby("hour")["traffic_volume"].mean()
    weekend = df[df["is_weekend"] == 1].groupby("hour")["traffic_volume"].mean()

    ax.plot(weekday.index, weekday.values, marker="o", label="Weekday", linewidth=2)
    ax.plot(weekend.index, weekend.values, marker="s", label="Weekend", linewidth=2)
    ax.set_xlabel("Hour of Day")
    ax.set_ylabel("Average Traffic Volume")
    ax.set_title("Average Traffic Volume by Hour of Day")
    ax.legend()
    ax.set_xticks(range(24))
    ax.grid(True, alpha=0.3)

    _save_figure(fig, "traffic_by_hour.png")
    plt.close(fig)


def plot_traffic_distribution(df: pd.DataFrame) -> None:
    """Traffic volume distribution histogram with congestion threshold."""
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.hist(df["traffic_volume"], bins=50, edgecolor="black", alpha=0.7, color="steelblue")
    ax.axvline(x=5500, color="red", linestyle="--", linewidth=2, label="Congestion Threshold (5500)")
    ax.set_xlabel("Traffic Volume")
    ax.set_ylabel("Frequency")
    ax.set_title("Traffic Volume Distribution")
    ax.legend()
    ax.grid(True, alpha=0.3)

    _save_figure(fig, "traffic_distribution.png")
    plt.close(fig)


def plot_temp_vs_traffic(df: pd.DataFrame) -> None:
    """Temperature vs traffic scatter plot colored by weather condition."""
    fig, ax = plt.subplots(figsize=(10, 6))

    top_weather = df["weather_main"].value_counts().head(5).index
    colors = plt.cm.Set2(np.linspace(0, 1, len(top_weather)))

    for weather, color in zip(top_weather, colors):
        subset = df[df["weather_main"] == weather]
        ax.scatter(subset["temp"], subset["traffic_volume"],
                   alpha=0.15, s=8, label=weather, color=color)

    ax.set_xlabel("Temperature (Kelvin)")
    ax.set_ylabel("Traffic Volume")
    ax.set_title("Temperature vs Traffic Volume by Weather Condition")
    ax.legend(markerscale=4)
    ax.grid(True, alpha=0.3)

    _save_figure(fig, "temp_vs_traffic.png")
    plt.close(fig)


def plot_traffic_heatmap(df: pd.DataFrame) -> None:
    """Heatmap of average traffic by day of week and hour."""
    fig, ax = plt.subplots(figsize=(12, 6))

    pivot = df.pivot_table(values="traffic_volume", index="day_of_week",
                           columns="hour", aggfunc="mean")

    day_labels = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    im = ax.imshow(pivot.values, aspect="auto", cmap="YlOrRd")
    ax.set_yticks(range(7))
    ax.set_yticklabels(day_labels)
    ax.set_xticks(range(24))
    ax.set_xticklabels(range(24))
    ax.set_xlabel("Hour of Day")
    ax.set_ylabel("Day of Week")
    ax.set_title("Average Traffic Volume by Day of Week and Hour")
    fig.colorbar(im, ax=ax, label="Avg Traffic Volume")

    _save_figure(fig, "traffic_heatmap.png")
    plt.close(fig)


def run_visualizations(debug: bool = False) -> None:
    """Generate all visualizations."""
    setup_logging(debug=debug)

    os.makedirs(FIGURES_DIR, exist_ok=True)
    os.makedirs(PART2_FIGURES_DIR, exist_ok=True)
    logger.info("Starting visualization generation")

    try:
        df = load_data()
        plot_traffic_by_hour(df)
        plot_traffic_distribution(df)
        plot_temp_vs_traffic(df)
        plot_traffic_heatmap(df)
        logger.info("All visualizations generated successfully")
    except Exception:
        logger.error("Visualization generation failed", exc_info=True)
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate traffic visualizations")
    parser.add_argument("--debug", action="store_true", help="Enable DEBUG logging")
    args = parser.parse_args()
    run_visualizations(debug=args.debug)
