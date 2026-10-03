"""
visualizations.py - Generate traffic analysis visualizations.
"""

import logging
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

FEATURED_DATA_PATH = os.path.join("capstone_part2", "featured_traffic.csv")
FIGURES_DIR = "figures"


def load_data() -> pd.DataFrame:
    """Load featured dataset."""
    try:
        df = pd.read_csv(FEATURED_DATA_PATH, parse_dates=["date_time"])
        logger.info("Loaded featured data: %d rows", len(df))
        return df
    except Exception:
        logger.error("Failed to load featured data", exc_info=True)
        raise


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

    path = os.path.join(FIGURES_DIR, "traffic_by_hour.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved figure: %s", path)
    # Interpretation: Weekday traffic shows clear morning (7-8am) and evening
    # (4-5pm) rush-hour peaks, while weekend traffic has a single midday plateau.


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

    path = os.path.join(FIGURES_DIR, "traffic_distribution.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved figure: %s", path)
    # Interpretation: The distribution is bimodal, with peaks near low and
    # high traffic. Most traffic falls below the 5500 congestion threshold.


def plot_temp_vs_traffic(df: pd.DataFrame) -> None:
    """Temperature vs traffic scatter plot colored by weather condition."""
    fig, ax = plt.subplots(figsize=(10, 6))

    # Use top 5 weather conditions for clarity
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

    path = os.path.join(FIGURES_DIR, "temp_vs_traffic.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved figure: %s", path)
    # Interpretation: Traffic volume is relatively stable across temperatures,
    # but drops notably at extreme cold. Weather condition has limited visual
    # impact on this relationship.


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

    path = os.path.join(FIGURES_DIR, "traffic_heatmap.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved figure: %s", path)
    # Interpretation: The heatmap confirms weekday rush-hour patterns
    # (Mon-Fri 7-8am, 4-5pm) and the absence of these peaks on weekends.


def run_visualizations() -> None:
    """Generate all visualizations."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(module)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    os.makedirs(FIGURES_DIR, exist_ok=True)
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
    run_visualizations()
