"""
app.py - Command-line mini-application for querying I-94 traffic data.
"""

import argparse
import logging
import os
import sys

import pandas as pd

import sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); from logging_config import setup_logging

logger = logging.getLogger(__name__)

FEATURED_DATA_PATH = os.path.join("capstone_part2", "featured_traffic.csv")

HELP_TEXT = """
Available commands:
  traffic <date>              Show hourly traffic for a date (e.g. 2017-06-15)
  peak <weekday|weekend>      Top 5 high-traffic hours for weekday or weekend
  compare <month>             Compare weekday vs weekend avg traffic for month (1-12)
  recommend <day_type> <weather>  Recommend best travel times (day_type: weekday/weekend)
  help                        Show this help message
  quit                        Exit the application
"""


def load_data() -> pd.DataFrame:
    """Load the featured dataset."""
    try:
        df = pd.read_csv(FEATURED_DATA_PATH, parse_dates=["date_time"])
        logger.info("Loaded data: %d rows", len(df))
        return df
    except Exception:
        logger.error("Failed to load data from %s", FEATURED_DATA_PATH, exc_info=True)
        raise


def cmd_traffic(df: pd.DataFrame, date_str: str) -> None:
    """Show hourly traffic for a specific date."""
    try:
        target_date = pd.to_datetime(date_str).date()
    except ValueError:
        logger.error("Invalid date format: %s (expected YYYY-MM-DD)", date_str)
        print(f"Error: Invalid date format '{date_str}'. Use YYYY-MM-DD.")
        return

    logger.info("Querying traffic for date: %s", target_date)
    day_data = df[df["date_time"].dt.date == target_date].sort_values("date_time")

    if day_data.empty:
        print(f"No data found for {target_date}")
        return

    print(f"\nHourly traffic for {target_date}:")
    print(f"{'Hour':>6}  {'Volume':>8}  {'Weather':>15}  {'Temp (K)':>10}")
    print("-" * 45)
    for _, row in day_data.iterrows():
        print(f"{row['hour']:>6}  {row['traffic_volume']:>8}  "
              f"{row['weather_main']:>15}  {row['temp']:>10.1f}")


def cmd_peak(df: pd.DataFrame, day_type: str) -> None:
    """Identify top 5 high-traffic hours."""
    if day_type not in ("weekday", "weekend"):
        logger.error("Invalid day type: %s", day_type)
        print("Error: day_type must be 'weekday' or 'weekend'.")
        return

    logger.info("Finding peak hours for: %s", day_type)
    is_weekend = 1 if day_type == "weekend" else 0
    subset = df[df["is_weekend"] == is_weekend]
    hourly_avg = subset.groupby("hour")["traffic_volume"].mean().sort_values(ascending=False)

    print(f"\nTop 5 high-traffic hours ({day_type}):")
    print(f"{'Hour':>6}  {'Avg Volume':>12}")
    print("-" * 22)
    for hour, vol in hourly_avg.head(5).items():
        print(f"{hour:>6}  {vol:>12.0f}")


def cmd_compare(df: pd.DataFrame, month_str: str) -> None:
    """Compare weekday vs weekend average traffic for a given month."""
    try:
        month = int(month_str)
        if not 1 <= month <= 12:
            raise ValueError("Month out of range")
    except ValueError:
        logger.error("Invalid month: %s", month_str)
        print("Error: month must be an integer 1-12.")
        return

    logger.info("Comparing traffic for month: %d", month)
    month_data = df[df["date_time"].dt.month == month]

    if month_data.empty:
        print(f"No data found for month {month}")
        return

    weekday_avg = month_data[month_data["is_weekend"] == 0]["traffic_volume"].mean()
    weekend_avg = month_data[month_data["is_weekend"] == 1]["traffic_volume"].mean()

    month_name = pd.Timestamp(2024, month, 1).strftime("%B")
    print(f"\nTraffic comparison for {month_name}:")
    print(f"  Weekday average: {weekday_avg:,.0f}")
    print(f"  Weekend average: {weekend_avg:,.0f}")
    print(f"  Difference:      {weekday_avg - weekend_avg:+,.0f}")


def cmd_recommend(df: pd.DataFrame, day_type: str, weather: str) -> None:
    """Recommend best travel times given day type and weather."""
    if day_type not in ("weekday", "weekend"):
        logger.error("Invalid day type: %s", day_type)
        print("Error: day_type must be 'weekday' or 'weekend'.")
        return

    logger.info("Generating recommendations for %s, weather=%s", day_type, weather)
    is_weekend = 1 if day_type == "weekend" else 0
    subset = df[(df["is_weekend"] == is_weekend) & (df["weather_main"].str.lower() == weather.lower())]

    if subset.empty:
        print(f"No data for {day_type} with weather '{weather}'.")
        available = df["weather_main"].unique()
        print(f"Available weather types: {', '.join(sorted(available))}")
        return

    hourly_avg = subset.groupby("hour")["traffic_volume"].mean()
    # Restrict to realistic travel hours (06:00-22:00)
    realistic = hourly_avg[(hourly_avg.index >= 6) & (hourly_avg.index <= 22)]
    if realistic.empty:
        realistic = hourly_avg
    best_hours = realistic.sort_values().head(5)
    peak_vol = realistic.max()
    peak_hour = realistic.idxmax()

    print(f"\nBest travel times ({day_type}, {weather} weather, 06:00-22:00):")
    print(f"{'Hour':>6}  {'Avg Volume':>12}  {'vs Peak'}")
    print("-" * 45)
    for hour, vol in best_hours.items():
        pct = (1 - vol / peak_vol) * 100 if peak_vol > 0 else 0
        print(f"{hour:02d}:00  {vol:>12,.0f}  {pct:.0f}% below {peak_hour:02d}:00 peak")


def run_app(debug: bool = False) -> None:
    """Run the interactive command-line application."""
    setup_logging(debug=debug)

    try:
        df = load_data()
    except Exception:
        print("Failed to load data. Run pipeline.py and feature_engineering.py first.")
        sys.exit(1)

    print("I-94 Traffic Analysis CLI")
    print("Type 'help' for available commands, 'quit' to exit.")

    while True:
        try:
            user_input = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue

        parts = user_input.split()
        command = parts[0].lower()

        try:
            if command == "quit":
                print("Goodbye!")
                break
            elif command == "help":
                print(HELP_TEXT)
            elif command == "traffic":
                if len(parts) < 2:
                    print("Usage: traffic <date>  (e.g. traffic 2017-06-15)")
                else:
                    cmd_traffic(df, parts[1])
            elif command == "peak":
                if len(parts) < 2:
                    print("Usage: peak <weekday|weekend>")
                else:
                    cmd_peak(df, parts[1].lower())
            elif command == "compare":
                if len(parts) < 2:
                    print("Usage: compare <month>  (1-12)")
                else:
                    cmd_compare(df, parts[1])
            elif command == "recommend":
                if len(parts) < 3:
                    print("Usage: recommend <weekday|weekend> <weather>")
                else:
                    cmd_recommend(df, parts[1].lower(), parts[2])
            else:
                print(f"Unknown command: '{command}'. Type 'help' for available commands.")
                logger.error("Unknown command: %s", command)
        except Exception:
            logger.error("Error processing command: %s", user_input, exc_info=True)
            print("An error occurred processing your command. Check logs for details.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="I-94 Traffic Analysis CLI")
    parser.add_argument("--debug", action="store_true", help="Enable DEBUG logging")
    args = parser.parse_args()
    run_app(debug=args.debug)
