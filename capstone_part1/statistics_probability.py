"""
Statistics & Probability Analysis for Smart City Traffic Capstone - Part 1
Computes descriptive statistics, correlation, and probability measures on I-94 traffic data.
"""

import csv
import math
import logging
from pathlib import Path
from collections import Counter

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / ".." / "data" / "Metro_Interstate_Traffic_Volume.csv"

# ---------------------------------------------------------------------------
# Helper functions (no numpy/scipy dependency)
# ---------------------------------------------------------------------------

def mean(values):
    return sum(values) / len(values)

def median(values):
    s = sorted(values)
    n = len(s)
    if n % 2 == 1:
        return s[n // 2]
    return (s[n // 2 - 1] + s[n // 2]) / 2

def variance(values, ddof=1):
    m = mean(values)
    return sum((x - m) ** 2 for x in values) / (len(values) - ddof)

def stdev(values, ddof=1):
    return math.sqrt(variance(values, ddof))

def correlation(xs, ys):
    n = len(xs)
    mx, my = mean(xs), mean(ys)
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / (n - 1)
    return cov / (stdev(xs) * stdev(ys))


def load_data(csv_path: Path) -> list[dict]:
    """Load CSV into a list of dicts with typed values."""
    logger.info("Loading data from %s", csv_path)
    rows = []
    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({
                "holiday": row["holiday"],
                "temp": float(row["temp"]),
                "rain_1h": float(row["rain_1h"]),
                "snow_1h": float(row["snow_1h"]),
                "clouds_all": int(row["clouds_all"]),
                "weather_main": row["weather_main"],
                "weather_description": row["weather_description"],
                "date_time": row["date_time"],
                "traffic_volume": int(row["traffic_volume"]),
            })
    logger.info("Loaded %d rows", len(rows))
    return rows


def task_2_1(data: list[dict]) -> None:
    """Task 2.1: Descriptive Statistics of traffic_volume."""
    print("\n" + "#" * 70)
    print("  TASK 2.1: Descriptive Statistics of Traffic Volume")
    print("#" * 70)

    volumes = [r["traffic_volume"] for r in data]
    n = len(volumes)
    m = mean(volumes)
    med = median(volumes)
    sd = stdev(volumes)
    var = variance(volumes)
    rng = max(volumes) - min(volumes)

    print(f"\n  Count:      {n:>12,}")
    print(f"  Mean:       {m:>12,.2f}")
    print(f"  Median:     {med:>12,.2f}")
    print(f"  Std Dev:    {sd:>12,.2f}")
    print(f"  Variance:   {var:>12,.2f}")
    print(f"  Min:        {min(volumes):>12,}")
    print(f"  Max:        {max(volumes):>12,}")
    print(f"  Range:      {rng:>12,}")

    print("\n--- Interpretation ---")
    print(
        f"The mean traffic volume ({m:,.0f} vehicles/hour) is slightly lower than the\n"
        f"  median ({med:,.0f}), suggesting left-skewness — a concentration of low-volume\n"
        f"  overnight hours pulls the mean downward. The large standard deviation ({sd:,.0f})\n"
        f"  and range ({rng:,}) indicate substantial variability in hourly traffic,\n"
        f"  reflecting the difference between overnight lulls and rush-hour peaks.\n"
        f"  This high variance is important for capacity planning: average-based\n"
        f"  models will under-predict peak demand."
    )


def task_2_2(data: list[dict]) -> None:
    """Task 2.2: Correlation between temperature and traffic volume."""
    print("\n" + "#" * 70)
    print("  TASK 2.2: Correlation — Temperature vs Traffic Volume")
    print("#" * 70)

    temps = [r["temp"] for r in data]
    volumes = [r["traffic_volume"] for r in data]
    r = correlation(temps, volumes)

    print(f"\n  Pearson correlation coefficient (r): {r:.4f}")

    # Interpret direction and strength
    direction = "positive" if r > 0 else "negative"
    abs_r = abs(r)
    if abs_r < 0.1:
        strength = "negligible"
    elif abs_r < 0.3:
        strength = "weak"
    elif abs_r < 0.5:
        strength = "moderate"
    elif abs_r < 0.7:
        strength = "strong"
    else:
        strength = "very strong"

    print(f"  Direction:  {direction}")
    print(f"  Strength:   {strength} (|r| = {abs_r:.4f})")

    print("\n--- Interpretation ---")
    print(
        f"The {strength} {direction} correlation (r = {r:.4f}) suggests that warmer\n"
        f"  temperatures are associated with somewhat higher traffic volumes. This is\n"
        f"  plausible in Minnesota where severe cold discourages travel.\n"
        f"\n"
        f"  However, correlation does not imply causation. The relationship may be\n"
        f"  confounded by seasonality: summer months bring both warmer temperatures\n"
        f"  AND more travel due to vacations, outdoor activities, and longer daylight\n"
        f"  hours. Temperature itself may not directly cause more driving; rather,\n"
        f"  both variables respond to seasonal patterns. A controlled study or\n"
        f"  regression model with seasonal indicators would be needed to isolate\n"
        f"  the independent effect of temperature on traffic volume."
    )


def task_3_1(data: list[dict]) -> None:
    """Task 3.1: Marginal and Joint Probabilities."""
    print("\n" + "#" * 70)
    print("  TASK 3.1: Probability — Marginal and Joint")
    print("#" * 70)

    n = len(data)
    congested = [r for r in data if r["traffic_volume"] > 5500]
    clear = [r for r in data if r["weather_main"] == "Clear"]
    congested_and_clear = [r for r in data if r["traffic_volume"] > 5500 and r["weather_main"] == "Clear"]

    p_congestion = len(congested) / n
    p_clear = len(clear) / n
    p_cong_and_clear = len(congested_and_clear) / n

    print(f"\n  Total observations:                  {n:>10,}")
    print(f"  Congested (volume > 5500):           {len(congested):>10,}")
    print(f"  Clear weather:                       {len(clear):>10,}")
    print(f"  Congested AND Clear weather:         {len(congested_and_clear):>10,}")
    print(f"\n  P(Congestion):                       {p_congestion:>10.4f}  ({p_congestion*100:.2f}%)")
    print(f"  P(Clear Weather):                    {p_clear:>10.4f}  ({p_clear*100:.2f}%)")
    print(f"  P(Congestion AND Clear Weather):     {p_cong_and_clear:>10.4f}  ({p_cong_and_clear*100:.2f}%)")

    print("\n--- Interpretation ---")
    print(
        f"About {p_congestion*100:.1f}% of all observed hours experience congestion (volume > 5500).\n"
        f"  Clear weather occurs in {p_clear*100:.1f}% of hours. The joint probability of\n"
        f"  congestion during clear weather is {p_cong_and_clear*100:.2f}%, which will be\n"
        f"  compared to conditional and independence expectations in Task 3.2."
    )

    return p_congestion, p_clear, p_cong_and_clear, congested, clear, congested_and_clear


def task_3_2(data, p_congestion, p_clear, p_cong_and_clear, congested, clear, congested_and_clear):
    """Task 3.2: Conditional Probability, Independence Test, Odds Ratio."""
    print("\n" + "#" * 70)
    print("  TASK 3.2: Conditional Probability & Independence")
    print("#" * 70)

    n = len(data)

    # P(Clear | Congestion)
    clear_given_congested = len(congested_and_clear) / len(congested) if congested else 0
    print(f"\n  P(Clear | Congestion) = {clear_given_congested:.4f}  ({clear_given_congested*100:.2f}%)")

    # P(High Temp | Congestion) where high temp > 292 K
    high_temp_threshold = 292.0
    congested_and_high_temp = [r for r in data if r["traffic_volume"] > 5500 and r["temp"] > high_temp_threshold]
    p_high_temp_given_congestion = len(congested_and_high_temp) / len(congested) if congested else 0
    print(f"  P(High Temp > 292K | Congestion) = {p_high_temp_given_congestion:.4f}  ({p_high_temp_given_congestion*100:.2f}%)")

    # Independence test: P(A ∩ B) vs P(A) × P(B)
    p_a_times_b = p_congestion * p_clear
    print(f"\n  Independence Test for Congestion and Clear Weather:")
    print(f"    P(Congestion) x P(Clear) = {p_a_times_b:.6f}")
    print(f"    P(Congestion AND Clear)  = {p_cong_and_clear:.6f}")
    diff = abs(p_cong_and_clear - p_a_times_b)
    print(f"    Difference:                {diff:.6f}")

    if diff < 0.005:
        independence = "approximately independent"
    else:
        independence = "NOT independent (dependent)"
    print(f"    Conclusion: The events are {independence}.")

    # Odds ratio: congestion in clear vs cloudy weather
    cloudy = [r for r in data if r["weather_main"] == "Clouds"]
    congested_and_cloudy = [r for r in data if r["traffic_volume"] > 5500 and r["weather_main"] == "Clouds"]

    # Odds of congestion in clear weather
    n_clear = len(clear)
    n_cong_clear = len(congested_and_clear)
    n_not_cong_clear = n_clear - n_cong_clear

    n_cloudy = len(cloudy)
    n_cong_cloudy = len(congested_and_cloudy)
    n_not_cong_cloudy = n_cloudy - n_cong_cloudy

    if n_not_cong_clear > 0 and n_not_cong_cloudy > 0:
        odds_clear = n_cong_clear / n_not_cong_clear
        odds_cloudy = n_cong_cloudy / n_not_cong_cloudy
        odds_ratio = odds_clear / odds_cloudy

        print(f"\n  Odds Ratio (Congestion: Clear vs Cloudy weather):")
        print(f"    Clear:  {n_cong_clear} congested / {n_not_cong_clear} not = odds {odds_clear:.4f}")
        print(f"    Cloudy: {n_cong_cloudy} congested / {n_not_cong_cloudy} not = odds {odds_cloudy:.4f}")
        print(f"    Odds Ratio = {odds_ratio:.4f}")

        if odds_ratio > 1:
            print(f"    Interpretation: Congestion is {odds_ratio:.2f}x more likely in clear weather than cloudy.")
        elif odds_ratio < 1:
            print(f"    Interpretation: Congestion is {1/odds_ratio:.2f}x more likely in cloudy weather than clear.")
        else:
            print("    Interpretation: Congestion odds are equal in both conditions.")
    else:
        print("\n  Odds ratio could not be computed (division by zero).")

    print("\n--- Summary ---")
    print(
        "The conditional probability analysis reveals whether weather conditions\n"
        "  meaningfully shift congestion risk. The independence test shows whether\n"
        "  clear weather and congestion co-occur at rates different from what pure\n"
        "  chance would predict. The odds ratio quantifies the relative risk of\n"
        "  congestion across weather types — useful for traffic management systems\n"
        "  that adjust signal timing or route recommendations based on conditions."
    )


def main():
    logger.info("Starting Statistics & Probability Analysis")

    csv_path = DATA_PATH.resolve()
    if not csv_path.exists():
        logger.error("CSV file not found at %s", csv_path)
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    data = load_data(csv_path)

    task_2_1(data)
    task_2_2(data)
    p_cong, p_clear, p_cong_clear, congested, clear, cong_clear = task_3_1(data)
    task_3_2(data, p_cong, p_clear, p_cong_clear, congested, clear, cong_clear)

    logger.info("Statistics & probability analysis complete")


if __name__ == "__main__":
    main()
