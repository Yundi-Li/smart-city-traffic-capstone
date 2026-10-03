"""
Unsupervised Machine Learning Analysis for Metro Interstate Traffic Volume.

Performs K-Means clustering and Association Rule Mining using the
Part 2 featured dataset via data_loader.
"""

import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import MinMaxScaler

from data_loader import load_featured_data, FIGURES_DIR

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# K-Means Clustering
# ---------------------------------------------------------------------------

def run_kmeans(df: pd.DataFrame) -> pd.DataFrame:
    """Run K-Means clustering on hour_sin, hour_cos, weather_severity, traffic_volume (all scaled)."""
    cluster_df = df[["hour", "hour_sin", "hour_cos", "weather_severity", "traffic_volume"]].dropna().copy()

    features = ["hour_sin", "hour_cos", "weather_severity", "traffic_volume"]
    logger.info("Clustering on %d rows with features: %s", len(cluster_df), features)

    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(cluster_df[features])

    # Elbow + silhouette (k=2..8)
    k_range = range(2, 9)
    inertias, silhouettes = [], []
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(scaled)
        inertias.append(km.inertia_)
        sil = silhouette_score(scaled, labels, sample_size=5000, random_state=42)
        silhouettes.append(sil)
        logger.info("k=%d  inertia=%.2f  silhouette=%.4f", k, km.inertia_, sil)

    best_k_sil = list(k_range)[np.argmax(silhouettes)]
    logger.info("Best k by silhouette: %d (score=%.4f)", best_k_sil, max(silhouettes))

    # Use k=4 for richer, more operationally useful clusters even if k=2 has
    # a higher silhouette score. Two clusters simply split "busy" vs "quiet",
    # whereas four clusters reveal distinct operational regimes (night lull,
    # shoulder hours, midday moderate, peak commute).
    chosen_k = 4
    logger.info("Using k=%d (richer operational clusters vs k=%d best silhouette)", chosen_k, best_k_sil)

    # Save elbow/silhouette plot
    os.makedirs(FIGURES_DIR, exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.plot(list(k_range), inertias, "bo-")
    ax1.set_xlabel("k")
    ax1.set_ylabel("Inertia")
    ax1.set_title("Elbow Method")
    ax2.plot(list(k_range), silhouettes, "ro-")
    ax2.set_xlabel("k")
    ax2.set_ylabel("Silhouette Score")
    ax2.set_title("Silhouette Analysis")
    plt.tight_layout()
    elbow_path = os.path.join(FIGURES_DIR, "elbow_method.png")
    plt.savefig(elbow_path, dpi=150)
    plt.close()
    logger.info("Saved elbow/silhouette plot to %s", elbow_path)

    # Fit with chosen k
    km_final = KMeans(n_clusters=chosen_k, random_state=42, n_init=10)
    cluster_df["cluster"] = km_final.fit_predict(scaled)

    # Sort clusters by mean traffic volume for consistent naming
    cluster_means = cluster_df.groupby("cluster")[["hour", "weather_severity", "traffic_volume"]].mean()
    cluster_means = cluster_means.sort_values("traffic_volume")
    rank = {old: new for new, old in enumerate(cluster_means.index)}
    cluster_df["cluster"] = cluster_df["cluster"].map(rank)

    # Cluster profiles with practical names
    cluster_profiles = cluster_df.groupby("cluster").agg(
        size=("hour", "size"),
        mean_hour=("hour", "mean"),
        mean_severity=("weather_severity", "mean"),
        mean_volume=("traffic_volume", "mean"),
    ).sort_index()

    cluster_names = {}
    for c, row in cluster_profiles.iterrows():
        h = row["mean_hour"]
        v = row["mean_volume"]
        if v < 1500:
            name = "Night lull"
            action = "Low-demand window suitable for roadwork and lane closures"
        elif v < 3000 and (h > 18 or h < 8):
            name = "Shoulder hours"
            action = "Transitional period; ramp metering can smooth flow"
        elif v >= 4500:
            name = "Peak commute"
            action = "Signal priority and congestion pricing most effective here"
        else:
            name = "Midday moderate"
            action = "Stable flow; standard signal timing sufficient"
        cluster_names[c] = (name, action)

    print("\n=== K-Means Cluster Profiles (k=4) ===")
    print(f"{'Cluster':<10} {'Name':<30} {'Size':>6} {'Avg Hour':>9} {'Avg Severity':>13} {'Avg Volume':>11}")
    print("-" * 90)
    for c, row in cluster_profiles.iterrows():
        name, action = cluster_names[c]
        print(f"{c:<10} {name:<30} {int(row['size']):>6} {row['mean_hour']:>9.1f} {row['mean_severity']:>13.2f} {row['mean_volume']:>11.0f}")
        print(f"{'':>10} -> {action}")

    # Scatter plot
    plt.figure(figsize=(10, 6))
    for c in range(chosen_k):
        mask = cluster_df["cluster"] == c
        name = cluster_names[c][0]
        plt.scatter(
            cluster_df.loc[mask, "hour"], cluster_df.loc[mask, "traffic_volume"],
            label=f"C{c}: {name}", alpha=0.4, s=10,
        )
    plt.xlabel("Hour of Day")
    plt.ylabel("Traffic Volume")
    plt.title("K-Means Traffic Condition Clusters (k=4)")
    plt.legend(fontsize=8)
    plt.tight_layout()
    cluster_path = os.path.join(FIGURES_DIR, "kmeans_clusters.png")
    plt.savefig(cluster_path, dpi=150)
    plt.close()
    logger.info("Saved cluster plot to %s", cluster_path)

    return cluster_df


# ---------------------------------------------------------------------------
# Association Rule Mining
# ---------------------------------------------------------------------------

def _categorize_hour(hour: int) -> str:
    if 6 <= hour < 12:
        return "morning"
    if 12 <= hour < 17:
        return "afternoon"
    if 17 <= hour < 21:
        return "evening"
    return "night"


def _categorize_weather(weather_main: str) -> str:
    w = str(weather_main).lower()
    if w in ("clear",):
        return "clear"
    if w in ("clouds", "mist", "haze", "fog", "smoke"):
        return "cloudy"
    if w in ("rain", "drizzle", "thunderstorm"):
        return "rain"
    if w in ("snow",):
        return "snow"
    return "other"


def run_association_rules(df: pd.DataFrame) -> pd.DataFrame:
    """Discretize features, mine frequent itemsets, and extract association rules.

    Filters rules so consequent contains only congestion_category items.
    Reports top 5 by lift.
    """
    arm_df = df.copy()

    arm_df["time_of_day"] = arm_df["hour"].apply(_categorize_hour)
    arm_df["day_type"] = arm_df["is_weekend"].apply(
        lambda x: "weekend" if x >= 1 else "weekday"
    )
    arm_df["weather"] = arm_df["weather_main"].apply(_categorize_weather)

    # Use the existing congestion_category from featured data
    congestion_col = "congestion_category"
    if congestion_col not in arm_df.columns:
        # Fallback: compute from quartiles
        quartiles = arm_df["traffic_volume"].quantile([0.25, 0.5, 0.75])
        q1, q2, q3 = quartiles.iloc[0], quartiles.iloc[1], quartiles.iloc[2]
        arm_df[congestion_col] = pd.cut(
            arm_df["traffic_volume"],
            bins=[-np.inf, q1, q2, q3, np.inf],
            labels=["Low", "Medium", "High", "Severe"],
        )

    # Build transactions
    transactions = []
    for _, row in arm_df.iterrows():
        transactions.append([
            f"time={row['time_of_day']}",
            f"day={row['day_type']}",
            f"weather={row['weather']}",
            f"congestion={row[congestion_col]}",
        ])

    te = TransactionEncoder()
    te_array = te.fit_transform(transactions)
    te_df = pd.DataFrame(te_array, columns=te.columns_)

    logger.info("Mining frequent itemsets ...")
    freq = apriori(te_df, min_support=0.01, use_colnames=True)
    logger.info("Found %d frequent itemsets", len(freq))

    rules = association_rules(freq, metric="lift", min_threshold=1.0)
    rules = rules.sort_values("lift", ascending=False)

    # Filter: consequent must contain only congestion_category items
    congestion_items = {c for c in te_df.columns if c.startswith("congestion=")}
    filtered = rules[rules["consequents"].apply(
        lambda cs: len(cs) > 0 and cs.issubset(congestion_items)
    )]
    filtered = filtered.sort_values("lift", ascending=False)

    top5 = filtered.head(5)
    print("\n=== Top 5 Association Rules by Lift (consequent = congestion) ===")
    for i, (_, r) in enumerate(top5.iterrows(), 1):
        ant = ", ".join(sorted(r["antecedents"]))
        con = ", ".join(sorted(r["consequents"]))
        print(
            f"{i:2d}. {ant}  =>  {con}  "
            f"(support={r['support']:.4f}, confidence={r['confidence']:.4f}, lift={r['lift']:.4f})"
        )

    return filtered


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    os.makedirs(FIGURES_DIR, exist_ok=True)

    try:
        df = load_featured_data()
        run_kmeans(df)
        run_association_rules(df)
        logger.info("Unsupervised ML analysis complete.")
    except FileNotFoundError:
        logger.error("Featured dataset not found. Run Part 2 pipeline first.")
        sys.exit(1)
    except Exception:
        logger.exception("Unexpected error during analysis")
        sys.exit(1)


if __name__ == "__main__":
    main()
