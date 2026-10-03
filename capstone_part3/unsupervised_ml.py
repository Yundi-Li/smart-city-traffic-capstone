"""
Unsupervised Machine Learning Analysis for Metro Interstate Traffic Volume.

Performs K-Means clustering and Association Rule Mining on traffic data.
"""

import logging
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler

logger = logging.getLogger(__name__)


def load_data(filepath: str) -> pd.DataFrame:
    """Load the traffic volume dataset."""
    logger.info("Loading data from %s", filepath)
    df = pd.read_csv(filepath)
    df["date_time"] = pd.to_datetime(df["date_time"])
    df["hour"] = df["date_time"].dt.hour
    df["day_of_week"] = df["date_time"].dt.dayofweek
    logger.info("Loaded %d rows", len(df))
    return df


# ---------------------------------------------------------------------------
# K-Means Clustering
# ---------------------------------------------------------------------------

def run_kmeans(df: pd.DataFrame, figures_dir: str) -> pd.DataFrame:
    """Run K-Means clustering with elbow method, save plots, return labelled df."""
    features = ["hour", "temp", "traffic_volume", "clouds_all"]
    cluster_df = df[features].dropna().copy()
    logger.info("Clustering on %d rows after dropping NaNs", len(cluster_df))

    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(cluster_df)
    scaled_cols = [f"{c}_normalized" for c in features]
    scaled_df = pd.DataFrame(scaled, columns=scaled_cols, index=cluster_df.index)

    # Elbow method (k=2..8)
    k_range = range(2, 9)
    inertias = []
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(scaled_df)
        inertias.append(km.inertia_)
        logger.info("k=%d  inertia=%.2f", k, km.inertia_)

    plt.figure(figsize=(8, 5))
    plt.plot(list(k_range), inertias, "bo-")
    plt.xlabel("Number of Clusters (k)")
    plt.ylabel("Inertia")
    plt.title("Elbow Method for Optimal k")
    plt.xticks(list(k_range))
    plt.tight_layout()
    elbow_path = os.path.join(figures_dir, "elbow_method.png")
    plt.savefig(elbow_path, dpi=150)
    plt.close()
    logger.info("Saved elbow plot to %s", elbow_path)

    # Fit with k=4
    chosen_k = 4
    km_final = KMeans(n_clusters=chosen_k, random_state=42, n_init=10)
    cluster_df["cluster"] = km_final.fit_predict(scaled_df)

    # Cluster descriptions
    print("\n=== K-Means Cluster Descriptions (k=4) ===")
    for c in range(chosen_k):
        subset = cluster_df[cluster_df["cluster"] == c]
        print(f"\nCluster {c} (n={len(subset)}):")
        for feat in features:
            print(f"  {feat}: mean={subset[feat].mean():.2f}")

    # Scatter visualisation (hour vs traffic_volume coloured by cluster)
    plt.figure(figsize=(10, 6))
    for c in range(chosen_k):
        mask = cluster_df["cluster"] == c
        plt.scatter(
            cluster_df.loc[mask, "hour"],
            cluster_df.loc[mask, "traffic_volume"],
            label=f"Cluster {c}",
            alpha=0.4,
            s=10,
        )
    plt.xlabel("Hour of Day")
    plt.ylabel("Traffic Volume")
    plt.title("K-Means Clusters (k=4)")
    plt.legend()
    plt.tight_layout()
    cluster_path = os.path.join(figures_dir, "kmeans_clusters.png")
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
    """Discretize features, mine frequent itemsets, and extract association rules."""
    arm_df = df.copy()

    # Discretize
    arm_df["time_of_day"] = arm_df["hour"].apply(_categorize_hour)
    arm_df["day_type"] = arm_df["day_of_week"].apply(
        lambda d: "weekend" if d >= 5 else "weekday"
    )
    arm_df["weather"] = arm_df["weather_main"].apply(_categorize_weather)

    quartiles = arm_df["traffic_volume"].quantile([0.25, 0.5, 0.75])
    q1, q2, q3 = quartiles.iloc[0], quartiles.iloc[1], quartiles.iloc[2]

    def _congestion(vol):
        if vol < q1:
            return "Low"
        if vol < q2:
            return "Medium"
        if vol < q3:
            return "High"
        return "Severe"

    arm_df["congestion"] = arm_df["traffic_volume"].apply(_congestion)

    # Build transactions
    transactions = []
    for _, row in arm_df.iterrows():
        transactions.append([
            f"time={row['time_of_day']}",
            f"day={row['day_type']}",
            f"weather={row['weather']}",
            f"congestion={row['congestion']}",
        ])

    te = TransactionEncoder()
    te_array = te.fit_transform(transactions)
    te_df = pd.DataFrame(te_array, columns=te.columns_)

    logger.info("Mining frequent itemsets …")
    freq = apriori(te_df, min_support=0.01, use_colnames=True)
    logger.info("Found %d frequent itemsets", len(freq))

    rules = association_rules(freq, metric="lift", min_threshold=1.0)
    rules = rules.sort_values("lift", ascending=False)

    top10 = rules.head(10)
    print("\n=== Top 10 Association Rules by Lift ===")
    for i, (_, r) in enumerate(top10.iterrows(), 1):
        ant = ", ".join(sorted(r["antecedents"]))
        con = ", ".join(sorted(r["consequents"]))
        print(
            f"{i:2d}. {ant}  =>  {con}  "
            f"(support={r['support']:.4f}, confidence={r['confidence']:.4f}, lift={r['lift']:.4f})"
        )

    return rules


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    data_path = os.path.join(repo_root, "capstone_part2", "cleaned_traffic.csv")
    figures_dir = os.path.join(repo_root, "capstone_part3", "figures")
    os.makedirs(figures_dir, exist_ok=True)

    try:
        df = load_data(data_path)
        run_kmeans(df, figures_dir)
        run_association_rules(df)
        logger.info("Unsupervised ML analysis complete.")
    except FileNotFoundError:
        logger.error("Dataset not found at %s", data_path)
        sys.exit(1)
    except Exception:
        logger.exception("Unexpected error during analysis")
        sys.exit(1)


if __name__ == "__main__":
    main()
