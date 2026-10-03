"""
Model monitoring module for Metro Interstate Traffic Volume prediction.

Simulates production monitoring by checking for prediction error drift
and feature distribution drift. Uses data_loader for featured data.

Time splits:
  - Train: 2012-2016
  - Holdout: 2016 (last 20% of train for baseline MAE)
  - Production: 2018
"""

import json
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.ensemble import GradientBoostingRegressor

from data_loader import load_featured_data, get_feature_columns, FIGURES_DIR

logger = logging.getLogger(__name__)


def check_feature_drift(train_data, test_data, features, alpha=0.05):
    """Run KS test on key features between training and production sets."""
    results = []
    for feat in features:
        stat, p_value = ks_2samp(train_data[feat].dropna(), test_data[feat].dropna())
        status = "ALERT" if p_value < alpha else "PASS"
        results.append({
            "feature": feat,
            "ks_statistic": stat,
            "p_value": p_value,
            "status": status,
        })
        logger.info("Feature drift [%s]: KS=%.4f, p=%.4f, status=%s", feat, stat, p_value, status)
    return results


def save_drift_visualization(train_data, prod_data, features, save_path):
    """Save histograms comparing train vs production distributions."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()

    for i, feat in enumerate(features):
        ax = axes[i]
        ax.hist(train_data[feat].dropna(), bins=50, alpha=0.5, label="Train (2012-2016)", density=True)
        ax.hist(prod_data[feat].dropna(), bins=50, alpha=0.5, label="Production (2018)", density=True)
        ax.set_title(f"{feat} Distribution")
        ax.set_xlabel(feat)
        ax.set_ylabel("Density")
        ax.legend()

    plt.suptitle("Feature Distribution Drift: Train vs Production (2018)", fontsize=14)
    plt.tight_layout()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Drift visualization saved to %s", save_path)


def run_monitoring() -> None:
    """Execute time-based drift monitoring."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    reports_dir = os.path.join(script_dir, "reports")
    report_path = os.path.join(reports_dir, "monitoring_report.txt")
    status_path = os.path.join(reports_dir, "monitoring_status.json")
    figure_path = os.path.join(FIGURES_DIR, "monitoring_drift.png")

    df = load_featured_data()
    feature_cols = get_feature_columns(df)

    df = df.sort_values("date_time").reset_index(drop=True)

    # Time-based splits
    train_mask = df["date_time"].dt.year <= 2016
    prod_mask = df["date_time"].dt.year == 2018
    df_train = df[train_mask].copy()
    df_prod = df[prod_mask].copy()

    logger.info("Training period: 2012-2016 (%d rows)", len(df_train))
    logger.info("Production period: 2018 (%d rows)", len(df_prod))

    if len(df_prod) == 0:
        logger.error("No 2018 data found for production monitoring")
        return

    X_train = df_train[feature_cols]
    y_train = df_train["traffic_volume"]
    X_prod = df_prod[feature_cols]
    y_prod = df_prod["traffic_volume"]

    # Holdout from end of training period (last 20% of 2016 data chronologically)
    from sklearn.model_selection import train_test_split as tts
    X_tr, X_holdout, y_tr, y_holdout = tts(
        X_train, y_train, test_size=0.2, random_state=42, shuffle=False
    )

    logger.info("Training GBR on %d samples, holdout %d samples", len(X_tr), len(X_holdout))
    model = GradientBoostingRegressor(n_estimators=200, max_depth=5, random_state=42)
    model.fit(X_tr, y_tr)

    holdout_pred = model.predict(X_holdout)
    holdout_mae = float(np.abs(y_holdout.values - holdout_pred).mean())

    prod_pred = model.predict(X_prod)
    prod_mae = float(np.abs(y_prod.values - prod_pred).mean())

    pct_change = (prod_mae - holdout_mae) / holdout_mae if holdout_mae > 0 else 0
    # ALERT only on degradation (error increase > 20%), not improvement
    drift_status = "ALERT" if pct_change > 0.20 else "PASS"
    drift_note = ""
    if pct_change < 0:
        drift_note = " (error decreased — no concern)"

    if drift_status == "ALERT":
        logger.warning(
            "Prediction error drift detected: holdout MAE=%.2f, production MAE=%.2f, change=%+.1f%%",
            holdout_mae, prod_mae, pct_change * 100,
        )
    else:
        logger.info(
            "Prediction error drift: holdout MAE=%.2f, production MAE=%.2f, change=%+.1f%%, status=%s%s",
            holdout_mae, prod_mae, pct_change * 100, drift_status, drift_note,
        )

    # Feature distribution drift: KS test on key features
    drift_features = ["temp", "clouds_all", "rain_1h", "hour"]
    feature_drift = check_feature_drift(df_train, df_prod, drift_features)

    for fd in feature_drift:
        if fd["status"] == "ALERT":
            logger.warning("Feature drift ALERT on %s: KS=%.4f, p=%.4f",
                           fd["feature"], fd["ks_statistic"], fd["p_value"])

    # Overall system status
    any_alert = drift_status == "ALERT" or any(fd["status"] == "ALERT" for fd in feature_drift)
    overall = "ALERT / Requires investigation" if any_alert else "PASS / Normal"

    if any_alert:
        logger.warning("Overall system status: %s", overall)
    else:
        logger.info("Overall system status: %s", overall)

    # Generate report
    lines = [
        "=" * 60,
        "MODEL MONITORING REPORT",
        "=" * 60,
        "",
        "Training period:    2012-2016",
        f"Production period:  2018 ({len(df_prod)} records)",
        "",
        "--- Prediction Error Drift ---",
        f"  Holdout MAE (2012-2016 tail):  {holdout_mae:.2f}",
        f"  Production MAE (2018):         {prod_mae:.2f}",
        f"  Change:                        {pct_change * 100:+.2f}%{drift_note}",
        f"  Status:                        [{drift_status}]",
        "",
        "--- Feature Distribution Drift (KS Test: train vs 2018) ---",
    ]
    for fd in feature_drift:
        lines.append(
            f"  {fd['feature']:20s}  KS={fd['ks_statistic']:.4f}  "
            f"p={fd['p_value']:.4f}  [{fd['status']}]"
        )
    lines.extend([
        "",
        "--- OVERALL SYSTEM STATUS ---",
        f"  {overall}",
        "",
        "=" * 60,
    ])
    report = "\n".join(lines)
    print(report)

    # Save report and status JSON
    os.makedirs(reports_dir, exist_ok=True)
    with open(report_path, "w") as f:
        f.write(report)
    logger.info("Monitoring report saved to %s", report_path)

    status_data = {
        "overall_status": "ALERT" if any_alert else "PASS",
        "overall_message": overall,
        "prediction_error_drift": {
            "status": drift_status,
            "holdout_mae": round(holdout_mae, 2),
            "production_mae": round(prod_mae, 2),
            "pct_change": round(pct_change * 100, 2),
        },
        "feature_drift": [
            {"feature": fd["feature"], "status": fd["status"],
             "ks_statistic": round(fd["ks_statistic"], 4),
             "p_value": round(fd["p_value"], 4)}
            for fd in feature_drift
        ],
    }
    with open(status_path, "w") as f:
        json.dump(status_data, f, indent=2)
    logger.info("Monitoring status JSON saved to %s", status_path)

    # Save drift visualization
    os.makedirs(FIGURES_DIR, exist_ok=True)
    save_drift_visualization(df_train, df_prod, drift_features, figure_path)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    try:
        run_monitoring()
    except FileNotFoundError as e:
        logger.error("Data file not found: %s", e)
        raise
    except Exception as e:
        logger.error("Monitoring failed: %s", e)
        raise
